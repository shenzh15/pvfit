"""Analyze truth-plane track residuals after a residual chi2 selection.

The denominators remain the reported POCA uncertainties, as in the fitter.
These are scaled residuals, not pulls using a fully propagated covariance.
"""

import json
from numbers import Real
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from ..config_loader import load_config
from .utils import mean_with_error, std_with_error

C_MM_PER_NS = 299.792458
DATA_FIELDS = (
    "pull_x",
    "pull_y",
    "pull_t",
    "error_x",
    "error_y",
    "error_t",
    "residual_x",
    "residual_y",
    "residual_t",
    "delta_z",
    "chi2",
)


def _collect_track_pulls(hdf5_path):
    """Keep all finite, key-associated tracks with positive POCA variances."""
    chunks = {name: [] for name in DATA_FIELDS}
    counts = {"n_associated": 0, "n_invalid": 0}
    with h5py.File(hdf5_path, "r") as f:
        for event_id, event in f.items():
            tracks = event["tracks"]
            truth = event["true_vertices"]
            # Use persisted key membership, not an origin-coordinate lookup.
            owner = np.full(len(tracks["x"]), -1, dtype=np.int64)
            for vertex_index, indices in enumerate(truth["track_indices"]):
                owner[indices] = vertex_index
            indices = np.flatnonzero(owner >= 0)
            counts["n_associated"] += len(indices)
            if not len(indices):
                continue
            missing = sorted({"tx", "ty", "backward"} - set(tracks))
            if missing:
                raise ValueError(
                    f"Event {event_id}: track pull analysis requires tracks/{', '.join(missing)} "
                    "for transport to the truth z plane. This HDF5 predates those fields; "
                    "re-run reconstruction with the updated code "
                    "(pvfit fit --config <your-config.yaml>)."
                )

            def state(name, tracks=tracks, indices=indices):
                # Promote saved float32 states before transport and subtraction.
                return tracks[name][:][indices].astype(np.float64)

            matched = owner[indices]
            target = {
                name: truth[name][:][matched].astype(np.float64)
                for name in ("x", "y", "z", "t")
            }
            dz = target["z"] - state("z")
            tx, ty = state("tx"), state("ty")
            backward = state("backward")
            time_slope = (
                np.where(backward, -1.0, 1.0) * np.sqrt(1 + tx**2 + ty**2) / C_MM_PER_NS
            )
            residual = np.column_stack(
                [
                    state("x") + dz * tx - target["x"],
                    state("y") + dz * ty - target["y"],
                    state("t") + dz * time_slope - target["t"],
                ]
            )
            covariance = np.column_stack(
                [state("cov_xx"), state("cov_yy"), state("cov_tt")]
            )
            valid = (
                np.isfinite(residual).all(axis=1)
                & np.isfinite(covariance).all(axis=1)
                & (covariance > 0).all(axis=1)
                & np.isfinite(dz)
                & np.isfinite(backward)
            )
            counts["n_invalid"] += int((~valid).sum())
            error = np.sqrt(covariance[valid])
            residual = residual[valid]
            scaled = residual / error
            for column, name in enumerate(("x", "y", "t")):
                chunks[f"residual_{name}"].append(residual[:, column])
                chunks[f"error_{name}"].append(error[:, column])
                chunks[f"pull_{name}"].append(scaled[:, column])
            chunks["delta_z"].append(dz[valid])
            chunks["chi2"].append(np.sum(scaled**2, axis=1))
    return {
        name: np.concatenate(values) if values else np.empty(0)
        for name, values in chunks.items()
    }, counts


def _statistics(values):
    """Report full-sample statistics, irrespective of plotting limits.

    ``mean_error`` is s/√n and ``std_error`` comes from the fourth central
    moment (see :func:`~pvfit.analysis.utils.std_with_error`); both are
    ``None`` below two entries. ``std`` itself is the population form (ddof=0)
    of the collected sample.
    """
    if not len(values):
        return {
            "n": 0,
            "mean": None,
            "mean_error": None,
            "std": None,
            "std_error": None,
            "central68_half_width": None,
        }
    low, high = np.quantile(values, [0.16, 0.84])
    mean, mean_error = mean_with_error(values)
    std_error = std_with_error(values)[1] if len(values) > 1 else None
    return {
        "n": len(values),
        "mean": mean,
        "mean_error": mean_error,
        "std": float(np.std(values)),
        "std_error": std_error,
        "central68_half_width": float((high - low) / 2),
    }


def _resolve_max_chi2(hdf5_path, override=None):
    """Read the cut used by reconstruction; an explicit analysis override wins."""
    value = override
    source = "explicit analysis override"
    if value is None:
        with h5py.File(hdf5_path, "r") as f:
            value = f.attrs.get("reconstruction_max_chi2")
        source = "HDF5 attribute reconstruction_max_chi2"
        if value is None:
            snapshot = Path(hdf5_path).expanduser().with_name("reconstruction.yaml")
            if not snapshot.is_file():
                raise ValueError(
                    "Cannot determine the reconstruction max_chi2: HDF5 has no "
                    "reconstruction_max_chi2 attribute and no reconstruction.yaml "
                    "exists beside it. Supply the original reconstruction.yaml "
                    "snapshot or set analysis.track_pull_chi2 explicitly."
                )
            algorithm = load_config(snapshot).get("algorithm", {})
            value = algorithm.get("max_chi2") if isinstance(algorithm, dict) else None
            source = f"{snapshot}: algorithm.max_chi2"
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{source} must specify a positive finite max_chi2")
    return float(value), source


def analyze_track_pull(hdf5_path, output_dir, max_chi2=None):
    """Save truth-plane residual diagnostics and full-sample statistics.

    Keep finite key-associated tracks with positive variances and
    pull_x**2 + pull_y**2 + pull_t**2 <= max_chi2 AFTER transport to truth z.
    Covariances remain at POCA; time transport assumes beta = 1, as in the
    fitter. This fixed truth-residual selection uses no Tukey weights.
    Old HDF5 files missing track directions must be regenerated.
    By default, read the reconstruction threshold from HDF5 metadata, or from
    reconstruction.yaml beside legacy HDF5 files. A numeric max_chi2 overrides it.

    Returns:
        dict: Counts and full-sample statistics also written to the JSON summary.
    """
    max_chi2, threshold_source = _resolve_max_chi2(hdf5_path, max_chi2)
    print("\nStarting truth-plane input track residual analysis...")
    print(f"Residual chi2 <= {max_chi2:g}; threshold from {threshold_source}")
    data, counts = _collect_track_pulls(hdf5_path)
    n_valid = len(data["chi2"])
    selected = data["chi2"] <= max_chi2
    n_selected = int(selected.sum())
    data = {name: values[selected] for name, values in data.items()}
    summary = {
        "normalization": "Truth-plane residual / reported POCA uncertainty; covariance not propagated",
        "transport": "Straight line to matched truth z; forward/backward sign; beta = 1",
        "matching": "Stored key-based true_vertices/track_indices",
        "selection": "pull_x**2 + pull_y**2 + pull_t**2 <= max_chi2, evaluated at truth z; no Tukey weights",
        "max_chi2": max_chi2,
        "max_chi2_source": threshold_source,
        "statistics_scope": "All tracks passing the residual chi2 selection, including values outside plot limits",
        **counts,
        "n_valid": n_valid,
        "n_selected": n_selected,
        "n_rejected_by_chi2": n_valid - n_selected,
        "statistics": {name: _statistics(values) for name, values in data.items()},
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "pull_analysis_input_track_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    _plot_track_pulls(data, output, max_chi2=max_chi2)
    print(
        f"Key-associated: {counts['n_associated']}; invalid: {counts['n_invalid']}; "
        f"valid: {n_valid}; chi2 <= {max_chi2:g}: {n_selected}; "
        f"rejected: {n_valid - n_selected}"
    )
    print(f"Track residual plots and summary saved to {output}/")
    return summary


def _pm(value, error):
    """``value ± error`` for titles; the error is omitted when unavailable."""
    return f"{value:.3g}" if error is None else f"{value:.3g} ± {error:.2g}"


def _plot_track_pulls(data, output_dir, *, max_chi2):
    """Show the core, label overflow, and retain tails in the displayed statistics."""
    fig, axes = plt.subplots(3, 4, figsize=(20, 13), layout="constrained")
    fig.suptitle(
        f"Tracks transported to truth z | Residual chi2 <= {max_chi2:g} | N = {len(data['chi2']):,}\n"
        "Scaled by reported POCA uncertainties (covariance not propagated)",
        fontsize=15,
    )

    def histogram(ax, name, label, limits=None):
        values = data[name]
        ax.set_xlabel(label)
        ax.set_ylabel("Probability density")
        ax.grid(True, alpha=0.25)
        if not len(values):
            ax.text(
                0.5,
                0.5,
                "No tracks pass the selection",
                ha="center",
                transform=ax.transAxes,
            )
            return
        counts, edges = np.histogram(values, bins=60, range=limits)
        # Normalize by ALL entries, not just entries within the display window.
        ax.stairs(counts / (len(values) * np.diff(edges)), edges, fill=True, alpha=0.6)
        stats = _statistics(values)
        outside = 100 * (1 - counts.sum() / len(values))
        ax.set_title(
            f"Mean {_pm(stats['mean'], stats['mean_error'])}\n"
            f"std {_pm(stats['std'], stats['std_error'])} | "
            f"68% half-width {stats['central68_half_width']:.3g}\n"
            f"Outside display: {outside:.2f}%",
            fontsize=10,
        )
        if limits is not None:
            ax.set_xlim(limits)

    for column, name in enumerate(("x", "y", "t")):
        unit = "ns" if name == "t" else "mm"
        limits = (-0.15, 0.15) if name == "t" else (-0.75, 0.75)
        histogram(
            axes[0, column],
            f"residual_{name}",
            f"Residual {name} at truth z [{unit}]",
            limits,
        )
        histogram(
            axes[1, column], f"error_{name}", f"Reported POCA sigma({name}) [{unit}]"
        )
        histogram(
            axes[2, column],
            f"pull_{name}",
            f"Residual {name} / POCA sigma({name})",
            (-5, 5),
        )
    histogram(axes[0, 3], "delta_z", "Truth z - POCA z [mm]", (-5, 5))
    histogram(axes[1, 3], "chi2", "Sum of squared scaled residuals", (0, 30))
    axes[2, 3].axis("off")
    axes[2, 3].text(
        0.0,
        0.9,
        "Statistics include all selected tracks,\nincluding values outside the\n"
        "displayed ranges.\n\n"
        "68% half-width = (q84 - q16) / 2.\n\n"
        "The residual chi2 cut includes x, y, t.\n"
        "No Tukey weights are applied.",
        va="top",
        fontsize=11,
        transform=axes[2, 3].transAxes,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(output / f"pull_analysis_input_track.{extension}", dpi=120)
    plt.close(fig)
