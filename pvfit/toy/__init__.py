"""Independent parametric toy data for exercising the PVfit workflow.

Profiles contain aggregate quantiles and correlations, never original events;
the bundled one (profile.json, loaded by default_profile) was fitted to a
100-event reference sample with calibrate_profile. Track states obey the same
straight-line transport used by the PV fitter.
"""

import json
from importlib import resources
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot
from scipy.special import ndtr, ndtri
from scipy.stats import rankdata

from ..fit.truth_matching import read_mcpv_dataframe

C_MM_PER_NS = 299.792458


def default_profile():
    """The profile shipped with the package."""
    return json.loads(resources.files(__package__).joinpath("profile.json").read_text())


MOMENTUM_BIN_EDGES = np.array([0, 500, 1000, 2000, 5000, 10000, np.inf])
PROBABILITIES = np.array(
    [
        0.001,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        0.95,
        0.975,
        0.99,
        0.995,
        0.999,
    ]
)
FEATURES = [
    "log_theta",
    "log_p",
    "log_sigma_xy",
    "log_sigma_t",
    "log_chi2",
    "pull_x",
    "pull_y",
    "pull_t",
]


def fit_distribution(values, discrete_columns=()):
    """Gaussian rank copula with piecewise quantile marginals (bounded tails)."""
    values = np.asarray(values, dtype=float)
    if len(values) < 10 or not np.all(np.isfinite(values)):
        raise ValueError("Need >=10 finite rows to calibrate a distribution")
    normal = np.column_stack([ndtri((rankdata(v) - 0.5) / len(v)) for v in values.T])
    corr = np.corrcoef(normal.T)
    distribution = {
        "quantiles": np.quantile(values, PROBABILITIES, axis=0).round(6).tolist(),
        "normal_correlation": corr.round(6).tolist(),
        "discrete_marginals": {},
    }
    for column in discrete_columns:
        levels, counts = np.unique(values[:, column], return_counts=True)
        distribution["discrete_marginals"][str(column)] = {
            "values": levels.round(6).tolist(),
            "cumulative_probability": (np.cumsum(counts) / counts.sum()).tolist(),
        }
    return distribution


def sample_distribution(rng, distribution, size):
    corr = np.asarray(distribution["normal_correlation"])
    # Rounding can introduce a tiny negative eigenvalue; project back to PSD.
    values, vectors = np.linalg.eigh(corr)
    normals = (
        rng.standard_normal((size, len(corr)))
        @ (vectors * np.sqrt(np.maximum(values, 0))).T
    )
    probabilities = ndtr(normals)
    table = np.asarray(distribution["quantiles"])
    sampled = np.column_stack(
        [
            np.interp(probabilities[:, i], PROBABILITIES, table[:, i])
            for i in range(table.shape[1])
        ]
    )
    for column, marginal in distribution.get("discrete_marginals", {}).items():
        i = int(column)
        indices = np.searchsorted(
            marginal["cumulative_probability"], probabilities[:, i]
        )
        sampled[:, i] = np.asarray(marginal["values"])[
            np.minimum(indices, len(marginal["values"]) - 1)
        ]
    return sampled


def fit_momentum_distribution(features):
    """Fit joint feature distributions within momentum bands, retaining their rates.

    A single copula across all momenta smears out the strong momentum dependence
    of residual widths. Sample a band first, then its joint feature distribution.
    """
    features = np.asarray(features, dtype=float)
    if len(features) < 10 or not np.all(np.isfinite(features)):
        raise ValueError("Need >=10 finite tracks for momentum calibration")
    momentum = np.exp(features[:, 1])
    bins = []
    for low, high in zip(MOMENTUM_BIN_EDGES[:-1], MOMENTUM_BIN_EDGES[1:], strict=False):
        selected = (momentum >= low) & (momentum < high)
        count = int(selected.sum())
        if not count:
            continue
        if count < 10:
            raise ValueError(
                f"Only {count} tracks in momentum band [{low:g}, {high:g}) MeV; "
                "need >=10 to calibrate its residual distribution"
            )
        bins.append(
            {
                "p_min_MeV": float(low),
                "p_max_MeV": float(high) if np.isfinite(high) else None,
                "probability": count / len(features),
                "distribution": fit_distribution(
                    features[selected], discrete_columns=(3,)
                ),
            }
        )
    return {"momentum_bins": bins}


def sample_momentum_distribution(rng, distribution, size):
    """Draw new tracks from the aggregate, momentum-conditioned profile."""
    bins = distribution["momentum_bins"]
    selected_bins = rng.choice(len(bins), size=size, p=[b["probability"] for b in bins])
    features = np.empty((size, len(FEATURES)))
    for index, band in enumerate(bins):
        selected = selected_bins == index
        if selected.any():
            features[selected] = sample_distribution(
                rng, band["distribution"], int(selected.sum())
            )
    return features


def track_features(tracks):
    """Transport to the true origin and scale by POCA errors (not full-covariance pulls)."""
    theta = np.hypot(tracks.beamPOCA_tx, tracks.beamPOCA_ty)
    sigma_xy = np.sqrt(tracks.c00)
    sigma_t = np.sqrt(tracks.c55)
    dz = tracks.ovtx_z - tracks.beamPOCA_z
    time_slope = np.where(tracks.backward, -1, 1) * np.sqrt(1 + theta**2) / C_MM_PER_NS
    return np.column_stack(
        [
            np.log(theta),
            np.log(tracks.p),
            np.log(sigma_xy),
            np.log(sigma_t),
            np.log(np.maximum(tracks.chi2, 1e-6)),
            (tracks.beamPOCA_x + dz * tracks.beamPOCA_tx - tracks.ovtx_x) / sigma_xy,
            (tracks.beamPOCA_y + dz * tracks.beamPOCA_ty - tracks.ovtx_y) / sigma_xy,
            (tracks.beamPOCA_t + dz * time_slope - tracks.ovtx_t) / sigma_t,
        ]
    )


def calibrate_profile(input_path):
    with uproot.open(input_path) as root:
        tracks = pd.DataFrame(
            root["velo_kalman_filter/monitor_tree"].arrays(library="np")
        )
        vertices = read_mcpv_dataframe(root["mc_pv_dumper/MCPV_effective_tree"])
    owners = {
        (int(event), int(key))
        for event, keys in zip(vertices.event_number, vertices.track_keys, strict=False)
        for key in keys
    }
    prompt = np.array(
        [
            (int(event), int(key)) in owners
            for event, key in zip(tracks.event_number, tracks.mc_key, strict=False)
        ]
    )
    points = vertices[["x", "y", "z", "time"]].to_numpy()
    event_counts = (
        vertices.groupby("event_number")
        .size()
        .reindex(tracks.event_number.unique(), fill_value=0)
    )
    multiplicity = vertices.n_valid_tracks.to_numpy()
    profile = {
        "schema_version": 2,
        "description": "Synthetic demonstration profile calibrated from aggregate key-enabled sample statistics; not detector simulation.",
        "calibration_events": int(tracks.event_number.nunique()),
        "units": {"position": "mm", "time": "ns", "momentum": "MeV"},
        "vertex_mean": points.mean(axis=0).round(6).tolist(),
        "vertex_covariance": np.cov(points.T).round(8).tolist(),
        "vertices_per_event": {
            "mean": float(event_counts.mean()),
            "variance": float(event_counts.var()),
        },
        "tracks_per_vertex": {
            "minimum": 4,
            "mean": float(multiplicity.mean()),
            "variance": float(multiplicity.var()),
        },
        "background_per_prompt_track": float((~prompt).sum() / prompt.sum()),
        "quantile_probabilities": PROBABILITIES.tolist(),
        "track_features": FEATURES,
        "populations": {},
        "background_origins": fit_distribution(
            tracks.loc[~prompt, ["ovtx_x", "ovtx_y", "ovtx_z", "ovtx_t"]]
        ),
        "background_from_pv_fraction": float(tracks.loc[~prompt, "fromPV"].mean()),
        "assumptions": [
            "Gaussian luminous region in x,y,z,t; negative-binomial multiplicities.",
            "Momentum-binned, forward/backward-specific Gaussian rank copulas for track kinematics, errors and transported scaled residuals.",
            "Uniform azimuth; equal x/y state uncertainties; fixed small slope variance; no slope-position covariance.",
            "Independent tracks conditional on their PV and population; no decay-chain or hit-level simulation.",
            "Continuous marginal tails clipped at calibration quantiles 0.001 and 0.999; time uncertainties retain discrete resolution levels.",
            "Background means tracks absent from effective-PV key lists, not necessarily fake detector tracks.",
        ],
    }
    features = track_features(tracks)
    for label, mask in [("prompt", prompt), ("background", ~prompt)]:
        subset = {"backward_fraction": float(tracks.loc[mask, "backward"].mean())}
        for backward in (False, True):
            subset["backward" if backward else "forward"] = fit_momentum_distribution(
                features[mask & (tracks.backward.to_numpy() == backward)]
            )
        profile["populations"][label] = subset
    return profile


def draw_counts(rng, mean, variance, size=None, minimum=0):
    extra_mean = mean - minimum
    if extra_mean <= 0:
        return np.full(size if size is not None else (), minimum, dtype=int)
    if variance > extra_mean:
        shape = extra_mean**2 / (variance - extra_mean)
        return minimum + rng.negative_binomial(
            shape, shape / (shape + extra_mean), size
        )
    return minimum + rng.poisson(extra_mean, size)


def make_tracks(rng, origins, distribution, event, first_key, from_pv):
    n = len(origins)
    backward = rng.random(n) < distribution["backward_fraction"]
    features = np.empty((n, len(FEATURES)))
    for flag, name in [(False, "forward"), (True, "backward")]:
        selected = backward == flag
        features[selected] = sample_momentum_distribution(
            rng, distribution[name], int(selected.sum())
        )
    theta, p, sigma_xy, sigma_t, chi2 = np.exp(features[:, :5]).T
    phi = rng.uniform(-np.pi, np.pi, n)
    tx, ty = theta * np.cos(phi), theta * np.sin(phi)
    # Draw measured x/y/t at the origin plane, then transport the same line to
    # its transverse point of closest approach to x=y=0. This couples z and t.
    x0 = origins[:, 0] + sigma_xy * features[:, 5]
    y0 = origins[:, 1] + sigma_xy * features[:, 6]
    t0 = origins[:, 3] + sigma_t * features[:, 7]
    dz = -(x0 * tx + y0 * ty) / theta**2
    time_slope = np.where(backward, -1, 1) * np.sqrt(1 + theta**2) / C_MM_PER_NS
    return pd.DataFrame(
        {
            "event_number": np.full(n, event, dtype=np.int64),
            "mc_key": np.arange(first_key, first_key + n, dtype=np.int64),
            "beamPOCA_x": x0 + dz * tx,
            "beamPOCA_y": y0 + dz * ty,
            "beamPOCA_z": origins[:, 2] + dz,
            "beamPOCA_t": t0 + dz * time_slope,
            "beamPOCA_tx": tx,
            "beamPOCA_ty": ty,
            "backward": backward,
            "c00": sigma_xy**2,
            "c11": sigma_xy**2,
            "c55": sigma_t**2,
            "c22": np.full(n, 3e-4**2),
            "c33": np.full(n, 3e-4**2),
            "c20": np.zeros(n),
            "c31": np.zeros(n),
            "ovtx_x": origins[:, 0],
            "ovtx_y": origins[:, 1],
            "ovtx_z": origins[:, 2],
            "ovtx_t": origins[:, 3],
            "p": p,
            "pt": p * theta / np.sqrt(1 + theta**2),
            "chi2": chi2,
            "fromPV": np.broadcast_to(from_pv, n).copy(),
            "true_eta": np.where(backward, -1, 1)
            * -np.log(np.tan(np.arctan(theta) / 2)),
            "true_phi": phi,
        }
    )


def generate_event(rng, profile, event):
    n_pvs = int(draw_counts(rng, **profile["vertices_per_event"], minimum=1))
    points = rng.multivariate_normal(
        profile["vertex_mean"], profile["vertex_covariance"], n_pvs
    )
    counts = draw_counts(rng, **profile["tracks_per_vertex"], size=n_pvs)
    origins = np.repeat(points, counts, axis=0)
    prompt = make_tracks(rng, origins, profile["populations"]["prompt"], event, 0, True)
    n_background = int(
        rng.poisson(len(prompt) * profile["background_per_prompt_track"])
    )
    background_origins = sample_distribution(
        rng, profile["background_origins"], n_background
    )
    background = make_tracks(
        rng,
        background_origins,
        profile["populations"]["background"],
        event,
        len(prompt),
        rng.random(n_background) < profile["background_from_pv_fraction"],
    )
    starts = np.concatenate(([0], np.cumsum(counts)))
    vertices = pd.DataFrame(
        {
            "event_number": np.full(n_pvs, event, dtype=np.int64),
            "effective_pv_key": np.arange(n_pvs, dtype=np.int64),
            "x": points[:, 0],
            "y": points[:, 1],
            "z": points[:, 2],
            "time": points[:, 3],
            "n_valid_tracks": counts,
            "track_keys": [
                np.arange(starts[i], starts[i + 1], dtype=np.int64)
                for i in range(n_pvs)
            ],
        }
    )
    # The row position and MC key deliberately differ, exercising key matching.
    tracks = (
        pd.concat([prompt, background], ignore_index=True)
        .iloc[rng.permutation(len(prompt) + n_background)]
        .reset_index(drop=True)
    )
    return tracks, vertices


def generate_sample(profile, output, events=10, seed=20260922, histogram_params=None):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            f"Refusing to overwrite {output}; choose a new output path"
        )
    if events < 1:
        raise ValueError("events must be positive")
    if (
        profile["schema_version"] != 2
        or profile["track_features"] != FEATURES
        or profile["quantile_probabilities"] != PROBABILITIES.tolist()
    ):
        raise ValueError(
            "Unsupported toy profile schema; use a version 2 momentum-conditioned profile"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_tracks = n_vertices = 0
    with uproot.recreate(output) as root:
        root["toy_metadata"] = json.dumps(
            {
                "sample_type": "synthetic_parametric_toy",
                "seed": seed,
                "events": events,
                "profile": profile,
                "histogram_source": "PVfit Python (not Allen)"
                if histogram_params
                else None,
                "histogram_parameters": histogram_params,
            }
        )
        for event in range(events):
            tracks, vertices = generate_event(rng, profile, event)
            arrays = {k: tracks[k].to_numpy() for k in tracks}
            pv_arrays = {
                k: vertices[k].to_numpy() for k in vertices if k != "track_keys"
            }
            pv_arrays["track_keys"] = ak.Array(vertices.track_keys.tolist())
            if event == 0:
                root.mktree(
                    "velo_kalman_filter/monitor_tree",
                    {k: v.dtype for k, v in arrays.items()},
                )
                root.mktree(
                    "mc_pv_dumper/MCPV_effective_tree",
                    {
                        **{
                            k: v.dtype
                            for k, v in pv_arrays.items()
                            if k != "track_keys"
                        },
                        "track_keys": "var * int64",
                    },
                )
                if histogram_params:
                    root.mktree(
                        "pv_beamline_histo/histo_tree",
                        {
                            "event_number": "int64",
                            "z_bin": "float64",
                            "t_bin": "float64",
                            "weight": "float64",
                        },
                    )
                    root.mktree(
                        "pv_beamline_peak/peak_tree",
                        {
                            "event_number": "int64",
                            "z_peak": "float64",
                            "t_peak": "float64",
                        },
                    )
            root["velo_kalman_filter/monitor_tree"].extend(arrays)
            root["mc_pv_dumper/MCPV_effective_tree"].extend(pv_arrays)
            if histogram_params:
                from ..fit.seedings_finder import (
                    fill_histogram_for_seedings,
                    find_seedings,
                )
                from ..fit.truth_matching import select_tracks

                h = fill_histogram_for_seedings(
                    select_tracks(tracks, histogram_params), histogram_params
                )
                iz, it = np.nonzero(h)
                root["pv_beamline_histo/histo_tree"].extend(
                    {
                        "event_number": np.full(len(iz), event, dtype=np.int64),
                        "z_bin": histogram_params["zmin"]
                        + (iz + 0.5) * histogram_params["bin_size_z"],
                        "t_bin": histogram_params["tmin"]
                        + (it + 0.5) * histogram_params["bin_size_t"],
                        "weight": h[iz, it],
                    }
                )
                peaks = np.asarray(find_seedings(h, histogram_params)).reshape(-1, 2)
                root["pv_beamline_peak/peak_tree"].extend(
                    {
                        "event_number": np.full(len(peaks), event, dtype=np.int64),
                        "z_peak": peaks[:, 0],
                        "t_peak": peaks[:, 1],
                    }
                )
            n_tracks += len(tracks)
            n_vertices += len(vertices)
    return {
        "output": str(output),
        "events": events,
        "tracks": n_tracks,
        "effective_vertices": n_vertices,
        "seed": seed,
    }
