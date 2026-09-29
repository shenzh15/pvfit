"""Rates binned in track multiplicity: the machinery every rate analysis shares.

An analysis module decides which vertices count as successes (that is the
physics: overlap-based or distance-based matching) and returns RateResult
objects. This module provides the bins, the counters with Clopper-Pearson
errors, the HDF5 event access, and the plot / CSV / summary writers that
run_analysis applies to every result in the same way.
"""

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
from matplotlib.ticker import (
    AutoMinorLocator,
    MaxNLocator,
    MultipleLocator,
    StrMethodFormatter,
)

from .utils import calculate_clopper_pearson_errors, find_best_match


def make_bins(min_ntracks, max_ntracks, width):
    """Equal-width, inclusive (start, end) bins in track multiplicity."""
    num_bins = (max_ntracks - min_ntracks) // width + 1
    return [
        (min_ntracks + i * width, min_ntracks + (i + 1) * width - 1)
        for i in range(num_bins)
    ]


def find_bin(bins, n_tracks):
    """The (start, end) bin containing n_tracks, or None if outside all bins."""
    for bin_ in bins:
        if bin_[0] <= n_tracks <= bin_[1]:
            return bin_
    return None


@dataclass
class EventVertices:
    """One event's true and reconstructed vertices as the analyses see them.

    Coordinates and errors are arrays indexed like the track sets; a field the
    file does not store is None (only z, t and track_indices are required).
    """

    event_id: int
    true_tracks: list  # set of track row indices per true vertex
    true_x: np.ndarray
    true_y: np.ndarray
    true_z: np.ndarray
    true_t: np.ndarray
    rec_tracks: list  # set of track row indices per reconstructed vertex
    rec_x: np.ndarray
    rec_y: np.ndarray
    rec_z: np.ndarray
    rec_t: np.ndarray
    rec_x_err: np.ndarray
    rec_y_err: np.ndarray
    rec_z_err: np.ndarray
    rec_t_err: np.ndarray


def _optional(group, name):
    return group[name][:] if name in group else None


def iter_events(hdf5_path):
    """Yield EventVertices for every event in the file, in event order."""
    with h5py.File(hdf5_path, "r") as f:
        for event_id in sorted(int(key) for key in f):
            group = f[str(event_id)]
            true = group["true_vertices"]
            rec = group["rec_vertices"]
            yield EventVertices(
                event_id=event_id,
                true_tracks=[set(idx) for idx in true["track_indices"][:]],
                true_x=_optional(true, "x"),
                true_y=_optional(true, "y"),
                true_z=true["z"][:],
                true_t=true["t"][:],
                rec_tracks=[set(idx) for idx in rec["track_indices"][:]],
                rec_x=_optional(rec, "x"),
                rec_y=_optional(rec, "y"),
                rec_z=rec["z"][:],
                rec_t=rec["t"][:],
                rec_x_err=_optional(rec, "x_err"),
                rec_y_err=_optional(rec, "y_err"),
                rec_z_err=_optional(rec, "z_err"),
                rec_t_err=_optional(rec, "t_err"),
            )


def track_matched_pairs(event, recall_threshold, purity_threshold):
    """(true_id, rec_id) for every reconstructed vertex with an overlap match.

    Each reconstructed vertex pairs with at most one true vertex (see
    utils.find_best_match); a true vertex may appear in several pairs.
    """
    true_vertex_info = {
        true_id: {"track_indices": tracks}
        for true_id, tracks in enumerate(event.true_tracks)
    }
    for rec_id, rec_tracks in enumerate(event.rec_tracks):
        if not rec_tracks:
            continue
        true_id = find_best_match(
            rec_tracks, true_vertex_info, recall_threshold, purity_threshold
        )
        if true_id is not None:
            yield true_id, rec_id


class BinnedCounts:
    """Numerator and denominator per track-multiplicity bin."""

    def __init__(self, bins):
        self.bins = list(bins)
        self.denominator = dict.fromkeys(self.bins, 0)
        self.numerator = dict.fromkeys(self.bins, 0)

    def add(self, n_tracks, success):
        """Count one object in its bin; objects outside every bin are ignored."""
        bin_ = find_bin(self.bins, n_tracks)
        if bin_ is not None:
            self.denominator[bin_] += 1
            if success:
                self.numerator[bin_] += 1

    @staticmethod
    def _rate(numerator, denominator):
        rate = numerator / denominator if denominator > 0 else 0.0
        lower, upper = calculate_clopper_pearson_errors(numerator, denominator, rate)
        return rate, lower, upper

    def rate(self, bin_):
        """(rate, error_lower, error_upper) of one bin."""
        return self._rate(self.numerator[bin_], self.denominator[bin_])

    def overall(self):
        """(rate, error_lower, error_upper, numerator, denominator) over all bins."""
        numerator = sum(self.numerator.values())
        denominator = sum(self.denominator.values())
        return (*self._rate(numerator, denominator), numerator, denominator)

    def table(self):
        """One row per bin, including empty bins."""
        rows = []
        for start, end in self.bins:
            rate, lower, upper = self.rate((start, end))
            rows.append(
                {
                    "bin_start": start,
                    "bin_end": end,
                    "bin_center": (start + end) / 2,
                    "rate": rate,
                    "error_lower": lower,
                    "error_upper": upper,
                    "denominator": self.denominator[start, end],
                    "numerator": self.numerator[start, end],
                }
            )
        return pd.DataFrame(rows)


@dataclass
class RateResult:
    """A binned rate plus what it measures, ready to be written out."""

    metric: str  # e.g. "efficiency", "fake_rate"; used in file names
    binned_by: str  # "true" or "rec": which vertices the bins count
    counts: BinnedCounts
    parameters: dict = field(default_factory=dict)  # matching settings, for the CSV
    label: str = ""  # plot title / legend text for this definition
    matching: str = "track"  # "track" (overlap), "geo" or "reso" (distance)

    # Optional population histogram, independent of rate numerator/denominator.
    reconstructed_counts: dict[tuple[int, int], int] | None = None

    @property
    def stem(self):
        return f"{self.metric}_{self.binned_by}_binned"

    def method_name(self, base_name):
        """Method label carried in file names, e.g. PVfit_Track_Based."""
        return f"{base_name}_{self.matching.title()}_Based"

    def summary_line(self, base_name):
        rate, lower, upper, numerator, denominator = self.counts.overall()
        return (
            f"{self.metric:<11} {self.binned_by:>4}-binned  {self.matching:<5}  "
            f"{rate:.3f} +{upper:.3f}/-{lower:.3f}  ({numerator}/{denominator})"
        )


def _slug(method_name):
    return method_name.lower().replace(" ", "_")


def write_rate_csv(result, output_dir, base_name):
    """Write <stem>_<slug>.csv (per bin) and <metric>_overall_<...>.csv."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    method_name = result.method_name(base_name)
    slug = _slug(method_name)

    result.counts.table().to_csv(output_path / f"{result.stem}_{slug}.csv", index=False)

    rate, lower, upper, numerator, denominator = result.counts.overall()
    overall = {
        "numerator": numerator,
        "denominator": denominator,
        "rate": rate,
        "error_lower": lower,
        "error_upper": upper,
        **result.parameters,
        "method_name": method_name,
    }
    pd.DataFrame([overall]).to_csv(
        output_path / f"{result.metric}_overall_{result.binned_by}_binned_{slug}.csv",
        index=False,
    )


def count_vertex_tracks(hdf5_path, bins):
    """Raw true/rec populations by their own track count, without matching."""
    bins = list(bins)
    true_counts = dict.fromkeys(bins, 0)
    reconstructed_counts = dict.fromkeys(bins, 0)
    for event in iter_events(hdf5_path):
        for vertices, counts in (
            (event.true_tracks, true_counts),
            (event.rec_tracks, reconstructed_counts),
        ):
            for tracks in vertices:
                if not tracks:
                    continue
                bin_ = find_bin(bins, len(tracks))
                if bin_ is not None:
                    counts[bin_] += 1
    return true_counts, reconstructed_counts


def write_vertex_track_count_csv(
    true_counts, reconstructed_counts, output_dir, base_name
):
    """Save both raw populations once, with no matching suffix or display scaling."""
    if true_counts.keys() != reconstructed_counts.keys():
        raise ValueError(
            "True and reconstructed track-count distributions need the same bins"
        )
    columns = [
        "bin_start",
        "bin_end",
        "bin_center",
        "true_count",
        "reconstructed_count",
    ]
    rows = [
        (
            start,
            end,
            (start + end) / 2,
            true_counts[start, end],
            reconstructed_counts[start, end],
        )
        for start, end in sorted(true_counts)
    ]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_file = output_path / f"vertex_track_count_distribution_{_slug(base_name)}.csv"
    pd.DataFrame(rows, columns=columns).to_csv(csv_file, index=False)
    return csv_file


_VERTEX_WORD = {"true": "true", "rec": "reconstructed"}


def plot_efficiency_lhcb2(result, output_dir, base_name, max_ntracks=150):
    """Efficiency with asymmetric 1-sigma bars and a reconstructed-vertex spectrum."""
    return plot_rate_lhcb2(result, output_dir, base_name, max_ntracks)


def _rate_display_ymax(table):
    """Leave headroom above central values, independently of long intervals."""
    target = max(0.001, 1.25 * float(table["rate"].max()))
    step = 10 ** np.floor(np.log10(target)) / 2
    return min(1.05, float(np.ceil(target / step) * step))


def plot_rate_lhcb2(result, output_dir, base_name, max_ntracks=150, ymax=None):
    """Compact rate plot with CP bars and an optional vertex spectrum.

    Efficiency uses a separately counted reconstructed-vertex spectrum;
    other true-binned rates show their truth population. Keep complete bins.
    Large intervals remain intact and get an overflow marker when they
    extend beyond the displayed y range.
    """
    table = result.counts.table()
    table = table[(table["denominator"] > 0) & (table["bin_end"] <= max_ntracks)]
    if table.empty:
        return None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stem = f"{result.stem}_{_slug(result.method_name(base_name))}_lhcb2"
    plot_file = output_path / f"{stem}.pdf"
    is_efficiency = result.metric == "efficiency"
    spectrum = None
    if is_efficiency:
        if result.reconstructed_counts is None:
            raise ValueError("Efficiency plots require reconstructed-vertex counts")
        spectrum = pd.DataFrame(
            [
                (start, end, count)
                for (start, end), count in result.reconstructed_counts.items()
            ],
            columns=["bin_start", "bin_end", "count"],
        )
        spectrum = spectrum[
            (spectrum["count"] > 0) & (spectrum["bin_end"] <= max_ntracks)
        ].sort_values("bin_start")
        spectrum_label = "Reconstructed-vertex distribution (scaled)"
    elif result.binned_by == "true":
        spectrum = table.rename(columns={"denominator": "count"})
        spectrum_label = "True-vertex distribution (scaled)"
    if ymax is None:
        ymax = 1.10 if is_efficiency else _rate_display_ymax(table)
    if not np.isfinite(ymax) or ymax <= 0:
        raise ValueError("Rate plot ymax must be positive and finite")
    if ymax < table["rate"].max():
        raise ValueError("Rate plot ymax must include all displayed central values")
    label = f"{base_name}: {result.label}"
    if result.matching == "geo" and "m_r" in result.parameters:
        label = rf"{base_name}: geometric matching ($m_r={result.parameters['m_r']:g}$)"

    with plt.rc_context():
        hep.style.use(hep.style.LHCb2)
        # Compact paper styling: thin frames and short inward ticks.
        plt.rcParams.update(
            {
                "axes.linewidth": 0.7,
                "font.weight": "normal",
                "xtick.top": False,
                "ytick.right": False,
                "xtick.major.size": 4.5,
                "ytick.major.size": 4.5,
                "xtick.minor.size": 2.2,
                "ytick.minor.size": 2.2,
                "xtick.major.width": 0.65,
                "ytick.major.width": 0.65,
                "xtick.minor.width": 0.55,
                "ytick.minor.width": 0.55,
                "xtick.major.pad": 3,
                "ytick.major.pad": 3,
            }
        )
        fig, ax = plt.subplots(figsize=(6.6, 4.6))
        if spectrum is not None and not spectrum.empty:
            # Scale only the background shape, independently of the rate.
            peak = 0.65 if is_efficiency else 0.6 * ymax
            distribution = peak * spectrum["count"] / spectrum["count"].max()
            ax.bar(
                (spectrum["bin_start"] + spectrum["bin_end"]) / 2,
                distribution,
                width=spectrum["bin_end"] - spectrum["bin_start"] + 1,
                color="0.8",
                edgecolor="none",
                linewidth=0,
                zorder=0,
                label=spectrum_label,
            )
        ax.errorbar(
            table["bin_center"],
            table["rate"],
            yerr=table[["error_lower", "error_upper"]].to_numpy().T,
            fmt="o",
            color="#2446b5",
            markersize=3.0,
            markerfacecolor="none",
            markeredgewidth=0.65,
            elinewidth=0.8,
            capsize=2.0,
            capthick=0.8,
            barsabove=True,
            label=label,
        )
        above = table["rate"] + table["error_upper"] > ymax
        if above.any():
            ax.plot(
                table.loc[above, "bin_center"],
                np.full(above.sum(), 0.97 * ymax),
                linestyle="none",
                marker="^",
                markersize=3,
                color="#2446b5",
                label="Interval extends above axis",
            )
        ax.set_xlim(0, max_ntracks)
        ax.set_ylim(0, ymax)
        ax.set_xlabel(
            "Number of tracks per vertex"
            if is_efficiency
            else f"Number of tracks per {_VERTEX_WORD[result.binned_by]} vertex",
            fontsize=16,
            labelpad=3,
        )
        ax.set_ylabel(
            result.metric.replace("_", " ").capitalize(), fontsize=17, labelpad=4
        )
        ax.xaxis.set_major_locator(MultipleLocator(25))
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_major_locator(
            MultipleLocator(0.2) if is_efficiency else MaxNLocator(nbins=5)
        )
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.grid(False)
        ax.legend(
            loc="lower left",
            bbox_to_anchor=(0, 1.025),
            fontsize=11.5,
            borderaxespad=0,
            handlelength=1.4,
            handletextpad=0.45,
        )
        fig.tight_layout(pad=0.7)
        fig.savefig(plot_file, bbox_inches="tight", pad_inches=0.04)
        fig.savefig(
            output_path / f"{stem}.png",
            dpi=240,
            bbox_inches="tight",
            pad_inches=0.04,
        )
        plt.close(fig)
    return plot_file
