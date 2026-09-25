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
import numpy as np
import pandas as pd

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


def _points(counts):
    """Bins with entries: (x, rate, lower, upper, denominator, numerator) lists."""
    x, rate, lower, upper, denominator, numerator = [], [], [], [], [], []
    for start, end in counts.bins:
        if counts.denominator[start, end] == 0:
            continue
        r, lo, hi = counts.rate((start, end))
        x.append((start + end) / 2)
        rate.append(r)
        lower.append(lo)
        upper.append(hi)
        denominator.append(counts.denominator[start, end])
        numerator.append(counts.numerator[start, end])
    return x, rate, lower, upper, denominator, numerator


_VERTEX_WORD = {"true": "true", "rec": "reconstructed"}


def plot_rate(result, output_dir, base_name, bin_width):
    """Rate vs track multiplicity with its 1-sigma band and the bin counts."""
    x, rate, lower, upper, denominator, numerator = _points(result.counts)
    if not x:
        print(f"No {result.metric} data to plot")
        return None

    _, ax1 = plt.subplots(figsize=(9, 6))
    ax2 = ax1.twinx()
    metric_text = result.metric.replace("_", " ").title()

    ax1.plot(x, rate, "bo-", linewidth=2, markersize=8, label=metric_text)
    ax1.fill_between(
        x,
        [r - e for r, e in zip(rate, lower, strict=True)],
        [r + e for r, e in zip(rate, upper, strict=True)],
        alpha=0.3,
        color="blue",
        label="Clopper-Pearson 1-sigma CI",
    )
    ax1.set_xlabel(
        f"Number of tracks per {_VERTEX_WORD[result.binned_by]} vertex", fontsize=12
    )
    ax1.set_ylabel(metric_text, fontsize=12)
    ax1.set_title(f"{metric_text} vs Track Count\n{result.label}", fontsize=14)
    ax1.set_ylim(0, 1.27)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    # Bin populations on a hidden second axis, numerator on top of denominator
    x_array = np.array(x)
    max_count = max(max(denominator), max(numerator))
    y_max = max_count * 2.3
    ax2.bar(
        x_array,
        denominator,
        bin_width,
        alpha=0.8,
        color="lightblue",
        label=f"Total {_VERTEX_WORD[result.binned_by]} vertices",
        linewidth=0.5,
    )
    ax2.bar(
        x_array,
        numerator,
        bin_width,
        alpha=0.9,
        color="darkorange",
        label=f"{metric_text} numerator",
        linewidth=0.5,
    )
    ax2.set_ylim(0, y_max)
    ax2.set_yticklabels([])
    ax2.tick_params(axis="y", which="both", length=0)
    ax2.legend(loc="upper right")

    label_box = {"boxstyle": "round,pad=0.2", "alpha": 0.8, "edgecolor": "none"}
    for x_pos, n_denominator, n_numerator in zip(
        x_array, denominator, numerator, strict=True
    ):
        ax2.text(
            x_pos,
            n_denominator + y_max * 0.03,
            str(n_denominator),
            ha="center",
            va="bottom",
            fontsize=7,
            color="darkblue",
            bbox={**label_box, "facecolor": "white"},
        )
        if n_numerator > 0:
            y_pos = (
                n_numerator / 2
                if n_numerator >= max_count * 0.15
                else n_numerator + y_max * 0.01
            )
            ax2.text(
                x_pos,
                y_pos,
                str(n_numerator),
                ha="center",
                va="center",
                fontsize=7,
                weight="bold",
                color="white",
                bbox={**label_box, "facecolor": "darkred", "alpha": 0.9},
            )

    plt.tight_layout()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_file = (
        output_path / f"{result.stem}_{_slug(result.method_name(base_name))}.pdf"
    )
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()
    return plot_file


def plot_rate_comparison(results, output_dir, base_name, name):
    """Overlay several rates of the same metric, plus their ratio to the first."""
    if len({r.metric for r in results}) != 1:
        raise ValueError("compared results must share a metric")
    metric = results[0].metric
    metric_text = metric.replace("_", " ").title()
    binned_by = results[0].binned_by
    slug = _slug(base_name)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    points = [_points(r.counts) for r in results]
    if any(not p[0] for p in points):
        print(f"No {metric} data to compare")
        return

    _, ax = plt.subplots(figsize=(9, 6))
    for result, (x, rate, lower, upper, _, _), fmt in zip(
        results, points, ("o-", "s-", "d-", "^-"), strict=False
    ):
        ax.errorbar(x, rate, yerr=[lower, upper], fmt=fmt, label=result.label)
    ax.set_xlabel(f"Number of tracks per {_VERTEX_WORD[binned_by]} vertex")
    ax.set_ylabel(metric_text)
    ax.set_title(f"{metric_text} Comparison")
    ax.set_ylim(0, 1.27)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.savefig(
        output_path / f"{metric}_comparison_{name}_{slug}.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    reference_x, reference_rate = points[0][0], points[0][1]
    _, ax = plt.subplots(figsize=(9, 6))
    for result, (x, rate, *_), fmt in zip(
        results[1:], points[1:], ("d-", "^-", "v-"), strict=False
    ):
        if x != reference_x:
            raise ValueError("compared results must use the same bins")
        ratio = [
            r / g if g > 0 else 0 for g, r in zip(reference_rate, rate, strict=True)
        ]
        ax.plot(x, ratio, fmt, label=f"{result.label} / {results[0].label}")
    ax.axhline(1.0, linestyle="--", color="gray")
    ax.set_xlabel(f"Number of tracks per {_VERTEX_WORD[binned_by]} vertex")
    ax.set_ylabel(f"{metric_text} Ratio")
    ax.set_title(f"{metric_text} Ratio")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.savefig(
        output_path / f"{metric}_ratio_{name}_{slug}.pdf", dpi=300, bbox_inches="tight"
    )
    plt.close()
