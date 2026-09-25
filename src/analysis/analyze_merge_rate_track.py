from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from .utils import calculate_clopper_pearson_errors


def analyze_vertex_merge_rate(
    hdf5_path,
    output_dir,
    recall_threshold,
    plot_min_ntracks,
    plot_max_ntracks,
    plot_bin_width_ntracks,
    method_name,
):
    """Analyze vertex merge rate from HDF5 results.

    Merge definition:
      - Find reconstructed (rec) vertices such that:
          there exist MORE THAN 1 true vertices with recall > threshold
        where recall = (# shared tracks) / (# tracks in true vertex)
      - Record:
          (1) these rec vertices as "merged rec"
          (2) the true vertices that satisfy recall > threshold w.r.t those rec vertices
              as "merged true"

    Then produce TWO merge rates:
      (A) True-vertex merge rate vs true nTracks
      (B) Rec-vertex merge rate vs rec nTracks

    Args:
        hdf5_path: Path to the HDF5 file containing rec_vertices and true_vertices
        output_dir: Output directory for plots and CSV files
        recall_threshold: Recall threshold for considering a true vertex "merged into" a rec vertex
        plot_min_ntracks: Minimum nTracks for binning/plots
        plot_max_ntracks: Maximum nTracks for binning/plots
        plot_bin_width_ntracks: Bin width in nTracks
        method_name: Name of the method for comparison plots
    """
    # Create equal-width bins for track count
    num_bins = (plot_max_ntracks - plot_min_ntracks) // plot_bin_width_ntracks + 1
    bins = [
        (
            plot_min_ntracks + i * plot_bin_width_ntracks,
            plot_min_ntracks + (i + 1) * plot_bin_width_ntracks - 1,
        )
        for i in range(num_bins)
    ]

    # -------------------------
    # Per-bin counters (TRUE)
    # -------------------------
    true_vertices_per_bin = dict.fromkeys(bins, 0)  # denominator (true)
    merged_true_vertices_per_bin = dict.fromkeys(bins, 0)  # numerator   (true)

    # -------------------------
    # Per-bin counters (REC)
    # -------------------------
    rec_vertices_per_bin = dict.fromkeys(bins, 0)  # denominator (rec)
    merged_rec_vertices_per_bin = dict.fromkeys(bins, 0)  # numerator   (rec)

    print(f"Loading HDF5 file: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as file:
        event_ids = sorted(int(key) for key in file)
        print(f"Found {len(event_ids)} events")

        for event_id in tqdm(event_ids, desc="Processing events"):
            event_group = file[str(event_id)]

            # Load true and reconstructed vertices from new HDF5 format
            true_verts = event_group["true_vertices"]
            true_track_indices = true_verts["track_indices"][:]  # vlen array

            rec_verts = event_group["rec_vertices"]
            rec_track_indices = rec_verts["track_indices"][:]  # vlen array

            # Build true vertex info
            true_vertex_info = {}
            for vertex_idx, track_indices in enumerate(true_track_indices):
                true_vertex_info[vertex_idx] = {
                    "track_indices": set(track_indices),
                    "num_tracks": len(track_indices),
                }

            # Count denominators and identify merged vertices
            _count_merged_vertices_by_bin(
                true_vertex_info,
                rec_track_indices,
                recall_threshold,
                bins,
                true_vertices_per_bin,
                merged_true_vertices_per_bin,
                rec_vertices_per_bin,
                merged_rec_vertices_per_bin,
            )

    # ---- compute rates + errors (two families) ----
    true_rates, true_rate_errors = _calculate_rates(
        bins, true_vertices_per_bin, merged_true_vertices_per_bin
    )
    rec_rates, rec_rate_errors = _calculate_rates(
        bins, rec_vertices_per_bin, merged_rec_vertices_per_bin
    )

    # ---- overall stats ----
    overall_true_stats = _calculate_overall_rate(
        true_vertices_per_bin, merged_true_vertices_per_bin, name="true"
    )
    overall_rec_stats = _calculate_overall_rate(
        rec_vertices_per_bin, merged_rec_vertices_per_bin, name="rec"
    )

    # ---- plotting data ----
    true_plot_data = _prepare_plot_data(
        bins,
        true_rates,
        true_rate_errors,
        true_vertices_per_bin,
        merged_true_vertices_per_bin,
    )
    rec_plot_data = _prepare_plot_data(
        bins,
        rec_rates,
        rec_rate_errors,
        rec_vertices_per_bin,
        merged_rec_vertices_per_bin,
    )

    if true_plot_data is None and rec_plot_data is None:
        print("No merge-rate data to plot")
        return

    # ---- plots ----
    if true_plot_data is not None:
        _create_merge_rate_plot(
            plot_data=true_plot_data,
            bin_width=plot_bin_width_ntracks,
            threshold=recall_threshold,
            output_dir=output_dir,
            method_name=method_name,
            mode="true",
        )

    if rec_plot_data is not None:
        _create_merge_rate_plot(
            plot_data=rec_plot_data,
            bin_width=plot_bin_width_ntracks,
            threshold=recall_threshold,
            output_dir=output_dir,
            method_name=method_name,
            mode="rec",
        )

    # ---- print summaries ----
    _print_results(
        recall_threshold,
        overall_true_stats,
        overall_rec_stats,
        bins,
        true_rate_errors,
        rec_rate_errors,
        true_vertices_per_bin,
        merged_true_vertices_per_bin,
        rec_vertices_per_bin,
        merged_rec_vertices_per_bin,
    )

    # ---- save csv ----
    _save_merge_rate_to_csv(
        bins=bins,
        plot_data=true_plot_data,
        rates=true_rates,
        rate_errors=true_rate_errors,
        denom_per_bin=true_vertices_per_bin,
        numer_per_bin=merged_true_vertices_per_bin,
        overall_stats=overall_true_stats,
        threshold=recall_threshold,
        output_dir=output_dir,
        method_name=method_name,
        mode="true",
    )

    _save_merge_rate_to_csv(
        bins=bins,
        plot_data=rec_plot_data,
        rates=rec_rates,
        rate_errors=rec_rate_errors,
        denom_per_bin=rec_vertices_per_bin,
        numer_per_bin=merged_rec_vertices_per_bin,
        overall_stats=overall_rec_stats,
        threshold=recall_threshold,
        output_dir=output_dir,
        method_name=method_name,
        mode="rec",
    )


def _count_merged_vertices_by_bin(
    true_vertex_info,
    rec_track_indices,
    recall_threshold,
    bins,
    true_vertices_per_bin,
    merged_true_vertices_per_bin,
    rec_vertices_per_bin,
    merged_rec_vertices_per_bin,
):
    """Identify merged vertices and count them by track count bin.

    Merge definition:
      - A rec vertex is "merged" if it overlaps (recall > threshold) with >1 true vertices
      - A true vertex is "merged" if it participates in such a merge

    Denominators:
      - true_vertices_per_bin: all true vertices
      - rec_vertices_per_bin: all rec vertices

    Numerators:
      - merged_rec_vertices_per_bin: merged rec vertices
      - merged_true_vertices_per_bin: merged true vertices
    """
    # -------------------------
    # TRUE denominator
    # -------------------------
    for vertex_data in true_vertex_info.values():
        num_tracks = vertex_data["num_tracks"]
        for bin_start, bin_end in bins:
            if bin_start <= num_tracks <= bin_end:
                true_vertices_per_bin[bin_start, bin_end] += 1
                break

    # -------------------------
    # REC denominator + MERGE detection
    # -------------------------
    merged_true_ids_all = set()

    for rec_idx, rec_tracks in enumerate(rec_track_indices):
        rec_track_set = set(rec_tracks)
        num_rec_tracks = len(rec_track_set)

        if num_rec_tracks == 0:
            continue

        # REC denominator bin fill
        for bin_start, bin_end in bins:
            if bin_start <= num_rec_tracks <= bin_end:
                rec_vertices_per_bin[bin_start, bin_end] += 1
                break

        # Find which true vertices have recall > threshold w.r.t this rec vertex
        merged_true_ids = _find_true_vertices_with_recall_above_threshold(
            rec_track_set,
            true_vertex_info,
            recall_threshold,
        )

        # Merge condition: more than 1 true vertex satisfies recall > threshold
        if len(merged_true_ids) > 1:
            # Merged REC numerator bin fill
            for bin_start, bin_end in bins:
                if bin_start <= num_rec_tracks <= bin_end:
                    merged_rec_vertices_per_bin[bin_start, bin_end] += 1
                    break

            # Record all involved true vertices
            merged_true_ids_all.update(merged_true_ids)

    # -------------------------
    # TRUE numerator (merged true vertices, binned by reconstructed num_tracks)
    # -------------------------
    for true_id in merged_true_ids_all:
        num_tracks = true_vertex_info[true_id]["num_tracks"]
        for bin_start, bin_end in bins:
            if bin_start <= num_tracks <= bin_end:
                merged_true_vertices_per_bin[bin_start, bin_end] += 1
                break


def _find_true_vertices_with_recall_above_threshold(
    rec_track_indices, true_vertex_info, recall_threshold
):
    """Return set of true vertex IDs with recall >= threshold.

    recall = (# shared tracks) / (# tracks in true vertex)
    """
    merged_true_ids = set()

    for vertex_id, vertex_data in true_vertex_info.items():
        true_track_indices = vertex_data["track_indices"]
        num_true_tracks = len(true_track_indices)
        if num_true_tracks == 0:
            continue

        num_shared = len(rec_track_indices & true_track_indices)
        recall = num_shared / num_true_tracks

        if recall >= recall_threshold:
            merged_true_ids.add(vertex_id)

    return merged_true_ids


def _calculate_rates(bins, denom_per_bin, numer_per_bin):
    """Calculate rate and Clopper-Pearson confidence intervals for each bin."""
    rates = {}
    errors = {}

    for bin_tuple in bins:
        denom = denom_per_bin[bin_tuple]
        numer = numer_per_bin[bin_tuple]
        rate = numer / denom if denom > 0 else 0.0
        rates[bin_tuple] = rate

        error_lower, error_upper = calculate_clopper_pearson_errors(numer, denom, rate)
        errors[bin_tuple] = {"lower": error_lower, "upper": error_upper}

    return rates, errors


def _calculate_overall_rate(denom_per_bin, numer_per_bin, name=""):
    """Calculate overall rate and error bounds."""
    total_denom = sum(denom_per_bin.values())
    total_numer = sum(numer_per_bin.values())
    overall_rate = total_numer / total_denom if total_denom > 0 else 0.0

    error_lower, error_upper = calculate_clopper_pearson_errors(
        total_numer, total_denom, overall_rate
    )

    return {
        "name": name,
        "rate": overall_rate,
        "error_lower": error_lower,
        "error_upper": error_upper,
        "total_denom": total_denom,
        "total_numer": total_numer,
    }


def _prepare_plot_data(bins, rates, rate_errors, denom_per_bin, numer_per_bin):
    """Prepare data for plotting, filtering out bins with no data."""
    bin_centers = [(start + end) / 2 for start, end in bins]
    denom_list = [denom_per_bin[bin_tuple] for bin_tuple in bins]
    numer_list = [numer_per_bin[bin_tuple] for bin_tuple in bins]

    x_values, rate_values, err_lower, err_upper = [], [], [], []
    denom_filtered, numer_filtered = [], []

    for idx, bin_tuple in enumerate(bins):
        if denom_list[idx] > 0:
            x_values.append(bin_centers[idx])
            rate_values.append(rates[bin_tuple])
            err_lower.append(rate_errors[bin_tuple]["lower"])
            err_upper.append(rate_errors[bin_tuple]["upper"])
            denom_filtered.append(denom_list[idx])
            numer_filtered.append(numer_list[idx])

    if not x_values:
        return None

    return {
        "x": x_values,
        "rate": rate_values,
        "error_lower": err_lower,
        "error_upper": err_upper,
        "denom_filtered": denom_filtered,
        "numer_filtered": numer_filtered,
        "bin_centers": bin_centers,
        "denom_all": denom_list,
        "numer_all": numer_list,
    }


def _create_merge_rate_plot(
    plot_data, bin_width, threshold, output_dir, method_name, mode="true"
):
    """Create and save the merge-rate plot with confidence intervals"""
    x = plot_data["x"]
    rate = plot_data["rate"]

    _, ax1 = plt.subplots(figsize=(9, 6))

    # rate + CI
    lower_bounds = [r - e for r, e in zip(rate, plot_data["error_lower"], strict=True)]
    upper_bounds = [r + e for r, e in zip(rate, plot_data["error_upper"], strict=True)]

    label_main = (
        "Merge rate (True-vertex binned)"
        if mode == "true"
        else "Merge rate (Rec-vertex binned)"
    )

    ax1.plot(x, rate, "bo-", linewidth=2, markersize=8, label=label_main)
    ax1.fill_between(
        x,
        lower_bounds,
        upper_bounds,
        alpha=0.3,
        color="blue",
        label="Clopper-Pearson 1-sigma CI",
    )

    ax1.set_xlabel(
        "Number of tracks per true vertex"
        if mode == "true"
        else "Number of tracks per rec vertex",
        fontsize=12,
    )
    ax1.set_ylabel("Merge rate", fontsize=12)
    ax1.set_title(
        f"Vertex Merge Rate vs Track Count\nRecall threshold: {threshold}",
        fontsize=14,
    )
    ax1.set_ylim(0, 1.27)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    plt.tight_layout()

    # save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    suffix = "true_binned" if mode == "true" else "rec_binned"
    plot_file = (
        output_path / f"merge_rate_{suffix}_{method_name.lower().replace(' ', '_')}.pdf"
    )
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()


def _print_results(
    recall_threshold,
    overall_true_stats,
    overall_rec_stats,
    bins,
    true_rate_errors,
    rec_rate_errors,
    true_vertices_per_bin,
    merged_true_vertices_per_bin,
    rec_vertices_per_bin,
    merged_rec_vertices_per_bin,
):
    """Print detailed merge-rate summary including per-bin results."""
    print(f"\n{'-' * 80}")
    print("MERGE RATE ANALYSIS SUMMARY - HDF5 DATA")
    print(f"{'-' * 80}")
    print(f"Recall threshold: {recall_threshold:.0%}")

    # -------------------------
    # Overall summary
    # -------------------------
    print("\nOverall Merge Rates:")
    print(
        f"  [TRUE] Merge rate: {overall_true_stats['rate']:.3f} "
        f"+{overall_true_stats['error_upper']:.3f} "
        f"-{overall_true_stats['error_lower']:.3f} "
        f"({overall_true_stats['total_numer']}/{overall_true_stats['total_denom']})"
    )
    print(
        f"  [REC ] Merge rate: {overall_rec_stats['rate']:.3f} "
        f"+{overall_rec_stats['error_upper']:.3f} "
        f"-{overall_rec_stats['error_lower']:.3f} "
        f"({overall_rec_stats['total_numer']}/{overall_rec_stats['total_denom']})"
    )

    # -------------------------
    # TRUE-binned breakdown
    # -------------------------
    print("\nMerge Rate by TRUE Vertex Track Count:")
    print(
        f"{'Track Count':<15} {'Total':<8} {'Merged':<8} {'Merge Rate':<12} {'Err(+/-)':<18}"
    )
    print(f"{'-' * 55}")

    for bin_start, bin_end in bins:
        total = true_vertices_per_bin[bin_start, bin_end]
        merged = merged_true_vertices_per_bin[bin_start, bin_end]
        if total > 0:
            rate = merged / total
            err = true_rate_errors[bin_start, bin_end]
            print(
                f"[{bin_start:3d}, {bin_end:3d}]".ljust(15)
                + f"{total:<8d} {merged:<8d} {rate:<12.3f} +{err['upper']:.3f} -{err['lower']:.3f}"
            )

    # -------------------------
    # REC-binned breakdown
    # -------------------------
    print("\nMerge Rate by REC Vertex Track Count:")
    print(
        f"{'Track Count':<15} {'Total':<8} {'Merged':<8} {'Merge Rate':<12} {'Err(+/-)':<18}"
    )
    print(f"{'-' * 55}")

    for bin_start, bin_end in bins:
        total = rec_vertices_per_bin[bin_start, bin_end]
        merged = merged_rec_vertices_per_bin[bin_start, bin_end]
        if total > 0:
            rate = merged / total
            err = rec_rate_errors[bin_start, bin_end]
            print(
                f"[{bin_start:3d}, {bin_end:3d}]".ljust(15)
                + f"{total:<8d} {merged:<8d} {rate:<12.3f} +{err['upper']:.3f} -{err['lower']:.3f}"
            )

    print(f"{'-' * 80}\n")


def _save_merge_rate_to_csv(
    bins,
    plot_data,
    rates,
    rate_errors,
    denom_per_bin,
    numer_per_bin,
    overall_stats,
    threshold,
    output_dir,
    method_name,
    mode="true",
):
    """Save merge-rate results to CSV (bin-level + overall)."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # If plot_data is None, we still save full-bin table (centers need bins)
    bin_centers = [(start + end) / 2 for start, end in bins]

    bin_data = []
    for i, bin_tuple in enumerate(bins):
        bin_start, bin_end = bin_tuple
        rate = rates[bin_tuple]
        error_lower = rate_errors[bin_tuple]["lower"]
        error_upper = rate_errors[bin_tuple]["upper"]
        denom = denom_per_bin[bin_tuple]
        numer = numer_per_bin[bin_tuple]

        bin_data.append(
            {
                "bin_start": bin_start,
                "bin_end": bin_end,
                "bin_center": bin_centers[i],
                "merge_rate": rate,
                "error_lower": error_lower,
                "error_upper": error_upper,
                "denominator": denom,
                "numerator": numer,
            }
        )

    df = pd.DataFrame(bin_data)

    suffix = "true_binned" if mode == "true" else "rec_binned"
    csv_path = (
        output_path / f"merge_rate_{suffix}_{method_name.lower().replace(' ', '_')}.csv"
    )
    df.to_csv(csv_path, index=False)
    print(f"Saved bin-level merge-rate results to: {csv_path}")

    summary_df = pd.DataFrame(
        [
            {
                "mode": mode,
                "total_numerator": overall_stats["total_numer"],
                "total_denominator": overall_stats["total_denom"],
                "overall_merge_rate": overall_stats["rate"],
                "overall_error_lower": overall_stats["error_lower"],
                "overall_error_upper": overall_stats["error_upper"],
                "recall_threshold": threshold,
                "method_name": method_name,
            }
        ]
    )

    overall_path = (
        output_path
        / f"merge_rate_overall_{suffix}_{method_name.lower().replace(' ', '_')}.csv"
    )
    summary_df.to_csv(overall_path, index=False)
    print(f"Saved overall merge-rate statistics to: {overall_path}")
