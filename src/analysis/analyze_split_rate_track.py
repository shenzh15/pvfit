from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from .utils import calculate_clopper_pearson_errors


def analyze_vertex_split_rate(
    hdf5_path,
    output_dir="output",
    purity_threshold=0.7,
    plot_min_ntracks=4,
    plot_max_ntracks=200,
    plot_bin_width_ntracks=5,
    method_name="HepATTn",
):
    """Analyze vertex split rate from HDF5 results.

    Split definition:
      - Find TRUE vertices such that:
          there exist MORE THAN 1 reconstructed vertices with purity > threshold
        where purity = (# shared tracks) / (# tracks in rec vertex)
      - Record:
          (1) these rec vertices as "split rec"
          (2) the true vertices that satisfy purity > threshold w.r.t those split rec vertices
              as "split true"

    Then produce TWO split rates:
      (A) True-vertex split rate vs true nTracks
      (B) Rec-vertex split rate vs rec nTracks

    Args:
        hdf5_path: Path to the HDF5 file containing rec_vertices and true_vertices
        output_dir: Output directory for plots and CSV files
        purity_threshold: Purity threshold for considering a true vertex "split into" a rec vertex
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
    split_true_vertices_per_bin = dict.fromkeys(bins, 0)  # numerator   (true)

    # -------------------------
    # Per-bin counters (REC)
    # -------------------------
    rec_vertices_per_bin = dict.fromkeys(bins, 0)  # denominator (rec)
    split_rec_vertices_per_bin = dict.fromkeys(bins, 0)  # numerator   (rec)

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

            # Count denominators and identify split vertices
            _count_split_vertices_by_bin(
                true_vertex_info,
                rec_track_indices,
                purity_threshold,
                bins,
                true_vertices_per_bin,
                split_true_vertices_per_bin,
                rec_vertices_per_bin,
                split_rec_vertices_per_bin,
            )

    # ---- compute rates + errors ----
    true_rates, true_rate_errors = _calculate_rates(
        bins, true_vertices_per_bin, split_true_vertices_per_bin
    )
    rec_rates, rec_rate_errors = _calculate_rates(
        bins, rec_vertices_per_bin, split_rec_vertices_per_bin
    )

    # ---- overall stats ----
    overall_true_stats = _calculate_overall_rate(
        true_vertices_per_bin, split_true_vertices_per_bin, name="true"
    )
    overall_rec_stats = _calculate_overall_rate(
        rec_vertices_per_bin, split_rec_vertices_per_bin, name="rec"
    )

    # ---- plotting data ----
    true_plot_data = _prepare_plot_data(
        bins,
        true_rates,
        true_rate_errors,
        true_vertices_per_bin,
        split_true_vertices_per_bin,
    )
    rec_plot_data = _prepare_plot_data(
        bins,
        rec_rates,
        rec_rate_errors,
        rec_vertices_per_bin,
        split_rec_vertices_per_bin,
    )

    if true_plot_data is None and rec_plot_data is None:
        print("No split-rate data to plot")
        return

    # ---- plots ----
    if true_plot_data is not None:
        _create_split_rate_plot(
            plot_data=true_plot_data,
            bin_width=plot_bin_width_ntracks,
            threshold=purity_threshold,
            output_dir=output_dir,
            method_name=method_name,
            mode="true",
        )

    if rec_plot_data is not None:
        _create_split_rate_plot(
            plot_data=rec_plot_data,
            bin_width=plot_bin_width_ntracks,
            threshold=purity_threshold,
            output_dir=output_dir,
            method_name=method_name,
            mode="rec",
        )

    # ---- print summaries ----
    _print_results(
        purity_threshold,
        overall_true_stats,
        overall_rec_stats,
        bins,
        true_rate_errors,
        rec_rate_errors,
        true_vertices_per_bin,
        split_true_vertices_per_bin,
        rec_vertices_per_bin,
        split_rec_vertices_per_bin,
    )

    # ---- save csv ----
    _save_split_rate_to_csv(
        bins=bins,
        plot_data=true_plot_data,
        rates=true_rates,
        rate_errors=true_rate_errors,
        denom_per_bin=true_vertices_per_bin,
        numer_per_bin=split_true_vertices_per_bin,
        overall_stats=overall_true_stats,
        threshold=purity_threshold,
        output_dir=output_dir,
        method_name=method_name,
        mode="true",
    )

    _save_split_rate_to_csv(
        bins=bins,
        plot_data=rec_plot_data,
        rates=rec_rates,
        rate_errors=rec_rate_errors,
        denom_per_bin=rec_vertices_per_bin,
        numer_per_bin=split_rec_vertices_per_bin,
        overall_stats=overall_rec_stats,
        threshold=purity_threshold,
        output_dir=output_dir,
        method_name=method_name,
        mode="rec",
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)


def _count_split_vertices_by_bin(
    true_vertex_info,
    rec_track_indices,
    purity_threshold,
    bins,
    true_vertices_per_bin,
    split_true_vertices_per_bin,
    rec_vertices_per_bin,
    split_rec_vertices_per_bin,
):
    """Identify split vertices and count them by track count bin.

    Split definition:
      - A true vertex is "split" if it overlaps (purity > threshold) with >1 rec vertices
      - A rec vertex is "split" if it participates in such a split

    Denominators:
      - true_vertices_per_bin: all true vertices
      - rec_vertices_per_bin: all rec vertices

    Numerators:
      - split_true_vertices_per_bin: split true vertices
      - split_rec_vertices_per_bin: split rec vertices
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
    # TRUE denominator + SPLIT detection
    # -------------------------
    split_rec_ids_all = set()

    for true_id, true_data in true_vertex_info.items():
        true_tracks = true_data["track_indices"]
        num_true_tracks = len(true_tracks)
        if num_true_tracks == 0:
            continue

        # Find which rec vertices have purity > threshold w.r.t this true vertex
        good_rec_ids = _find_rec_vertices_with_purity_above_threshold(
            true_tracks=true_tracks,
            rec_track_indices=rec_track_indices,
            purity_threshold=purity_threshold,
        )

        # Split condition: more than 1 rec vertex satisfies purity > threshold
        if len(good_rec_ids) > 1:
            # Split TRUE numerator bin fill
            num_tracks = true_data["num_tracks"]
            for bin_start, bin_end in bins:
                if bin_start <= num_tracks <= bin_end:
                    split_true_vertices_per_bin[bin_start, bin_end] += 1
                    break

            # Record all involved rec vertices
            split_rec_ids_all.update(good_rec_ids)

    # -------------------------
    # REC denominator and numerator (split rec vertices)
    # -------------------------
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

        # REC numerator bin fill (if this rec vertex is involved in any split)
        if rec_idx in split_rec_ids_all:
            for bin_start, bin_end in bins:
                if bin_start <= num_rec_tracks <= bin_end:
                    split_rec_vertices_per_bin[bin_start, bin_end] += 1
                    break


def _find_rec_vertices_with_purity_above_threshold(
    true_tracks, rec_track_indices, purity_threshold
):
    """Return set of rec vertex IDs with purity >= threshold.

    purity = (# shared tracks) / (# tracks in rec vertex)
    """
    good_rec_ids = set()

    for rec_idx, rec_tracks in enumerate(rec_track_indices):
        rec_track_set = set(rec_tracks)
        num_rec_tracks = len(rec_track_set)
        if num_rec_tracks == 0:
            continue

        num_shared = len(true_tracks & rec_track_set)
        purity = num_shared / num_rec_tracks

        if purity >= purity_threshold:
            good_rec_ids.add(rec_idx)

    return good_rec_ids


def _calculate_rates(bins, denom_per_bin, numer_per_bin):
    rates, errors = {}, {}
    for b in bins:
        denom = denom_per_bin[b]
        numer = numer_per_bin[b]
        rate = numer / denom if denom > 0 else 0.0
        rates[b] = rate
        lo, hi = calculate_clopper_pearson_errors(numer, denom, rate)
        errors[b] = {"lower": lo, "upper": hi}
    return rates, errors


def _calculate_overall_rate(denom_per_bin, numer_per_bin, name=""):
    total_denom = sum(denom_per_bin.values())
    total_numer = sum(numer_per_bin.values())
    rate = total_numer / total_denom if total_denom > 0 else 0.0
    lo, hi = calculate_clopper_pearson_errors(total_numer, total_denom, rate)
    return {
        "name": name,
        "rate": rate,
        "error_lower": lo,
        "error_upper": hi,
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


def _create_split_rate_plot(
    plot_data, bin_width, threshold, output_dir, method_name, mode="true"
):
    """Create and save the split-rate plot with confidence intervals"""
    x = plot_data["x"]
    rate = plot_data["rate"]

    _, ax1 = plt.subplots(figsize=(9, 6))

    # rate + CI
    lower_bounds = [r - e for r, e in zip(rate, plot_data["error_lower"], strict=True)]
    upper_bounds = [r + e for r, e in zip(rate, plot_data["error_upper"], strict=True)]

    label_main = (
        "Split rate (True-vertex binned)"
        if mode == "true"
        else "Split rate (Rec-vertex binned)"
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
    ax1.set_ylabel("Split rate", fontsize=12)
    ax1.set_title(
        f"Vertex Split Rate vs Track Count\nPurity threshold: {threshold}",
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
        output_path / f"split_rate_{suffix}_{method_name.lower().replace(' ', '_')}.pdf"
    )
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()


def _print_results(
    purity_threshold,
    overall_true_stats,
    overall_rec_stats,
    bins,
    true_rate_errors,
    rec_rate_errors,
    true_vertices_per_bin,
    split_true_vertices_per_bin,
    rec_vertices_per_bin,
    split_rec_vertices_per_bin,
):
    """Print detailed split-rate summary including per-bin results."""
    print(f"\n{'-' * 80}")
    print("SPLIT RATE ANALYSIS SUMMARY - HDF5 DATA")
    print(f"{'-' * 80}")
    print(f"Purity threshold: {purity_threshold:.0%}")

    # -------------------------
    # Overall summary
    # -------------------------
    print("\nOverall Split Rates:")
    print(
        f"  [TRUE] Split rate: {overall_true_stats['rate']:.3f} "
        f"+{overall_true_stats['error_upper']:.3f} "
        f"-{overall_true_stats['error_lower']:.3f} "
        f"({overall_true_stats['total_numer']}/{overall_true_stats['total_denom']})"
    )
    print(
        f"  [REC ] Split rate: {overall_rec_stats['rate']:.3f} "
        f"+{overall_rec_stats['error_upper']:.3f} "
        f"-{overall_rec_stats['error_lower']:.3f} "
        f"({overall_rec_stats['total_numer']}/{overall_rec_stats['total_denom']})"
    )

    # -------------------------
    # TRUE-binned breakdown
    # -------------------------
    print("\nSplit Rate by TRUE Vertex Track Count:")
    print(
        f"{'Track Count':<15} {'Total':<8} {'Split':<8} {'Split Rate':<12} {'Err(+/-)':<18}"
    )
    print(f"{'-' * 55}")

    for bin_start, bin_end in bins:
        total = true_vertices_per_bin[bin_start, bin_end]
        split = split_true_vertices_per_bin[bin_start, bin_end]
        if total > 0:
            rate = split / total
            err = true_rate_errors[bin_start, bin_end]
            print(
                f"[{bin_start:3d}, {bin_end:3d}]".ljust(15)
                + f"{total:<8d} {split:<8d} {rate:<12.3f} +{err['upper']:.3f} -{err['lower']:.3f}"
            )

    # -------------------------
    # REC-binned breakdown
    # -------------------------
    print("\nSplit Rate by REC Vertex Track Count:")
    print(
        f"{'Track Count':<15} {'Total':<8} {'Split':<8} {'Split Rate':<12} {'Err(+/-)':<18}"
    )
    print(f"{'-' * 55}")

    for bin_start, bin_end in bins:
        total = rec_vertices_per_bin[bin_start, bin_end]
        split = split_rec_vertices_per_bin[bin_start, bin_end]
        if total > 0:
            rate = split / total
            err = rec_rate_errors[bin_start, bin_end]
            print(
                f"[{bin_start:3d}, {bin_end:3d}]".ljust(15)
                + f"{total:<8d} {split:<8d} {rate:<12.3f} +{err['upper']:.3f} -{err['lower']:.3f}"
            )

    print(f"{'-' * 80}\n")


def _save_split_rate_to_csv(
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
    """Save split-rate results to CSV (bin-level + overall)."""
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
                "split_rate": rate,
                "error_lower": error_lower,
                "error_upper": error_upper,
                "denominator": denom,
                "numerator": numer,
            }
        )

    df = pd.DataFrame(bin_data)

    suffix = "true_binned" if mode == "true" else "rec_binned"
    csv_path = (
        output_path / f"split_rate_{suffix}_{method_name.lower().replace(' ', '_')}.csv"
    )
    df.to_csv(csv_path, index=False)
    print(f"Saved bin-level split-rate results to: {csv_path}")

    summary_df = pd.DataFrame(
        [
            {
                "mode": mode,
                "total_numerator": overall_stats["total_numer"],
                "total_denominator": overall_stats["total_denom"],
                "overall_split_rate": overall_stats["rate"],
                "overall_error_lower": overall_stats["error_lower"],
                "overall_error_upper": overall_stats["error_upper"],
                "purity_threshold": threshold,
                "method_name": method_name,
            }
        ]
    )

    summary_path = (
        output_path
        / f"split_rate_overall_{suffix}_{method_name.lower().replace(' ', '_')}.csv"
    )
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved overall split-rate statistics to: {summary_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze vertex split rate from HDF5 predictions file"
    )
    parser.add_argument(
        "--hdf5_path",
        type=str,
        required=True,
        help="Path to the HDF5 file containing predictions and targets",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Output directory for plots and CSV files",
    )
    parser.add_argument(
        "--purity_threshold",
        type=float,
        default=0.7,
        help="Purity threshold for split definition",
    )
    parser.add_argument(
        "--plot_min_ntracks",
        type=int,
        default=4,
        help="Minimum nTracks for binning/plots",
    )
    parser.add_argument(
        "--plot_max_ntracks",
        type=int,
        default=200,
        help="Maximum nTracks for binning/plots",
    )
    parser.add_argument(
        "--plot_bin_width_ntracks", type=int, default=5, help="Bin width in nTracks"
    )
    parser.add_argument(
        "--method_name",
        type=str,
        default="HepATTn",
        help="Method name for plots and CSV files",
    )
    args = parser.parse_args()

    analyze_vertex_split_rate(
        hdf5_path=args.hdf5_path,
        output_dir=args.output_dir,
        purity_threshold=args.purity_threshold,
        plot_min_ntracks=args.plot_min_ntracks,
        plot_max_ntracks=args.plot_max_ntracks,
        plot_bin_width_ntracks=args.plot_bin_width_ntracks,
        method_name=args.method_name,
    )
