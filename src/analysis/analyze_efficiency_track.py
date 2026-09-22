import operator
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import calculate_clopper_pearson_errors, find_best_match


def analyze_vertex_efficiency(
    hdf5_path,
    output_dir="output",
    recall_threshold=0.7,
    purity_threshold=0.7,
    plot_min_ntracks=4,
    plot_max_ntracks=200,
    plot_bin_width_ntracks=5,
    method_name="HepATTn",
):
    """Analyze vertex reconstruction efficiency from HDF5 results.

    Uses track-based matching: a match is successful if both recall and purity
    of shared tracks exceed their respective thresholds.

    Args:
        hdf5_path: Path to the HDF5 file containing rec_vertices and true_vertices
        output_dir: Output directory for plots and CSV files
        recall_threshold: Recall threshold for matching (intersection / num_true_tracks)
        purity_threshold: Purity threshold for matching (intersection / num_rec_tracks)
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

    # Initialize per-bin counters (using bin tuples as keys)
    true_vertices_per_bin = dict.fromkeys(bins, 0)
    matched_vertices_per_bin = dict.fromkeys(bins, 0)

    print(f"Loading HDF5 file: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as file:
        event_ids = sorted(int(key) for key in file)
        print(f"Found {len(event_ids)} events")

        for event_id in tqdm(event_ids, desc="Processing events"):
            event_data = file[str(event_id)]

            # Extract true and reconstructed vertices from HDF5
            true_vertices = event_data["true_vertices"]
            true_track_indices = true_vertices["track_indices"][:]  # vlen array

            rec_vertices = event_data["rec_vertices"]
            rec_track_indices = rec_vertices["track_indices"][:]  # vlen array

            # Build true vertex info using reconstructed track count for binning
            true_vertex_info = {}
            for vertex_idx, track_indices in enumerate(true_track_indices):
                true_vertex_info[vertex_idx] = {
                    "track_indices": set(track_indices),
                    "num_tracks": len(track_indices),
                }

            # Identify matched vertices and count them by bin
            _count_matched_vertices_by_bin(
                true_vertex_info,
                rec_track_indices,
                recall_threshold,
                purity_threshold,
                bins,
                true_vertices_per_bin,
                matched_vertices_per_bin,
            )

    # Calculate efficiency and errors
    efficiencies, efficiency_errors = _calculate_efficiencies(
        bins, true_vertices_per_bin, matched_vertices_per_bin
    )

    # Calculate overall efficiency and error bounds
    overall_stats = _calculate_overall_efficiency(
        true_vertices_per_bin, matched_vertices_per_bin
    )

    # Prepare plotting data (filter bins with no data)
    plot_data = _prepare_plot_data(
        bins,
        efficiencies,
        efficiency_errors,
        true_vertices_per_bin,
        matched_vertices_per_bin,
    )

    if plot_data is None:
        print("No efficiency data to plot")
        return

    # Create and save plot
    _create_efficiency_plot(
        plot_data,
        plot_bin_width_ntracks,
        recall_threshold,
        purity_threshold,
        output_dir,
        method_name,
    )

    # Print summary results
    _print_results(
        recall_threshold,
        purity_threshold,
        overall_stats,
        plot_data["efficiency"],
        bins,
        efficiency_errors,
        true_vertices_per_bin,
    )

    # Save results to CSV
    _save_efficiency_to_csv(
        bins,
        plot_data,
        efficiencies,
        efficiency_errors,
        true_vertices_per_bin,
        matched_vertices_per_bin,
        overall_stats,
        recall_threshold,
        purity_threshold,
        output_dir,
        method_name,
    )


def _count_matched_vertices_by_bin(
    true_vertex_info,
    rec_track_indices,
    recall_threshold,
    purity_threshold,
    bins,
    true_vertices_per_bin,
    matched_vertices_per_bin,
):
    """Identify matched true vertices and count them by track count bin.

    Each reconstructed vertex matches to at most one true vertex. Counts both total
    true vertices (denominator) and successfully matched vertices (numerator) for
    efficiency calculation. Updates true_vertices_per_bin and matched_vertices_per_bin in place.

    Args:
        true_vertex_info: Dict of vertex_id to {track_indices, num_tracks}
        rec_track_indices: Variable-length array of track indices per rec vertex
        recall_threshold: Recall threshold for matching
        purity_threshold: Purity threshold for matching
        bins: List of (bin_start, bin_end) tuples
        true_vertices_per_bin: Dict to update with true vertex counts (denominator)
        matched_vertices_per_bin: Dict to update with matched vertex counts (numerator)
    """
    # Count true vertices per bin (denominator) using reconstructed track count
    for vertex_data in true_vertex_info.values():
        num_tracks = vertex_data["num_tracks"]

        # Update bin counts (denominator)
        for bin_start, bin_end in bins:
            if bin_start <= num_tracks <= bin_end:
                true_vertices_per_bin[bin_start, bin_end] += 1
                break

    # Match reconstructed vertices and collect all matched true vertex IDs (numerator)
    matched_true_ids = set()
    for track_indices in rec_track_indices:
        rec_track_set = set(track_indices)

        if not rec_track_set:
            continue

        best_match_id = find_best_match(
            rec_track_set, true_vertex_info, recall_threshold, purity_threshold
        )
        if best_match_id is not None:
            matched_true_ids.add(best_match_id)

    # Fill all collected matched vertices per bin (numerator) using reconstructed track count
    for matched_id in matched_true_ids:
        num_tracks = true_vertex_info[matched_id]["num_tracks"]
        for bin_start, bin_end in bins:
            if bin_start <= num_tracks <= bin_end:
                matched_vertices_per_bin[bin_start, bin_end] += 1
                break


def _calculate_efficiencies(bins, true_vertices_per_bin, matched_vertices_per_bin):
    """Calculate efficiency and Clopper-Pearson confidence intervals for each bin."""
    efficiencies = {}
    errors = {}

    for bin_tuple in bins:
        num_true = true_vertices_per_bin[bin_tuple]
        num_matched = matched_vertices_per_bin[bin_tuple]
        efficiency = num_matched / num_true if num_true > 0 else 0.0
        efficiencies[bin_tuple] = efficiency

        error_lower, error_upper = calculate_clopper_pearson_errors(
            num_matched, num_true, efficiency
        )
        errors[bin_tuple] = {"lower": error_lower, "upper": error_upper}

    return efficiencies, errors


def _calculate_overall_efficiency(true_vertices_per_bin, matched_vertices_per_bin):
    """Calculate overall efficiency and error bounds.

    Returns:
        dict: Dictionary containing overall statistics:
            - efficiency: overall efficiency value
            - error_lower: lower error bound
            - error_upper: upper error bound
            - total_true: total number of true vertices
            - total_matched: total number of matched vertices
    """
    total_true = sum(true_vertices_per_bin.values())
    total_matched = sum(matched_vertices_per_bin.values())
    overall_efficiency = total_matched / total_true if total_true > 0 else 0.0

    # Calculate Clopper-Pearson confidence intervals
    error_lower, error_upper = calculate_clopper_pearson_errors(
        total_matched, total_true, overall_efficiency
    )

    return {
        "efficiency": overall_efficiency,
        "error_lower": error_lower,
        "error_upper": error_upper,
        "total_true": total_true,
        "total_matched": total_matched,
    }


def _prepare_plot_data(
    bins,
    efficiencies,
    efficiency_errors,
    true_vertices_per_bin,
    matched_vertices_per_bin,
):
    """Prepare data for plotting, filtering out bins with no data."""
    bin_centers = [(start + end) / 2 for start, end in bins]
    num_true_list = [true_vertices_per_bin[bin_tuple] for bin_tuple in bins]
    num_matched_list = [matched_vertices_per_bin[bin_tuple] for bin_tuple in bins]

    # Filter bins with data
    x_values, eff_values, err_lower, err_upper = [], [], [], []
    num_true_filtered, num_matched_filtered = [], []

    for idx, bin_tuple in enumerate(bins):
        if num_true_list[idx] > 0:
            x_values.append(bin_centers[idx])
            eff_values.append(efficiencies[bin_tuple])
            err_lower.append(efficiency_errors[bin_tuple]["lower"])
            err_upper.append(efficiency_errors[bin_tuple]["upper"])
            num_true_filtered.append(num_true_list[idx])
            num_matched_filtered.append(num_matched_list[idx])

    if not x_values:
        return None

    return {
        "x": x_values,
        "efficiency": eff_values,
        "error_lower": err_lower,
        "error_upper": err_upper,
        "num_true_filtered": num_true_filtered,
        "num_matched_filtered": num_matched_filtered,
        "bin_centers": bin_centers,
        "num_true_all": num_true_list,
        "num_matched_all": num_matched_list,
    }


def _create_efficiency_plot(
    plot_data, bin_width, recall_threshold, purity_threshold, output_dir, method_name
):
    """Create and save the efficiency plot with confidence intervals."""
    x = plot_data["x"]
    efficiency = plot_data["efficiency"]
    num_true = plot_data["num_true_filtered"]
    num_matched = plot_data["num_matched_filtered"]

    _, ax1 = plt.subplots(figsize=(9, 6))
    ax2 = ax1.twinx()

    # Plot efficiency with confidence intervals
    lower_bounds = [
        eff - err for eff, err in zip(efficiency, plot_data["error_lower"], strict=True)
    ]
    upper_bounds = [
        eff + err for eff, err in zip(efficiency, plot_data["error_upper"], strict=True)
    ]

    ax1.plot(
        x,
        efficiency,
        "bo-",
        linewidth=2,
        markersize=8,
        label="Efficiency (Track-based)",
    )
    ax1.fill_between(
        x,
        lower_bounds,
        upper_bounds,
        alpha=0.3,
        color="blue",
        label="Clopper-Pearson 1-sigma CI",
    )

    ax1.set_xlabel("Number of tracks per primary vertex", fontsize=12)
    ax1.set_ylabel("Efficiency", fontsize=12)
    ax1.set_title(
        f"Vertex Reconstruction Efficiency vs Track Count\nRecall≥{recall_threshold}, Purity≥{purity_threshold}",
        fontsize=14,
    )
    ax1.set_ylim(0, 1.27)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    # Add bar charts for counts
    x_array = np.array(x)
    max_count = max(max(num_true), max(num_matched))
    y_max = max_count * 2.3

    ax2.bar(
        x_array,
        num_true,
        bin_width,
        alpha=0.8,
        color="lightblue",
        label="Total true vertices",
        linewidth=0.5,
    )
    ax2.bar(
        x_array,
        num_matched,
        bin_width,
        alpha=0.9,
        color="darkorange",
        label="Matched vertices",
        linewidth=0.5,
    )

    ax2.set_ylim(0, y_max)
    ax2.set_yticklabels([])
    ax2.tick_params(axis="y", which="both", length=0)
    ax2.legend(loc="upper right")

    # Add numerical labels on bars
    for x_pos, n_true, n_matched in zip(x_array, num_true, num_matched, strict=True):
        if n_true > 0:
            ax2.text(
                x_pos,
                n_true + y_max * 0.03,
                str(n_true),
                ha="center",
                va="bottom",
                fontsize=7,
                color="darkblue",
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "white",
                    "alpha": 0.8,
                    "edgecolor": "none",
                },
            )
        if n_matched > 0:
            y_pos = (
                n_matched / 2
                if n_matched >= max_count * 0.15
                else n_matched + y_max * 0.01
            )
            ax2.text(
                x_pos,
                y_pos,
                str(n_matched),
                ha="center",
                va="center",
                fontsize=7,
                weight="bold",
                color="white",
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "darkred",
                    "alpha": 0.9,
                    "edgecolor": "none",
                },
            )

    plt.tight_layout()

    # Save plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    method_slug = method_name.lower().replace(" ", "_")
    plot_file = output_path / f"efficiency_true_binned_{method_slug}.pdf"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()


def _print_results(
    recall_threshold,
    purity_threshold,
    overall_stats,
    efficiency_values,
    bins,
    efficiency_errors,
    true_vertices_per_bin,
):
    """Print efficiency analysis summary to console."""
    print(f"\n{'-' * 60}")
    print("EFFICIENCY ANALYSIS SUMMARY - HDF5 DATA")
    print(f"{'-' * 60}")
    print(
        f"Matching criteria: Recall ≥ {recall_threshold}, Purity ≥ {purity_threshold}"
    )
    eff = overall_stats["efficiency"]
    err_upper = overall_stats["error_upper"]
    err_lower = overall_stats["error_lower"]
    matched = overall_stats["total_matched"]
    total = overall_stats["total_true"]
    print(
        f"Overall efficiency: {eff:.3f} +{err_upper:.3f}/-{err_lower:.3f} ({matched}/{total})"
    )

    if efficiency_values:
        # Map efficiency values to bins (only bins with data)
        bins_with_data = [
            bin_tuple for bin_tuple in bins if true_vertices_per_bin[bin_tuple] > 0
        ]

        # Find best efficiency
        best_idx = int(np.argmax(efficiency_values))
        best_bin = bins_with_data[best_idx]
        best_eff = efficiency_values[best_idx]
        best_err = efficiency_errors[best_bin]

        # Find worst efficiency (excluding zero)
        non_zero_effs = [(i, eff) for i, eff in enumerate(efficiency_values) if eff > 0]
        if non_zero_effs:
            worst_idx, worst_eff = min(non_zero_effs, key=operator.itemgetter(1))
            worst_bin = bins_with_data[worst_idx]
            worst_err = efficiency_errors[worst_bin]
            print(
                f"Best efficiency: {best_eff:.3f} +{best_err['upper']:.3f}/-{best_err['lower']:.3f} in bin [{best_bin[0]}, {best_bin[1]}]"
            )
            print(
                f"Worst efficiency: {worst_eff:.3f} +{worst_err['upper']:.3f}/-{worst_err['lower']:.3f} in bin [{worst_bin[0]}, {worst_bin[1]}]"
            )
        else:
            print(
                f"Best efficiency: {best_eff:.3f} +{best_err['upper']:.3f}/-{best_err['lower']:.3f} in bin [{best_bin[0]}, {best_bin[1]}]"
            )

    print(f"Total true vertices analyzed: {overall_stats['total_true']}")
    print(f"Total matched vertices: {overall_stats['total_matched']}")
    print(f"{'-' * 60}\n")


def _save_efficiency_to_csv(
    bins,
    plot_data,
    efficiencies,
    efficiency_errors,
    true_vertices_per_bin,
    matched_vertices_per_bin,
    overall_stats,
    recall_threshold,
    purity_threshold,
    output_dir,
    method_name,
):
    """Save efficiency results to CSV files for comparison with other methods."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save detailed per-bin results
    bin_data = []
    for i, bin_tuple in enumerate(bins):
        bin_start, bin_end = bin_tuple
        bin_center = plot_data["bin_centers"][i]
        efficiency = efficiencies[bin_tuple]
        error_lower = efficiency_errors[bin_tuple]["lower"]
        error_upper = efficiency_errors[bin_tuple]["upper"]
        num_true = true_vertices_per_bin[bin_tuple]
        num_matched = matched_vertices_per_bin[bin_tuple]

        bin_data.append(
            {
                "bin_start": bin_start,
                "bin_end": bin_end,
                "bin_center": bin_center,
                "efficiency": efficiency,
                "error_lower": error_lower,
                "error_upper": error_upper,
                "num_true": num_true,
                "num_matched": num_matched,
            }
        )

    df = pd.DataFrame(bin_data)
    method_slug = method_name.lower().replace(" ", "_")
    csv_path = output_path / f"efficiency_true_binned_{method_slug}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved bin-level efficiency results to: {csv_path}")

    # Save overall summary
    summary_df = pd.DataFrame(
        [
            {
                "total_matched": overall_stats["total_matched"],
                "total_true": overall_stats["total_true"],
                "overall_efficiency": overall_stats["efficiency"],
                "overall_error_lower": overall_stats["error_lower"],
                "overall_error_upper": overall_stats["error_upper"],
                "recall_threshold": recall_threshold,
                "purity_threshold": purity_threshold,
                "method_name": method_name,
            }
        ]
    )

    overall_path = output_path / f"efficiency_overall_true_binned_{method_slug}.csv"
    summary_df.to_csv(overall_path, index=False)
    print(f"Saved overall efficiency statistics to: {overall_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze vertex reconstruction efficiency from HDF5 results file"
    )
    parser.add_argument(
        "--hdf5_path",
        type=str,
        required=True,
        help="Path to the HDF5 file containing rec_vertices and true_vertices",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Output directory for plots and CSV files",
    )
    parser.add_argument(
        "--recall_threshold",
        type=float,
        default=0.7,
        help="Recall threshold for matching (default: 0.7)",
    )
    parser.add_argument(
        "--purity_threshold",
        type=float,
        default=0.7,
        help="Purity threshold for matching (default: 0.7)",
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
        help="Method name for comparison plots and CSV files",
    )
    args = parser.parse_args()

    analyze_vertex_efficiency(
        hdf5_path=args.hdf5_path,
        output_dir=args.output_dir,
        recall_threshold=args.recall_threshold,
        purity_threshold=args.purity_threshold,
        plot_min_ntracks=args.plot_min_ntracks,
        plot_max_ntracks=args.plot_max_ntracks,
        plot_bin_width_ntracks=args.plot_bin_width_ntracks,
        method_name=args.method_name,
    )
