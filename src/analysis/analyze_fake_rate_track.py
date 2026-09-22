from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from .utils import calculate_clopper_pearson_errors


def analyze_vertex_fake_rate(
    hdf5_path,
    output_dir="output",
    recall_threshold=0.7,
    purity_threshold=0.7,
    plot_min_ntracks=4,
    plot_max_ntracks=200,
    plot_bin_width_ntracks=5,
    method_name="HepATTn",
):
    """Analyze fake vertex rate from HDF5 results.

    A reconstructed vertex is considered a "fake vertex" if it is NOT claimed by any true vertex.
    This means: for all true vertices, neither recall nor purity exceeds the thresholds.

    Args:
        hdf5_path: Path to the HDF5 file containing rec_vertices and true_vertices
        output_dir: Output directory for plots and CSV files
        recall_threshold: Recall threshold for matching (intersection / num_true_tracks)
        purity_threshold: Purity threshold for matching (intersection / num_rec_tracks)
        plot_min_ntracks: Minimum nTracks for binning/plots
        plot_max_ntracks: Maximum nTracks for binning/plots
        plot_bin_width_ntracks: Bin width in nTracks
        method_name: Name of the method for plots and output filenames
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
    total_rec_per_bin = dict.fromkeys(bins, 0)
    fake_rec_per_bin = dict.fromkeys(bins, 0)

    print(f"Loading HDF5 file: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as f:
        event_ids = sorted([int(key) for key in f])
        print(f"Found {len(event_ids)} events")

        for event_id in tqdm(event_ids, desc="Processing events"):
            event_group = f[str(event_id)]

            # Load true vertices from HDF5 file
            true_verts = event_group["true_vertices"]
            true_track_indices = true_verts["track_indices"][:]

            # Build true vertex info
            true_vertex_info = {}
            for vertex_idx, track_indices in enumerate(true_track_indices):
                true_vertex_info[vertex_idx] = {
                    "track_indices": set(track_indices),
                }

            # Load reconstructed vertices from HDF5 file
            rec_verts = event_group["rec_vertices"]
            rec_track_indices = rec_verts["track_indices"][:]

            # Identify and count fake vertices by track count bin
            _count_fake_vertices_by_bin(
                true_vertex_info,
                rec_track_indices,
                recall_threshold,
                purity_threshold,
                bins,
                total_rec_per_bin,
                fake_rec_per_bin,
            )

    # Calculate fake rates and errors per bin
    fake_rates, fake_rate_errors = _calculate_fake_rates(
        bins, total_rec_per_bin, fake_rec_per_bin
    )

    # Calculate overall fake rate and error bounds
    overall_stats = _calculate_overall_fake_rate(total_rec_per_bin, fake_rec_per_bin)

    # Prepare plotting data (filter bins with no data)
    plot_data = _prepare_plot_data(
        bins, fake_rates, fake_rate_errors, total_rec_per_bin, fake_rec_per_bin
    )

    if plot_data is None:
        print("No fake rate data to plot")
        return

    # Create visualization
    _create_fake_rate_plots(
        plot_data,
        recall_threshold,
        purity_threshold,
        output_dir,
        method_name,
    )

    # Print summary
    _print_fake_rate_summary(
        overall_stats,
        recall_threshold,
        purity_threshold,
        fake_rate_errors,
        total_rec_per_bin,
        fake_rec_per_bin,
    )

    # Save results to CSV
    _save_fake_rate_to_csv(
        bins,
        plot_data,
        fake_rates,
        fake_rate_errors,
        total_rec_per_bin,
        fake_rec_per_bin,
        overall_stats,
        recall_threshold,
        purity_threshold,
        output_dir,
        method_name,
    )


def _count_fake_vertices_by_bin(
    true_vertex_info,
    rec_track_indices,
    recall_threshold,
    purity_threshold,
    bins,
    total_rec_per_bin,
    fake_rec_per_bin,
):
    """Identify fake reconstructed vertices and count them by track count bin.

    Each true vertex claims its best matching rec vertex. Unclaimed rec vertices
    are considered fake. Updates total_rec_per_bin and fake_rec_per_bin in place.
    """
    # Build rec vertex info from new HDF5 format
    rec_vertices = {}
    for rec_idx, track_indices in enumerate(rec_track_indices):
        num_tracks = len(track_indices)
        if num_tracks == 0:
            continue

        rec_vertices[rec_idx] = {
            "track_indices": set(track_indices),
            "num_tracks": num_tracks,
            "claimed_by": None,  # Will store (true_vertex_id, recall)
        }

    # Each true vertex claims its best rec vertex
    for true_vertex_id, true_vertex_data in true_vertex_info.items():
        true_track_indices = true_vertex_data["track_indices"]
        num_true_tracks = len(true_track_indices)

        best_rec_idx = None
        best_recall = 0.0

        # Find best rec vertex for this true vertex
        for rec_idx, rec_data in rec_vertices.items():
            rec_track_indices = rec_data["track_indices"]
            num_rec_tracks = rec_data["num_tracks"]

            num_shared = len(rec_track_indices & true_track_indices)
            if num_shared == 0:
                continue

            purity = num_shared / num_rec_tracks
            recall = num_shared / num_true_tracks

            # Qualified if either purity or recall meets their threshold (OR condition)
            if (
                purity >= purity_threshold or recall >= recall_threshold
            ) and recall > best_recall:
                best_recall = recall
                best_rec_idx = rec_idx

        # Claim the best rec vertex
        if best_rec_idx is not None:
            current_claim = rec_vertices[best_rec_idx]["claimed_by"]
            # If unclaimed or current claim has lower recall, claim it
            if current_claim is None or best_recall > current_claim[1]:
                rec_vertices[best_rec_idx]["claimed_by"] = (true_vertex_id, best_recall)

    # Count fakes (unclaimed rec vertices) and update bin counts
    for rec_data in rec_vertices.values():
        num_tracks = rec_data["num_tracks"]
        is_fake = rec_data["claimed_by"] is None

        # Update bin counts
        for bin_start, bin_end in bins:
            if bin_start <= num_tracks <= bin_end:
                total_rec_per_bin[bin_start, bin_end] += 1
                if is_fake:
                    fake_rec_per_bin[bin_start, bin_end] += 1
                break


def _calculate_fake_rates(bins, total_rec_per_bin, fake_rec_per_bin):
    """Calculate fake rate and Clopper-Pearson confidence intervals for each bin.

    Args:
        bins: List of (bin_start, bin_end) tuples
        total_rec_per_bin: Dictionary mapping bin tuples to total reconstructed vertex counts
        fake_rec_per_bin: Dictionary mapping bin tuples to fake vertex counts

    Returns:
        tuple: (fake_rates, errors) where both are dictionaries keyed by bin tuples
    """
    fake_rates = {}
    errors = {}

    for bin_tuple in bins:
        total = total_rec_per_bin[bin_tuple]
        fake = fake_rec_per_bin[bin_tuple]
        rate = fake / total if total > 0 else 0.0
        fake_rates[bin_tuple] = rate

        error_lower, error_upper = calculate_clopper_pearson_errors(fake, total, rate)
        errors[bin_tuple] = {"lower": error_lower, "upper": error_upper}

    return fake_rates, errors


def _calculate_overall_fake_rate(total_rec_per_bin, fake_rec_per_bin):
    """Calculate overall fake rate and error bounds.

    Returns:
        dict: Dictionary containing overall statistics:
            - fake_rate: overall fake rate value
            - error_lower: lower error bound
            - error_upper: upper error bound
            - total_rec: total number of reconstructed vertices
            - total_fake: total number of fake vertices
    """
    total_rec = sum(total_rec_per_bin.values())
    total_fake = sum(fake_rec_per_bin.values())
    fake_rate = total_fake / total_rec if total_rec > 0 else 0.0

    # Calculate Clopper-Pearson confidence intervals
    error_lower, error_upper = calculate_clopper_pearson_errors(
        total_fake, total_rec, fake_rate
    )

    return {
        "fake_rate": fake_rate,
        "error_lower": error_lower,
        "error_upper": error_upper,
        "total_rec": total_rec,
        "total_fake": total_fake,
    }


def _prepare_plot_data(
    bins, fake_rates, fake_rate_errors, total_rec_per_bin, fake_rec_per_bin
):
    """Prepare data for plotting, filtering out bins with no data.

    Args:
        bins: List of (bin_start, bin_end) tuples
        fake_rates: Dictionary mapping bin tuples to fake rates
        fake_rate_errors: Dictionary mapping bin tuples to error dicts with 'lower' and 'upper' keys
        total_rec_per_bin: Dictionary mapping bin tuples to total reconstructed vertex counts
        fake_rec_per_bin: Dictionary mapping bin tuples to fake vertex counts

    Returns:
        dict or None: Dictionary containing plot data, or None if no data
    """
    bin_centers = [(start + end) / 2 for start, end in bins]
    total_list = [total_rec_per_bin[bin_tuple] for bin_tuple in bins]
    fake_list = [fake_rec_per_bin[bin_tuple] for bin_tuple in bins]

    # Filter bins with data
    x_values, rate_values, err_lower, err_upper = [], [], [], []
    total_filtered, fake_filtered = [], []

    for idx, bin_tuple in enumerate(bins):
        if total_list[idx] > 0:
            x_values.append(bin_centers[idx])
            rate_values.append(fake_rates[bin_tuple])
            err_lower.append(fake_rate_errors[bin_tuple]["lower"])
            err_upper.append(fake_rate_errors[bin_tuple]["upper"])
            total_filtered.append(total_list[idx])
            fake_filtered.append(fake_list[idx])

    if not x_values:
        return None

    return {
        "x": x_values,
        "fake_rate": rate_values,
        "error_lower": err_lower,
        "error_upper": err_upper,
        "total_filtered": total_filtered,
        "fake_filtered": fake_filtered,
        "bin_centers": bin_centers,
        "total_all": total_list,
        "fake_all": fake_list,
    }


def _create_fake_rate_plots(
    plot_data,
    recall_threshold,
    purity_threshold,
    output_dir,
    method_name,
):
    """Create visualization plot for fake rate analysis."""
    x = plot_data["x"]
    fake_rates = plot_data["fake_rate"]

    # Create the plot
    _, ax = plt.subplots(figsize=(9, 6))

    # Plot fake rate with confidence intervals
    lower_bounds = [
        rate - err
        for rate, err in zip(fake_rates, plot_data["error_lower"], strict=True)
    ]
    upper_bounds = [
        rate + err
        for rate, err in zip(fake_rates, plot_data["error_upper"], strict=True)
    ]

    ax.plot(
        x,
        fake_rates,
        "ro-",
        linewidth=2,
        markersize=8,
        label="Fake Rate (not claimed by any true vertex)",
    )
    ax.fill_between(
        x,
        lower_bounds,
        upper_bounds,
        alpha=0.3,
        color="red",
        label="Clopper-Pearson 1-sigma CI",
    )

    ax.set_xlabel("Number of tracks per reconstructed vertex", fontsize=12)
    ax.set_ylabel("Fake Rate", fontsize=12)
    ax.set_title(
        f"Reconstructed Vertex Fake Rate vs Track Count\n"
        f"Recall threshold: {recall_threshold:.0%}, Purity threshold: {purity_threshold:.0%}",
        fontsize=14,
    )
    ax.set_ylim(0, 1.27)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    plt.tight_layout()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    method_slug = method_name.lower().replace(" ", "_")
    plt.savefig(
        output_path / f"fake_rate_rec_binned_{method_slug}.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def _print_fake_rate_summary(
    overall_stats,
    recall_threshold,
    purity_threshold,
    fake_rate_errors,
    total_rec_per_bin,
    fake_rec_per_bin,
):
    """Print detailed summary of fake rate analysis."""
    print(f"\n{'-' * 70}")
    print("FAKE RATE ANALYSIS SUMMARY")
    print(f"{'-' * 70}")
    print(
        f"Recall threshold: {recall_threshold:.0%}, Purity threshold: {purity_threshold:.0%}"
    )
    print(f"Total reconstructed vertices analyzed: {overall_stats['total_rec']}")
    print(f"Fake vertices detected: {overall_stats['total_fake']}")
    print("\nOverall Fake Rate:")
    print(
        f"  Fake rate: {overall_stats['fake_rate']:.3f} +{overall_stats['error_upper']:.3f} -{overall_stats['error_lower']:.3f}"
    )

    # Detailed breakdown by track count
    print("\nFake Rate by Track Count:")
    print(
        f"{'Track Count':<12} {'Total':<8} {'Fake':<10} {'Fake Rate':<12} {'Err(+/-)':<18}"
    )
    print(f"{'-' * 60}")

    track_counts = sorted(total_rec_per_bin.keys(), key=lambda b: (b[0] + b[1]) / 2)
    for bin_start, bin_end in track_counts:
        total = total_rec_per_bin[bin_start, bin_end]
        fake = fake_rec_per_bin[bin_start, bin_end]
        if total > 0:
            fake_rate_bin = fake / total
            err = fake_rate_errors[bin_start, bin_end]
            print(
                f"[{bin_start}, {bin_end}]".ljust(12)
                + f" {total:<8} {fake:<10} {fake_rate_bin:<12.3f} +{err['upper']:.3f} -{err['lower']:.3f}"
            )

    print(f"{'-' * 60}")
    print("Fake Definition:")
    print("  A reconstructed vertex is fake if:")
    print("  - It is NOT claimed by any true vertex")
    print("  - Claiming rule (true → rec):")
    print(
        f"    * A true vertex considers rec vertices with purity >= {purity_threshold:.0%} OR recall >= {recall_threshold:.0%}"
    )
    print(
        "    * It claims the candidate with the highest recall (ties resolved by higher recall only)"
    )
    print(f"{'-' * 70}\n")


def _save_fake_rate_to_csv(
    bins,
    plot_data,
    fake_rates,
    fake_rate_errors,
    total_rec_per_bin,
    fake_rec_per_bin,
    overall_stats,
    recall_threshold,
    purity_threshold,
    output_dir,
    method_name,
):
    """Save fake rate results to CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save detailed per-bin results
    bin_data = []
    for i, bin_tuple in enumerate(bins):
        bin_start, bin_end = bin_tuple
        bin_center = plot_data["bin_centers"][i]
        fake_rate = fake_rates[bin_tuple]
        error_lower = fake_rate_errors[bin_tuple]["lower"]
        error_upper = fake_rate_errors[bin_tuple]["upper"]
        num_total = total_rec_per_bin[bin_tuple]
        num_fake = fake_rec_per_bin[bin_tuple]

        bin_data.append(
            {
                "bin_start": bin_start,
                "bin_end": bin_end,
                "bin_center": bin_center,
                "fake_rate": fake_rate,
                "error_lower": error_lower,
                "error_upper": error_upper,
                "num_total": num_total,
                "num_fake": num_fake,
            }
        )

    df = pd.DataFrame(bin_data)
    method_slug = method_name.lower().replace(" ", "_")
    csv_path = output_path / f"fake_rate_rec_binned_{method_slug}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved bin-level fake rate results to: {csv_path}")

    # Save overall summary
    summary_df = pd.DataFrame(
        [
            {
                "total_fake": overall_stats["total_fake"],
                "total_rec": overall_stats["total_rec"],
                "overall_fake_rate": overall_stats["fake_rate"],
                "overall_error_lower": overall_stats["error_lower"],
                "overall_error_upper": overall_stats["error_upper"],
                "recall_threshold": recall_threshold,
                "purity_threshold": purity_threshold,
                "method_name": method_name,
            }
        ]
    )

    overall_path = output_path / f"fake_rate_overall_rec_binned_{method_slug}.csv"
    summary_df.to_csv(overall_path, index=False)
    print(f"Saved overall fake rate statistics to: {overall_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze fake vertex rate from HDF5 results file"
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
        help="Recall threshold for matching (intersection / num_true_tracks)",
    )
    parser.add_argument(
        "--purity_threshold",
        type=float,
        default=0.7,
        help="Purity threshold for matching (intersection / num_rec_tracks)",
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
        help="Method name for plots and output filenames",
    )
    args = parser.parse_args()

    analyze_vertex_fake_rate(
        hdf5_path=args.hdf5_path,
        output_dir=args.output_dir,
        recall_threshold=args.recall_threshold,
        purity_threshold=args.purity_threshold,
        plot_min_ntracks=args.plot_min_ntracks,
        plot_max_ntracks=args.plot_max_ntracks,
        plot_bin_width_ntracks=args.plot_bin_width_ntracks,
        method_name=args.method_name,
    )
