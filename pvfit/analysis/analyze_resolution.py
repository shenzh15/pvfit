"""Resolution analysis for vertex reconstruction using track-based matching.

This module calculates vertex position resolution (RMS) by analyzing residuals
between reconstructed and true vertices, using track-based matching criteria.
"""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import find_best_match


def analyze_vertex_resolution(
    hdf5_path,
    output_dir,
    recall_threshold,
    purity_threshold,
    plot_min_ntracks,
    plot_max_ntracks,
    plot_bin_width_ntracks,
    x_residual_range,
    y_residual_range,
    z_residual_range,
    t_residual_range,
    method_name,
):
    """Analyze vertex position resolution from HDF5 results.

    Uses track-based matching to pair reconstructed vertices with true vertices,
    then calculates position resolution (RMS of residuals) binned by true vertex
    track count.

    Args:
        hdf5_path: Path to the HDF5 file containing rec_vertices and true_vertices
        output_dir: Output directory for plots and CSV files
        recall_threshold: Recall threshold for matching (intersection / num_true_tracks)
        purity_threshold: Purity threshold for matching (intersection / num_rec_tracks)
        plot_min_ntracks: Minimum nTracks for binning/plots
        plot_max_ntracks: Maximum nTracks for binning/plots
        plot_bin_width_ntracks: Bin width in nTracks
        x_residual_range: X residual range limit in mm (default: 0.1)
        y_residual_range: Y residual range limit in mm (default: 0.1)
        z_residual_range: Z residual range limit in mm (default: 0.7)
        t_residual_range: T residual range limit in ns (default: 0.06)
        method_name: Name of the method for plots and CSV files

    Returns:
        None: Saves resolution plots and CSV files to output directory
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

    # Initialize dictionaries to store residuals for each bin
    bin_residuals = {
        "x": {bin_tuple: [] for bin_tuple in bins},
        "y": {bin_tuple: [] for bin_tuple in bins},
        "z": {bin_tuple: [] for bin_tuple in bins},
        "t": {bin_tuple: [] for bin_tuple in bins},
    }

    print(f"Loading HDF5 file: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as f:
        event_ids = sorted(int(key) for key in f)
        print(f"Found {len(event_ids)} events")

        for event_id in tqdm(event_ids, desc="Processing events"):
            event_group = f[str(event_id)]

            # Extract true vertices
            true_verts = event_group["true_vertices"]
            true_x = true_verts["x"][:]
            true_y = true_verts["y"][:]
            true_z = true_verts["z"][:]
            true_t = true_verts["t"][:]
            true_track_indices = true_verts["track_indices"][:]

            # Extract reconstructed vertices
            rec_verts = event_group["rec_vertices"]
            rec_x = rec_verts["x"][:]
            rec_y = rec_verts["y"][:]
            rec_z = rec_verts["z"][:]
            rec_t = rec_verts["t"][:]
            rec_track_indices = rec_verts["track_indices"][:]

            # Build true vertex info
            true_vertex_info = {}
            for vertex_idx, track_indices in enumerate(true_track_indices):
                true_vertex_info[vertex_idx] = {
                    "track_indices": set(track_indices),
                    "num_tracks": len(track_indices),
                    "x": true_x[vertex_idx],
                    "y": true_y[vertex_idx],
                    "z": true_z[vertex_idx],
                    "t": true_t[vertex_idx],
                }

            # Match rec vertices to true vertices and collect residuals
            _collect_residuals_by_bin(
                true_vertex_info,
                rec_x,
                rec_y,
                rec_z,
                rec_t,
                rec_track_indices,
                recall_threshold,
                purity_threshold,
                bins,
                bin_residuals,
                x_residual_range,
                y_residual_range,
                z_residual_range,
                t_residual_range,
            )

    # Calculate resolutions and errors for each coordinate
    resolutions = {}
    resolution_errors = {}
    for coord in ["x", "y", "z", "t"]:
        resolutions[coord], resolution_errors[coord] = _calculate_resolutions(
            bins, bin_residuals[coord]
        )

    # Calculate overall resolutions
    overall_stats = _calculate_overall_resolution(bin_residuals)

    # Prepare plotting data
    plot_data = _prepare_plot_data(bins, resolutions, resolution_errors, bin_residuals)

    if plot_data is None:
        print("No resolution data to plot")
        return

    # Create and save resolution plots
    _create_resolution_plots(
        plot_data,
        recall_threshold,
        purity_threshold,
        x_residual_range,
        y_residual_range,
        z_residual_range,
        t_residual_range,
        output_dir,
        method_name,
    )

    # Create residual histograms
    _create_residual_histograms(
        bins,
        bin_residuals,
        resolutions,
        x_residual_range,
        y_residual_range,
        z_residual_range,
        t_residual_range,
        output_dir,
        method_name,
    )

    # Print summary results
    _print_results(
        recall_threshold,
        purity_threshold,
        overall_stats,
        resolutions,
        resolution_errors,
        bins,
        bin_residuals,
    )

    # Save results to CSV
    _save_resolution_to_csv(
        bins,
        plot_data,
        resolutions,
        resolution_errors,
        bin_residuals,
        overall_stats,
        recall_threshold,
        purity_threshold,
        output_dir,
        method_name,
    )


def _collect_residuals_by_bin(
    true_vertex_info,
    rec_x,
    rec_y,
    rec_z,
    rec_t,
    rec_track_indices,
    recall_threshold,
    purity_threshold,
    bins,
    bin_residuals,
    x_residual_range,
    y_residual_range,
    z_residual_range,
    t_residual_range,
):
    """Match rec vertices to true vertices and collect residuals by bin.

    For each reconstructed vertex, finds the best matching true vertex
    (if any) and calculates position residuals, storing them in the
    appropriate track count bin.

    Args:
        true_vertex_info: Dict of vertex_id to {track_indices, num_tracks, x, y, z, t}
        rec_x, rec_y, rec_z, rec_t: Arrays of reconstructed vertex positions
        rec_track_indices: Variable-length array of track indices per rec vertex
        recall_threshold: Recall threshold for matching
        purity_threshold: Purity threshold for matching
        bins: List of (bin_start, bin_end) tuples
        bin_residuals: Dict to update with residuals per coordinate per bin
        x_residual_range: X residual range limit in mm
        y_residual_range: Y residual range limit in mm
        z_residual_range: Z residual range limit in mm
        t_residual_range: T residual range limit in ns
    """
    for rec_idx, track_indices in enumerate(rec_track_indices):
        rec_track_set = set(track_indices)

        if not rec_track_set:
            continue

        # Find best matching true vertex
        best_match_id = find_best_match(
            rec_track_set, true_vertex_info, recall_threshold, purity_threshold
        )

        if best_match_id is not None:
            # Get true vertex info
            true_vertex = true_vertex_info[best_match_id]
            num_tracks = true_vertex["num_tracks"]

            # Calculate residuals
            x_residual = rec_x[rec_idx] - true_vertex["x"]
            y_residual = rec_y[rec_idx] - true_vertex["y"]
            z_residual = rec_z[rec_idx] - true_vertex["z"]
            t_residual = rec_t[rec_idx] - true_vertex["t"]

            # Find which bin this vertex belongs to and store residuals
            for bin_start, bin_end in bins:
                if bin_start <= num_tracks <= bin_end:
                    bin_tuple = (bin_start, bin_end)

                    # Only store residuals if they are within the specified ranges
                    if -x_residual_range <= x_residual <= x_residual_range:
                        bin_residuals["x"][bin_tuple].append(x_residual)
                    if -y_residual_range <= y_residual <= y_residual_range:
                        bin_residuals["y"][bin_tuple].append(y_residual)
                    if -z_residual_range <= z_residual <= z_residual_range:
                        bin_residuals["z"][bin_tuple].append(z_residual)
                    if -t_residual_range <= t_residual <= t_residual_range:
                        bin_residuals["t"][bin_tuple].append(t_residual)
                    break


def _calculate_resolutions(bins, bin_residuals_coord):
    """Calculate resolution (RMS) and uncertainties for each bin.

    Uses the statistical formulas:
    - Resolution = RMS = sqrt(sample_variance with ddof=1)
    - Var(s^2) ≈ (1/n)[m4 - ((n-3)/(n-1))s^4]
    - Error on std: sqrt(Var(s^2)/(4s^2))

    Args:
        bins: List of (bin_start, bin_end) tuples
        bin_residuals_coord: Dict of bin_tuple to list of residuals for one coordinate

    Returns:
        tuple: (resolutions_dict, errors_dict)
    """
    resolutions = {}
    errors = {}

    for bin_tuple in bins:
        residuals = bin_residuals_coord[bin_tuple]

        if len(residuals) > 1:  # Need at least 2 points for variance
            residuals_array = np.array(residuals)
            n = len(residuals_array)

            # Calculate sample variance (unbiased estimator)
            sample_var = np.var(residuals_array, ddof=1)
            resolutions[bin_tuple] = np.sqrt(sample_var)

            # Calculate variance of sample variance using:
            # Var(s^2) ≈ (1/n) [ m4 - ((n-3)/(n-1)) s^4 ]
            mean_residual = np.mean(residuals_array)
            centered_residuals = residuals_array - mean_residual
            m4 = np.mean(centered_residuals**4)  # Fourth central moment

            var_of_sample_var = (1 / n) * (m4 - ((n - 3) / (n - 1)) * sample_var**2)

            # Error on standard deviation: Var(s) ≈ Var(s^2) / (4s^2)
            if sample_var > 0 and var_of_sample_var > 0:
                errors[bin_tuple] = np.sqrt(var_of_sample_var / (4 * sample_var))
            else:
                errors[bin_tuple] = 0.0
        else:
            resolutions[bin_tuple] = 0.0
            errors[bin_tuple] = 0.0

    return resolutions, errors


def _calculate_overall_resolution(bin_residuals):
    """Calculate overall resolution for each coordinate.

    Args:
        bin_residuals: Dict of coordinate to dict of bin_tuple to residuals

    Returns:
        dict: Dictionary containing overall resolution statistics per coordinate
    """
    overall_stats = {}

    for coord in ["x", "y", "z", "t"]:
        # Collect all residuals across bins
        all_residuals = []
        for residuals in bin_residuals[coord].values():
            all_residuals.extend(residuals)

        if len(all_residuals) > 1:
            residuals_array = np.array(all_residuals)
            n = len(residuals_array)

            # Calculate sample variance (unbiased estimator)
            sample_var = np.var(residuals_array, ddof=1)
            resolution = np.sqrt(sample_var)

            # Calculate error using fourth moment
            mean_residual = np.mean(residuals_array)
            centered_residuals = residuals_array - mean_residual
            m4 = np.mean(centered_residuals**4)
            var_of_sample_var = (1 / n) * (m4 - ((n - 3) / (n - 1)) * sample_var**2)

            if sample_var > 0 and var_of_sample_var > 0:
                error = np.sqrt(var_of_sample_var / (4 * sample_var))
            else:
                error = 0.0

            overall_stats[coord] = {
                "resolution": resolution,
                "error": error,
                "n_vertices": len(all_residuals),
            }
        else:
            overall_stats[coord] = {
                "resolution": 0.0,
                "error": 0.0,
                "n_vertices": 0,
            }

    return overall_stats


def _prepare_plot_data(bins, resolutions, resolution_errors, bin_residuals):
    """Prepare data for plotting, filtering out bins with no data."""
    bin_centers = [(start + end) / 2 for start, end in bins]

    # Prepare data for each coordinate
    plot_data = {coord: {} for coord in ["x", "y", "z", "t"]}

    for coord in ["x", "y", "z", "t"]:
        x_values, res_values, err_values, n_vertices_list = [], [], [], []

        for idx, bin_tuple in enumerate(bins):
            n_vertices = len(bin_residuals[coord][bin_tuple])
            if n_vertices > 0:
                x_values.append(bin_centers[idx])
                res_values.append(resolutions[coord][bin_tuple])
                err_values.append(resolution_errors[coord][bin_tuple])
                n_vertices_list.append(n_vertices)

        if x_values:
            plot_data[coord] = {
                "x": x_values,
                "resolution": res_values,
                "error": err_values,
                "n_vertices": n_vertices_list,
            }

    # Check if we have any data
    if not any(plot_data[coord] for coord in ["x", "y", "z", "t"]):
        return None

    return plot_data


def _create_resolution_plots(
    plot_data,
    recall_threshold,
    purity_threshold,
    x_residual_range,
    y_residual_range,
    z_residual_range,
    t_residual_range,
    output_dir,
    method_name,
):
    """Create and save resolution vs track count plots."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    method_slug = method_name.lower().replace(" ", "_")

    # Create a figure with four subplots (one for each coordinate)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    coords = ["x", "y", "z", "t"]
    colors = ["red", "green", "blue", "purple"]
    residual_ranges = [
        x_residual_range,
        y_residual_range,
        z_residual_range,
        t_residual_range,
    ]
    units = ["μm", "μm", "μm", "ps"]  # micrometers for x,y,z; picoseconds for t
    scale_factors = [1000, 1000, 1000, 1000]  # mm to μm, ns to ps

    for idx, (coord, color, res_range, unit, scale) in enumerate(
        zip(coords, colors, residual_ranges, units, scale_factors)
    ):
        ax = axes[idx]

        if coord not in plot_data or not plot_data[coord]:
            ax.text(
                0.5,
                0.5,
                f"No {coord.upper()} resolution data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        data = plot_data[coord]
        x = data["x"]
        resolution = data["resolution"]
        error = data["error"]
        n_vertices = data["n_vertices"]

        # Convert to display units (micrometers or picoseconds)
        resolution_scaled = [r * scale for r in resolution]
        error_scaled = [e * scale for e in error]

        # Plot with error bars
        ax.errorbar(
            x,
            resolution_scaled,
            yerr=error_scaled,
            fmt="o-",
            color=color,
            linewidth=2,
            markersize=8,
            capsize=5,
            capthick=2,
            label=f"{coord.upper()} Resolution (RMS)",
        )

        ax.set_xlabel("Number of tracks per primary vertex", fontsize=12)
        ax.set_ylabel(f"{coord.upper()} Resolution ({unit})", fontsize=12)

        # Format range display based on coordinate
        if coord == "t":
            range_str = f"[-{res_range}, {res_range}] ns"
        else:
            range_str = f"[-{res_range}, {res_range}] mm"

        ax.set_title(
            f"{coord.upper()} Resolution vs Track Count\n"
            f"(residuals in {range_str})\n"
            r"Error: $\sqrt{Var(s^2)/(4s^2)}$",
            fontsize=11,
        )
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(bottom=0)

        # Add annotations
        max_res = max(resolution_scaled) if resolution_scaled else 1
        for x_pos, res, n_vtx in zip(x, resolution_scaled, n_vertices):
            if n_vtx > 0:
                ax.text(
                    x_pos,
                    res + max_res * 0.05,
                    f"n={n_vtx}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    alpha=0.7,
                )

    plt.suptitle(
        f"Vertex Position Resolution Analysis\nRecall≥{recall_threshold}, Purity≥{purity_threshold}",
        fontsize=14,
        y=0.995,
    )
    plt.tight_layout()
    plot_file = output_path / f"resolution_vs_ntracks_{method_slug}.pdf"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved resolution plot to: {plot_file}")


def _create_residual_histograms(
    bins,
    bin_residuals,
    resolutions,
    x_residual_range,
    y_residual_range,
    z_residual_range,
    t_residual_range,
    output_dir,
    method_name,
):
    """Create residual histogram plots for each bin and coordinate."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    method_slug = method_name.lower().replace(" ", "_")

    coords = ["x", "y", "z", "t"]
    colors = ["red", "green", "blue", "purple"]
    residual_ranges = [
        x_residual_range,
        y_residual_range,
        z_residual_range,
        t_residual_range,
    ]
    units = ["mm", "mm", "mm", "ns"]

    for coord, color, res_range, unit in zip(coords, colors, residual_ranges, units):
        # Count bins with data
        bins_with_data = [
            bin_tuple for bin_tuple in bins if len(bin_residuals[coord][bin_tuple]) > 0
        ]

        if not bins_with_data:
            continue

        n_bins_hist = len(bins_with_data)
        n_cols = 2
        n_rows = (n_bins_hist + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        plot_idx = 0
        for bin_tuple in bins_with_data:
            if plot_idx < n_bins_hist:
                row = plot_idx // n_cols
                col = plot_idx % n_cols
                ax = axes[row, col]

                residuals = bin_residuals[coord][bin_tuple]
                bin_start, bin_end = bin_tuple

                ax.hist(residuals, bins=100, alpha=0.7, color=color, density=True)
                ax.axvline(0, color="black", linestyle="--", alpha=0.7)
                ax.set_xlabel(f"{coord.upper()} Residual ({unit})", fontsize=11)
                ax.set_ylabel("Density", fontsize=11)
                ax.set_title(
                    f"{coord.upper()} Residuals - [{bin_start}, {bin_end}] tracks\n"
                    f"σ={resolutions[coord][bin_tuple]:.4f} {unit}, n={len(residuals)}",
                    fontsize=11,
                )
                ax.set_xlim(-res_range, res_range)
                ax.grid(True, alpha=0.3)

                plot_idx += 1

        # Hide unused subplots
        for i in range(plot_idx, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)

        plt.tight_layout()
        hist_file = output_path / f"residual_histograms_{coord}_{method_slug}.pdf"
        plt.savefig(hist_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved {coord.upper()} residual histograms to: {hist_file}")


def _print_results(
    recall_threshold,
    purity_threshold,
    overall_stats,
    resolutions,
    resolution_errors,
    bins,
    bin_residuals,
):
    """Print resolution analysis summary to console."""
    print(f"\n{'-' * 60}")
    print("RESOLUTION ANALYSIS SUMMARY - HDF5 DATA")
    print(f"{'-' * 60}")
    print(
        f"Matching criteria: Recall ≥ {recall_threshold}, Purity ≥ {purity_threshold}"
    )

    for coord in ["x", "y", "z", "t"]:
        stats = overall_stats[coord]
        unit = "ns" if coord == "t" else "mm"
        print(
            f"\nOverall {coord.upper()} Resolution: {stats['resolution']:.4f} ± {stats['error']:.4f} {unit} "
            f"({stats['n_vertices']} vertices)"
        )

    # Print per-bin results
    for coord in ["x", "y", "z", "t"]:
        unit = "ns" if coord == "t" else "mm"
        print(f"\n{coord.upper()} Resolution by bin ({unit}):")
        for bin_tuple in bins:
            n_vertices = len(bin_residuals[coord][bin_tuple])
            if n_vertices > 1:
                bin_start, bin_end = bin_tuple
                res = resolutions[coord][bin_tuple]
                err = resolution_errors[coord][bin_tuple]
                print(
                    f"  [{bin_start}, {bin_end}]: {res:.4f} ± {err:.4f} {unit} (n={n_vertices} vertices)"
                )

    print(f"{'-' * 60}\n")


def _save_resolution_to_csv(
    bins,
    plot_data,
    resolutions,
    resolution_errors,
    bin_residuals,
    overall_stats,
    recall_threshold,
    purity_threshold,
    output_dir,
    method_name,
):
    """Save resolution results to CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    method_slug = method_name.lower().replace(" ", "_")

    # Save per-bin results for each coordinate
    for coord in ["x", "y", "z", "t"]:
        bin_data = []
        unit = "ns" if coord == "t" else "mm"

        for bin_tuple in bins:
            bin_start, bin_end = bin_tuple
            bin_center = (bin_start + bin_end) / 2
            resolution = resolutions[coord][bin_tuple]
            error = resolution_errors[coord][bin_tuple]
            n_vertices = len(bin_residuals[coord][bin_tuple])

            bin_data.append(
                {
                    "bin_start": bin_start,
                    "bin_end": bin_end,
                    "bin_center": bin_center,
                    f"resolution_{unit}": resolution,
                    f"resolution_error_{unit}": error,
                    "n_vertices": n_vertices,
                }
            )

        df = pd.DataFrame(bin_data)
        csv_path = output_path / f"resolution_{coord}_binned_{method_slug}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved {coord.upper()} bin-level resolution to: {csv_path}")

    # Save overall summary for each coordinate
    for coord in ["x", "y", "z", "t"]:
        stats = overall_stats[coord]
        unit = "ns" if coord == "t" else "mm"

        summary_df = pd.DataFrame(
            [
                {
                    "coordinate": coord,
                    f"overall_resolution_{unit}": stats["resolution"],
                    f"overall_error_{unit}": stats["error"],
                    "total_vertices": stats["n_vertices"],
                    "recall_threshold": recall_threshold,
                    "purity_threshold": purity_threshold,
                    "method_name": method_name,
                }
            ]
        )

        overall_path = output_path / f"resolution_{coord}_overall_{method_slug}.csv"
        summary_df.to_csv(overall_path, index=False)
        print(f"Saved {coord.upper()} overall resolution to: {overall_path}")
