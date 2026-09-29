"""Vertex position resolution and bias of residuals, with overlap matching.

Residuals rec − true of matched reconstructed-true pairs are binned by the true
vertex's track count; each coordinate keeps only residuals inside its window,
so every number below describes the core of the distribution, not its tails.

Two quantities are reported separately: the *resolution*, the sample standard
deviation of the residuals about their mean, and the *bias*, that mean. The
RMS about zero would mix the two (RMS² = resolution² + bias²).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .binned_rate import find_bin, iter_events, track_matched_pairs
from .utils import mean_with_error, std_with_error

COORDS = ("x", "y", "z", "t")


def analyze_vertex_resolution(
    hdf5_path,
    output_dir,
    recall_threshold,
    purity_threshold,
    bins,
    residual_ranges,
    method_name,
):
    """Resolution per coordinate vs true-vertex track count; plots and CSVs.

    Args:
        hdf5_path: HDF5 file with rec_vertices and true_vertices
        output_dir: where plots and CSVs go
        recall_threshold, purity_threshold: overlap matching thresholds
        bins: (start, end) track-multiplicity bins (binned_rate.make_bins)
        residual_ranges: {"x", "y", "z", "t"}: half-width of the residual
            window kept per coordinate (mm, mm, mm, ns)
        method_name: label used in file names
    """
    bin_residuals = {coord: {bin_: [] for bin_ in bins} for coord in COORDS}
    for event in iter_events(hdf5_path):
        for true_id, rec_id in track_matched_pairs(
            event, recall_threshold, purity_threshold
        ):
            bin_ = find_bin(bins, len(event.true_tracks[true_id]))
            if bin_ is None:
                continue
            for coord in COORDS:
                residual = (
                    getattr(event, f"rec_{coord}")[rec_id]
                    - getattr(event, f"true_{coord}")[true_id]
                )
                if -residual_ranges[coord] <= residual <= residual_ranges[coord]:
                    bin_residuals[coord][bin_].append(residual)

    resolutions = {}
    resolution_errors = {}
    biases = {}
    for coord in COORDS:
        resolutions[coord], resolution_errors[coord], biases[coord] = (
            _calculate_resolutions(bins, bin_residuals[coord])
        )
    overall_stats = _calculate_overall_resolution(bin_residuals)

    for coord in COORDS:
        stats = overall_stats[coord]
        unit = "ns" if coord == "t" else "mm"
        print(
            f"resolution  {coord}  {stats['resolution']:.4f} ± {stats['error']:.4f} {unit}"
            f"  bias {stats['bias']:.4f} ± {stats['bias_error']:.4f} {unit}"
            f"  ({stats['n_vertices']} vertices)"
        )

    plot_data = _prepare_plot_data(bins, resolutions, resolution_errors, bin_residuals)
    if plot_data is None:
        print("No resolution data to plot")
        return

    ranges = tuple(residual_ranges[coord] for coord in COORDS)
    _create_resolution_plots(
        plot_data, recall_threshold, purity_threshold, *ranges, output_dir, method_name
    )
    _create_residual_histograms(
        bins, bin_residuals, resolutions, biases, *ranges, output_dir, method_name
    )
    _save_resolution_to_csv(
        bins,
        plot_data,
        resolutions,
        resolution_errors,
        biases,
        bin_residuals,
        overall_stats,
        recall_threshold,
        purity_threshold,
        output_dir,
        method_name,
    )


def _bias(residuals):
    """(mean, error of the mean) of the residuals; (0.0, 0.0) below two values."""
    mean, error = mean_with_error(residuals)
    if mean is None or error is None:
        return 0.0, 0.0
    return mean, error


def _calculate_resolutions(bins, bin_residuals_coord):
    """Resolution, its error and (bias, bias error) per bin for one coordinate."""
    resolutions, errors, biases = {}, {}, {}
    for bin_ in bins:
        resolutions[bin_], errors[bin_] = std_with_error(bin_residuals_coord[bin_])
        biases[bin_] = _bias(bin_residuals_coord[bin_])
    return resolutions, errors, biases


def _calculate_overall_resolution(bin_residuals):
    """Resolution, bias, their errors and vertex count per coordinate over all bins."""
    overall_stats = {}
    for coord in COORDS:
        residuals = [r for values in bin_residuals[coord].values() for r in values]
        resolution, error = std_with_error(residuals)
        bias, bias_error = _bias(residuals)
        overall_stats[coord] = {
            "resolution": resolution,
            "error": error,
            "bias": bias,
            "bias_error": bias_error,
            "n_vertices": len(residuals) if len(residuals) > 1 else 0,
        }
    return overall_stats


def _prepare_plot_data(bins, resolutions, resolution_errors, bin_residuals):
    """Prepare data for plotting, filtering out bins with no data."""
    bin_centers = [(start + end) / 2 for start, end in bins]

    # Prepare data for each coordinate
    plot_data = {coord: {} for coord in ["x", "y", "z", "t"]}

    for coord in COORDS:
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
        zip(coords, colors, residual_ranges, units, scale_factors, strict=True)
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
            label=f"{coord.upper()} Resolution (std of residuals)",
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
            f"(std of residuals in {range_str}; bias reported separately)\n"
            r"Error: $\sqrt{Var(s^2)/(4s^2)}$",
            fontsize=11,
        )
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(bottom=0)

        # Add annotations
        max_res = max(resolution_scaled) if resolution_scaled else 1
        for x_pos, res, n_vtx in zip(x, resolution_scaled, n_vertices, strict=True):
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
    biases,
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

    for coord, color, res_range, unit in zip(
        coords, colors, residual_ranges, units, strict=True
    ):
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
                    f"std={resolutions[coord][bin_tuple]:.4f} {unit}, "
                    f"bias={biases[coord][bin_tuple][0]:.4f} {unit}, n={len(residuals)}",
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


def _save_resolution_to_csv(
    bins,
    plot_data,
    resolutions,
    resolution_errors,
    biases,
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
    for coord in COORDS:
        bin_data = []
        unit = "ns" if coord == "t" else "mm"

        for bin_tuple in bins:
            bin_start, bin_end = bin_tuple
            bin_center = (bin_start + bin_end) / 2
            resolution = resolutions[coord][bin_tuple]
            error = resolution_errors[coord][bin_tuple]
            bias, bias_error = biases[coord][bin_tuple]
            n_vertices = len(bin_residuals[coord][bin_tuple])

            bin_data.append(
                {
                    "bin_start": bin_start,
                    "bin_end": bin_end,
                    "bin_center": bin_center,
                    f"resolution_{unit}": resolution,
                    f"resolution_error_{unit}": error,
                    f"bias_{unit}": bias,
                    f"bias_error_{unit}": bias_error,
                    "n_vertices": n_vertices,
                }
            )

        df = pd.DataFrame(bin_data)
        csv_path = output_path / f"resolution_{coord}_binned_{method_slug}.csv"
        df.to_csv(csv_path, index=False)

    # Save overall summary for each coordinate
    for coord in COORDS:
        stats = overall_stats[coord]
        unit = "ns" if coord == "t" else "mm"

        summary_df = pd.DataFrame(
            [
                {
                    "coordinate": coord,
                    f"overall_resolution_{unit}": stats["resolution"],
                    f"overall_error_{unit}": stats["error"],
                    f"overall_bias_{unit}": stats["bias"],
                    f"overall_bias_error_{unit}": stats["bias_error"],
                    "total_vertices": stats["n_vertices"],
                    "recall_threshold": recall_threshold,
                    "purity_threshold": purity_threshold,
                    "method_name": method_name,
                }
            ]
        )

        overall_path = output_path / f"resolution_{coord}_overall_{method_slug}.csv"
        summary_df.to_csv(overall_path, index=False)
