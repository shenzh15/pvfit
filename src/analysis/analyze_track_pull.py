"""Input track pull analysis: Compare track parameters to true vertices."""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def analyze_track_pull(hdf5_path, output_dir="output", max_chi2=10):
    """Analyze input track pulls by comparing track parameters to true vertex positions.

    For each track, calculates residuals and pulls (normalized residuals) relative to
    its associated true vertex, accounting for track parameter uncertainties.

    Args:
        hdf5_path: Path to HDF5 file containing fit results and track data
        output_dir: Directory for saving plots
        max_chi2: Maximum chi2 threshold for track-vertex matching

    Returns:
        None: Saves track pull analysis plots to output directory
    """
    print("\nStarting input track pull analysis...")

    # Initialize storage for pull data
    pull_data = {
        "pull_x": [],
        "pull_y": [],
        "pull_t": [],
        "error_x": [],
        "error_y": [],
        "error_t": [],
        "residual_x": [],
        "residual_y": [],
        "residual_t": [],
        "delta_z": [],
    }

    with h5py.File(hdf5_path, "r") as f:
        n_events = f.attrs["n_events"]
        print(f"Processing {n_events} events...")

        for event_id_str in f.keys():
            event_group = f[event_id_str]

            # Load true vertices
            true_verts = event_group["true_vertices"]
            true_x = true_verts["x"][:]
            true_y = true_verts["y"][:]
            true_z = true_verts["z"][:]
            true_t = true_verts["t"][:]

            # Load tracks
            tracks = event_group["tracks"]
            track_x = tracks["x"][:]
            track_y = tracks["y"][:]
            track_z = tracks["z"][:]
            track_t = tracks["t"][:]
            track_cov_xx = tracks["cov_xx"][:]
            track_cov_yy = tracks["cov_yy"][:]
            track_cov_tt = tracks["cov_tt"][:]
            track_ovtx_x = tracks["ovtx_x"][:]
            track_ovtx_y = tracks["ovtx_y"][:]
            track_ovtx_z = tracks["ovtx_z"][:]
            track_ovtx_t = tracks["ovtx_t"][:]

            # Process each track
            for i in range(len(track_x)):
                # Find matching true vertex by comparing ovtx with true vertices
                # Use small tolerance for floating point comparison
                tolerance = 1e-6
                matched_idx = None
                for j in range(len(true_x)):
                    if (
                        abs(track_ovtx_x[i] - true_x[j]) < tolerance
                        and abs(track_ovtx_y[i] - true_y[j]) < tolerance
                        and abs(track_ovtx_z[i] - true_z[j]) < tolerance
                        and abs(track_ovtx_t[i] - true_t[j]) < tolerance
                    ):
                        matched_idx = j
                        break

                if matched_idx is None:
                    continue  # Track's ovtx not in true vertex list

                # Calculate residuals
                res_x = track_x[i] - true_x[matched_idx]
                res_y = track_y[i] - true_y[matched_idx]
                res_t = track_t[i] - true_t[matched_idx]
                delta_z = true_z[matched_idx] - track_z[i]

                # Check covariance validity
                if (
                    track_cov_xx[i] <= 0
                    or track_cov_yy[i] <= 0
                    or track_cov_tt[i] <= 0
                    or np.isnan(track_cov_xx[i])
                    or np.isnan(track_cov_yy[i])
                    or np.isnan(track_cov_tt[i])
                ):
                    continue

                # Calculate errors (square root of covariance)
                err_x = np.sqrt(track_cov_xx[i])
                err_y = np.sqrt(track_cov_yy[i])
                err_t = np.sqrt(track_cov_tt[i])

                # Calculate pulls
                pull_x = res_x / err_x
                pull_y = res_y / err_y
                pull_t = res_t / err_t

                # Calculate chi2 for this track
                chi2 = pull_x**2 + pull_y**2 + pull_t**2

                # Apply chi2 cut
                if chi2 > max_chi2:
                    continue

                # Store results
                pull_data["pull_x"].append(pull_x)
                pull_data["pull_y"].append(pull_y)
                pull_data["pull_t"].append(pull_t)
                pull_data["error_x"].append(err_x)
                pull_data["error_y"].append(err_y)
                pull_data["error_t"].append(err_t)
                pull_data["residual_x"].append(res_x)
                pull_data["residual_y"].append(res_y)
                pull_data["residual_t"].append(res_t)
                pull_data["delta_z"].append(delta_z)

    # Convert to numpy arrays
    for key in pull_data:
        pull_data[key] = np.array(pull_data[key])

    print(f"Collected {len(pull_data['pull_x'])} valid tracks for analysis")

    if len(pull_data["pull_x"]) == 0:
        print("No valid tracks found for pull analysis")
        return

    # Create plots
    _plot_track_pulls(pull_data, output_dir)

    print(f"Track pull analysis complete. Plots saved to {output_dir}/")


def _plot_track_pulls(pull_data, output_dir):
    """Create multi-panel figure showing track pull distributions."""

    # Create combined plot with 3 rows x 4 columns
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle("Track-level Analysis: Residuals, Errors, and Pulls", fontsize=16)

    pull_range_min, pull_range_max = -5, 5
    residual_xy_range = (-0.75, 0.75)
    residual_t_range = (-0.15, 0.15)
    delta_z_range = (-5, 5)

    # Filter data within ranges for statistics
    pull_x_in_range = pull_data["pull_x"][
        (pull_data["pull_x"] >= pull_range_min)
        & (pull_data["pull_x"] <= pull_range_max)
    ]
    pull_y_in_range = pull_data["pull_y"][
        (pull_data["pull_y"] >= pull_range_min)
        & (pull_data["pull_y"] <= pull_range_max)
    ]
    pull_t_in_range = pull_data["pull_t"][
        (pull_data["pull_t"] >= pull_range_min)
        & (pull_data["pull_t"] <= pull_range_max)
    ]

    residual_x_in_range = pull_data["residual_x"][
        (pull_data["residual_x"] >= residual_xy_range[0])
        & (pull_data["residual_x"] <= residual_xy_range[1])
    ]
    residual_y_in_range = pull_data["residual_y"][
        (pull_data["residual_y"] >= residual_xy_range[0])
        & (pull_data["residual_y"] <= residual_xy_range[1])
    ]
    residual_t_in_range = pull_data["residual_t"][
        (pull_data["residual_t"] >= residual_t_range[0])
        & (pull_data["residual_t"] <= residual_t_range[1])
    ]
    delta_z_in_range = pull_data["delta_z"][
        (pull_data["delta_z"] >= delta_z_range[0])
        & (pull_data["delta_z"] <= delta_z_range[1])
    ]

    # Row 1: Residual distributions
    axes[0, 0].hist(
        pull_data["residual_x"],
        bins=50,
        range=residual_xy_range,
        alpha=0.7,
        density=True,
    )
    axes[0, 0].set_xlabel("Residual X")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].set_title(
        f"Residual X (mean={np.mean(residual_x_in_range):.3f}, std={np.std(residual_x_in_range):.3f})"
    )
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(
        pull_data["residual_y"],
        bins=50,
        range=residual_xy_range,
        alpha=0.7,
        density=True,
    )
    axes[0, 1].set_xlabel("Residual Y")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].set_title(
        f"Residual Y (mean={np.mean(residual_y_in_range):.3f}, std={np.std(residual_y_in_range):.3f})"
    )
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].hist(
        pull_data["residual_t"],
        bins=50,
        range=residual_t_range,
        alpha=0.7,
        density=True,
    )
    axes[0, 2].set_xlabel("Residual T")
    axes[0, 2].set_ylabel("Density")
    axes[0, 2].set_title(
        f"Residual T (mean={np.mean(residual_t_in_range):.3f}, std={np.std(residual_t_in_range):.3f})"
    )
    axes[0, 2].grid(True, alpha=0.3)

    axes[0, 3].hist(
        pull_data["delta_z"], bins=50, range=delta_z_range, alpha=0.7, density=True
    )
    axes[0, 3].set_xlabel("Delta Z")
    axes[0, 3].set_ylabel("Density")
    axes[0, 3].set_title(
        f"Delta Z (mean={np.mean(delta_z_in_range):.3f}, std={np.std(delta_z_in_range):.3f})"
    )
    axes[0, 3].grid(True, alpha=0.3)

    # Row 2: Error distributions
    axes[1, 0].hist(pull_data["error_x"], bins=50, alpha=0.7, density=True)
    axes[1, 0].set_xlabel("Error X")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].set_title(
        f"Error X (mean={np.mean(pull_data['error_x']):.3f}, std={np.std(pull_data['error_x']):.3f})"
    )
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(pull_data["error_y"], bins=50, alpha=0.7, density=True)
    axes[1, 1].set_xlabel("Error Y")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].set_title(
        f"Error Y (mean={np.mean(pull_data['error_y']):.3f}, std={np.std(pull_data['error_y']):.3f})"
    )
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].hist(pull_data["error_t"], bins=50, alpha=0.7, density=True)
    axes[1, 2].set_xlabel("Error T")
    axes[1, 2].set_ylabel("Density")
    axes[1, 2].set_title(
        f"Error T (mean={np.mean(pull_data['error_t']):.3f}, std={np.std(pull_data['error_t']):.3f})"
    )
    axes[1, 2].grid(True, alpha=0.3)

    # Empty subplot for row 2, column 4
    axes[1, 3].axis("off")

    # Row 3: Pull distributions
    axes[2, 0].hist(
        pull_data["pull_x"],
        bins=50,
        range=(pull_range_min, pull_range_max),
        alpha=0.7,
        density=True,
    )
    axes[2, 0].set_xlabel("Pull X")
    axes[2, 0].set_ylabel("Density")
    axes[2, 0].set_title(
        f"Pull X (mean={np.mean(pull_x_in_range):.3f}, std={np.std(pull_x_in_range):.3f})"
    )
    axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].hist(
        pull_data["pull_y"],
        bins=50,
        range=(pull_range_min, pull_range_max),
        alpha=0.7,
        density=True,
    )
    axes[2, 1].set_xlabel("Pull Y")
    axes[2, 1].set_ylabel("Density")
    axes[2, 1].set_title(
        f"Pull Y (mean={np.mean(pull_y_in_range):.3f}, std={np.std(pull_y_in_range):.3f})"
    )
    axes[2, 1].grid(True, alpha=0.3)

    axes[2, 2].hist(
        pull_data["pull_t"],
        bins=50,
        range=(pull_range_min, pull_range_max),
        alpha=0.7,
        density=True,
    )
    axes[2, 2].set_xlabel("Pull T")
    axes[2, 2].set_ylabel("Density")
    axes[2, 2].set_title(
        f"Pull T (mean={np.mean(pull_t_in_range):.3f}, std={np.std(pull_t_in_range):.3f})"
    )
    axes[2, 2].grid(True, alpha=0.3)

    # Empty subplot for row 3, column 4
    axes[2, 3].axis("off")

    plt.tight_layout()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_path / "pull_analysis_input_track.pdf",
        format="pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(
        f"Track-level analysis plot saved as {output_path / 'pull_analysis_input_track.pdf'}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze input track pulls relative to true vertices"
    )
    parser.add_argument(
        "--hdf5_path",
        type=str,
        required=True,
        help="Path to the HDF5 file containing fit results",
    )
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Output directory for plots"
    )
    parser.add_argument(
        "--max_chi2",
        type=float,
        default=10,
        help="Maximum chi2 threshold for track-vertex matching",
    )

    args = parser.parse_args()

    analyze_track_pull(
        hdf5_path=args.hdf5_path, output_dir=args.output_dir, max_chi2=args.max_chi2
    )
