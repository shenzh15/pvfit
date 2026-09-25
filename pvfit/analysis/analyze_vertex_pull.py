"""Fitted vertex pull analysis: Compare fitted vertex parameters to true vertices."""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from .utils import find_best_match


def analyze_vertex_pull(hdf5_path, output_dir, recall_threshold, purity_threshold):
    """Analyze fitted vertex pulls by comparing fitted parameters to true vertices.

    For each matched reconstructed-true vertex pair, calculates residuals and pulls
    (normalized residuals) using the fitted vertex position errors.

    Args:
        hdf5_path: Path to HDF5 file containing fit results
        output_dir: Directory for saving plots
        recall_threshold: Minimum track overlap ratio for matching (true perspective)
        purity_threshold: Minimum track overlap ratio for matching (rec perspective)

    Returns:
        None: Saves vertex pull analysis plots to output directory
    """
    print("\nStarting fitted vertex pull analysis...")

    # Initialize storage for pull data
    pull_data = {
        "pull_x": [],
        "pull_y": [],
        "pull_z": [],
        "pull_t": [],
        "error_x": [],
        "error_y": [],
        "error_z": [],
        "error_t": [],
        "residual_x": [],
        "residual_y": [],
        "residual_z": [],
        "residual_t": [],
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
            true_track_indices = true_verts["track_indices"][:]

            # Load reconstructed vertices
            rec_verts = event_group["rec_vertices"]
            rec_x = rec_verts["x"][:]
            rec_y = rec_verts["y"][:]
            rec_z = rec_verts["z"][:]
            rec_t = rec_verts["t"][:]
            rec_x_err = rec_verts["x_err"][:]
            rec_y_err = rec_verts["y_err"][:]
            rec_z_err = rec_verts["z_err"][:]
            rec_t_err = rec_verts["t_err"][:]
            rec_track_indices = rec_verts["track_indices"][:]

            # Build true_vertex_info dict
            true_vertex_info = {}
            for idx, track_indices in enumerate(true_track_indices):
                true_vertex_info[idx] = {"track_indices": set(track_indices)}

            # Match each reconstructed vertex to its best true vertex
            for rec_idx, track_indices in enumerate(rec_track_indices):
                rec_track_set = set(track_indices)
                if not rec_track_set:
                    continue

                true_idx = find_best_match(
                    rec_track_set, true_vertex_info, recall_threshold, purity_threshold
                )
                if true_idx is None:
                    continue

                # Calculate residuals
                res_x = rec_x[rec_idx] - true_x[true_idx]
                res_y = rec_y[rec_idx] - true_y[true_idx]
                res_z = rec_z[rec_idx] - true_z[true_idx]
                res_t = rec_t[rec_idx] - true_t[true_idx]

                # Get fitted errors
                err_x = rec_x_err[rec_idx]
                err_y = rec_y_err[rec_idx]
                err_z = rec_z_err[rec_idx]
                err_t = rec_t_err[rec_idx]

                # Check error validity
                if (
                    err_x <= 0
                    or err_y <= 0
                    or err_z <= 0
                    or err_t <= 0
                    or np.isnan(err_x)
                    or np.isnan(err_y)
                    or np.isnan(err_z)
                    or np.isnan(err_t)
                ):
                    continue

                # Calculate pulls
                pull_x = res_x / err_x
                pull_y = res_y / err_y
                pull_z = res_z / err_z
                pull_t = res_t / err_t

                # Store results
                pull_data["pull_x"].append(pull_x)
                pull_data["pull_y"].append(pull_y)
                pull_data["pull_z"].append(pull_z)
                pull_data["pull_t"].append(pull_t)
                pull_data["error_x"].append(err_x)
                pull_data["error_y"].append(err_y)
                pull_data["error_z"].append(err_z)
                pull_data["error_t"].append(err_t)
                pull_data["residual_x"].append(res_x)
                pull_data["residual_y"].append(res_y)
                pull_data["residual_z"].append(res_z)
                pull_data["residual_t"].append(res_t)

    # Convert to numpy arrays
    for key in pull_data:
        pull_data[key] = np.array(pull_data[key])

    print(f"Collected {len(pull_data['pull_x'])} valid matched vertices for analysis")

    if len(pull_data["pull_x"]) == 0:
        print("No valid matched vertices found for pull analysis")
        return

    # Create plots
    _plot_vertex_pulls(pull_data, output_dir)

    print(f"Fitted vertex pull analysis complete. Plots saved to {output_dir}/")


def _plot_vertex_pulls(pull_data, output_dir):
    """Create multi-panel figure showing fitted vertex pull distributions."""

    # Create combined plot with 3 rows x 4 columns
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle("Fitted Vertex Analysis: Residuals, Errors, and Pulls", fontsize=16)

    pull_range_min, pull_range_max = -5, 5
    residual_xy_range = (-0.1, 0.1)
    residual_z_range = (-0.7, 0.7)
    residual_t_range = (-0.06, 0.06)

    # Filter data within ranges for statistics
    pull_x_in_range = pull_data["pull_x"][
        (pull_data["pull_x"] >= pull_range_min)
        & (pull_data["pull_x"] <= pull_range_max)
    ]
    pull_y_in_range = pull_data["pull_y"][
        (pull_data["pull_y"] >= pull_range_min)
        & (pull_data["pull_y"] <= pull_range_max)
    ]
    pull_z_in_range = pull_data["pull_z"][
        (pull_data["pull_z"] >= pull_range_min)
        & (pull_data["pull_z"] <= pull_range_max)
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
    residual_z_in_range = pull_data["residual_z"][
        (pull_data["residual_z"] >= residual_z_range[0])
        & (pull_data["residual_z"] <= residual_z_range[1])
    ]
    residual_t_in_range = pull_data["residual_t"][
        (pull_data["residual_t"] >= residual_t_range[0])
        & (pull_data["residual_t"] <= residual_t_range[1])
    ]

    # Row 1: Residual distributions
    axes[0, 0].hist(
        pull_data["residual_x"],
        bins=50,
        range=residual_xy_range,
        alpha=0.7,
        density=True,
    )
    axes[0, 0].set_xlabel("Residual X (mm)")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].set_title(
        f"Residual X (mean={np.mean(residual_x_in_range):.4f}, std={np.std(residual_x_in_range):.4f})"
    )
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(
        pull_data["residual_y"],
        bins=50,
        range=residual_xy_range,
        alpha=0.7,
        density=True,
    )
    axes[0, 1].set_xlabel("Residual Y (mm)")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].set_title(
        f"Residual Y (mean={np.mean(residual_y_in_range):.4f}, std={np.std(residual_y_in_range):.4f})"
    )
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].hist(
        pull_data["residual_z"],
        bins=50,
        range=residual_z_range,
        alpha=0.7,
        density=True,
    )
    axes[0, 2].set_xlabel("Residual Z (mm)")
    axes[0, 2].set_ylabel("Density")
    axes[0, 2].set_title(
        f"Residual Z (mean={np.mean(residual_z_in_range):.4f}, std={np.std(residual_z_in_range):.4f})"
    )
    axes[0, 2].grid(True, alpha=0.3)

    axes[0, 3].hist(
        pull_data["residual_t"],
        bins=50,
        range=residual_t_range,
        alpha=0.7,
        density=True,
    )
    axes[0, 3].set_xlabel("Residual T (ns)")
    axes[0, 3].set_ylabel("Density")
    axes[0, 3].set_title(
        f"Residual T (mean={np.mean(residual_t_in_range):.4f}, std={np.std(residual_t_in_range):.4f})"
    )
    axes[0, 3].grid(True, alpha=0.3)

    # Row 2: Error distributions
    axes[1, 0].hist(pull_data["error_x"], bins=50, alpha=0.7, density=True)
    axes[1, 0].set_xlabel("Error X (mm)")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].set_title(
        f"Error X (mean={np.mean(pull_data['error_x']):.4f}, std={np.std(pull_data['error_x']):.4f})"
    )
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(pull_data["error_y"], bins=50, alpha=0.7, density=True)
    axes[1, 1].set_xlabel("Error Y (mm)")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].set_title(
        f"Error Y (mean={np.mean(pull_data['error_y']):.4f}, std={np.std(pull_data['error_y']):.4f})"
    )
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].hist(pull_data["error_z"], bins=50, alpha=0.7, density=True)
    axes[1, 2].set_xlabel("Error Z (mm)")
    axes[1, 2].set_ylabel("Density")
    axes[1, 2].set_title(
        f"Error Z (mean={np.mean(pull_data['error_z']):.4f}, std={np.std(pull_data['error_z']):.4f})"
    )
    axes[1, 2].grid(True, alpha=0.3)

    axes[1, 3].hist(pull_data["error_t"], bins=50, alpha=0.7, density=True)
    axes[1, 3].set_xlabel("Error T (ns)")
    axes[1, 3].set_ylabel("Density")
    axes[1, 3].set_title(
        f"Error T (mean={np.mean(pull_data['error_t']):.4f}, std={np.std(pull_data['error_t']):.4f})"
    )
    axes[1, 3].grid(True, alpha=0.3)

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
        pull_data["pull_z"],
        bins=50,
        range=(pull_range_min, pull_range_max),
        alpha=0.7,
        density=True,
    )
    axes[2, 2].set_xlabel("Pull Z")
    axes[2, 2].set_ylabel("Density")
    axes[2, 2].set_title(
        f"Pull Z (mean={np.mean(pull_z_in_range):.3f}, std={np.std(pull_z_in_range):.3f})"
    )
    axes[2, 2].grid(True, alpha=0.3)

    axes[2, 3].hist(
        pull_data["pull_t"],
        bins=50,
        range=(pull_range_min, pull_range_max),
        alpha=0.7,
        density=True,
    )
    axes[2, 3].set_xlabel("Pull T")
    axes[2, 3].set_ylabel("Density")
    axes[2, 3].set_title(
        f"Pull T (mean={np.mean(pull_t_in_range):.3f}, std={np.std(pull_t_in_range):.3f})"
    )
    axes[2, 3].grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_path / "pull_analysis_fitted_vertex.pdf",
        format="pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(
        f"Fitted vertex pull analysis plot saved as {output_path / 'pull_analysis_fitted_vertex.pdf'}"
    )
