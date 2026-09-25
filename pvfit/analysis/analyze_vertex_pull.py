"""Fitted vertex pull analysis: Compare fitted vertex parameters to true vertices."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .binned_rate import iter_events, track_matched_pairs


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

    for event in iter_events(hdf5_path):
        for true_id, rec_id in track_matched_pairs(
            event, recall_threshold, purity_threshold
        ):
            errors = {
                coord: getattr(event, f"rec_{coord}_err")[rec_id] for coord in "xyzt"
            }
            if any(err <= 0 or np.isnan(err) for err in errors.values()):
                continue
            for coord, err in errors.items():
                residual = (
                    getattr(event, f"rec_{coord}")[rec_id]
                    - getattr(event, f"true_{coord}")[true_id]
                )
                pull_data[f"residual_{coord}"].append(residual)
                pull_data[f"error_{coord}"].append(err)
                pull_data[f"pull_{coord}"].append(residual / err)

    # Convert to numpy arrays
    for key in pull_data:
        pull_data[key] = np.array(pull_data[key])

    print(f"vertex_pull  {len(pull_data['pull_x'])} matched vertices with valid errors")

    if len(pull_data["pull_x"]) == 0:
        print("No valid matched vertices found for pull analysis")
        return

    # Create plots
    _plot_vertex_pulls(pull_data, output_dir)


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
