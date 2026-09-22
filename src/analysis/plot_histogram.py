"""Plot 2D histogram of vertex seeding and fitted positions."""

from pathlib import Path

import h5py
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import yaml


def plot_event_histogram(
    hdf5_path,
    event_id=None,
    output_dir="output",
    config_path="config.yaml",
    draw_fixed_ellipses=False,
    draw_fitted_ellipses=False,
):
    """Create a 2D visualization of vertex seeding histogram with overlaid vertex positions.

    Plots the track density histogram used for vertex seeding, with overlaid markers
    showing initial seeded positions, fitted vertex positions, and true vertex locations.
    Provides visual validation of the vertex finding and fitting pipeline.

    Args:
        hdf5_path: Path to the HDF5 file containing fit results
        event_id: Event ID to plot. If None, plots the first event in the file
        output_dir: Directory for saving plots
        config_path: Path to config.yaml file
        draw_fixed_ellipses: Whether to draw fixed matching ellipses around initial seeded positions
        draw_fitted_ellipses: Whether to draw 3-sigma error ellipses around fitted positions

    Returns:
        None: Saves 2D histogram plot to output directory
    """
    print("Plotting 2D Histogram...")

    # Load parameters from config.yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    m_r = config["algorithm"]["m_r"]
    z_scale = config["algorithm"]["z_scale"]
    t_scale = config["algorithm"]["t_scale"]

    print(
        f"Using parameters from config: m_r={m_r}, z_scale={z_scale}, t_scale={t_scale}"
    )

    with h5py.File(hdf5_path, "r") as f:
        # Get event ID
        if event_id is None:
            # Use first event if not specified
            event_ids = sorted([int(key) for key in f.keys()])
            if not event_ids:
                print("No events found in HDF5 file")
                return
            event_id = event_ids[0]
            print(f"Using first event: {event_id}")

        # Check if event exists
        if str(event_id) not in f:
            print(f"Event {event_id} not found in HDF5 file")
            available = sorted([int(key) for key in f.keys()])
            print(
                f"Available events: {available[:10]}{'...' if len(available) > 10 else ''}"
            )
            return

        event_group = f[str(event_id)]

        # Load histogram if it exists
        if "histogram" not in event_group:
            print(f"No histogram data found for event {event_id}")
            return

        H = event_group["histogram"][:]

        # Read histogram parameters from HDF5 attributes
        hist_attrs = event_group["histogram"].attrs
        z_range = (hist_attrs["z_min"], hist_attrs["z_max"])
        t_range = (hist_attrs["t_min"], hist_attrs["t_max"])
        z_bins = hist_attrs["n_bins_z"]
        t_bins = hist_attrs["n_bins_t"]

        print(
            f"Histogram parameters: z_range={z_range}, t_range={t_range}, z_bins={z_bins}, t_bins={t_bins}"
        )

        # Load true vertices
        true_verts = event_group["true_vertices"]
        true_z = true_verts["z"][:]
        true_t = true_verts["t"][:]
        true_ids = (
            true_verts["vertex_id"][:]
            if "vertex_id" in true_verts
            else np.arange(len(true_z))
        )
        true_track_indices = true_verts["track_indices"][:]
        true_ntracks = np.array([len(ti) for ti in true_track_indices])

        # Load reconstructed vertices
        rec_verts = event_group["rec_vertices"]
        rec_z = rec_verts["z"][:]
        rec_t = rec_verts["t"][:]

        # Load initial seeded positions
        if "z_seed" not in rec_verts or "t_seed" not in rec_verts:
            raise KeyError(
                f"Event {event_id}: 'z_seed' or 't_seed' not found in rec_vertices"
            )
        initial_z = rec_verts["z_seed"][:]
        initial_t = rec_verts["t_seed"][:]

        z_err = None
        t_err = None

        if draw_fitted_ellipses:
            if "z_err" in rec_verts and "t_err" in rec_verts:
                z_err = rec_verts["z_err"][:]
                t_err = rec_verts["t_err"][:]

    # Create a copy and set zero values to NaN for better visualization
    H_plot = np.copy(H)
    H_plot[H_plot == 0] = np.nan

    # Create edges for pcolormesh
    z_edges = np.linspace(z_range[0], z_range[1], z_bins + 1)
    t_edges = np.linspace(t_range[0], t_range[1], t_bins + 1)

    # Create figure
    fig, ax = plt.subplots(figsize=(15, 8))

    # Create the main plot using the histogram
    X, Y = np.meshgrid(z_edges, t_edges)
    im = ax.pcolormesh(X, Y, H_plot.T, cmap="jet", shading="auto")

    # Mark true vertices with red circles and vertex_id labels
    if len(true_z) > 0:
        ax.scatter(
            true_z,
            true_t,
            color="red",
            s=30,
            alpha=0.8,
            label="True Vertices",
            marker="o",
            edgecolors="darkred",
            linewidths=0.8,
        )
        for i in range(len(true_z)):
            ax.annotate(
                f"{int(true_ids[i])}({true_ntracks[i]})",
                xy=(true_z[i], true_t[i]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
                color="red",
                fontweight="bold",
            )

    # Mark initial reconstructed vertices with green X marks
    if len(initial_z) > 0:
        ax.scatter(
            initial_z,
            initial_t,
            color="green",
            s=20,
            alpha=0.8,
            label="Initial Seeded Vertices",
            marker="x",
            linewidths=1.5,
        )

    # Mark fitted vertices with orange squares
    if len(rec_z) > 0:
        ax.scatter(
            rec_z,
            rec_t,
            color="orange",
            s=15,
            alpha=0.9,
            label="Fitted Vertices",
            marker="s",
            edgecolors="darkorange",
            linewidths=0.8,
        )

    # Draw fixed matching ellipses around initial positions
    if draw_fixed_ellipses and len(initial_z) > 0:
        for i in range(len(initial_z)):
            ellipse = patches.Ellipse(
                xy=(initial_z[i], initial_t[i]),
                width=z_scale * m_r,  # diameter in z
                height=t_scale * m_r,  # diameter in t
                fill=False,
                edgecolor="cyan",
                linewidth=0.8,
                alpha=0.7,
                label="Fixed Matching Ellipse" if i == 0 else "",
            )
            ax.add_patch(ellipse)

            # Add label text for initial vertex
            # label_text = f"{i}"
            # ax.annotate(
            #    label_text,
            #    xy=(initial_z[i], initial_t[i]),
            #    xytext=(2, 2),  # Text offset in points
            #    textcoords='offset points',
            #    fontsize=8,
            #    color='green',
            #    fontweight='bold'
            # )

    # Draw ellipses around fitted positions with error bounds
    if draw_fitted_ellipses and z_err is not None and len(rec_z) > 0:
        for i in range(len(rec_z)):
            # 5-sigma ellipse
            width = 2 * 5 * z_err[i]  # diameter
            height = 2 * 5 * t_err[i]  # diameter

            ellipse_fitted = patches.Ellipse(
                xy=(rec_z[i], rec_t[i]),
                width=width,
                height=height,
                fill=False,
                edgecolor="magenta",
                linewidth=0.8,
                alpha=0.7,
                label="5σ Error Ellipse" if i == 0 else "",
            )
            ax.add_patch(ellipse_fitted)

    # Draw red arrows from initial to fitted positions
    # if len(initial_z) == len(rec_z):
    #     for i in range(len(rec_z)):
    #         if abs(initial_z[i] - rec_z[i]) > 0.1 or abs(initial_t[i] - rec_t[i]) > 0.001:
    #             arrowprops = dict(arrowstyle='-|>,head_length=0.4,head_width=0.3',
    #                             color='red', alpha=0.6, lw=1.5)
    #             ax.annotate('', xy=(rec_z[i], rec_t[i]),
    #                        xytext=(initial_z[i], initial_t[i]),
    #                        arrowprops=arrowprops)

    # Color bar & labels
    cbar = plt.colorbar(im, label="Track Density", ax=ax)
    cbar.set_label("Track Density (counts)", fontsize=18)
    cbar.ax.tick_params(labelsize=14)

    ax.set_xlabel("beamPOCA_z (mm)", fontsize=20)
    ax.set_ylabel("beamPOCA_t (ns)", fontsize=20)
    ax.set_title(
        f"2D Histogram with Vertex Positions for Event {event_id}", fontsize=22
    )

    # Increase tick label size
    ax.tick_params(axis="both", which="major", labelsize=16)

    # Larger legend with bigger markers
    ax.legend(loc="upper right", fontsize=16, markerscale=1.5)

    # Add statistics text box
    stats_info = f"True vertices: {len(true_z)}\n"
    stats_info += f"Initial seeded: {len(initial_z)}\n"
    stats_info += f"Fitted vertices: {len(rec_z)}"

    ax.text(
        0.02,
        0.98,
        stats_info,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        fontsize=18,
    )

    # Save plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / f"histogram_event_{event_id}.pdf"
    plt.tight_layout()
    plt.savefig(output_file, format="pdf", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved as {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot 2D histogram of vertex seeding and fitted positions"
    )
    parser.add_argument(
        "--hdf5_path",
        type=str,
        required=True,
        help="Path to the HDF5 file containing fit results",
    )
    parser.add_argument(
        "--event_id",
        type=int,
        default=0,
        help="Event ID to plot (if not specified, plots first event)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Output directory for plots"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="config.yaml",
        help="Path to config.yaml file",
    )
    parser.add_argument(
        "--draw_fixed_ellipses",
        action="store_true",
        help="Draw fixed matching ellipses around initial seeded positions",
    )
    parser.add_argument(
        "--draw_fitted_ellipses",
        action="store_true",
        help="Draw 3-sigma error ellipses around fitted positions",
    )

    args = parser.parse_args()

    plot_event_histogram(
        hdf5_path=args.hdf5_path,
        event_id=args.event_id,
        output_dir=args.output_dir,
        config_path=args.config_path,
        draw_fixed_ellipses=args.draw_fixed_ellipses,
        draw_fitted_ellipses=args.draw_fitted_ellipses,
    )
