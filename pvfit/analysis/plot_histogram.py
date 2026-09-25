"""Plot the 2D (z, t) seeding histogram with vertex overlays.

Works for any HDF5 file that follows the PVFit event layout, whichever program
produced it. Only ``histogram`` is required; each overlay is drawn when present:

    <event_id>/
    ├── histogram [n_bins_z, n_bins_t]
    │   @z_min, @z_max, @t_min, @t_max, @n_bins_z, @n_bins_t
    ├── seeds/          z, t                              (all seeds / peaks)
    ├── true_vertices/  z, t  [vertex_id, track_indices]  (labels)
    └── rec_vertices/   z, t  [z_err, t_err]              (error ellipses)

Files written before ``seeds/`` existed fall back to ``rec_vertices/{z,t}_seed``.
The optional file attribute ``@source`` (e.g. "PVFit", "Allen") labels the plot.
"""

from pathlib import Path

import h5py
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def _read_points(event_group, group_name, *optional, z="z", t="t"):
    """Read (z, t) and any optional per-point fields; None if not stored."""
    group = event_group.get(group_name)
    if group is None or z not in group or t not in group:
        return None
    points = {"z": group[z][:], "t": group[t][:]}
    for name in optional:
        if name in group:
            points[name] = group[name][:]
    return points


def load_event_histogram(hdf5_path, event_id=None):
    """Load one event's histogram and overlays from an HDF5 file.

    Args:
        hdf5_path: Path to an HDF5 file in the PVFit event layout
        event_id: Event ID to load. If None, loads the first event in the file

    Returns:
        dict with event_id, source, histogram, z_edges, t_edges and the overlays
        true / seeds / rec (each a dict of arrays, or None when not stored);
        None if the event or its histogram is missing.
    """
    with h5py.File(hdf5_path, "r") as f:
        available = sorted(int(key) for key in f.keys())
        if event_id is None:
            if not available:
                print("No events found in HDF5 file")
                return None
            event_id = available[0]
            print(f"Using first event: {event_id}")

        if str(event_id) not in f:
            print(f"Event {event_id} not found in HDF5 file")
            print(
                f"Available events: {available[:10]}{'...' if len(available) > 10 else ''}"
            )
            return None

        event_group = f[str(event_id)]
        if "histogram" not in event_group:
            print(f"No histogram data found for event {event_id}")
            return None

        H = event_group["histogram"][:]
        hist_attrs = event_group["histogram"].attrs
        z_bins = int(hist_attrs["n_bins_z"])
        t_bins = int(hist_attrs["n_bins_t"])
        if H.shape != (z_bins, t_bins):
            raise ValueError(
                f"Event {event_id}: histogram shape {H.shape} does not match "
                f"(n_bins_z, n_bins_t) = {(z_bins, t_bins)}"
            )

        seeds = _read_points(event_group, "seeds")
        if seeds is None:
            seeds = _read_points(event_group, "rec_vertices", z="z_seed", t="t_seed")

        source = f.attrs.get("source")
        return {
            "event_id": event_id,
            "source": source.decode() if isinstance(source, bytes) else source,
            "histogram": H,
            "z_edges": np.linspace(hist_attrs["z_min"], hist_attrs["z_max"], z_bins + 1),
            "t_edges": np.linspace(hist_attrs["t_min"], hist_attrs["t_max"], t_bins + 1),
            "true": _read_points(
                event_group, "true_vertices", "vertex_id", "track_indices"
            ),
            "seeds": seeds,
            "rec": _read_points(event_group, "rec_vertices", "z_err", "t_err"),
        }


def plot_event_histogram(hdf5_path, output_dir, settings, event_id=None, label=None):
    """Create a 2D visualization of the seeding histogram with vertex overlays.

    Plots the track density histogram used for vertex seeding, with overlaid markers
    for seeds, fitted vertices and true vertices when the file stores them.

    Args:
        hdf5_path: Path to the HDF5 file (PVFit event layout, any producer)
        output_dir: Directory for saving plots
        settings: AnalysisConfig; uses the matching scales and the draw_* flags
        event_id: Event ID to plot. If None, plots the first event in the file
        label: Title prefix; defaults to the file's @source attribute

    Returns:
        Path of the saved PDF, or None if the event could not be plotted
    """
    print("Plotting 2D Histogram...")

    draw_fixed_ellipses = settings.draw_fixed_ellipses
    draw_fitted_ellipses = settings.draw_fitted_ellipses
    if draw_fixed_ellipses:
        m_r, z_scale, t_scale = settings.m_r, settings.z_scale, settings.t_scale
        print(f"Using matching scales: m_r={m_r}, z_scale={z_scale}, t_scale={t_scale}")

    data = load_event_histogram(hdf5_path, event_id)
    if data is None:
        return None

    z_edges, t_edges = data["z_edges"], data["t_edges"]
    print(
        f"Histogram parameters: z_range=({z_edges[0]:g}, {z_edges[-1]:g}), "
        f"t_range=({t_edges[0]:g}, {t_edges[-1]:g}), "
        f"z_bins={len(z_edges) - 1}, t_bins={len(t_edges) - 1}"
    )

    fig, ax = plt.subplots(figsize=(15, 8))

    # Copy as float so empty bins can be hidden as NaN (also for integer input)
    H_plot = data["histogram"].astype(float).T
    H_plot[H_plot == 0] = np.nan
    im = ax.pcolormesh(z_edges, t_edges, H_plot, cmap="jet", shading="auto")

    true, seeds, rec = data["true"], data["seeds"], data["rec"]

    # Mark true vertices with red circles, labelled id(n_tracks)
    if true is not None and len(true["z"]) > 0:
        ax.scatter(
            true["z"],
            true["t"],
            color="red",
            s=30,
            alpha=0.8,
            label="True Vertices",
            marker="o",
            edgecolors="darkred",
            linewidths=0.8,
        )
        ids = true.get("vertex_id", np.arange(len(true["z"])))
        for i in range(len(true["z"])):
            text = f"{int(ids[i])}"
            if "track_indices" in true:
                text += f"({len(true['track_indices'][i])})"
            ax.annotate(
                text,
                xy=(true["z"][i], true["t"][i]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
                color="red",
                fontweight="bold",
            )

    # Mark seeds with green X marks
    if seeds is not None and len(seeds["z"]) > 0:
        ax.scatter(
            seeds["z"],
            seeds["t"],
            color="green",
            s=20,
            alpha=0.8,
            label="Seeds",
            marker="x",
            linewidths=1.5,
        )

    # Mark fitted vertices with orange squares
    if rec is not None and len(rec["z"]) > 0:
        ax.scatter(
            rec["z"],
            rec["t"],
            color="orange",
            s=15,
            alpha=0.9,
            label="Fitted Vertices",
            marker="s",
            edgecolors="darkorange",
            linewidths=0.8,
        )

    # Draw fixed matching ellipses around seeds
    if draw_fixed_ellipses and seeds is not None:
        for i in range(len(seeds["z"])):
            ellipse = patches.Ellipse(
                xy=(seeds["z"][i], seeds["t"][i]),
                width=z_scale * m_r,  # diameter in z
                height=t_scale * m_r,  # diameter in t
                fill=False,
                edgecolor="cyan",
                linewidth=0.8,
                alpha=0.7,
                label="Fixed Matching Ellipse" if i == 0 else "",
            )
            ax.add_patch(ellipse)

    # Draw 5-sigma ellipses around fitted positions
    if draw_fitted_ellipses and rec is not None and "z_err" in rec and "t_err" in rec:
        for i in range(len(rec["z"])):
            ellipse_fitted = patches.Ellipse(
                xy=(rec["z"][i], rec["t"][i]),
                width=2 * 5 * rec["z_err"][i],  # diameter
                height=2 * 5 * rec["t_err"][i],  # diameter
                fill=False,
                edgecolor="magenta",
                linewidth=0.8,
                alpha=0.7,
                label="5σ Error Ellipse" if i == 0 else "",
            )
            ax.add_patch(ellipse_fitted)

    # Color bar & labels
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Track Density (counts)", fontsize=18)
    cbar.ax.tick_params(labelsize=14)

    ax.set_xlabel("beamPOCA_z (mm)", fontsize=20)
    ax.set_ylabel("beamPOCA_t (ns)", fontsize=20)
    label = label or data["source"]
    title = f"2D Histogram with Vertex Positions for Event {data['event_id']}"
    ax.set_title(f"{label}: {title}" if label else title, fontsize=22)
    ax.tick_params(axis="both", which="major", labelsize=16)

    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right", fontsize=16, markerscale=1.5)

    # Statistics text box, only for overlays stored in the file
    stats = [
        f"{name}: {len(points['z'])}"
        for name, points in (
            ("True vertices", true),
            ("Seeds", seeds),
            ("Fitted vertices", rec),
        )
        if points is not None
    ]
    if stats:
        ax.text(
            0.02,
            0.98,
            "\n".join(stats),
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            fontsize=18,
        )

    # Save plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / f"histogram_event_{data['event_id']}.pdf"
    fig.tight_layout()
    fig.savefig(output_file, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Plot saved as {output_file}")
    return output_file
