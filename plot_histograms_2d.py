#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot
from matplotlib.colors import LogNorm

from src.fit.truth_matching import (
    build_true_vertex_info,
    read_mcpv_dataframe,
    select_tracks,
)

DEFAULT_ZMIN = -200.0
DEFAULT_ZMAX = 200.0
DEFAULT_DZ = 0.5
DEFAULT_TMIN = -0.6
DEFAULT_TMAX = 0.8
DEFAULT_DT = 0.025


def compute_uniform_edges(xmin, xmax, dx):
    nbins = int(round((xmax - xmin) / dx))
    edges = [xmin + i * dx for i in range(nbins + 1)]
    return edges


def load_all_rows(tree):
    arrays = tree.arrays(library="np")
    event_field = "event_number" if "event_number" in arrays else "event"
    rows = {}
    for ev, z, t, weight in zip(
        arrays[event_field], arrays["z_bin"], arrays["t_bin"], arrays["weight"]
    ):
        rows.setdefault(int(ev), []).append((float(z), float(t), float(weight)))
    return rows


def load_peaks(tree):
    arrays = tree.arrays(library="np")
    event_field = "event_number" if "event_number" in arrays else "event"
    peaks = {}
    for ev, z, t in zip(arrays[event_field], arrays["z_peak"], arrays["t_peak"]):
        peaks.setdefault(int(ev), []).append((float(z), float(t)))
    return peaks


def load_mcpvs(
    mcpv_tree,
    monitor_tree,
    min_reco_tracks=4,
    zmin=DEFAULT_ZMIN,
    zmax=DEFAULT_ZMAX,
    tmin=DEFAULT_TMIN,
    tmax=DEFAULT_TMAX,
):
    """Use exactly the same truth membership and z/t cuts as reconstruction."""
    vertices = read_mcpv_dataframe(mcpv_tree)
    tracks = pd.DataFrame(monitor_tree.arrays(library="np"))
    tracks = select_tracks(tracks, dict(zmin=zmin, zmax=zmax, tmin=tmin, tmax=tmax))
    track_groups = dict(tuple(tracks.groupby("event_number")))
    mcpvs = {}
    for ev, pvs in vertices.groupby("event_number"):
        info = build_true_vertex_info(
            track_groups.get(ev, tracks.iloc[:0]),
            pvs,
            min_tracks=min_reco_tracks,
        )
        mcpvs[int(ev)] = [(pv["z"], pv["t"]) for pv in info]
    return mcpvs


def build_histogram_grid(rows, z_edges, t_edges):
    hist = np.zeros((len(t_edges) - 1, len(z_edges) - 1), dtype=np.float32)

    for z_bin, t_bin, weight in rows:
        xbin = np.searchsorted(z_edges, z_bin, side="right") - 1
        ybin = np.searchsorted(t_edges, t_bin, side="right") - 1
        if 0 <= xbin < hist.shape[1] and 0 <= ybin < hist.shape[0]:
            hist[ybin, xbin] = weight

    return hist


def plot_event(rows, peaks, mcpvs, ev, args, z_edges, t_edges, output_dir):
    hist = build_histogram_grid(rows, z_edges, t_edges)
    hist_plot = hist.astype(np.float32, copy=True)
    hist_plot[hist_plot == 0] = np.nan
    filled_bins = int(np.count_nonzero(hist > 0))
    total_bins = int(hist.size)
    filled_ratio = 100.0 * filled_bins / total_bins if total_bins else 0.0

    fig, ax = plt.subplots(figsize=(15, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    cmap = plt.get_cmap("jet").copy()
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))

    positive = hist[np.isfinite(hist) & (hist > 0)]
    norm = None
    if args.logz and positive.size:
        norm = LogNorm(vmin=float(positive.min()), vmax=float(positive.max()))

    mesh = ax.pcolormesh(
        z_edges, t_edges, hist_plot, cmap=cmap, shading="auto", norm=norm
    )

    if mcpvs:
        mc_z = np.array([pv[0] for pv in mcpvs], dtype=np.float32)
        mc_t = np.array([pv[1] for pv in mcpvs], dtype=np.float32)
        ax.scatter(
            mc_z,
            mc_t,
            color="red",
            s=40,
            alpha=0.9,
            label="True PVs",
            marker="o",
            zorder=4,
        )

    if peaks:
        peak_z = np.array([peak[0] for peak in peaks], dtype=np.float32)
        peak_t = np.array([peak[1] for peak in peaks], dtype=np.float32)
        ax.scatter(
            peak_z,
            peak_t,
            color="green",
            s=56,
            alpha=0.95,
            label="Seed Peaks",
            marker="x",
            linewidths=2.0,
            zorder=3,
        )

        if len(peaks) <= 12:
            for idx, (z_peak, t_peak) in enumerate(peaks):
                ax.annotate(
                    f"P{idx}",
                    xy=(z_peak, t_peak),
                    xytext=(5, 4),
                    textcoords="offset points",
                    fontsize=10,
                    color="#14532d",
                    fontweight="bold",
                )

    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("Track Density (counts)", fontsize=17)
    cbar.ax.tick_params(labelsize=13)

    ax.set_xlim(args.zmin, args.zmax)
    ax.set_ylim(args.tmin, args.tmax)
    ax.set_xlabel("beamPOCA z (mm)", fontsize=19)
    ax.set_ylabel("beamPOCA t (ns)", fontsize=19)
    ax.set_title(
        f"PV Beamline Histogram for Event {ev} (MC key truth matching)",
        fontsize=21,
        pad=14,
    )
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.grid(color="#94a3b8", alpha=0.22, linewidth=0.8)

    if peaks or mcpvs:
        ax.legend(loc="upper right", fontsize=14, frameon=True, framealpha=0.92)

    stats_info = f"Filled bins: {filled_bins} / {total_bins} ({filled_ratio:.2f}%)\n"
    stats_info += f"True PVs: {len(mcpvs)}, Seed peaks: {len(peaks)}\n"
    stats_info += f"z bins: {len(z_edges) - 1}, t bins: {len(t_edges) - 1}"
    ax.text(
        0.02,
        0.98,
        stats_info,
        transform=ax.transAxes,
        va="top",
        bbox=dict(
            boxstyle="round", facecolor="#f5e6b3", edgecolor="#9c7b2f", alpha=0.92
        ),
        fontsize=14,
    )

    output_path = output_dir / f"pv_beamline_histo_2d_ev{ev}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "input" / "key_sample.root"
    default_output_dir = script_dir / "figs"

    parser = argparse.ArgumentParser(
        description="Plot the 2D pv_beamline_histo sparse histogram stored in a ROOT TTree."
    )
    parser.add_argument(
        "--input", default=str(default_input), help="Path to the ROOT file."
    )
    parser.add_argument(
        "--tree",
        default="pv_beamline_histo/histo_tree",
        help="Path to the TTree inside the ROOT file.",
    )
    parser.add_argument(
        "--peak-tree",
        default="pv_beamline_peak/peak_tree",
        help="Path to the peak TTree.",
    )
    parser.add_argument(
        "--mcpv-tree",
        default="mc_pv_dumper/MCPV_effective_tree",
        help="Effective MC PV tree containing track_keys.",
    )
    parser.add_argument(
        "--monitor-tree",
        default="velo_kalman_filter/monitor_tree",
        help="Path to the track monitor TTree.",
    )
    parser.add_argument(
        "--min-reco-tracks",
        type=int,
        default=4,
        help="Minimum reconstructed tracks matched to show a true PV.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_output_dir),
        help="Output directory for images.",
    )
    parser.add_argument(
        "--event",
        type=int,
        default=None,
        help="Only plot one event number. Default: plot all events separately.",
    )
    parser.add_argument(
        "--logz", action="store_true", help="Use log scale on the color axis."
    )
    parser.add_argument(
        "--zmin",
        type=float,
        default=DEFAULT_ZMIN,
        help="Minimum z shown on the histogram.",
    )
    parser.add_argument(
        "--zmax",
        type=float,
        default=DEFAULT_ZMAX,
        help="Maximum z shown on the histogram.",
    )
    parser.add_argument("--dz", type=float, default=DEFAULT_DZ, help="z bin width.")
    parser.add_argument(
        "--tmin",
        type=float,
        default=DEFAULT_TMIN,
        help="Minimum t shown on the histogram.",
    )
    parser.add_argument(
        "--tmax",
        type=float,
        default=DEFAULT_TMAX,
        help="Maximum t shown on the histogram.",
    )
    parser.add_argument("--dt", type=float, default=DEFAULT_DT, help="t bin width.")
    args = parser.parse_args()

    root_file = uproot.open(args.input)

    tree = root_file.get(args.tree)
    if tree is None:
        raise RuntimeError(f"Failed to find TTree: {args.tree}")

    all_rows = load_all_rows(tree)
    if not all_rows:
        raise RuntimeError(f"No rows found in {args.tree}")

    all_peaks = {}
    peak_tree = root_file.get(args.peak_tree)
    if peak_tree is not None:
        all_peaks = load_peaks(peak_tree)

    all_mcpvs = {}
    mcpv_tree = root_file.get(args.mcpv_tree)
    monitor_tree = root_file.get(args.monitor_tree)
    if mcpv_tree is None or monitor_tree is None:
        raise ValueError("Both truth and monitor trees are required for truth overlays")
    all_mcpvs = load_mcpvs(
        mcpv_tree,
        monitor_tree,
        args.min_reco_tracks,
        zmin=args.zmin,
        zmax=args.zmax,
        tmin=args.tmin,
        tmax=args.tmax,
    )
    root_file.close()

    if args.event is not None:
        if args.event not in all_rows:
            raise RuntimeError(
                f"Event {args.event} not found. Available: {sorted(all_rows.keys())}"
            )
        events_to_plot = [args.event]
    else:
        events_to_plot = sorted(all_rows.keys())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    z_edges = np.asarray(
        compute_uniform_edges(args.zmin, args.zmax, args.dz), dtype=np.float64
    )
    t_edges = np.asarray(
        compute_uniform_edges(args.tmin, args.tmax, args.dt), dtype=np.float64
    )

    for ev in events_to_plot:
        rows = all_rows[ev]
        peaks = all_peaks.get(ev, [])
        mcpvs = all_mcpvs.get(ev, [])
        out_path = plot_event(
            rows, peaks, mcpvs, ev, args, z_edges, t_edges, output_dir
        )
        print(
            f"Event {ev}: {len(rows)} filled bins, {len(peaks)} peaks, {len(mcpvs)} true PVs -> {out_path}"
        )

    print(f"\nInput: {args.input}")
    print(f"Output dir: {output_dir}")
    print(f"z range: [{args.zmin}, {args.zmax}] with dz={args.dz}")
    print(f"t range: [{args.tmin}, {args.tmax}] with dt={args.dt}")
    print(f"Total events: {len(events_to_plot)}")


if __name__ == "__main__":
    main()
