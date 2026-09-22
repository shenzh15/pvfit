from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from .utils import (
    calculate_clopper_pearson_errors,
    match_vertices_elliptical_zt,
)


def analyze_vertex_fake_rate(
    hdf5_path,
    output_dir="output",
    m_r=1.0,
    z_scale=1.5,
    t_scale=0.06,
    plot_min_ntracks=4,
    plot_max_ntracks=200,
    plot_bin_width_ntracks=5,
    method_name="PVfit_Geo_Based",
):
    """
    Analyze fake vertex rate using geometric (z,t) matching.

    A reconstructed vertex is considered fake if:
        closest_true_vertex is None
    after geometric matching in (z,t) space.
    """

    # ------------------------------------------------------------
    # Track-count bins
    # ------------------------------------------------------------
    num_bins = (plot_max_ntracks - plot_min_ntracks) // plot_bin_width_ntracks + 1
    bins = [
        (
            plot_min_ntracks + i * plot_bin_width_ntracks,
            plot_min_ntracks + (i + 1) * plot_bin_width_ntracks - 1,
        )
        for i in range(num_bins)
    ]

    total_rec_per_bin = dict.fromkeys(bins, 0)
    fake_rec_per_bin = dict.fromkeys(bins, 0)

    print(f"Loading HDF5 file: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as f:
        event_ids = sorted(int(k) for k in f.keys())
        print(f"Found {len(event_ids)} events")

        for event_id in tqdm(event_ids, desc="Processing events"):
            event_data = f[str(event_id)]

            # --------------------------------------------------
            # Load true vertices from new HDF5 structure
            # --------------------------------------------------
            true_verts = event_data["true_vertices"]
            true_z = true_verts["z"][:]
            true_t = true_verts["t"][:]

            true_vertices = []
            for i, (z, t) in enumerate(zip(true_z, true_t, strict=True)):
                true_vertices.append(
                    {
                        "vertex_id": i,
                        "z": z,
                        "t": t,
                    }
                )

            # --------------------------------------------------
            # Load reconstructed vertices from new HDF5 structure
            # --------------------------------------------------
            rec_verts = event_data["rec_vertices"]
            rec_z = rec_verts["z"][:]
            rec_t = rec_verts["t"][:]
            rec_track_indices = rec_verts["track_indices"][:]

            rec_vertices = []
            for i, (z, t, track_indices) in enumerate(
                zip(rec_z, rec_t, rec_track_indices, strict=True)
            ):
                n_tracks = len(track_indices)
                if n_tracks == 0:
                    continue

                rec_vertices.append(
                    {
                        "rec_vertex_id": i,
                        "rec_z": z,
                        "rec_t": t,
                        "num_tracks": n_tracks,
                    }
                )

            # --------------------------------------------------
            # Run geometric matching (analysis-time decision)
            # --------------------------------------------------
            match_vertices_elliptical_zt(
                rec_vertices=rec_vertices,
                true_vertices=true_vertices,
                m_r=m_r,
                z_scale=z_scale,
                t_scale=t_scale,
            )

            # --------------------------------------------------
            # Count fake vertices
            # --------------------------------------------------
            _process_rec_vertices_for_fake(
                rec_vertices,
                bins,
                total_rec_per_bin,
                fake_rec_per_bin,
            )

    # ------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------
    fake_rates, fake_rate_errors = _calculate_fake_rates(
        bins, total_rec_per_bin, fake_rec_per_bin
    )

    overall_stats = _calculate_overall_fake_rate(total_rec_per_bin, fake_rec_per_bin)

    plot_data = _prepare_plot_data(
        bins,
        fake_rates,
        fake_rate_errors,
        total_rec_per_bin,
        fake_rec_per_bin,
    )

    if plot_data is None:
        print("No fake rate data to plot")
        return

    _create_fake_rate_plots(
        plot_data,
        m_r,
        output_dir,
        method_name,
    )

    _print_fake_rate_summary(
        overall_stats,
        m_r,
        z_scale,
        t_scale,
        total_rec_per_bin,
        fake_rec_per_bin,
    )

    _save_fake_rate_to_csv(
        bins,
        plot_data,
        fake_rates,
        fake_rate_errors,
        total_rec_per_bin,
        fake_rec_per_bin,
        overall_stats,
        m_r,
        z_scale,
        t_scale,
        output_dir,
        method_name,
    )


# ======================================================================
# Helper functions (structure mirrors track-based fake rate)
# ======================================================================


def _process_rec_vertices_for_fake(
    rec_vertices,
    bins,
    total_rec_per_bin,
    fake_rec_per_bin,
):
    """Count fake reconstructed vertices after geometric matching."""
    for rec in rec_vertices:
        n_tracks = rec["num_tracks"]
        is_fake = rec["closest_true_vertex"] is None

        for bin_start, bin_end in bins:
            if bin_start <= n_tracks <= bin_end:
                total_rec_per_bin[(bin_start, bin_end)] += 1
                if is_fake:
                    fake_rec_per_bin[(bin_start, bin_end)] += 1
                break


def _calculate_fake_rates(bins, total_rec_per_bin, fake_rec_per_bin):
    fake_rates = {}
    errors = {}

    for bin_tuple in bins:
        total = total_rec_per_bin[bin_tuple]
        fake = fake_rec_per_bin[bin_tuple]
        rate = fake / total if total > 0 else 0.0

        err_lo, err_hi = calculate_clopper_pearson_errors(fake, total, rate)
        fake_rates[bin_tuple] = rate
        errors[bin_tuple] = {"lower": err_lo, "upper": err_hi}

    return fake_rates, errors


def _calculate_overall_fake_rate(total_rec_per_bin, fake_rec_per_bin):
    total_rec = sum(total_rec_per_bin.values())
    total_fake = sum(fake_rec_per_bin.values())
    fake_rate = total_fake / total_rec if total_rec > 0 else 0.0

    err_lo, err_hi = calculate_clopper_pearson_errors(total_fake, total_rec, fake_rate)

    return {
        "fake_rate": fake_rate,
        "error_lower": err_lo,
        "error_upper": err_hi,
        "total_rec": total_rec,
        "total_fake": total_fake,
    }


def _prepare_plot_data(
    bins,
    fake_rates,
    fake_rate_errors,
    total_rec_per_bin,
    fake_rec_per_bin,
):
    bin_centers = [(a + b) / 2 for a, b in bins]

    x, y, lo, hi = [], [], [], []
    total_f, fake_f = [], []

    for i, bin_tuple in enumerate(bins):
        total = total_rec_per_bin[bin_tuple]
        if total == 0:
            continue

        x.append(bin_centers[i])
        y.append(fake_rates[bin_tuple])
        lo.append(fake_rate_errors[bin_tuple]["lower"])
        hi.append(fake_rate_errors[bin_tuple]["upper"])
        total_f.append(total)
        fake_f.append(fake_rec_per_bin[bin_tuple])

    if not x:
        return None

    return {
        "x": x,
        "fake_rate": y,
        "error_lower": lo,
        "error_upper": hi,
        "total_filtered": total_f,
        "fake_filtered": fake_f,
        "bin_centers": bin_centers,
    }


def _create_fake_rate_plots(
    plot_data,
    m_r,
    output_dir,
    method_name,
):
    x = plot_data["x"]
    y = plot_data["fake_rate"]

    lower = [r - e for r, e in zip(y, plot_data["error_lower"], strict=True)]
    upper = [r + e for r, e in zip(y, plot_data["error_upper"], strict=True)]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x, y, "ro-", label="Fake rate (geo matching)")
    ax.fill_between(x, lower, upper, alpha=0.3, color="red")

    ax.set_xlabel("Number of tracks per reconstructed vertex")
    ax.set_ylabel("Fake rate")
    ax.set_title(f"Vertex Fake Rate vs Track Count (Geometric Matching)\nm_r={m_r}")
    ax.set_ylim(0, 1.27)
    ax.grid(alpha=0.3)
    ax.legend()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    method_slug = method_name.lower().replace(" ", "_")
    plt.savefig(
        Path(output_dir) / f"fake_rate_rec_binned_{method_slug}.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def _print_fake_rate_summary(
    overall_stats,
    m_r,
    z_scale,
    t_scale,
    total_rec_per_bin,
    fake_rec_per_bin,
):
    print("\n" + "-" * 70)
    print("FAKE RATE ANALYSIS SUMMARY (GEOMETRIC MATCHING)")
    print("-" * 70)
    print(f"Matching: elliptical z-t, m_r={m_r}, z_scale={z_scale}, t_scale={t_scale}")
    print(f"Total reconstructed vertices: {overall_stats['total_rec']}")
    print(f"Fake vertices: {overall_stats['total_fake']}")
    print(
        f"Overall fake rate: {overall_stats['fake_rate']:.3f} "
        f"+{overall_stats['error_upper']:.3f} "
        f"-{overall_stats['error_lower']:.3f}"
    )

    print("\nFake rate by track multiplicity:")
    for (a, b), total in total_rec_per_bin.items():
        if total > 0:
            fake = fake_rec_per_bin[(a, b)]
            print(f"[{a},{b}] : {fake}/{total} = {fake / total:.3f}")

    print("-" * 70 + "\n")


def _save_fake_rate_to_csv(
    bins,
    plot_data,
    fake_rates,
    fake_rate_errors,
    total_rec_per_bin,
    fake_rec_per_bin,
    overall_stats,
    m_r,
    z_scale,
    t_scale,
    output_dir,
    method_name,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    method_slug = method_name.lower().replace(" ", "_")

    rows = []
    for i, (a, b) in enumerate(bins):
        rows.append(
            {
                "bin_start": a,
                "bin_end": b,
                "bin_center": plot_data["bin_centers"][i],
                "fake_rate": fake_rates[(a, b)],
                "error_lower": fake_rate_errors[(a, b)]["lower"],
                "error_upper": fake_rate_errors[(a, b)]["upper"],
                "num_total": total_rec_per_bin[(a, b)],
                "num_fake": fake_rec_per_bin[(a, b)],
                "m_r": m_r,
                "z_scale": z_scale,
                "t_scale": t_scale,
            }
        )

    pd.DataFrame(rows).to_csv(
        Path(output_dir) / f"fake_rate_rec_binned_{method_slug}.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "total_rec": overall_stats["total_rec"],
                "total_fake": overall_stats["total_fake"],
                "overall_fake_rate": overall_stats["fake_rate"],
                "error_lower": overall_stats["error_lower"],
                "error_upper": overall_stats["error_upper"],
                "m_r": m_r,
                "z_scale": z_scale,
                "t_scale": t_scale,
            }
        ]
    ).to_csv(
        Path(output_dir) / f"fake_rate_overall_rec_binned_{method_slug}.csv",
        index=False,
    )
