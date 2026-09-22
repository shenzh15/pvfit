import copy
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from .utils import (
    calculate_clopper_pearson_errors,
    match_vertices_elliptical_zt,
    match_vertices_resolution_zt,
)


def analyze_vertex_efficiency(
    hdf5_path,
    output_dir="output",
    m_r=0.8,
    z_scale=1.5,
    t_scale=0.06,
    nsigma=3.0,
    plot_min_ntracks=4,
    plot_max_ntracks=200,
    plot_bin_width_ntracks=5,
    method_name="PVfit_Geo_Based",
):
    """
    Analyze vertex reconstruction efficiency using geometric (z,t) matching.

    A true vertex is considered reconstructed if:
        it is the closest_true_vertex of at least one reconstructed vertex
    after geometric matching in (z,t) space.
    """

    # ------------------------------------------------------------
    # Track-count bins (same as fake_rate_geo)
    # ------------------------------------------------------------
    num_bins = (plot_max_ntracks - plot_min_ntracks) // plot_bin_width_ntracks + 1
    bins = [
        (
            plot_min_ntracks + i * plot_bin_width_ntracks,
            plot_min_ntracks + (i + 1) * plot_bin_width_ntracks - 1,
        )
        for i in range(num_bins)
    ]

    total_true_per_bin = dict.fromkeys(bins, 0)
    matched_true_per_bin_geo = dict.fromkeys(bins, 0)

    # --- RESOLUTION ---
    matched_true_per_bin_reso = dict.fromkeys(bins, 0)

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
            true_track_indices = true_verts["track_indices"][:]

            true_vertices = []
            for i, (z, t, track_indices) in enumerate(
                zip(true_z, true_t, true_track_indices, strict=True)
            ):
                if len(track_indices) == 0:
                    continue

                true_vertices.append(
                    {
                        "vertex_id": i,
                        "z": z,
                        "t": t,
                        "num_tracks": len(track_indices),
                    }
                )

            # --------------------------------------------------
            # Load reconstructed vertices from new HDF5 structure
            # --------------------------------------------------
            rec_verts = event_data["rec_vertices"]
            rec_z = rec_verts["z"][:]
            rec_t = rec_verts["t"][:]
            z_err = rec_verts["z_err"][:]
            t_err = rec_verts["t_err"][:]

            rec_vertices = []
            for i, (z, t, ze, te) in enumerate(
                zip(rec_z, rec_t, z_err, t_err, strict=True)
            ):
                rec_vertices.append(
                    {
                        "rec_vertex_id": i,
                        "rec_z": z,
                        "rec_t": t,
                        "z_err": ze,
                        "t_err": te,
                    }
                )

            # --------------------------------------------------
            # Fill DENOMINATOR (true vertices)
            # --------------------------------------------------
            _process_true_vertices_for_denominator(
                true_vertices,
                bins,
                total_true_per_bin,
            )

            if not rec_vertices:
                continue

            rec_vertices_geo = copy.deepcopy(rec_vertices)
            rec_vertices_reso = copy.deepcopy(rec_vertices)

            # --------------------------------------------------
            # Run geometric matching (analysis-time decision)
            # --------------------------------------------------
            match_vertices_elliptical_zt(
                rec_vertices=rec_vertices_geo,
                true_vertices=true_vertices,
                m_r=m_r,
                z_scale=z_scale,
                t_scale=t_scale,
            )

            # --------------------------------------------------
            # Fill NUMERATOR (matched true vertices, unique per event)
            # --------------------------------------------------
            _process_true_vertices_for_numerator(
                rec_vertices_geo,
                true_vertices,
                bins,
                matched_true_per_bin_geo,
            )

            # --------------------------------------------------
            # Run resolution matching
            # --------------------------------------------------
            match_vertices_resolution_zt(
                rec_vertices=rec_vertices_reso,
                true_vertices=true_vertices,
                nsigma=nsigma,
            )

            # --------------------------------------------------
            # Fill NUMERATOR (matched true vertices, unique per event)
            # --------------------------------------------------
            _process_true_vertices_for_numerator(
                rec_vertices_reso,
                true_vertices,
                bins,
                matched_true_per_bin_reso,
            )

    # ------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------
    eff_geo, err_geo = _calculate_efficiencies(
        bins, total_true_per_bin, matched_true_per_bin_geo
    )

    eff_reso, err_reso = _calculate_efficiencies(
        bins, total_true_per_bin, matched_true_per_bin_reso
    )

    overall_stats = _calculate_overall_efficiency(
        total_true_per_bin, matched_true_per_bin_geo, matched_true_per_bin_reso
    )

    plot_data_geo = _prepare_plot_data(
        bins,
        eff_geo,
        err_geo,
        total_true_per_bin,
        matched_true_per_bin_geo,
    )

    plot_data_reso = _prepare_plot_data(
        bins,
        eff_reso,
        err_reso,
        total_true_per_bin,
        matched_true_per_bin_reso,
    )

    if plot_data_geo is None or plot_data_reso is None:
        print("No efficiency data to plot")
        return

    _create_efficiency_plots(
        plot_data_geo,
        plot_data_reso,
        m_r,
        nsigma,
        output_dir,
        method_name,
    )

    _print_efficiency_summary(
        overall_stats,
        m_r,
        z_scale,
        t_scale,
        total_true_per_bin,
        matched_true_per_bin_geo,
        matched_true_per_bin_reso,
    )

    _save_efficiency_to_csv(
        bins,
        eff_geo,
        err_geo,
        eff_reso,
        err_reso,
        total_true_per_bin,
        matched_true_per_bin_geo,
        matched_true_per_bin_reso,
        overall_stats,
        m_r,
        z_scale,
        t_scale,
        output_dir,
        method_name,
    )


# ======================================================================
# Helper functions (structure mirrors fake_rate_geo.py)
# ======================================================================


def _process_true_vertices_for_denominator(
    true_vertices,
    bins,
    total_true_per_bin,
):
    """Count true vertices (denominator)."""
    for tv in true_vertices:
        n_tracks = tv["num_tracks"]
        for bin_start, bin_end in bins:
            if bin_start <= n_tracks <= bin_end:
                total_true_per_bin[(bin_start, bin_end)] += 1
                break


def _process_true_vertices_for_numerator(
    rec_vertices,
    true_vertices,
    bins,
    matched_true_per_bin,
):
    """Count matched true vertices (numerator), unique per event."""
    matched_true_ids = set()

    for rec in rec_vertices:
        if rec["closest_true_vertex"] is None:
            continue
        matched_true_ids.add(rec["closest_true_vertex"]["vertex_id"])

    for tv in true_vertices:
        if tv["vertex_id"] not in matched_true_ids:
            continue
        n_tracks = tv["num_tracks"]
        for bin_start, bin_end in bins:
            if bin_start <= n_tracks <= bin_end:
                matched_true_per_bin[(bin_start, bin_end)] += 1
                break


def _calculate_efficiencies(bins, total_true_per_bin, matched_true_per_bin):
    efficiencies = {}
    errors = {}

    for bin_tuple in bins:
        total = total_true_per_bin[bin_tuple]
        matched = matched_true_per_bin[bin_tuple]
        eff = matched / total if total > 0 else 0.0

        err_lo, err_hi = calculate_clopper_pearson_errors(matched, total, eff)
        efficiencies[bin_tuple] = eff
        errors[bin_tuple] = {"lower": err_lo, "upper": err_hi}

    return efficiencies, errors


def _calculate_overall_efficiency(
    total_true_per_bin, matched_true_per_bin_geo, matched_true_per_bin_reso
):
    total_true = sum(total_true_per_bin.values())
    total_matched_geo = sum(matched_true_per_bin_geo.values())
    total_matched_reso = sum(matched_true_per_bin_reso.values())
    efficiency_geo = total_matched_geo / total_true if total_true > 0 else 0.0
    efficiency_reso = total_matched_reso / total_true if total_true > 0 else 0.0

    err_lo_geo, err_hi_geo = calculate_clopper_pearson_errors(
        total_matched_geo, total_true, efficiency_geo
    )

    err_lo_reso, err_hi_reso = calculate_clopper_pearson_errors(
        total_matched_reso, total_true, efficiency_reso
    )

    return {
        "efficiency_geo": efficiency_geo,
        "efficiency_reso": efficiency_reso,
        "error_lower_geo": err_lo_geo,
        "error_upper_geo": err_hi_geo,
        "error_lower_reso": err_lo_reso,
        "error_upper_reso": err_hi_reso,
        "total_true": total_true,
        "total_matched_geo": total_matched_geo,
        "total_matched_reso": total_matched_reso,
    }


def _prepare_plot_data(
    bins,
    efficiencies,
    efficiency_errors,
    total_true_per_bin,
    matched_true_per_bin,
):
    bin_centers = [(a + b) / 2 for a, b in bins]

    x, y, lo, hi = [], [], [], []
    total_f, matched_f = [], []

    for i, bin_tuple in enumerate(bins):
        total = total_true_per_bin[bin_tuple]
        if total == 0:
            continue

        x.append(bin_centers[i])
        y.append(efficiencies[bin_tuple])
        lo.append(efficiency_errors[bin_tuple]["lower"])
        hi.append(efficiency_errors[bin_tuple]["upper"])
        total_f.append(total)
        matched_f.append(matched_true_per_bin[bin_tuple])

    if not x:
        return None

    return {
        "x": x,
        "efficiency": y,
        "error_lower": lo,
        "error_upper": hi,
        "total_filtered": total_f,
        "matched_filtered": matched_f,
        "bin_centers": bin_centers,
    }


def _create_efficiency_plots(
    plot_data_geo,
    plot_data_reso,
    m_r,
    nsigma,
    output_dir,
    method_name,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    method_slug = method_name.lower().replace(" ", "_")

    x = plot_data_geo["x"]
    assert plot_data_geo["x"] == plot_data_reso["x"], "Geo and Reso bins mismatch!"

    y_geo = plot_data_geo["efficiency"]
    y_reso = plot_data_reso["efficiency"]

    err_geo_lo = plot_data_geo["error_lower"]
    err_geo_hi = plot_data_geo["error_upper"]

    err_reso_lo = plot_data_reso["error_lower"]
    err_reso_hi = plot_data_reso["error_upper"]

    # ============================================================
    # GEO ONLY
    # ============================================================
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.errorbar(x, y_geo, yerr=[err_geo_lo, err_geo_hi], fmt="o-", label="Geo matching")

    ax.set_xlabel("Number of tracks per true vertex")
    ax.set_ylabel("Efficiency")
    ax.set_title(f"Efficiency (Geometric Matching)\n m_r={m_r}")
    ax.set_ylim(0, 1.27)
    ax.grid(alpha=0.3)
    ax.legend()

    plt.savefig(
        Path(output_dir) / f"efficiency_geo_{method_slug}.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # ============================================================
    # RESOLUTION ONLY
    # ============================================================
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.errorbar(
        x,
        y_reso,
        yerr=[err_reso_lo, err_reso_hi],
        fmt="s-",
        label=f"Resolution matching ({nsigma}σ)",
    )

    ax.set_xlabel("Number of tracks per true vertex")
    ax.set_ylabel("Efficiency")
    ax.set_title(f"Efficiency (Resolution Matching, {nsigma}σ)")
    ax.set_ylim(0, 1.27)
    ax.grid(alpha=0.3)
    ax.legend()

    plt.savefig(
        Path(output_dir) / f"efficiency_reso_{method_slug}.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # ============================================================
    # COMPARISON
    # ============================================================
    fig, ax = plt.subplots(figsize=(9, 6))

    # GEO
    ax.errorbar(
        x,
        y_geo,
        yerr=[err_geo_lo, err_geo_hi],
        fmt="o-",
        label=f"Geo matching (m_r={m_r})",
    )

    # RESO
    ax.errorbar(
        x,
        y_reso,
        yerr=[err_reso_lo, err_reso_hi],
        fmt="s-",
        label=f"Resolution matching ({nsigma}σ)",
    )

    ax.set_xlabel("Number of tracks per true vertex")
    ax.set_ylabel("Efficiency")
    ax.set_title("Efficiency Comparison (Geo vs Resolution)")
    ax.set_ylim(0, 1.27)
    ax.grid(alpha=0.3)
    ax.legend()

    plt.savefig(
        Path(output_dir) / f"efficiency_comparison_{method_slug}.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # ============================================================
    # RATIO
    # ============================================================
    ratio = []
    for g, r in zip(y_geo, y_reso):
        if g > 0:
            ratio.append(r / g)
        else:
            ratio.append(0)

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(x, ratio, "d-", label="Resolution / Geo")

    ax.axhline(1.0, linestyle="--", color="gray")

    ax.set_xlabel("Number of tracks per true vertex")
    ax.set_ylabel("Efficiency Ratio")
    ax.set_title("Efficiency Ratio (Resolution / Geo)")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.savefig(
        Path(output_dir) / f"efficiency_ratio_{method_slug}.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def _print_efficiency_summary(
    overall_stats,
    m_r,
    z_scale,
    t_scale,
    total_true_per_bin,
    matched_true_per_bin_geo,
    matched_true_per_bin_reso,
):
    print("\n" + "-" * 70)
    print("EFFICIENCY ANALYSIS SUMMARY (GEOMETRIC MATCHING)")
    print("-" * 70)
    print(f"Matching: elliptical z-t, m_r={m_r}, z_scale={z_scale}, t_scale={t_scale}")
    print(f"Total true vertices: {overall_stats['total_true']}")
    print(f"Matched true vertices: {overall_stats['total_matched_geo']}")
    print(f"Matched true vertices: {overall_stats['total_matched_reso']}")
    print(
        f"Overall efficiency (geo): {overall_stats['efficiency_geo']:.3f} "
        f"+{overall_stats['error_upper_geo']:.3f} "
        f"-{overall_stats['error_lower_geo']:.3f}"
    )
    print(
        f"Overall efficiency (reso): {overall_stats['efficiency_reso']:.3f} "
        f"+{overall_stats['error_upper_reso']:.3f} "
        f"-{overall_stats['error_lower_reso']:.3f}"
    )

    print("\nEfficiency by track multiplicity:")
    for (a, b), total in total_true_per_bin.items():
        if total > 0:
            matched_geo = matched_true_per_bin_geo[(a, b)]
            matched_reso = matched_true_per_bin_reso[(a, b)]
            print(f"[{a},{b}] Geo : {matched_geo}/{total} = {matched_geo / total:.3f}")
            print(
                f"[{a},{b}] Reso : {matched_reso}/{total} = {matched_reso / total:.3f}"
            )

    print("-" * 70 + "\n")


def _save_efficiency_to_csv(
    bins,
    eff_geo,
    err_geo,
    eff_reso,
    err_reso,
    total_true_per_bin,
    matched_true_per_bin_geo,
    matched_true_per_bin_reso,
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
                "bin_center": (a + b) / 2,
                "efficiency_geo": eff_geo[(a, b)],
                "error_lower_geo": err_geo[(a, b)]["lower"],
                "error_upper_geo": err_geo[(a, b)]["upper"],
                "efficiency_reso": eff_reso[(a, b)],
                "error_lower_reso": err_reso[(a, b)]["lower"],
                "error_upper_reso": err_reso[(a, b)]["upper"],
                "num_true": total_true_per_bin[(a, b)],
                "num_matched_geo": matched_true_per_bin_geo[(a, b)],
                "num_matched_reso": matched_true_per_bin_reso[(a, b)],
                "m_r": m_r,
                "z_scale": z_scale,
                "t_scale": t_scale,
            }
        )

    pd.DataFrame(rows).to_csv(
        Path(output_dir) / f"efficiency_true_binned_{method_slug}.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "total_true": overall_stats["total_true"],
                "total_matched_geo": overall_stats["total_matched_geo"],
                "total_matched_reso": overall_stats["total_matched_reso"],
                "overall_efficiency_geo": overall_stats["efficiency_geo"],
                "error_lower_geo": overall_stats["error_lower_geo"],
                "error_upper_geo": overall_stats["error_upper_geo"],
                "overall_efficiency_reso": overall_stats["efficiency_reso"],
                "error_lower_reso": overall_stats["error_lower_reso"],
                "error_upper_reso": overall_stats["error_upper_reso"],
                "m_r": m_r,
                "z_scale": z_scale,
                "t_scale": t_scale,
            }
        ]
    ).to_csv(
        Path(output_dir) / f"efficiency_overall_true_binned_{method_slug}.csv",
        index=False,
    )
