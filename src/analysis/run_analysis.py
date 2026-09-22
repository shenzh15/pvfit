# src/analysis/run_analysis.py

"""
run_analysis.py

Central entry point for HDF5-based reconstruction performance studies.
"""

import os
from typing import Iterable, Optional

from .analyze_efficiency_geo import (
    analyze_vertex_efficiency as analyze_vertex_efficiency_geo,
)
from .analyze_efficiency_track import (
    analyze_vertex_efficiency as analyze_vertex_efficiency_track,
)
from .analyze_fake_rate_geo import (
    analyze_vertex_fake_rate as analyze_vertex_fake_rate_geo,
)
from .analyze_fake_rate_track import (
    analyze_vertex_fake_rate as analyze_vertex_fake_rate_track,
)
from .analyze_merge_rate_track import (
    analyze_vertex_merge_rate as analyze_vertex_merge_rate_track,
)
from .analyze_resolution import analyze_vertex_resolution
from .analyze_split_rate_track import (
    analyze_vertex_split_rate as analyze_vertex_split_rate_track,
)
from .analyze_track_pull import analyze_track_pull
from .analyze_vertex_pull import analyze_vertex_pull
from .plot_histogram import plot_event_histogram

DEFAULT_ANALYSES = (
    "efficiency_track",
    "efficiency_geo",
    "fake_rate_track",
    "fake_rate_geo",
    "merge_rate",
    "split_rate",
    "resolution",
    "plot_histogram",
    "track_pull",
    "vertex_pull",
)


def run_analysis(
    hdf5_file: str,
    output_dir: str,
    params: dict,
    analyses: Optional[Iterable[str]] = None,
):
    """
    Run reconstruction performance analyses based on an HDF5 file.

    Args:
        hdf5_file: Path to the HDF5 file (e.g. fit_results.h5)
        output_dir: Directory to save plots / CSVs
        params: configuration dictionary (from config.yaml)
        analyses: list of analyses to run
                  (None → run all default analyses)
    """

    if analyses is None:
        analyses = DEFAULT_ANALYSES

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 80}")
    print("RUNNING HDF5-BASED PERFORMANCE ANALYSIS")
    print(f"{'=' * 80}")
    print(f"HDF5 file: {hdf5_file}")
    print(f"Output dir: {output_dir}")
    print(f"Analyses: {list(analyses)}")
    print(f"Recall threshold: {params['recall_threshold']}")
    print(f"Purity threshold: {params['purity_threshold']}")
    print(f"{'=' * 80}\n")

    # ------------------------------------------------------------
    # Efficiency
    # ------------------------------------------------------------
    if "efficiency_track" in analyses:
        print("▶ Running vertex efficiency (track) analysis...")
        analyze_vertex_efficiency_track(
            hdf5_path=hdf5_file,
            output_dir=output_dir,
            recall_threshold=params["recall_threshold"],
            purity_threshold=params["purity_threshold"],
            plot_min_ntracks=4,
            plot_max_ntracks=200,
            plot_bin_width_ntracks=5,
            method_name="PVfit_Track_Based",
        )
    if "efficiency_geo" in analyses:
        print("▶ Running vertex efficiency (geo) analysis...")
        analyze_vertex_efficiency_geo(
            hdf5_path=hdf5_file,
            output_dir=output_dir,
            m_r=params["m_r"],
            nsigma=params["nsigma"],
            z_scale=params.get("z_scale", 1.5),
            t_scale=params.get("t_scale", 0.06),
            plot_min_ntracks=4,
            plot_max_ntracks=200,
            plot_bin_width_ntracks=5,
            method_name="PVfit_Geo_Based",
        )

    # ------------------------------------------------------------
    # Fake rate
    # ------------------------------------------------------------
    if "fake_rate_track" in analyses:
        print("▶ Running vertex fake rate (track) analysis...")
        analyze_vertex_fake_rate_track(
            hdf5_path=hdf5_file,
            output_dir=output_dir,
            recall_threshold=params["recall_threshold"],
            purity_threshold=params["purity_threshold"],
            plot_min_ntracks=4,
            plot_max_ntracks=200,
            plot_bin_width_ntracks=5,
            method_name="PVfit_Track_Based",
        )
    if "fake_rate_geo" in analyses:
        print("▶ Running vertex fake rate (geo) analysis...")
        analyze_vertex_fake_rate_geo(
            hdf5_path=hdf5_file,
            output_dir=output_dir,
            m_r=params["m_r"],
            z_scale=params.get("z_scale", 1.5),
            t_scale=params.get("t_scale", 0.06),
            plot_min_ntracks=4,
            plot_max_ntracks=200,
            plot_bin_width_ntracks=5,
            method_name="PVfit_Geo_Based",
        )

    # ------------------------------------------------------------
    # Merge rate
    # ------------------------------------------------------------
    if "merge_rate" in analyses:
        print("▶ Running vertex merge rate (track) analysis...")
        analyze_vertex_merge_rate_track(
            hdf5_path=hdf5_file,
            output_dir=output_dir,
            recall_threshold=params["recall_threshold"],
            plot_min_ntracks=4,
            plot_max_ntracks=200,
            plot_bin_width_ntracks=5,
            method_name="PVfit_Track_Based",
        )

    # ------------------------------------------------------------
    # Split rate
    # ------------------------------------------------------------
    if "split_rate" in analyses:
        print("▶ Running vertex split rate (track) analysis...")
        analyze_vertex_split_rate_track(
            hdf5_path=hdf5_file,
            output_dir=output_dir,
            purity_threshold=params["purity_threshold"],
            plot_min_ntracks=4,
            plot_max_ntracks=200,
            plot_bin_width_ntracks=5,
            method_name="PVfit_Track_Based",
        )

    # ------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------
    if "resolution" in analyses:
        print("▶ Running vertex resolution analysis...")
        analyze_vertex_resolution(
            hdf5_path=hdf5_file,
            output_dir=output_dir,
            recall_threshold=params["recall_threshold"],
            purity_threshold=params["purity_threshold"],
            plot_min_ntracks=4,
            plot_max_ntracks=200,
            plot_bin_width_ntracks=5,
            x_residual_range=params.get("x_residual_range", 0.1),
            y_residual_range=params.get("y_residual_range", 0.1),
            z_residual_range=params.get("z_residual_range", 0.7),
            t_residual_range=params.get("t_residual_range", 0.06),
            method_name="PVfit_Track_Based",
        )

    # ------------------------------------------------------------
    # Plot histogram
    # ------------------------------------------------------------
    if "plot_histogram" in analyses:
        print("▶ Plotting 2D histogram...")
        # Get event_id from params if specified, otherwise use None (first event)
        event_id = params.get("plot_event_id", 0)
        draw_fixed_ellipses = params.get("draw_fixed_ellipses", False)
        draw_fitted_ellipses = params.get("draw_fitted_ellipses", False)

        plot_event_histogram(
            hdf5_path=hdf5_file,
            event_id=event_id,
            output_dir=output_dir,
            config_path="config.yaml",
            draw_fixed_ellipses=draw_fixed_ellipses,
            draw_fitted_ellipses=draw_fitted_ellipses,
        )

    # ------------------------------------------------------------
    # Track pull analysis
    # ------------------------------------------------------------
    if "track_pull" in analyses:
        print("▶ Running input track pull analysis...")
        analyze_track_pull(
            hdf5_path=hdf5_file,
            output_dir=output_dir,
            max_chi2=params["max_chi2"],
        )

    # ------------------------------------------------------------
    # Vertex pull analysis
    # ------------------------------------------------------------
    if "vertex_pull" in analyses:
        print("▶ Running fitted vertex pull analysis...")
        analyze_vertex_pull(
            hdf5_path=hdf5_file,
            output_dir=output_dir,
            recall_threshold=params["recall_threshold"],
            purity_threshold=params["purity_threshold"],
        )

    print(f"\n{'=' * 80}")
    print("COMPLETED HDF5 PERFORMANCE ANALYSIS")
    print(f"{'=' * 80}")
    print(f"Results saved to: {output_dir}\n")
