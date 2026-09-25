"""Map analysis names to modules; the analysis stage of ``pvfit analyze``."""

import os

from ..config_loader import AnalysisConfig
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


def run_analysis(hdf5_file: str, output_dir: str, settings: AnalysisConfig):
    """Run the analyses ``settings.analyses`` selects on an HDF5 file.

    Args:
        hdf5_file: Path to the HDF5 file (e.g. fit_results.h5)
        output_dir: Directory to save plots / CSVs
        settings: analysis settings, independent of any reconstruction config
    """
    analyses = settings.analyses
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 80}")
    print("RUNNING HDF5-BASED PERFORMANCE ANALYSIS")
    print(f"{'=' * 80}")
    print(f"HDF5 file: {hdf5_file}")
    print(f"Output dir: {output_dir}")
    print(f"Analyses: {list(analyses)}")
    print(f"Recall threshold: {settings.recall_threshold}")
    print(f"Purity threshold: {settings.purity_threshold}")
    print(f"{'=' * 80}\n")

    binning = dict(
        plot_min_ntracks=settings.plot_min_ntracks,
        plot_max_ntracks=settings.plot_max_ntracks,
        plot_bin_width_ntracks=settings.plot_bin_width_ntracks,
    )
    track_based = f"{settings.method_name}_Track_Based"
    geo_based = f"{settings.method_name}_Geo_Based"

    if "efficiency_track" in analyses:
        print("▶ Running vertex efficiency (track) analysis...")
        analyze_vertex_efficiency_track(
            hdf5_path=hdf5_file,
            output_dir=output_dir,
            recall_threshold=settings.recall_threshold,
            purity_threshold=settings.purity_threshold,
            **binning,
            method_name=track_based,
        )
    if "efficiency_geo" in analyses:
        print("▶ Running vertex efficiency (geo) analysis...")
        analyze_vertex_efficiency_geo(
            hdf5_path=hdf5_file,
            output_dir=output_dir,
            m_r=settings.m_r,
            nsigma=settings.nsigma,
            z_scale=settings.z_scale,
            t_scale=settings.t_scale,
            **binning,
            method_name=geo_based,
        )

    if "fake_rate_track" in analyses:
        print("▶ Running vertex fake rate (track) analysis...")
        analyze_vertex_fake_rate_track(
            hdf5_path=hdf5_file,
            output_dir=output_dir,
            recall_threshold=settings.recall_threshold,
            purity_threshold=settings.purity_threshold,
            **binning,
            method_name=track_based,
        )
    if "fake_rate_geo" in analyses:
        print("▶ Running vertex fake rate (geo) analysis...")
        analyze_vertex_fake_rate_geo(
            hdf5_path=hdf5_file,
            output_dir=output_dir,
            m_r=settings.m_r,
            z_scale=settings.z_scale,
            t_scale=settings.t_scale,
            **binning,
            method_name=geo_based,
        )

    if "merge_rate" in analyses:
        print("▶ Running vertex merge rate (track) analysis...")
        analyze_vertex_merge_rate_track(
            hdf5_path=hdf5_file,
            output_dir=output_dir,
            recall_threshold=settings.recall_threshold,
            **binning,
            method_name=track_based,
        )

    if "split_rate" in analyses:
        print("▶ Running vertex split rate (track) analysis...")
        analyze_vertex_split_rate_track(
            hdf5_path=hdf5_file,
            output_dir=output_dir,
            purity_threshold=settings.purity_threshold,
            **binning,
            method_name=track_based,
        )

    if "resolution" in analyses:
        print("▶ Running vertex resolution analysis...")
        analyze_vertex_resolution(
            hdf5_path=hdf5_file,
            output_dir=output_dir,
            recall_threshold=settings.recall_threshold,
            purity_threshold=settings.purity_threshold,
            **binning,
            x_residual_range=settings.x_residual_range,
            y_residual_range=settings.y_residual_range,
            z_residual_range=settings.z_residual_range,
            t_residual_range=settings.t_residual_range,
            method_name=track_based,
        )

    if "plot_histogram" in analyses:
        print("▶ Plotting 2D histogram...")
        plot_event_histogram(
            hdf5_path=hdf5_file,
            output_dir=output_dir,
            settings=settings,
            event_id=settings.plot_event_id,
        )

    if "track_pull" in analyses:
        print("▶ Running input track pull analysis...")
        analyze_track_pull(hdf5_path=hdf5_file, output_dir=output_dir)

    if "vertex_pull" in analyses:
        print("▶ Running fitted vertex pull analysis...")
        analyze_vertex_pull(
            hdf5_path=hdf5_file,
            output_dir=output_dir,
            recall_threshold=settings.recall_threshold,
            purity_threshold=settings.purity_threshold,
        )

    print(f"\n{'=' * 80}")
    print("COMPLETED HDF5 PERFORMANCE ANALYSIS")
    print(f"{'=' * 80}")
    print(f"Results saved to: {output_dir}\n")
