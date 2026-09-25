"""
Event-level processing for PVfit (track-based version).

This module contains:
  - process_event: physics logic for a single event
  - process_event_wrapper: multiprocessing-compatible wrapper
"""

import pandas as pd

# Seed finding & vertex fitting
from .seedings_finder import fill_histogram_for_seedings, find_seedings
from .truth_matching import build_true_vertex_info, select_tracks
from .vertex_fitter import fit_reconstructed_vertices, merge_close_vertices


def process_event_wrapper(args):
    """
    Wrapper for multiprocessing.

    Args:
        args (tuple): (event_number, event_track_data, event_mcpv_data, params)

    Returns:
        dict: output of process_event
    """
    event_number, event_track_data, event_mcpv_data, params = args
    return process_event(event_number, event_track_data, event_mcpv_data, params)


def process_event(
    event_number: int,
    event_track_data: pd.DataFrame,
    event_mcpv_data: pd.DataFrame,
    params: dict,
) -> dict:
    """
    Process a single event:
      - read true vertices from MCPV data
      - apply z/t cuts to tracks
      - run seeding
      - run vertex fitting
      - build true / reco vertex structures

    NOTE:
      - Truth membership only; no reconstructed-to-true vertex matching
      - No efficiency / fake-rate analysis
      - Pure reconstruction stage

    Args:
        event_number: event ID
        event_track_data: pandas DataFrame with per-track information from monitor_tree
        event_mcpv_data: pandas DataFrame with MC true vertices from MCPV_effective_tree
        params: configuration dictionary

    Returns:
        dict containing:
          - event_number
          - n_true
          - n_seedings
          - seeds: every (z, t) seed, including ones later dropped or merged
          - n_fitted
          - histogram
          - vertex_true_info
          - vertex_rec_info
    """

    # ------------------------------------------------------------
    # 0. Extract configuration parameters
    # ------------------------------------------------------------
    max_chi2 = params["max_chi2"]
    is_adaptive_tukey = params["is_adaptive_tukey"]
    drop_low_track_seeds = params["drop_low_track_seeds"]

    # ------------------------------------------------------------
    # 1. Select tracks and build truth membership on the same row order
    # ------------------------------------------------------------
    event_data_selected = select_tracks(event_track_data, params)
    vertex_true_info = build_true_vertex_info(
        event_data_selected,
        event_mcpv_data,
    )

    # ------------------------------------------------------------
    # 3. Seeding
    # ------------------------------------------------------------
    H = fill_histogram_for_seedings(event_data_selected, params)
    seedings = find_seedings(H, params)

    # ------------------------------------------------------------
    # 4. Initialize reconstructed vertices
    # ------------------------------------------------------------
    vertex_rec_info = []

    for i, (rec_z, rec_t) in enumerate(seedings):
        vertex_rec_info.append(
            {
                "rec_vertex_id": i,
                "rec_z": rec_z,
                "rec_t": rec_t,
                "rec_x": 0.0,
                "rec_y": 0.0,
                "track_index": [],
            }
        )

    # ------------------------------------------------------------
    # 5. Vertex fitting
    # ------------------------------------------------------------
    fit_reconstructed_vertices(
        event_data_selected,
        vertex_rec_info,
        max_chi2,
        is_adaptive_tukey,
        drop_low_track_seeds,
    )

    # ------------------------------------------------------------
    # 5b. Merge close vertices in (z, t) space
    # ------------------------------------------------------------
    if params["merge_vertices"]:
        nsigma_merge = params["nsigma_merge"]
        merge_close_vertices(
            event_data_selected,
            vertex_rec_info,
            max_chi2,
            is_adaptive_tukey,
            nsigma_merge,
        )

    # ------------------------------------------------------------
    # 7. Collect summary numbers
    # ------------------------------------------------------------
    n_fitted = len([v for v in vertex_rec_info if "Params_rec_x" in v])

    return {
        "event_number": event_number,
        "n_true": len(vertex_true_info),
        "n_seedings": len(seedings),
        "seeds": seedings,
        "n_fitted": n_fitted,
        "histogram": H,
        "vertex_true_info": vertex_true_info,
        "vertex_rec_info": vertex_rec_info,
        "event_data_selected": event_data_selected.reset_index(drop=True),
    }
