"""
vertex_fitter.py - Vertex fitting functions for PVfit
"""

import numpy as np
import pandas as pd

from .utils import getTukeyWeight


def extract_track_info_for_chi2(event_data, track_index):
    """Extract track parameters and covariance elements for chi2 calculation."""
    return {
        "backward": event_data["backward"].iloc[track_index],
        "tx": event_data["beamPOCA_tx"].iloc[track_index],
        "ty": event_data["beamPOCA_ty"].iloc[track_index],
        "z": event_data["beamPOCA_z"].iloc[track_index],
        "x": event_data["beamPOCA_x"].iloc[track_index],
        "y": event_data["beamPOCA_y"].iloc[track_index],
        "t": event_data["beamPOCA_t"].iloc[track_index],
        "c00": event_data["c00"].iloc[track_index],
        "c22": event_data["c22"].iloc[track_index],
        "c20": event_data["c20"].iloc[track_index],
        "c11": event_data["c11"].iloc[track_index],
        "c33": event_data["c33"].iloc[track_index],
        "c31": event_data["c31"].iloc[track_index],
        "c55": event_data["c55"].iloc[track_index],
    }


def calculate_track_vertex_chi2(track_info, vertex_info):
    """
    Calculate chi-squared between track and vertex positions.

    Args:
        track_info: Dictionary with track parameters and covariance elements
        vertex_info: Dictionary with vertex position (rec_z, rec_x, rec_y, rec_t)

    Returns:
        dict: Chi2 value, covariances, and residuals
    """
    # Extract data from dictionaries
    backward = track_info["backward"]
    tx = track_info["tx"]
    ty = track_info["ty"]
    track_z = track_info["z"]
    track_x = track_info["x"]
    track_y = track_info["y"]
    track_t = track_info["t"]
    c00 = track_info["c00"]  # x_x covariance
    # c22 = track_info['c22'] # tx_tx covariance
    # c20 = track_info['c20'] # tx_x covariance
    c11 = track_info["c11"]  # y_y covariance
    # c33 = track_info['c33'] # ty_ty covariance
    # c31 = track_info['c31'] # ty_y covariance
    c55 = track_info["c55"]  # t_t covariance

    vertex_rec_z = vertex_info["rec_z"]
    vertex_rec_x = vertex_info["rec_x"]
    vertex_rec_y = vertex_info["rec_y"]
    vertex_rec_t = vertex_info["rec_t"]

    # Calculate the sign and time slope
    sgn = -1 if backward else 1
    tb_tmp = sgn * np.sqrt(1 + tx**2 + ty**2) / 299.792458  # mm/ns

    # Calculate delta z and residuals
    delta_z = vertex_rec_z - track_z
    res_x = track_x - vertex_rec_x + delta_z * tx
    res_y = track_y - vertex_rec_y + delta_z * ty
    res_t = track_t - vertex_rec_t + delta_z * tb_tmp

    # Calculate track covariances
    # Switch 1: extrapolate the error to the vertex
    # Track_cov_xx = c00 + delta_z**2 * c22 + 2 * delta_z * c20
    # Track_cov_yy = c11 + delta_z**2 * c33 + 2 * delta_z * c31
    # Track_cov_tt = c55
    # Switch 2: not extrapolate the error to the vertex
    Track_cov_xx = c00
    Track_cov_yy = c11
    Track_cov_tt = c55

    # Calculate chi-squared
    trackvtxchi2 = (
        res_x**2 / Track_cov_xx + res_y**2 / Track_cov_yy + res_t**2 / Track_cov_tt
    )

    return {
        "chi2": trackvtxchi2,
        "cov_xx": Track_cov_xx,
        "cov_yy": Track_cov_yy,
        "cov_tt": Track_cov_tt,
        "tb_tmp": tb_tmp,
        "res_x": res_x,
        "res_y": res_y,
        "res_t": res_t,
    }


def _assign_track_to_closest_vertex(
    event_data_selected, i, vertex_rec_info, max_chi2, cache
):
    """Find closest vertex for track i, assign it, and update cache.

    Returns True if the track was assigned, False otherwise.
    """
    c00 = event_data_selected["c00"]
    c11 = event_data_selected["c11"]
    c55 = event_data_selected["c55"]
    tx = event_data_selected["beamPOCA_tx"]
    ty = event_data_selected["beamPOCA_ty"]

    if c00.iloc[i] <= 0 or c11.iloc[i] <= 0 or c55.iloc[i] <= 0:
        print(
            f"Warning: Non-positive covariance at index {i}: c00={c00.iloc[i]}, c11={c11.iloc[i]}, c55={c55.iloc[i]}"
        )
        return False

    vertex_index_closest = None
    trackvtxchi2_min = max_chi2

    for vertex in vertex_rec_info:
        track_info = extract_track_info_for_chi2(event_data_selected, i)
        result = calculate_track_vertex_chi2(track_info, vertex)
        if result["chi2"] < trackvtxchi2_min:
            trackvtxchi2_min = result["chi2"]
            vertex_index_closest = vertex["rec_vertex_id"]

    if vertex_index_closest is None:
        return False

    for vertex in vertex_rec_info:
        if vertex["rec_vertex_id"] == vertex_index_closest:
            vertex["track_index"].append(i)
            track_info = extract_track_info_for_chi2(event_data_selected, i)
            result = calculate_track_vertex_chi2(track_info, vertex)
            cache["trackvtxchi2_cache"].iloc[i] = result["chi2"]
            tb_tmp = result["tb_tmp"]
            cache["HWH00_cache"].iloc[i] = 1 / c00.iloc[i]
            cache["HWH11_cache"].iloc[i] = 1 / c11.iloc[i]
            cache["HWH22_cache"].iloc[i] = (
                tx.iloc[i] ** 2 / c00.iloc[i]
                + ty.iloc[i] ** 2 / c11.iloc[i]
                + tb_tmp**2 / c55.iloc[i]
            )
            cache["HWH02_cache"].iloc[i] = -tx.iloc[i] / c00.iloc[i]
            cache["HWH12_cache"].iloc[i] = -ty.iloc[i] / c11.iloc[i]
            cache["HWH23_cache"].iloc[i] = -tb_tmp / c55.iloc[i]
            cache["HWH33_cache"].iloc[i] = 1 / c55.iloc[i]
            break

    return True


def _build_cache_for_vertex(event_data_selected, vertex):
    """Build HWH cache for a single vertex using its existing track_index.

    Unlike assign_tracks_to_vertices, this does NOT re-assign tracks.
    It only computes the cached matrix elements needed by fit_vertices.

    Args:
        event_data_selected: DataFrame with filtered track data
        vertex: vertex dict with 'track_index' already set

    Returns:
        dict: Cache containing chi2 values and HWH matrices
    """
    z = event_data_selected["beamPOCA_z"]
    c00 = event_data_selected["c00"]
    c11 = event_data_selected["c11"]
    c55 = event_data_selected["c55"]
    tx = event_data_selected["beamPOCA_tx"]
    ty = event_data_selected["beamPOCA_ty"]

    cache = {
        "trackvtxchi2_cache": pd.Series(np.full(len(z), np.inf), index=z.index),
        "HWH00_cache": pd.Series(np.zeros(len(z)), index=z.index),
        "HWH11_cache": pd.Series(np.zeros(len(z)), index=z.index),
        "HWH22_cache": pd.Series(np.zeros(len(z)), index=z.index),
        "HWH02_cache": pd.Series(np.zeros(len(z)), index=z.index),
        "HWH12_cache": pd.Series(np.zeros(len(z)), index=z.index),
        "HWH23_cache": pd.Series(np.zeros(len(z)), index=z.index),
        "HWH33_cache": pd.Series(np.zeros(len(z)), index=z.index),
    }

    for i in vertex["track_index"]:
        if c00.iloc[i] <= 0 or c11.iloc[i] <= 0 or c55.iloc[i] <= 0:
            continue

        track_info = extract_track_info_for_chi2(event_data_selected, i)
        result = calculate_track_vertex_chi2(track_info, vertex)
        tb_tmp = result["tb_tmp"]

        cache["trackvtxchi2_cache"].iloc[i] = result["chi2"]
        cache["HWH00_cache"].iloc[i] = 1 / c00.iloc[i]
        cache["HWH11_cache"].iloc[i] = 1 / c11.iloc[i]
        cache["HWH22_cache"].iloc[i] = (
            tx.iloc[i] ** 2 / c00.iloc[i]
            + ty.iloc[i] ** 2 / c11.iloc[i]
            + tb_tmp**2 / c55.iloc[i]
        )
        cache["HWH02_cache"].iloc[i] = -tx.iloc[i] / c00.iloc[i]
        cache["HWH12_cache"].iloc[i] = -ty.iloc[i] / c11.iloc[i]
        cache["HWH23_cache"].iloc[i] = -tb_tmp / c55.iloc[i]
        cache["HWH33_cache"].iloc[i] = 1 / c55.iloc[i]

    return cache


def assign_tracks_to_vertices(
    event_data_selected, vertex_rec_info, max_chi2=10, drop_low_track_seeds=True
):
    """
    Assign tracks to closest vertices using chi2 distance.

    Args:
        event_data_selected: DataFrame with filtered track data
        vertex_rec_info: List of reconstructed vertex dictionaries
        max_chi2: Maximum chi2 threshold for track-vertex matching
        drop_low_track_seeds: If True, remove seeds with < 4 tracks and reassign
                              their tracks to the nearest remaining vertex

    Returns:
        dict: Cache containing chi2 values and HWH matrices for fitting
    """
    z = event_data_selected["beamPOCA_z"]

    # Initialize cache
    cache = {
        "trackvtxchi2_cache": pd.Series(np.full(len(z), np.inf), index=z.index),
        "HWH00_cache": pd.Series(np.zeros(len(z)), index=z.index),
        "HWH11_cache": pd.Series(np.zeros(len(z)), index=z.index),
        "HWH22_cache": pd.Series(np.zeros(len(z)), index=z.index),
        "HWH02_cache": pd.Series(np.zeros(len(z)), index=z.index),
        "HWH12_cache": pd.Series(np.zeros(len(z)), index=z.index),
        "HWH23_cache": pd.Series(np.zeros(len(z)), index=z.index),
        "HWH33_cache": pd.Series(np.zeros(len(z)), index=z.index),
    }

    # Initial track-to-vertex assignment
    for i in range(len(z)):
        _assign_track_to_closest_vertex(
            event_data_selected, i, vertex_rec_info, max_chi2, cache
        )

    # Remove vertices with < 4 tracks, reassign orphaned tracks
    if drop_low_track_seeds:
        orphaned_tracks = []
        vertices_to_remove = [v for v in vertex_rec_info if len(v["track_index"]) < 4]
        for vertex in vertices_to_remove:
            orphaned_tracks.extend(vertex["track_index"])
            vertex_rec_info.remove(vertex)

        # Clear cache and reassign orphaned tracks to remaining vertices
        for i in orphaned_tracks:
            cache["trackvtxchi2_cache"].iloc[i] = np.inf
            for key in (
                "HWH00_cache",
                "HWH11_cache",
                "HWH22_cache",
                "HWH02_cache",
                "HWH12_cache",
                "HWH23_cache",
                "HWH33_cache",
            ):
                cache[key].iloc[i] = 0

        if vertex_rec_info:
            for i in orphaned_tracks:
                _assign_track_to_closest_vertex(
                    event_data_selected, i, vertex_rec_info, max_chi2, cache
                )

    return cache


def fit_vertices(
    event_data_selected,
    vertex_rec_info,
    cache_data,
    max_chi2=10,
    is_adaptive_tukey=True,
):
    """
    Fit vertex parameters using assigned tracks with Tukey reweighting.

    Args:
        event_data_selected: DataFrame with filtered track data
        vertex_rec_info: List of reconstructed vertex dictionaries
        cache_data: Dictionary with cached chi2 values and HWH matrices
        max_chi2: Maximum chi2 threshold for Tukey weight
        is_adaptive_tukey: If True, recompute chi2 each iteration for Tukey weight;
                        if False, use cached chi2 from initial track assignment

    Returns:
        None: Modifies vertex_rec_info in-place with fitted parameters
    """
    # Extract cached data from dictionary
    trackvtxchi2_cache = cache_data["trackvtxchi2_cache"]
    HWH00_cache = cache_data["HWH00_cache"]
    HWH11_cache = cache_data["HWH11_cache"]
    HWH22_cache = cache_data["HWH22_cache"]
    HWH02_cache = cache_data["HWH02_cache"]
    HWH12_cache = cache_data["HWH12_cache"]
    HWH23_cache = cache_data["HWH23_cache"]
    HWH33_cache = cache_data["HWH33_cache"]

    # Get event data columns (needed for fit_vertices function)
    c00 = event_data_selected["c00"]
    c11 = event_data_selected["c11"]
    c55 = event_data_selected["c55"]
    tx = event_data_selected["beamPOCA_tx"]
    ty = event_data_selected["beamPOCA_ty"]

    for vertex in vertex_rec_info:
        Params_rec_x = vertex["rec_x"]  # Initial values
        Params_rec_y = vertex["rec_y"]  # Initial values
        Params_rec_z = vertex["rec_z"]  # Initial values
        Params_rec_t = vertex["rec_t"]  # Initial values

        # At least 4 tracks needed
        if len(vertex["track_index"]) < 4:
            continue

        maxIter = 20
        minIter = 1
        Iter = 0
        maxDeltaZ = 0.0005
        converged = False

        # Iterative fitting
        while Iter < minIter or (Iter < maxIter and (not converged)):
            sum_weight = 0
            halfD2Chi2DX2_00 = 0.0
            halfD2Chi2DX2_02 = 0.0
            halfD2Chi2DX2_11 = 0.0
            halfD2Chi2DX2_12 = 0.0
            halfD2Chi2DX2_22 = 0.0
            halfD2Chi2DX2_23 = 0.0
            halfD2Chi2DX2_33 = 0.0
            halfDChi2DX_0 = 0
            halfDChi2DX_1 = 0
            halfDChi2DX_2 = 0
            halfDChi2DX_3 = 0
            totol_chi2 = 0

            for i in vertex[
                "track_index"
            ]:  # Loop over all tracks assigned to the vertex
                track_data = extract_track_info_for_chi2(event_data_selected, i)
                vertex_data = {
                    "rec_z": Params_rec_z,
                    "rec_x": Params_rec_x,
                    "rec_y": Params_rec_y,
                    "rec_t": Params_rec_t,
                }
                result = calculate_track_vertex_chi2(track_data, vertex_data)

                if is_adaptive_tukey:
                    weight = getTukeyWeight(result["chi2"], max_chi2=max_chi2)
                else:
                    weight = getTukeyWeight(
                        trackvtxchi2_cache.iloc[i], max_chi2=max_chi2
                    )
                sum_weight += weight
                totol_chi2 += weight * result["chi2"]

                halfD2Chi2DX2_00 += weight * HWH00_cache.iloc[i]
                halfD2Chi2DX2_02 += weight * HWH02_cache.iloc[i]
                halfD2Chi2DX2_11 += weight * HWH11_cache.iloc[i]
                halfD2Chi2DX2_12 += weight * HWH12_cache.iloc[i]
                halfD2Chi2DX2_22 += weight * HWH22_cache.iloc[i]
                halfD2Chi2DX2_23 += weight * HWH23_cache.iloc[i]
                halfD2Chi2DX2_33 += weight * HWH33_cache.iloc[i]

                # Calculate gradient
                halfDChi2DX_0 += weight * (-result["res_x"] / c00.iloc[i])
                halfDChi2DX_1 += weight * (-result["res_y"] / c11.iloc[i])
                halfDChi2DX_2 += weight * (
                    result["res_y"] * ty.iloc[i] / c11.iloc[i]
                    + result["res_x"] * tx.iloc[i] / c00.iloc[i]
                    + result["res_t"] * result["tb_tmp"] / c55.iloc[i]
                )
                halfDChi2DX_3 += weight * (-result["res_t"] / c55.iloc[i])

            # Matrix inversion
            a00 = halfD2Chi2DX2_00
            a11 = halfD2Chi2DX2_11
            a22 = halfD2Chi2DX2_22
            a33 = halfD2Chi2DX2_33
            a02 = halfD2Chi2DX2_02
            a12 = halfD2Chi2DX2_12
            a23 = halfD2Chi2DX2_23

            # Gradient vector
            grad_x = halfDChi2DX_0
            grad_y = halfDChi2DX_1
            grad_z = halfDChi2DX_2
            grad_t = halfDChi2DX_3

            # Calculate determinant
            det = (
                a00 * a11 * a22 * a33
                - a00 * a11 * a23**2
                - a00 * a12**2 * a33
                - a02**2 * a11 * a33
            )

            if abs(det) < 1e-12:
                break

            # Inverse matrix elements
            cov_00 = (a11 * a22 * a33 - a11 * a23**2 - a12**2 * a33) / det
            cov_10 = a02 * a12 * a33 / det
            cov_11 = (a00 * a22 * a33 - a00 * a23**2 - a02**2 * a33) / det
            cov_20 = -a02 * a11 * a33 / det
            cov_21 = -a00 * a12 * a33 / det
            cov_22 = a00 * a11 * a33 / det
            cov_30 = a02 * a11 * a23 / det
            cov_31 = a00 * a12 * a23 / det
            cov_32 = -a00 * a11 * a23 / det
            cov_33 = (a00 * a11 * a22 - a00 * a12**2 - a02**2 * a11) / det

            delta_x = (
                cov_00 * grad_x + cov_10 * grad_y + cov_20 * grad_z + cov_30 * grad_t
            )
            delta_y = (
                cov_10 * grad_x + cov_11 * grad_y + cov_21 * grad_z + cov_31 * grad_t
            )
            delta_z = (
                cov_20 * grad_x + cov_21 * grad_y + cov_22 * grad_z + cov_32 * grad_t
            )
            delta_t = (
                cov_30 * grad_x + cov_31 * grad_y + cov_32 * grad_z + cov_33 * grad_t
            )

            # Update parameters
            Params_rec_x -= delta_x
            Params_rec_y -= delta_y
            Params_rec_z -= delta_z
            Params_rec_t -= delta_t

            converged = abs(delta_z) < maxDeltaZ
            Iter += 1

        # Save fitted parameters
        vertex["Params_rec_x"] = Params_rec_x
        vertex["Params_rec_y"] = Params_rec_y
        vertex["Params_rec_z"] = Params_rec_z
        vertex["Params_rec_t"] = Params_rec_t
        vertex["cov_00"] = cov_00
        vertex["cov_11"] = cov_11
        vertex["cov_22"] = cov_22
        vertex["cov_33"] = cov_33


def fit_reconstructed_vertices(
    event_data_selected,
    vertex_rec_info,
    max_chi2=10,
    is_adaptive_tukey=True,
    drop_low_track_seeds=True,
):
    """
    Fit all reconstructed vertices in the event.

    Args:
        event_data_selected: DataFrame with filtered track data
        vertex_rec_info: List of reconstructed vertex dictionaries
        max_chi2: Maximum chi2 threshold for track-vertex matching and Tukey weight
        is_adaptive_tukey: If True, recompute chi2 each iteration for Tukey weight;
                        if False, use cached chi2 from initial track assignment
        drop_low_track_seeds: If True, remove seeds with < 4 tracks and reassign

    Returns:
        None: Modifies vertex_rec_info in-place with fitted parameters
    """

    # Step 1: Assign tracks to vertices, we cache some data here for fitting
    cache_data = assign_tracks_to_vertices(
        event_data_selected, vertex_rec_info, max_chi2, drop_low_track_seeds
    )

    # Step 2: Fit vertices
    fit_vertices(
        event_data_selected, vertex_rec_info, cache_data, max_chi2, is_adaptive_tukey
    )


def merge_close_vertices(
    event_data_selected,
    vertex_rec_info,
    max_chi2=10,
    is_adaptive_tukey=True,
    nsigma_merge=5,
):
    """
    Merge reconstructed vertices that are close in (z, t) space after fitting.

    For each pair of fitted vertices, compute a chi2 distance:
        chi2 = (dz / sigma_z)^2 + (dt / sigma_t)^2
    where sigma_z = sqrt(cov_zz_i + cov_zz_j), sigma_t = sqrt(cov_tt_i + cov_tt_j).
    Pairs with chi2 < nsigma_merge^2 are merged: their track lists are combined,
    and the merged vertex is re-fitted.

    Single pass: find all pairs within threshold, merge each pair directly.

    Args:
        event_data_selected: DataFrame with filtered track data
        vertex_rec_info: List of reconstructed vertex dictionaries (modified in-place)
        max_chi2: Maximum chi2 threshold for Tukey weight during re-fit
        is_adaptive_tukey: Tukey mode for re-fit
        nsigma_merge: Sigma threshold for merging (chi2 < nsigma_merge^2)

    Returns:
        None: Modifies vertex_rec_info in-place
    """
    # Only consider fitted vertices (those with covariance information)
    fitted = [
        v
        for v in vertex_rec_info
        if "Params_rec_z" in v and "cov_22" in v and "cov_33" in v
    ]

    if len(fitted) < 2:
        return

    chi2_threshold = nsigma_merge**2

    # ------------------------------------------------------------------
    # Find pairs to merge
    # ------------------------------------------------------------------
    pairs_to_merge = []
    for idx_a in range(len(fitted)):
        for idx_b in range(idx_a + 1, len(fitted)):
            vi = fitted[idx_a]
            vj = fitted[idx_b]

            dz = vi["Params_rec_z"] - vj["Params_rec_z"]
            dt = vi["Params_rec_t"] - vj["Params_rec_t"]
            sigma_z_sq = vi["cov_22"] + vj["cov_22"]
            sigma_t_sq = vi["cov_33"] + vj["cov_33"]

            if sigma_z_sq <= 0 or sigma_t_sq <= 0:
                continue

            chi2 = dz**2 / sigma_z_sq + dt**2 / sigma_t_sq

            if chi2 < chi2_threshold:
                pairs_to_merge.append((vi, vj, chi2))

    if not pairs_to_merge:
        return

    # ------------------------------------------------------------------
    # Merge each pair: combine tracks, keep the one with more tracks as seed
    # ------------------------------------------------------------------
    already_merged = set()  # rec_vertex_id's that have been consumed

    for vi, vj, chi2 in sorted(pairs_to_merge, key=lambda x: x[2]):
        if (
            vi["rec_vertex_id"] in already_merged
            or vj["rec_vertex_id"] in already_merged
        ):
            continue

        # Determine which vertex has more tracks → keep as seed
        if len(vi["track_index"]) >= len(vj["track_index"]):
            keep, drop = vi, vj
        else:
            keep, drop = vj, vi

        # Combine tracks (deduplicate)
        combined = list(set(keep["track_index"] + drop["track_index"]))

        # Build merged vertex using the kept vertex's fitted position as seed
        merged = {
            "rec_vertex_id": keep["rec_vertex_id"],
            "rec_z": keep["Params_rec_z"],
            "rec_t": keep["Params_rec_t"],
            "rec_x": keep["Params_rec_x"],
            "rec_y": keep["Params_rec_y"],
            "track_index": combined,
        }

        # Replace keep with merged, remove drop
        idx_keep = vertex_rec_info.index(keep)
        vertex_rec_info[idx_keep] = merged
        vertex_rec_info.remove(drop)

        already_merged.add(vi["rec_vertex_id"])
        already_merged.add(vj["rec_vertex_id"])

        # Re-fit only this merged vertex
        cache_data = _build_cache_for_vertex(event_data_selected, merged)
        fit_vertices(
            event_data_selected, [merged], cache_data, max_chi2, is_adaptive_tukey
        )
