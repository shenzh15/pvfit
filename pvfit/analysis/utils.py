"""Utility functions for vertex reconstruction performance analysis."""

import numpy as np
from scipy.stats import beta


def calculate_clopper_pearson_errors(k, n, rate):
    """Calculate Clopper-Pearson 1-sigma confidence interval errors.

    Args:
        k: Number of successes
        n: Total number of trials
        rate: Observed rate (k/n)

    Returns:
        tuple: (error_lower, error_upper)
    """
    if n == 0:
        return 0.0, 0.0

    lower_bound = beta.ppf(0.1585, k, n - k + 1) if k > 0 else 0.0
    upper_bound = beta.ppf(0.8415, k + 1, n - k) if k < n else 1.0

    if np.isnan(lower_bound) or np.isnan(upper_bound):
        return 0.0, 0.0

    return rate - lower_bound, upper_bound - rate


def match_vertices_elliptical_zt(
    rec_vertices: list[dict],
    true_vertices: list[dict],
    m_r: float,
    z_scale: float = 1.5,
    t_scale: float = 0.06,
) -> None:
    """
    Match reconstructed vertices to true vertices using an elliptical
    cut in (z, t) space.

    The matching follows the criterion:

        (dz / sigma_z)^2 + (dt / sigma_t)^2 <= 1

    where:
        dz = z_true - z_rec
        dt = t_true - t_rec
        sigma_z = z_scale * m_r / 2
        sigma_t = t_scale * m_r / 2

    Among all matched true vertices, the closest one is selected using
    a scaled distance metric.

    Parameters
    ----------
    rec_vertices : list of dict
        Each dict must contain:
            - 'rec_z'
            - 'rec_t'

        This function adds the following keys in-place:
            - 'matched_true_vertices'
            - 'closest_true_vertex'
            - 'n_matches'

    true_vertices : list of dict
        Each dict must contain:
            - 'vertex_id'
            - 'z'
            - 't'

    m_r : float
        Matching radius parameter (mm).

    z_scale : float, optional
        Scale factor for z ellipse diameter. The ellipse semi-axis in z is
        sigma_z = z_scale * m_r / 2 (default: 1.5).

    t_scale : float, optional
        Scale factor for t ellipse diameter. The ellipse semi-axis in t is
        sigma_t = t_scale * m_r / 2 (default: 0.06).
    """

    sigma_z = z_scale * m_r / 2.0
    sigma_t = t_scale * m_r / 2.0

    for rec in rec_vertices:
        rec_z = rec["rec_z"]
        rec_t = rec["rec_t"]

        matched = []
        closest: dict | None = None
        min_dist = float("inf")

        for tru in true_vertices:
            dz = tru["z"] - rec_z
            dt = tru["t"] - rec_t

            # Elliptical cut
            if (dz * dz) / (sigma_z * sigma_z) + (dt * dt) / (sigma_t * sigma_t) <= 1.0:
                matched.append(tru)

                # Distance metric for closest match
                dist = (dz / sigma_z) ** 2 + (dt / sigma_t) ** 2
                if dist < min_dist:
                    min_dist = dist
                    closest = tru

        rec["matched_true_vertices"] = matched
        rec["closest_true_vertex"] = closest
        rec["n_matches"] = len(matched)


def match_vertices_resolution_zt(
    rec_vertices: list[dict],
    true_vertices: list[dict],
    nsigma: float = 5.0,
) -> None:
    """
    Match reconstructed vertices to true vertices using a resolution-based
    elliptical cut in (z, t) space.

    The matching criterion is:

        (dz / (nsigma * z_err))^2 + (dt / (nsigma * t_err))^2 <= 1

    i.e. an ellipse with semi-axes:
        a_z = nsigma * z_err
        a_t = nsigma * t_err

    Parameters
    ----------
    rec_vertices : list of dict
        Each dict must contain:
            - 'rec_z'
            - 'rec_t'
            - 'z_err'
            - 't_err'

    true_vertices : list of dict
        Each dict must contain:
            - 'vertex_id'
            - 'z'
            - 't'

    nsigma : float
        Size of matching window in units of resolution (default: 5σ)
    """

    for rec in rec_vertices:
        rec_z = rec["rec_z"]
        rec_t = rec["rec_t"]

        z_err = rec.get("z_err", None)
        t_err = rec.get("t_err", None)

        # --- Safety: skip bad fits ---
        if z_err is None or t_err is None or z_err <= 0 or t_err <= 0:
            rec["matched_true_vertices"] = []
            rec["closest_true_vertex"] = None
            rec["n_matches"] = 0
            continue

        sigma_z = nsigma * z_err
        sigma_t = nsigma * t_err

        matched = []
        closest: dict | None = None
        min_dist = float("inf")

        for tru in true_vertices:
            dz = tru["z"] - rec_z
            dt = tru["t"] - rec_t

            # --- Elliptical cut (resolution-based) ---
            val = (dz / sigma_z) ** 2 + (dt / sigma_t) ** 2

            if val <= 1.0:
                matched.append(tru)

                # --- normalized distance ---
                if val < min_dist:
                    min_dist = val
                    closest = tru

        rec["matched_true_vertices"] = matched
        rec["closest_true_vertex"] = closest
        rec["n_matches"] = len(matched)


def find_best_match(
    rec_track_indices, true_vertex_info, recall_threshold, purity_threshold
):
    """Find the best matching true vertex for a reconstructed vertex.

    A match requires BOTH:
    - Recall = intersection / num_true_tracks >= recall_threshold
    - Purity = intersection / num_rec_tracks >= purity_threshold

    Args:
        rec_track_indices: set of track indices for the reconstructed vertex
        true_vertex_info: dict {vertex_id: {"track_indices": set, ...}}
        recall_threshold: minimum recall for a match
        purity_threshold: minimum purity for a match

    Returns:
        Best matching vertex ID or None if no match found
    """
    best_match_id = None
    best_recall = 0.0

    num_rec_tracks = len(rec_track_indices)
    if num_rec_tracks == 0:
        return None

    for vertex_id, vertex_data in true_vertex_info.items():
        true_track_indices = vertex_data["track_indices"]
        num_true_tracks = len(true_track_indices)

        if num_true_tracks == 0:
            continue

        # Calculate overlap metrics
        num_shared = len(rec_track_indices & true_track_indices)
        recall = num_shared / num_true_tracks
        purity = num_shared / num_rec_tracks

        # Both recall and purity must exceed their thresholds, and recall should be best so far
        if (
            recall >= recall_threshold
            and purity >= purity_threshold
            and recall > best_recall
        ):
            best_recall = recall
            best_match_id = vertex_id

    return best_match_id
