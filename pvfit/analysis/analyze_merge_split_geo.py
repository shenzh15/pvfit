"""Merge and split rates with distance-based (z, t) matching.

The overlap-based definitions ask how many vertices share tracks; here the
question is how many vertices fall inside each other's matching ellipse. As
for the geometric efficiency and fake rate, a fixed ellipse ("geo") and one
scaled by the fit uncertainty ("reso") are both evaluated.
"""

from .binned_rate import BinnedCounts, RateResult, iter_events
from .utils import match_vertices_elliptical_zt, match_vertices_resolution_zt


def _matched_event(event, matcher):
    """True and reconstructed vertices of one event after running ``matcher``."""
    true_vertices = [
        {"vertex_id": i, "z": z, "t": t, "num_tracks": len(tracks)}
        for i, (z, t, tracks) in enumerate(
            zip(event.true_z, event.true_t, event.true_tracks, strict=True)
        )
    ]
    rec_vertices = [
        {
            "rec_vertex_id": i,
            "rec_z": z,
            "rec_t": t,
            "z_err": z_err,
            "t_err": t_err,
            "num_tracks": len(tracks),
        }
        for i, (z, t, z_err, t_err, tracks) in enumerate(
            zip(
                event.rec_z,
                event.rec_t,
                event.rec_z_err,
                event.rec_t_err,
                event.rec_tracks,
                strict=True,
            )
        )
        if len(tracks) > 0
    ]
    matcher(rec_vertices, true_vertices)
    return true_vertices, rec_vertices


def _definitions(m_r, z_scale, t_scale, nsigma):
    """(matching, parameters, label, matcher) for the two ellipse definitions."""
    return [
        (
            "geo",
            {"m_r": m_r, "z_scale": z_scale, "t_scale": t_scale},
            f"Geo matching (m_r={m_r})",
            lambda rec, true: match_vertices_elliptical_zt(
                rec, true, m_r=m_r, z_scale=z_scale, t_scale=t_scale
            ),
        ),
        (
            "reso",
            {"nsigma": nsigma},
            f"Resolution matching ({nsigma}σ)",
            lambda rec, true: match_vertices_resolution_zt(rec, true, nsigma=nsigma),
        ),
    ]


def analyze_vertex_merge_rate(hdf5_path, bins, m_r, z_scale, t_scale, nsigma):
    """Vertices involved in a merge: more than one true vertex inside a rec ellipse.

    A reconstructed vertex is merged when its ellipse contains more than one
    true vertex; those true vertices count as merged too. Returned binned by
    true-vertex and by rec-vertex track count, for both ellipse definitions.
    """
    results = []
    for matching, parameters, label, matcher in _definitions(
        m_r, z_scale, t_scale, nsigma
    ):
        by_true = BinnedCounts(bins)
        by_rec = BinnedCounts(bins)
        for event in iter_events(hdf5_path):
            true_vertices, rec_vertices = _matched_event(event, matcher)
            merged_true_ids = set()
            for rec in rec_vertices:
                is_merged = rec["n_matches"] > 1
                by_rec.add(rec["num_tracks"], is_merged)
                if is_merged:
                    merged_true_ids |= {
                        tv["vertex_id"] for tv in rec["matched_true_vertices"]
                    }
            for tv in true_vertices:
                by_true.add(tv["num_tracks"], tv["vertex_id"] in merged_true_ids)
        results += [
            RateResult("merge_rate", "true", by_true, parameters, label, matching),
            RateResult("merge_rate", "rec", by_rec, parameters, label, matching),
        ]
    return results


def analyze_vertex_split_rate(hdf5_path, bins, m_r, z_scale, t_scale, nsigma):
    """Vertices involved in a split: one true vertex inside more than one rec ellipse.

    A true vertex is split when it lies inside the ellipses of more than one
    reconstructed vertex; those reconstructed vertices count as split too.
    Returned binned by true-vertex and by rec-vertex track count, for both
    ellipse definitions.
    """
    results = []
    for matching, parameters, label, matcher in _definitions(
        m_r, z_scale, t_scale, nsigma
    ):
        by_true = BinnedCounts(bins)
        by_rec = BinnedCounts(bins)
        for event in iter_events(hdf5_path):
            true_vertices, rec_vertices = _matched_event(event, matcher)
            containing_recs = {tv["vertex_id"]: 0 for tv in true_vertices}
            for rec in rec_vertices:
                for tv in rec["matched_true_vertices"]:
                    containing_recs[tv["vertex_id"]] += 1
            split_true_ids = {
                vertex_id for vertex_id, n in containing_recs.items() if n > 1
            }
            for tv in true_vertices:
                by_true.add(tv["num_tracks"], tv["vertex_id"] in split_true_ids)
            for rec in rec_vertices:
                is_split = any(
                    tv["vertex_id"] in split_true_ids
                    for tv in rec["matched_true_vertices"]
                )
                by_rec.add(rec["num_tracks"], is_split)
        results += [
            RateResult("split_rate", "true", by_true, parameters, label, matching),
            RateResult("split_rate", "rec", by_rec, parameters, label, matching),
        ]
    return results
