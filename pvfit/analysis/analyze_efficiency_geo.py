"""Reconstruction efficiency with distance-based (z, t) matching.

Two definitions are evaluated on the same events: a fixed ellipse
(``match_vertices_elliptical_zt``) and one scaled by each reconstructed
vertex's fit uncertainty (``match_vertices_resolution_zt``).
"""

from .binned_rate import BinnedCounts, RateResult, find_bin, iter_events
from .utils import match_vertices_elliptical_zt, match_vertices_resolution_zt


def _true_vertices(event):
    return [
        {"vertex_id": i, "z": z, "t": t, "num_tracks": len(tracks)}
        for i, (z, t, tracks) in enumerate(
            zip(event.true_z, event.true_t, event.true_tracks, strict=True)
        )
    ]


def _rec_vertices(event):
    return [
        {"rec_vertex_id": i, "rec_z": z, "rec_t": t, "z_err": z_err, "t_err": t_err}
        for i, (z, t, z_err, t_err) in enumerate(
            zip(event.rec_z, event.rec_t, event.rec_z_err, event.rec_t_err, strict=True)
        )
    ]


def _matched_true_ids(rec_vertices):
    """True vertices that are the closest match of some reconstructed vertex."""
    return {
        rec["closest_true_vertex"]["vertex_id"]
        for rec in rec_vertices
        if rec["closest_true_vertex"] is not None
    }


def analyze_vertex_efficiency(hdf5_path, bins, m_r, z_scale, t_scale, nsigma):
    """Fraction of true vertices that are some reconstructed vertex's closest match.

    Returns the fixed-ellipse ("geo") and resolution-scaled ("reso") results,
    both binned by the true vertex's reconstructed track count.
    """
    by_geo = BinnedCounts(bins)
    by_reso = BinnedCounts(bins)
    reconstructed_counts = dict.fromkeys(by_geo.bins, 0)
    for event in iter_events(hdf5_path):
        # All nonempty reconstructed vertices, including unmatched ones;
        # the spectrum uses their own multiplicity, independently of truth.
        for tracks in event.rec_tracks:
            if tracks:
                bin_ = find_bin(by_geo.bins, len(tracks))
                if bin_ is not None:
                    reconstructed_counts[bin_] += 1
        true_vertices = _true_vertices(event)
        matched_geo = matched_reso = set()
        if len(event.rec_tracks) > 0:
            rec_geo = _rec_vertices(event)
            match_vertices_elliptical_zt(
                rec_geo, true_vertices, m_r=m_r, z_scale=z_scale, t_scale=t_scale
            )
            matched_geo = _matched_true_ids(rec_geo)
            rec_reso = _rec_vertices(event)
            match_vertices_resolution_zt(rec_reso, true_vertices, nsigma=nsigma)
            matched_reso = _matched_true_ids(rec_reso)
        for vertex in true_vertices:
            by_geo.add(vertex["num_tracks"], vertex["vertex_id"] in matched_geo)
            by_reso.add(vertex["num_tracks"], vertex["vertex_id"] in matched_reso)

    return [
        RateResult(
            metric="efficiency",
            binned_by="true",
            counts=by_geo,
            parameters={"m_r": m_r, "z_scale": z_scale, "t_scale": t_scale},
            label=f"Geo matching (m_r={m_r})",
            matching="geo",
            reconstructed_counts=reconstructed_counts,
        ),
        RateResult(
            metric="efficiency",
            binned_by="true",
            counts=by_reso,
            parameters={"nsigma": nsigma},
            label=f"Resolution matching ({nsigma}σ)",
            matching="reso",
            reconstructed_counts=reconstructed_counts,
        ),
    ]
