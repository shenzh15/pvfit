"""Fake rate with distance-based (z, t) matching.

Like the geometric efficiency, two definitions are evaluated: a fixed
ellipse (``match_vertices_elliptical_zt``) and one scaled by each
reconstructed vertex's fit uncertainty (``match_vertices_resolution_zt``).
"""

from .binned_rate import BinnedCounts, RateResult, iter_events
from .utils import match_vertices_elliptical_zt, match_vertices_resolution_zt


def _rec_vertices(event):
    return [
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


def analyze_vertex_fake_rate(hdf5_path, bins, m_r, z_scale, t_scale, nsigma):
    """Fraction of reconstructed vertices with no true vertex inside the ellipse.

    Returns the fixed-ellipse ("geo") and resolution-scaled ("reso") results,
    both binned by the reconstructed vertex's track count.
    """
    by_geo = BinnedCounts(bins)
    by_reso = BinnedCounts(bins)
    for event in iter_events(hdf5_path):
        true_vertices = [
            {"vertex_id": i, "z": z, "t": t}
            for i, (z, t) in enumerate(zip(event.true_z, event.true_t, strict=True))
        ]
        rec_geo = _rec_vertices(event)
        match_vertices_elliptical_zt(
            rec_geo, true_vertices, m_r=m_r, z_scale=z_scale, t_scale=t_scale
        )
        for rec in rec_geo:
            by_geo.add(rec["num_tracks"], rec["closest_true_vertex"] is None)

        rec_reso = _rec_vertices(event)
        match_vertices_resolution_zt(rec_reso, true_vertices, nsigma=nsigma)
        for rec in rec_reso:
            by_reso.add(rec["num_tracks"], rec["closest_true_vertex"] is None)

    return [
        RateResult(
            metric="fake_rate",
            binned_by="rec",
            counts=by_geo,
            parameters={"m_r": m_r, "z_scale": z_scale, "t_scale": t_scale},
            label=f"Geo matching (m_r={m_r})",
            matching="geo",
        ),
        RateResult(
            metric="fake_rate",
            binned_by="rec",
            counts=by_reso,
            parameters={"nsigma": nsigma},
            label=f"Resolution matching ({nsigma}σ)",
            matching="reso",
        ),
    ]
