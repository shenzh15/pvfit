"""Fake rate with distance-based (z, t) matching."""

from .binned_rate import BinnedCounts, RateResult, iter_events
from .utils import match_vertices_elliptical_zt


def analyze_vertex_fake_rate(hdf5_path, bins, m_r, z_scale, t_scale):
    """Fraction of reconstructed vertices with no true vertex inside the ellipse.

    Binned by the reconstructed vertex's track count.
    """
    counts = BinnedCounts(bins)
    for event in iter_events(hdf5_path):
        true_vertices = [
            {"vertex_id": i, "z": z, "t": t}
            for i, (z, t) in enumerate(zip(event.true_z, event.true_t, strict=True))
        ]
        rec_vertices = [
            {"rec_vertex_id": i, "rec_z": z, "rec_t": t, "num_tracks": len(tracks)}
            for i, (z, t, tracks) in enumerate(
                zip(event.rec_z, event.rec_t, event.rec_tracks, strict=True)
            )
            if len(tracks) > 0
        ]
        match_vertices_elliptical_zt(
            rec_vertices, true_vertices, m_r=m_r, z_scale=z_scale, t_scale=t_scale
        )
        for rec in rec_vertices:
            counts.add(rec["num_tracks"], rec["closest_true_vertex"] is None)

    return [
        RateResult(
            metric="fake_rate",
            binned_by="rec",
            counts=counts,
            parameters={"m_r": m_r, "z_scale": z_scale, "t_scale": t_scale},
            label=f"Geo matching (m_r={m_r})",
            matching="geo",
        )
    ]
