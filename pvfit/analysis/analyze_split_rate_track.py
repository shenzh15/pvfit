"""Split rate with overlap (track-based) matching."""

from .binned_rate import BinnedCounts, RateResult, iter_events


def _rec_vertices_with_purity_above(true_tracks, rec_track_sets, purity_threshold):
    """Ids of rec vertices whose purity (shared / rec tracks) reaches the threshold."""
    return {
        rec_id
        for rec_id, rec_tracks in enumerate(rec_track_sets)
        if rec_tracks
        and len(true_tracks & rec_tracks) / len(rec_tracks) >= purity_threshold
    }


def analyze_vertex_split_rate(hdf5_path, bins, purity_threshold):
    """Vertices involved in a split, as a fraction of true and of rec vertices.

    A true vertex is split when more than one reconstructed vertex has purity
    at or above the threshold with respect to it; those reconstructed vertices
    count as split too. Returned binned by true-vertex and by rec-vertex track
    count.
    """
    by_true = BinnedCounts(bins)
    by_rec = BinnedCounts(bins)
    for event in iter_events(hdf5_path):
        split_rec_ids = set()
        for true_tracks in event.true_tracks:
            pieces = (
                _rec_vertices_with_purity_above(
                    true_tracks, event.rec_tracks, purity_threshold
                )
                if true_tracks
                else set()
            )
            is_split = len(pieces) > 1
            by_true.add(len(true_tracks), is_split)
            if is_split:
                split_rec_ids |= pieces
        for rec_id, rec_tracks in enumerate(event.rec_tracks):
            if not rec_tracks:
                continue
            by_rec.add(len(rec_tracks), rec_id in split_rec_ids)

    parameters = {"purity_threshold": purity_threshold}
    label = f"Purity threshold: {purity_threshold}"
    return [
        RateResult("split_rate", "true", by_true, parameters, label),
        RateResult("split_rate", "rec", by_rec, parameters, label),
    ]
