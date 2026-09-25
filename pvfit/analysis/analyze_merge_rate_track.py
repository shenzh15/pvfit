"""Merge rate with overlap (track-based) matching."""

from .binned_rate import BinnedCounts, RateResult, iter_events


def _true_vertices_with_recall_above(rec_tracks, true_track_sets, recall_threshold):
    """Ids of true vertices whose recall (shared / true tracks) reaches the threshold."""
    return {
        true_id
        for true_id, true_tracks in enumerate(true_track_sets)
        if true_tracks
        and len(rec_tracks & true_tracks) / len(true_tracks) >= recall_threshold
    }


def analyze_vertex_merge_rate(hdf5_path, bins, recall_threshold):
    """Vertices involved in a merge, as a fraction of true and of rec vertices.

    A reconstructed vertex is merged when more than one true vertex has recall
    at or above the threshold with respect to it; those true vertices count as
    merged too. Returned binned by true-vertex and by rec-vertex track count.
    """
    by_true = BinnedCounts(bins)
    by_rec = BinnedCounts(bins)
    for event in iter_events(hdf5_path):
        merged_true_ids = set()
        for rec_tracks in event.rec_tracks:
            if not rec_tracks:
                continue
            overlapping = _true_vertices_with_recall_above(
                rec_tracks, event.true_tracks, recall_threshold
            )
            is_merged = len(overlapping) > 1
            by_rec.add(len(rec_tracks), is_merged)
            if is_merged:
                merged_true_ids |= overlapping
        for true_id, true_tracks in enumerate(event.true_tracks):
            by_true.add(len(true_tracks), true_id in merged_true_ids)

    parameters = {"recall_threshold": recall_threshold}
    label = f"Recall threshold: {recall_threshold}"
    return [
        RateResult("merge_rate", "true", by_true, parameters, label),
        RateResult("merge_rate", "rec", by_rec, parameters, label),
    ]
