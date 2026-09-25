"""Reconstruction efficiency with overlap (track-based) matching."""

from .binned_rate import BinnedCounts, RateResult, iter_events
from .utils import find_best_match


def analyze_vertex_efficiency(hdf5_path, bins, recall_threshold, purity_threshold):
    """Fraction of true vertices matched by some reconstructed vertex.

    A reconstructed vertex matches the true vertex sharing the most tracks when
    both recall (shared / true tracks) and purity (shared / rec tracks) reach
    their thresholds; each reconstructed vertex matches at most one true vertex.
    Binned by the true vertex's reconstructed track count.
    """
    counts = BinnedCounts(bins)
    for event in iter_events(hdf5_path):
        true_vertex_info = {
            vertex_id: {"track_indices": tracks, "num_tracks": len(tracks)}
            for vertex_id, tracks in enumerate(event.true_tracks)
        }
        matched_true_ids = set()
        for rec_tracks in event.rec_tracks:
            if not rec_tracks:
                continue
            best_match_id = find_best_match(
                rec_tracks, true_vertex_info, recall_threshold, purity_threshold
            )
            if best_match_id is not None:
                matched_true_ids.add(best_match_id)
        for vertex_id, tracks in enumerate(event.true_tracks):
            counts.add(len(tracks), vertex_id in matched_true_ids)

    return [
        RateResult(
            metric="efficiency",
            binned_by="true",
            counts=counts,
            parameters={
                "recall_threshold": recall_threshold,
                "purity_threshold": purity_threshold,
            },
            label=f"Recall≥{recall_threshold}, Purity≥{purity_threshold}",
        )
    ]
