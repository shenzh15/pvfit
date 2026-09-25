"""Fake rate with overlap (track-based) matching."""

from .binned_rate import BinnedCounts, RateResult, iter_events


def analyze_vertex_fake_rate(hdf5_path, bins, recall_threshold, purity_threshold):
    """Fraction of reconstructed vertices no true vertex claims.

    Each true vertex claims, among the reconstructed vertices sharing tracks
    with it and reaching either the purity or the recall threshold, the one
    with the highest recall; a contested reconstructed vertex goes to the claim
    with the higher recall. Unclaimed reconstructed vertices are fake. Binned
    by the reconstructed vertex's track count.
    """
    counts = BinnedCounts(bins)
    for event in iter_events(hdf5_path):
        rec_vertices = {
            rec_id: {
                "track_indices": tracks,
                "num_tracks": len(tracks),
                "claimed_by": None,
            }
            for rec_id, tracks in enumerate(event.rec_tracks)
            if len(tracks) > 0
        }

        # Each true vertex claims its best reconstructed vertex
        for true_id, true_tracks in enumerate(event.true_tracks):
            num_true_tracks = len(true_tracks)
            best_rec_id = None
            best_recall = 0.0
            for rec_id, rec in rec_vertices.items():
                num_shared = len(rec["track_indices"] & true_tracks)
                if num_shared == 0:
                    continue
                purity = num_shared / rec["num_tracks"]
                recall = num_shared / num_true_tracks
                if (
                    purity >= purity_threshold or recall >= recall_threshold
                ) and recall > best_recall:
                    best_recall = recall
                    best_rec_id = rec_id
            if best_rec_id is not None:
                current_claim = rec_vertices[best_rec_id]["claimed_by"]
                if current_claim is None or best_recall > current_claim[1]:
                    rec_vertices[best_rec_id]["claimed_by"] = (true_id, best_recall)

        for rec in rec_vertices.values():
            counts.add(rec["num_tracks"], rec["claimed_by"] is None)

    return [
        RateResult(
            metric="fake_rate",
            binned_by="rec",
            counts=counts,
            parameters={
                "recall_threshold": recall_threshold,
                "purity_threshold": purity_threshold,
            },
            label=f"Recall≥{recall_threshold} or Purity≥{purity_threshold} to be claimed",
        )
    ]
