"""Each rate analysis counts hand-built events the way its definition says."""

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from pvfit.analysis import analyze_merge_split_geo as merge_split_geo
from pvfit.analysis.analyze_efficiency_geo import (
    analyze_vertex_efficiency as efficiency_geo,
)
from pvfit.analysis.analyze_efficiency_track import (
    analyze_vertex_efficiency as efficiency_track,
)
from pvfit.analysis.analyze_fake_rate_geo import analyze_vertex_fake_rate as fake_geo
from pvfit.analysis.analyze_fake_rate_track import (
    analyze_vertex_fake_rate as fake_track,
)
from pvfit.analysis.analyze_merge_rate_track import (
    analyze_vertex_merge_rate as merge_track,
)
from pvfit.analysis.analyze_split_rate_track import (
    analyze_vertex_split_rate as split_track,
)
from pvfit.analysis.binned_rate import BinnedCounts, make_bins

BINS = make_bins(4, 13, 5)  # (4, 8) and (9, 13)
LOW, HIGH = BINS


def write_events(path, events):
    """events: list of (true_vertices, rec_vertices); a vertex is a dict with
    z, t, tracks and, for rec vertices, z_err/t_err."""
    with h5py.File(path, "w") as f:
        for event_id, (true, rec) in enumerate(events):
            group = f.create_group(str(event_id))
            for name, vertices in (("true_vertices", true), ("rec_vertices", rec)):
                g = group.create_group(name)
                for field in ("x", "y", "z", "t", "z_err", "t_err"):
                    if name == "true_vertices" and field.endswith("_err"):
                        continue
                    g[field] = np.array([v.get(field, 0.0) for v in vertices])
                ds = g.create_dataset(
                    "track_indices", (len(vertices),), dtype=h5py.vlen_dtype(np.int32)
                )
                for i, v in enumerate(vertices):
                    ds[i] = np.array(sorted(v["tracks"]), dtype=np.int32)


def counts(result):
    """{bin: (numerator, denominator)} of a RateResult."""
    c = result.counts
    return {b: (c.numerator[b], c.denominator[b]) for b in c.bins}


def by_key(results):
    return {(r.matching, r.binned_by): r for r in results}


class BinnedCountsTests(unittest.TestCase):
    def test_bins_counts_and_overall(self):
        self.assertEqual(BINS, [(4, 8), (9, 13)])
        c = BinnedCounts(BINS)
        for n_tracks, success in ((4, True), (8, False), (13, True), (14, True)):
            c.add(n_tracks, success)  # 14 is outside every bin: ignored
        self.assertEqual(c.numerator, {LOW: 1, HIGH: 1})
        self.assertEqual(c.denominator, {LOW: 2, HIGH: 1})
        rate, lower, upper, numerator, denominator = c.overall()
        self.assertEqual((rate, numerator, denominator), (2 / 3, 2, 3))
        self.assertGreater(lower, 0)
        self.assertGreater(upper, 0)
        self.assertEqual(list(c.table()["rate"]), [0.5, 1.0])


class TrackBasedRateTests(unittest.TestCase):
    """Thresholds 0.7/0.7. Event 0: true0 (tracks 0-9), true1 (10-15),
    true2 (16-21); rec A (0-7) matches true0, rec B (10-21) covers true1 and
    true2 with purity 0.5, rec C (22-25) shares nothing. Event 1: true0 (0-9)
    is cut in half by rec D (0-4) and rec E (5-9)."""

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.path = Path(cls.directory.name) / "track.h5"
        write_events(
            cls.path,
            [
                (
                    [
                        {"tracks": range(0, 10)},
                        {"tracks": range(10, 16)},
                        {"tracks": range(16, 22)},
                    ],
                    [
                        {"tracks": range(0, 8)},
                        {"tracks": range(10, 22)},
                        {"tracks": range(22, 26)},
                    ],
                ),
                (
                    [{"tracks": range(0, 10)}],
                    [{"tracks": range(0, 5)}, {"tracks": range(5, 10)}],
                ),
            ],
        )

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_efficiency(self):
        (result,) = efficiency_track(self.path, BINS, 0.7, 0.7)
        # only true0 of event 0 is matched (rec A: recall 0.8, purity 1.0);
        # rec B fails purity, D and E fail recall
        self.assertEqual(counts(result), {LOW: (0, 2), HIGH: (1, 2)})
        self.assertEqual(result.parameters["recall_threshold"], 0.7)

    def test_fake_rate(self):
        (result,) = fake_track(self.path, BINS, 0.7, 0.7)
        # rec C is unclaimed; B is claimed by true1 (recall 1.0 suffices);
        # true0 of event 1 claims D (first of the tie), so E is fake
        self.assertEqual(counts(result), {LOW: (2, 4), HIGH: (0, 1)})

    def test_merge_rate(self):
        results = by_key(merge_track(self.path, BINS, 0.7))
        # rec B has recall 1.0 to both true1 and true2
        self.assertEqual(counts(results["track", "rec"]), {LOW: (0, 4), HIGH: (1, 1)})
        self.assertEqual(counts(results["track", "true"]), {LOW: (2, 2), HIGH: (0, 2)})

    def test_split_rate(self):
        results = by_key(split_track(self.path, BINS, 0.7))
        # true0 of event 1 has two pure pieces D and E
        self.assertEqual(counts(results["track", "true"]), {LOW: (0, 2), HIGH: (1, 2)})
        self.assertEqual(counts(results["track", "rec"]), {LOW: (2, 4), HIGH: (0, 1)})


class DistanceBasedRateTests(unittest.TestCase):
    """Fixed ellipse: m_r=0.8, z_scale=1.5, t_scale=0.06 -> semi-axes 0.6 mm,
    0.024 ns. Resolution ellipse: 3 sigma with z_err=0.05 -> 0.15 mm.
    true0 at z=0 (10 tracks) and true1 at z=0.5 (6 tracks) sit inside one
    fixed ellipse; rec A (z=0.1, 8 tracks) and rec C (z=0.45, 5 tracks)
    each contain both in the fixed ellipse but only their nearest in the
    3 sigma one; rec B (z=10, 4 tracks) is far from everything."""

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.path = Path(cls.directory.name) / "geo.h5"
        err = {"z_err": 0.05, "t_err": 0.005}
        write_events(
            cls.path,
            [
                (
                    [
                        {"z": 0.0, "t": 0.0, "tracks": range(0, 10)},
                        {"z": 0.5, "t": 0.0, "tracks": range(10, 16)},
                    ],
                    [
                        {"z": 0.1, "t": 0.0, "tracks": range(0, 8), **err},
                        {"z": 10.0, "t": 0.0, "tracks": range(20, 24), **err},
                        {"z": 0.45, "t": 0.0, "tracks": range(10, 15), **err},
                    ],
                )
            ],
        )
        cls.args = (cls.path, BINS, 0.8, 1.5, 0.06, 3.0)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_efficiency(self):
        results = by_key(efficiency_geo(*self.args))
        # A's closest is true0, C's closest is true1, under both definitions
        for matching in ("geo", "reso"):
            self.assertEqual(
                counts(results[matching, "true"]), {LOW: (1, 1), HIGH: (1, 1)}
            )

    def test_fake_rate(self):
        results = by_key(fake_geo(*self.args))
        for matching in ("geo", "reso"):
            self.assertEqual(
                counts(results[matching, "rec"]), {LOW: (1, 3), HIGH: (0, 0)}
            )
        self.assertEqual(results["reso", "rec"].parameters, {"nsigma": 3.0})

    def test_merge_rate(self):
        results = by_key(merge_split_geo.analyze_vertex_merge_rate(*self.args))
        # fixed ellipse: A and C each contain both true vertices
        self.assertEqual(counts(results["geo", "rec"]), {LOW: (2, 3), HIGH: (0, 0)})
        self.assertEqual(counts(results["geo", "true"]), {LOW: (1, 1), HIGH: (1, 1)})
        # 3 sigma: each rec contains only its nearest true vertex
        self.assertEqual(counts(results["reso", "rec"]), {LOW: (0, 3), HIGH: (0, 0)})
        self.assertEqual(counts(results["reso", "true"]), {LOW: (0, 1), HIGH: (0, 1)})

    def test_split_rate(self):
        results = by_key(merge_split_geo.analyze_vertex_split_rate(*self.args))
        # fixed ellipse: each true vertex lies inside A's and C's ellipse
        self.assertEqual(counts(results["geo", "true"]), {LOW: (1, 1), HIGH: (1, 1)})
        self.assertEqual(counts(results["geo", "rec"]), {LOW: (2, 3), HIGH: (0, 0)})
        self.assertEqual(counts(results["reso", "true"]), {LOW: (0, 1), HIGH: (0, 1)})
        self.assertEqual(counts(results["reso", "rec"]), {LOW: (0, 3), HIGH: (0, 0)})


if __name__ == "__main__":
    unittest.main()
