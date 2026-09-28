"""Events are identified by (run_number, event_number) across all input files."""

import unittest

import numpy as np
import pandas as pd

from pvfit.fit.event_identity import require_run_numbers, unify_event_identity
from pvfit.fit.truth_matching import build_true_vertex_info


def tracks_frame(runs, events, keys):
    return pd.DataFrame(
        {
            "run_number": runs,
            "event_number": events,
            "mc_key": keys,
            "beamPOCA_z": np.zeros(len(keys)),
            "beamPOCA_t": np.zeros(len(keys)),
        }
    )


def vertices_frame(runs, events, pv_keys, track_keys):
    return pd.DataFrame(
        {
            "run_number": runs,
            "event_number": events,
            "effective_pv_key": pv_keys,
            "x": np.zeros(len(pv_keys)),
            "y": np.zeros(len(pv_keys)),
            "z": np.zeros(len(pv_keys)),
            "time": np.zeros(len(pv_keys)),
            "n_valid_tracks": [len(k) for k in track_keys],
            "track_keys": [np.array(k) for k in track_keys],
        }
    )


class EventIdentityTests(unittest.TestCase):
    def setUp(self):
        # Two runs, both restarting at event 0, both reusing effective_pv_key 5.
        self.tracks = tracks_frame(
            runs=[101] * 4 + [102] * 4,
            events=[0] * 4 + [0] * 4,
            keys=[1, 2, 3, 4, 1, 2, 3, 4],
        )
        self.vertices = vertices_frame(
            runs=[101, 102],
            events=[0, 0],
            pv_keys=[5, 5],
            track_keys=[[1, 2, 3, 4], [1, 2, 3, 4]],
        )

    def test_multi_run_events_get_distinct_ids_and_keep_their_origin(self):
        tracks, vertices, identity = unify_event_identity(
            [(self.tracks, self.vertices, "a.root")]
        )
        self.assertEqual(sorted(tracks["event_number"].unique()), [0, 1])
        self.assertEqual(sorted(vertices["event_number"].unique()), [0, 1])
        self.assertEqual(list(identity["event_number"]), [0, 1])
        self.assertEqual(list(identity["run_number"]), [101, 102])
        self.assertEqual(list(identity["source_event_number"]), [0, 0])
        self.assertEqual(list(identity["source_file"]), ["a.root", "a.root"])
        self.assertEqual(list(tracks["source_event_number"]), [0] * 8)
        # Tracks of run 102 now share an id only with the truth of run 102.
        run_102 = tracks[tracks["run_number"] == 102]["event_number"].unique()
        self.assertEqual(list(run_102), [1])
        self.assertEqual(
            list(vertices[vertices["run_number"] == 102]["event_number"]), [1]
        )
        # The inputs are not modified in place.
        self.assertEqual(list(self.tracks["event_number"].unique()), [0])

    def test_multi_run_truth_matching_no_longer_sees_duplicate_keys(self):
        with self.assertRaisesRegex(ValueError, "appears in runs"):
            build_true_vertex_info(self.tracks, self.vertices, min_tracks=0)
        tracks, vertices, _ = unify_event_identity(
            [(self.tracks, self.vertices, "a.root")]
        )
        for event in (0, 1):
            info = build_true_vertex_info(
                tracks[tracks["event_number"] == event],
                vertices[vertices["event_number"] == event],
                min_tracks=0,
            )
            self.assertEqual(len(info), 1)
            self.assertEqual(info[0]["track_indices"], [0, 1, 2, 3])

    def test_ids_follow_run_then_event_order(self):
        tracks = tracks_frame([7, 7, 3], [9, 3, 5], [1, 2, 3])
        vertices = vertices_frame([7, 7, 3], [9, 3, 5], [5, 6, 7], [[1], [2], [3]])
        _, _, identity = unify_event_identity([(tracks, vertices, "a.root")])
        self.assertEqual(list(identity["run_number"]), [3, 7, 7])
        self.assertEqual(list(identity["source_event_number"]), [5, 3, 9])
        self.assertEqual(list(identity["event_number"]), [0, 1, 2])

    def test_files_share_one_id_space_and_remember_their_file(self):
        second_tracks = tracks_frame([103], [0], [1])
        second_vertices = vertices_frame([103], [0], [5], [[1]])
        tracks, vertices, identity = unify_event_identity(
            [
                (second_tracks, second_vertices, "b.root"),
                (self.tracks, self.vertices, "a.root"),
            ]
        )
        self.assertEqual(list(identity["event_number"]), [0, 1, 2])
        self.assertEqual(list(identity["run_number"]), [101, 102, 103])
        self.assertEqual(list(identity["source_file"]), ["a.root", "a.root", "b.root"])
        self.assertEqual(sorted(tracks["event_number"].unique()), [0, 1, 2])
        self.assertEqual(sorted(vertices["event_number"].unique()), [0, 1, 2])

    def test_same_run_and_event_in_two_files_is_an_error(self):
        with self.assertRaisesRegex(ValueError, r"\(101, 0\) appears in more than one"):
            unify_event_identity(
                [
                    (self.tracks, self.vertices, "a.root"),
                    (self.tracks, self.vertices, "copy.root"),
                ]
            )

    def test_missing_run_numbers_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "track tree lacks run_number"):
            require_run_numbers(self.tracks.drop(columns="run_number"), self.vertices)
        with self.assertRaisesRegex(ValueError, "truth tree lacks run_number"):
            require_run_numbers(self.tracks, self.vertices.drop(columns="run_number"))
        with self.assertRaisesRegex(ValueError, "track and truth trees lack"):
            unify_event_identity(
                [
                    (
                        self.tracks.drop(columns="run_number"),
                        self.vertices.drop(columns="run_number"),
                        "old.root",
                    )
                ]
            )
        with self.assertRaisesRegex(ValueError, "No input data"):
            unify_event_identity([])


if __name__ == "__main__":
    unittest.main()
