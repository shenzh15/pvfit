"""Regression checks for truth association, transport, and uncut diagnostics."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd

from pvfit.analysis.analyze_track_pull import analyze_track_pull
from pvfit.fit.save_results_to_hdf5 import save_all_events_to_hdf5
from pvfit.fit.truth_matching import build_true_vertex_info


class TrackPullMembershipTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "input.h5"

    def write_tracks(self, n=2, indices=None, states=None, truth=None):
        defaults = {name: np.zeros(n) for name in ("x", "y", "z", "t", "tx", "ty")}
        defaults["backward"] = np.zeros(n, dtype=bool)
        defaults.update({name: np.ones(n) for name in ("cov_xx", "cov_yy", "cov_tt")})
        defaults.update(states or {})
        with h5py.File(self.path, "w") as f:
            f.attrs["n_events"] = 1
            event = f.create_group("0")
            true = event.create_group("true_vertices")
            for name in ("x", "y", "z", "t"):
                true[name] = [(truth or {}).get(name, 0.0)]
            membership = true.create_dataset(
                "track_indices", (1,), dtype=h5py.vlen_dtype(np.int32)
            )
            membership[0] = np.asarray(
                np.arange(n) if indices is None else indices, dtype=np.int32
            )
            tracks = event.create_group("tracks")
            for name, values in defaults.items():
                tracks[name] = values

    def analyze(self):
        with patch("pvfit.analysis.analyze_track_pull._plot_track_pulls") as plot:
            summary = analyze_track_pull(self.path, self.directory.name)
        plot.assert_called_once()
        return summary, plot.call_args.args[0]

    def test_uses_track_indices_without_origin_coordinate_fields(self):
        self.write_tracks(indices=[1], states={"x": [100.0, 0.1]})
        summary, data = self.analyze()
        np.testing.assert_array_equal(data["residual_x"], [0.1])
        self.assertEqual(summary["n_associated"], 1)

    def test_transport_to_truth_plane_for_forward_and_backward_tracks(self):
        # Two exact lines through the same truth point. Opposite time slopes
        # require opposite propagation signs; both must give zero residuals.
        flight_time = 2 * np.sqrt(2) / 299.792458
        self.write_tracks(
            states={
                "x": [8.0, 10.0],
                "y": [-5.0, -7.0],
                "z": [0.0, 0.0],
                "t": [7 - flight_time, 7 + flight_time],
                "tx": [1.0, 0.0],
                "ty": [0.0, 1.0],
                "backward": [False, True],
            },
            truth={"x": 10.0, "y": -5.0, "z": 2.0, "t": 7.0},
        )
        _, data = self.analyze()
        for axis in ("x", "y", "t"):
            np.testing.assert_allclose(data[f"residual_{axis}"], 0, atol=1e-12)
        np.testing.assert_array_equal(data["delta_z"], [2.0, 2.0])

    def test_large_residual_is_kept_in_primary_plot_and_statistics(self):
        self.write_tracks(states={"x": [0.5, 8.0]})
        summary, data = self.analyze()
        np.testing.assert_array_equal(data["pull_x"], [0.5, 8.0])
        self.assertEqual(summary["n_valid"], 2)
        self.assertEqual(summary["statistics"]["pull_x"]["std"], 3.75)
        stored = json.loads(
            (
                Path(self.directory.name) / "pull_analysis_input_track_summary.json"
            ).read_text()
        )
        self.assertEqual(stored, summary)

    def test_large_residual_after_transport_is_retained(self):
        # Transport produces a 4-sigma residual; the analysis must retain the track.
        self.write_tracks(n=1, states={"tx": [2.0]}, truth={"z": 2.0})
        summary, data = self.analyze()
        np.testing.assert_array_equal(data["pull_x"], [4.0])
        self.assertEqual(summary["n_valid"], 1)
        self.assertEqual(summary["statistics"]["pull_x"]["mean"], 4.0)

    def test_invalid_states_and_covariances_are_excluded_and_counted(self):
        self.write_tracks(
            n=6,
            states={
                "cov_xx": [1.0, 0.0, -1.0, np.nan, np.inf, 1.0],
                "x": [0.0, 0.0, 0.0, 0.0, 0.0, np.nan],
            },
        )
        summary, data = self.analyze()
        self.assertEqual(summary["n_invalid"], 5)
        self.assertEqual(len(data["pull_x"]), 1)

    def test_legacy_hdf5_requires_regeneration(self):
        self.write_tracks()
        with h5py.File(self.path, "a") as f:
            del f["0/tracks/tx"]
        with self.assertRaisesRegex(ValueError, "re-run reconstruction"):
            self.analyze()

    def test_empty_membership_has_defined_empty_statistics(self):
        self.write_tracks(indices=[])
        summary, data = self.analyze()
        self.assertEqual(summary["n_valid"], 0)
        self.assertEqual(len(data["pull_x"]), 0)
        self.assertIsNone(summary["statistics"]["pull_x"]["std"])

    def test_writer_preserves_directions_in_selected_row_order(self):
        # Two truth memberships with different multiplicities exercise the
        # writer's ragged arrays as in the original toy-based regression case.
        # Coordinates coincide deliberately: the associations must follow keys.
        z = np.array([-2.0, 3.0, -1.0, 5.0, -3.0, 2.0, 4.0, -4.0, 6.0])
        tx = np.array([0.1, -0.2, 0.3, -0.4, 0.2, 0.1, -0.1, 0.3, -0.2])
        ty = np.array([0.2, 0.1, -0.1, -0.2, 0.3, -0.2, 0.1, 0.2, -0.1])
        backward = np.array([False, True, False, True, True, False, True, False, True])
        time_slope = np.where(backward, -1, 1) * np.sqrt(1 + tx**2 + ty**2) / 299.792458
        n = len(z)
        tracks = pd.DataFrame(
            {
                "event_number": np.zeros(n, dtype=int),
                "mc_key": [12, 20, 10, 23, 13, 21, 11, 24, 22],
                "beamPOCA_x": z * tx,
                "beamPOCA_y": z * ty,
                "beamPOCA_z": z,
                "beamPOCA_t": z * time_slope,
                "beamPOCA_tx": tx,
                "beamPOCA_ty": ty,
                "backward": backward,
                "c00": np.ones(n),
                "c11": np.ones(n),
                "c55": np.ones(n),
                "ovtx_x": np.zeros(n),
                "ovtx_y": np.zeros(n),
                "ovtx_z": np.zeros(n),
                "ovtx_t": np.zeros(n),
                "p": np.full(n, 1000.0),
                "pt": 1000 * np.hypot(tx, ty) / np.sqrt(1 + tx**2 + ty**2),
                "chi2": np.zeros(n),
                "fromPV": np.ones(n, dtype=bool),
            },
            index=[22, 7, 14, 5, 30, 4, 17, 2, 8],
        )
        vertices = pd.DataFrame(
            {
                "event_number": [0, 0],
                "effective_pv_key": [99, 100],
                "x": [0.0, 0.0],
                "y": [0.0, 0.0],
                "z": [0.0, 0.0],
                "time": [0.0, 0.0],
                "n_valid_tracks": [4, 5],
                "track_keys": [
                    np.array([10, 11, 12, 13]),
                    np.array([20, 21, 22, 23, 24]),
                ],
            }
        )
        truth = build_true_vertex_info(tracks, vertices)
        result = {
            "event_number": 0,
            "vertex_true_info": truth,
            "vertex_rec_info": [],
            "histogram": np.zeros((2, 2)),
            "n_seedings": 0,
            "seeds": [],
            "event_data_selected": tracks,
        }
        params = {
            "zmin": -200,
            "zmax": 200,
            "tmin": -0.6,
            "tmax": 0.8,
            "Nbins_z": 2,
            "Nbins_t": 2,
        }
        save_all_events_to_hdf5([result], self.path, params)
        with h5py.File(self.path, "r") as f:
            saved = f["0/tracks"]
            for name in ("tx", "ty"):
                np.testing.assert_array_equal(
                    saved[name][:],
                    tracks[f"beamPOCA_{name}"].to_numpy(dtype=np.float32),
                )
            np.testing.assert_array_equal(saved["backward"][:], tracks.backward)
        summary, _ = self.analyze()
        self.assertEqual(
            summary["n_associated"], sum(len(v["track_indices"]) for v in truth)
        )
        self.assertEqual(summary["n_invalid"], 0)


if __name__ == "__main__":
    unittest.main()
