"""The histogram plot reads any producer's HDF5 in the PVFit event layout."""

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from src.analysis.plot_histogram import load_event_histogram, plot_event_histogram
from src.config_loader import AnalysisConfig
from src.fit.save_results_to_hdf5 import save_all_events_to_hdf5


class PlotHistogramTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / "input.h5"

    def write(self, histogram, groups=None, source=None, n_bins=None):
        with h5py.File(self.path, "w") as f:
            if source:
                f.attrs["source"] = source
            event = f.create_group("0")
            ds = event.create_dataset("histogram", data=histogram)
            n_z, n_t = n_bins or histogram.shape
            ds.attrs.update(
                z_min=-10.0, z_max=10.0, t_min=-1.0, t_max=1.0, n_bins_z=n_z, n_bins_t=n_t
            )
            for name, fields in (groups or {}).items():
                group = event.create_group(name)
                for field, values in fields.items():
                    group[field] = values

    def plot(self, **flags):
        output = plot_event_histogram(
            self.path, self.root / "plots", AnalysisConfig(**flags), event_id=0
        )
        self.assertIsNotNone(output)
        self.assertTrue(output.exists())

    def test_pvfit_writer_stores_all_seeds_and_source(self):
        tracks = pd.DataFrame(
            {
                name: np.zeros(1)
                for name in (
                    "beamPOCA_x", "beamPOCA_y", "beamPOCA_z", "beamPOCA_t",
                    "beamPOCA_tx", "beamPOCA_ty", "c00", "c11", "c55",
                    "ovtx_x", "ovtx_y", "ovtx_z", "ovtx_t", "p", "pt", "chi2",
                )
            }
            | {"mc_key": [1], "backward": [False], "fromPV": [True]}
        )
        rec = {"rec_z": 1.0, "rec_t": 0.1, "track_index": [0]}
        result = {
            "event_number": 0,
            "vertex_true_info": [],
            "vertex_rec_info": [rec],
            "histogram": np.ones((4, 2)),
            "n_seedings": 3,
            # Two seeds were dropped or merged; only one became a vertex.
            "seeds": [(1.0, 0.1), (5.0, -0.2), (-3.0, 0.3)],
            "event_data_selected": tracks,
        }
        params = dict(zmin=-10, zmax=10, tmin=-1, tmax=1, Nbins_z=4, Nbins_t=2)
        save_all_events_to_hdf5([result], self.path, params)

        data = load_event_histogram(self.path, 0)
        self.assertEqual(data["source"], "PVFit")
        np.testing.assert_allclose(data["seeds"]["z"], [1.0, 5.0, -3.0])
        self.assertEqual(len(data["rec"]["z"]), 1)
        self.plot(draw_fixed_ellipses=True, draw_fitted_ellipses=True)

    def test_other_producer_with_peaks_and_no_errors_or_membership(self):
        self.write(
            np.arange(8, dtype=np.int32).reshape(4, 2),
            groups={
                "seeds": {"z": [1.0, 2.0, 3.0], "t": [0.0, 0.1, 0.2]},
                "rec_vertices": {"z": [1.5], "t": [0.05]},
                "true_vertices": {"z": [1.4], "t": [0.04]},
            },
            source="Allen",
        )
        data = load_event_histogram(self.path, 0)
        self.assertEqual(data["source"], "Allen")
        self.assertEqual(len(data["seeds"]["z"]), 3)
        self.plot(draw_fixed_ellipses=True, draw_fitted_ellipses=True)

    def test_older_files_fall_back_to_rec_seed_positions(self):
        self.write(
            np.ones((4, 2)),
            groups={
                "rec_vertices": {"z": [1.5], "t": [0.05], "z_seed": [1.0], "t_seed": [0.0]}
            },
        )
        data = load_event_histogram(self.path, 0)
        np.testing.assert_allclose(data["seeds"]["z"], [1.0])
        self.assertIsNone(data["source"])

    def test_histogram_only(self):
        self.write(np.ones((4, 2)))
        data = load_event_histogram(self.path, 0)
        self.assertIsNone(data["true"])
        self.assertIsNone(data["seeds"])
        self.assertIsNone(data["rec"])
        self.plot(draw_fitted_ellipses=True)

    def test_transposed_histogram_is_rejected(self):
        self.write(np.ones((2, 4)), n_bins=(4, 2))
        with self.assertRaisesRegex(ValueError, "does not match"):
            load_event_histogram(self.path, 0)


if __name__ == "__main__":
    unittest.main()
