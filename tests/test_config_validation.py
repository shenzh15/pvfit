"""Configuration values that would silently distort results are rejected."""

import copy
import math
import unittest
from dataclasses import replace
from pathlib import Path

import yaml

from pvfit.config_loader import AnalysisConfig, ReconstructionConfig

CONFIGS = Path(__file__).parents[1] / "configs"


class ReconstructionConfigValidationTests(unittest.TestCase):
    def setUp(self):
        self.base = yaml.safe_load((CONFIGS / "reconstruction.yaml").read_text())

    def build(self, **algorithm):
        mapping = copy.deepcopy(self.base)
        mapping["algorithm"].update(algorithm)
        return ReconstructionConfig.from_mapping(mapping)

    def test_shipped_configuration_is_valid(self):
        self.build()

    def test_round_trips_and_derives_bin_quantities(self):
        cfg = self.build()
        self.assertEqual(ReconstructionConfig.from_mapping(cfg.to_mapping()), cfg)
        self.assertAlmostEqual(cfg.bin_size_z, 0.5)
        self.assertAlmostEqual(cfg.bin_size_t, 0.025)
        self.assertAlmostEqual(cfg.m_minTracks, cfg.minTracks)
        halved = replace(cfg, bins_z=400)
        self.assertAlmostEqual(halved.m_minTracks, 2 * cfg.minTracks)
        with self.assertRaisesRegex(ValueError, "bins.z must be a positive"):
            replace(cfg, bins_z=0)

    def test_analysis_keys_are_rejected_and_never_reach_fit_params(self):
        fit_params = self.build().as_params()
        self.assertNotIn("purity_threshold", fit_params)
        self.assertNotIn("m_r", fit_params)
        with self.assertRaisesRegex(ValueError, "Unknown algorithm keys: m_r"):
            self.build(m_r=0.8)

    def test_non_finite_numbers_are_rejected(self):
        for name in ("max_chi2", "minTracks", "nsigma_merge", "max_track_bl_chi2"):
            for value in (math.nan, math.inf):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, f"algorithm.{name}"):
                        self.build(**{name: value})
        with self.assertRaisesRegex(ValueError, "z_range requires min < max"):
            self.build(z_range={"min": -math.inf, "max": 200})

    def test_flags_must_be_real_booleans(self):
        for name in (
            "beamline_chi2_cut",
            "z0_err_cut",
            "is_adaptive_tukey",
            "drop_low_track_seeds",
            "merge_vertices",
        ):
            for value in ("false", "true", 0, 1, None):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, "must be true or false"):
                        self.build(**{name: value})
        self.assertFalse(self.build(beamline_chi2_cut=False).beamline_chi2_cut)


class AnalysisConfigValidationTests(unittest.TestCase):
    def test_shipped_configuration_is_valid(self):
        AnalysisConfig.from_mapping(
            yaml.safe_load((CONFIGS / "analysis.yaml").read_text())
        )

    def test_fit_keys_are_rejected_and_defaults_do_not_inherit_fit_thresholds(self):
        settings = AnalysisConfig.from_mapping({"analysis": {"m_r": 0.5}})
        self.assertEqual(settings.purity_threshold, 0.7)
        self.assertFalse(hasattr(settings, "max_chi2"))
        with self.assertRaisesRegex(ValueError, "Unknown analysis keys: max_chi2"):
            AnalysisConfig.from_mapping({"analysis": {"max_chi2": 4}})

    def test_thresholds_are_fractions(self):
        for name in ("recall_threshold", "purity_threshold"):
            for value in (1.5, -0.1, math.nan, "0.7", True):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                        AnalysisConfig(**{name: value})
            AnalysisConfig(**{name: 0.0})
            AnalysisConfig(**{name: 1.0})

    def test_scales_and_windows_are_positive_and_finite(self):
        for name in (
            "m_r",
            "nsigma",
            "z_scale",
            "t_scale",
            "x_residual_range",
            "y_residual_range",
            "z_residual_range",
            "t_residual_range",
        ):
            for value in (0, -1, math.nan, math.inf, "1"):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, "positive finite"):
                        AnalysisConfig(**{name: value})

    def test_flags_and_labels(self):
        for name in ("draw_fixed_ellipses", "draw_fitted_ellipses"):
            with self.assertRaisesRegex(ValueError, "must be true or false"):
                AnalysisConfig(**{name: "false"})
        with self.assertRaisesRegex(ValueError, "method_name"):
            AnalysisConfig(method_name="  ")
        with self.assertRaisesRegex(ValueError, "plot_min_ntracks"):
            AnalysisConfig(plot_min_ntracks=10, plot_max_ntracks=5)


if __name__ == "__main__":
    unittest.main()
