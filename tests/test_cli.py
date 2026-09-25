"""CLI dispatch, independent analysis, and failure handling."""

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import yaml

from src.cli import main, parse_args
from src.config_loader import AnalysisConfig, ReconstructionConfig


class CLITests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.input = self.root / "results.h5"
        self.input.touch()
        self.output = self.root / "analysis"
        self.config = self.root / "reconstruction.yaml"
        self.analysis_config = self.root / "analysis.yaml"
        self.analysis_config.write_text(
            yaml.safe_dump({"analysis": {"analyses": ["resolution"]}})
        )

    def invoke(self, args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(args)
        return code, stdout.getvalue(), stderr.getvalue()

    def write_config(self, config):
        self.config.write_text(yaml.safe_dump(config))
        return str(self.config)

    def fit_config(self):
        config = yaml.safe_load(
            (Path(__file__).parents[1] / "configs/reconstruction.yaml").read_text()
        )
        (self.root / "b.root").touch()
        (self.root / "a.root").touch()
        config["paths"]["input_root"] = str(self.root / "*.root")
        config["paths"]["output_dir"] = str(self.root / "fit")
        return config

    def test_help_and_version_are_available(self):
        for args in (
            ["--help"],
            ["analyze", "--help"],
            ["optimize", "--help"],
            ["--version"],
        ):
            with self.subTest(args=args), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    parse_args(args)
                self.assertEqual(error.exception.code, 0)

    def test_bad_cli_values_fail_before_execution(self):
        for args in (
            ["fit", "-w", "0"],
            ["fit", "-e", "-2"],
            ["analyze", "-a", "not_an_analysis"],
            ["fit", "--analyses", "resolution"],
        ):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    parse_args(args)
                self.assertEqual(error.exception.code, 2)

    def test_analysis_runs_without_config_or_reconstruction(self):
        with (
            patch("src.cli._run_fit") as fit,
            patch("src.cli._run_analysis") as analyze,
        ):
            code, _, stderr = self.invoke(
                [
                    "analyze",
                    "--input",
                    str(self.input),
                    "--output",
                    str(self.output),
                    "-a",
                    "resolution",
                ]
            )
        self.assertEqual((code, stderr), (0, ""))
        fit.assert_not_called()
        options = analyze.call_args.args[0]
        self.assertEqual(options["hdf5_file"], str(self.input))
        self.assertEqual(options["settings"].analyses, ("resolution",))
        self.assertFalse(hasattr(options["settings"], "m_minTracks"))
        self.assertTrue((self.output / "analysis.done").is_file())

    def test_analysis_only_config_and_overrides(self):
        self.write_config(
            {
                "analysis": {
                    "analyses": ["resolution"],
                    "purity_threshold": 0.8,
                    "method_name": "ExternalReco",
                }
            }
        )
        with patch("src.cli._run_analysis") as analyze:
            code, _, _ = self.invoke(
                [
                    "analyze",
                    "--input",
                    str(self.input),
                    "--output",
                    str(self.output),
                    "-c",
                    str(self.config),
                    "-a",
                    "track_pull",
                ]
            )
        self.assertEqual(code, 0)
        options = analyze.call_args.args[0]
        self.assertEqual(options["settings"].analyses, ("track_pull",))
        self.assertEqual(options["settings"].purity_threshold, 0.8)
        self.assertEqual(options["settings"].method_name, "ExternalReco")
        self.assertFalse(hasattr(options["settings"], "max_chi2"))

    def test_analysis_uses_configured_hdf5_and_output(self):
        self.write_config(
            {
                "paths": {
                    "input_hdf5": str(self.input),
                    "output_dir": str(self.output),
                },
                "analysis": {"purity_threshold": 0.9, "analyses": ["efficiency_track"]},
            }
        )
        with patch("src.cli._run_analysis") as analyze:
            code, _, _ = self.invoke(["analyze", "-c", str(self.config)])
        self.assertEqual(code, 0)
        options = analyze.call_args.args[0]
        self.assertEqual(options["hdf5_file"], str(self.input))
        self.assertEqual(options["output_dir"], str(self.output))
        self.assertEqual(options["settings"].purity_threshold, 0.9)

    def test_fit_overrides_and_sorted_inputs(self):
        self.write_config(self.fit_config())
        with (
            patch("src.cli._run_fit", return_value="saved.h5") as fit,
            patch("src.cli._run_analysis") as analyze,
        ):
            code, _, _ = self.invoke(
                [
                    "fit",
                    "-c",
                    str(self.config),
                    "-w",
                    "2",
                    "-e",
                    "7",
                    "--output",
                    str(self.output),
                ]
            )
        self.assertEqual(code, 0)
        analyze.assert_not_called()
        options = fit.call_args.args[0]
        self.assertEqual(options["max_workers"], 2)
        self.assertEqual(options["event_number"], 7)
        self.assertEqual(options["output_dir"], str(self.output))
        self.assertEqual(
            [Path(p).name for p in options["input_files"]], ["a.root", "b.root"]
        )

    def test_run_passes_actual_fit_output_to_analysis(self):
        self.write_config(self.fit_config())
        with (
            patch("src.cli._run_fit", return_value=str(self.input)) as fit,
            patch("src.cli._run_analysis") as analyze,
        ):
            code, _, _ = self.invoke(
                [
                    "run",
                    "-c",
                    str(self.config),
                    "--analysis-config",
                    str(self.analysis_config),
                    "--output",
                    str(self.output),
                ]
            )
        self.assertEqual(code, 0)
        fit.assert_called_once()
        self.assertEqual(analyze.call_args.args[0]["hdf5_file"], str(self.input))
        self.assertEqual(
            analyze.call_args.args[0]["output_dir"], str(self.output / "analysis")
        )

    def test_config_analysis_typo_fails_before_fit(self):
        config = self.fit_config()
        self.write_config(config)
        self.analysis_config.write_text(
            yaml.safe_dump({"analysis": {"analyses": ["typo"]}})
        )
        with patch("src.cli._run_fit") as fit:
            code, _, stderr = self.invoke(
                [
                    "run",
                    "-c",
                    str(self.config),
                    "--analysis-config",
                    str(self.analysis_config),
                ]
            )
        self.assertEqual(code, 2)
        self.assertIn("Unknown analyses", stderr)
        fit.assert_not_called()

    def test_missing_hdf5_has_clear_error_without_traceback(self):
        code, _, stderr = self.invoke(
            ["analyze", "--input", str(self.root / "missing.h5")]
        )
        self.assertEqual(code, 2)
        self.assertIn("HDF5 input does not exist", stderr)
        self.assertNotIn("Traceback", stderr)
        code, _, stderr = self.invoke(["analyze"])
        self.assertEqual(code, 2)
        self.assertIn("requires --input", stderr)

    def test_empty_and_non_mapping_yaml_fail_cleanly(self):
        for contents in ("", "- item\n", "analysis: []\n"):
            with self.subTest(contents=contents):
                self.config.write_text(contents)
                code, _, stderr = self.invoke(
                    ["analyze", "--input", str(self.input), "-c", str(self.config)]
                )
                self.assertEqual(code, 2)
                self.assertIn("mapping", stderr)

    def test_invalid_workflow_workers_fail_before_fit(self):
        config = self.fit_config()
        config["workflow"]["max_workers"] = 0
        self.write_config(config)
        with patch("src.cli._run_fit") as fit:
            code, _, stderr = self.invoke(["fit", "-c", str(self.config)])
        self.assertEqual(code, 2)
        self.assertIn("positive integer", stderr)
        fit.assert_not_called()

    def test_failure_clears_old_completion_marker(self):
        self.output.mkdir()
        marker = self.output / "analysis.done"
        marker.touch()
        with patch(
            "src.cli._run_analysis", side_effect=RuntimeError("cannot read result")
        ):
            code, _, stderr = self.invoke(
                ["analyze", "--input", str(self.input), "--output", str(self.output)]
            )
        self.assertEqual(code, 1)
        self.assertIn("cannot read result", stderr)
        self.assertFalse(marker.exists())

    def test_debug_retains_traceback(self):
        with patch("src.cli._run_analysis", side_effect=RuntimeError("test failure")):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                self.invoke(
                    [
                        "analyze",
                        "--input",
                        str(self.input),
                        "--output",
                        str(self.output),
                        "--debug",
                    ]
                )

    def test_run_keeps_fit_and_analysis_parameters_separate(self):
        config = self.fit_config()
        config["algorithm"]["max_chi2"] = 4
        self.write_config(config)
        self.analysis_config.write_text(
            yaml.safe_dump(
                {
                    "analysis": {
                        "analyses": ["track_pull"],
                        "purity_threshold": 0.9,
                    }
                }
            )
        )
        with (
            patch("src.cli._run_fit", return_value=str(self.input)) as fit,
            patch("src.cli._run_analysis") as analyze,
        ):
            code, _, _ = self.invoke(
                [
                    "run",
                    "-c",
                    str(self.config),
                    "--analysis-config",
                    str(self.analysis_config),
                ]
            )
        self.assertEqual(code, 0)
        self.assertEqual(fit.call_args.args[0]["params"]["max_chi2"], 4)
        self.assertNotIn("purity_threshold", fit.call_args.args[0]["params"])
        settings = analyze.call_args.args[0]["settings"]
        self.assertFalse(hasattr(settings, "max_chi2"))
        self.assertEqual(settings.purity_threshold, 0.9)
        self.assertEqual(settings.analyses, ("track_pull",))

    def test_fit_does_not_load_analysis_configuration(self):
        self.write_config(self.fit_config())
        self.analysis_config.unlink()
        with patch("src.cli._run_fit", return_value=str(self.input)) as fit:
            code, _, _ = self.invoke(["fit", "-c", str(self.config)])
        self.assertEqual(code, 0)
        fit.assert_called_once()

    def test_run_rejects_missing_analysis_config_before_fitting(self):
        self.write_config(self.fit_config())
        self.analysis_config.unlink()
        with patch("src.cli._run_fit") as fit:
            code, _, _ = self.invoke(
                [
                    "run",
                    "-c",
                    str(self.config),
                    "--analysis-config",
                    str(self.analysis_config),
                ]
            )
        self.assertEqual(code, 2)
        fit.assert_not_called()

    def test_analysis_rejects_reconstruction_yaml(self):
        self.write_config(self.fit_config())
        with patch("src.cli._run_analysis") as analyze:
            code, _, stderr = self.invoke(
                ["analyze", "--input", str(self.input), "-c", str(self.config)]
            )
        self.assertEqual(code, 2)
        self.assertIn("separate analysis YAML", stderr)
        analyze.assert_not_called()

    def test_analysis_defaults_do_not_inherit_fit_thresholds(self):
        settings = AnalysisConfig.from_mapping({"analysis": {"m_r": 0.5}})
        self.assertEqual(settings.purity_threshold, 0.7)
        self.assertFalse(hasattr(settings, "max_chi2"))
        with self.assertRaisesRegex(ValueError, "Unknown analysis keys: max_chi2"):
            AnalysisConfig.from_mapping({"analysis": {"max_chi2": 4}})
        fit_params = ReconstructionConfig.from_mapping(self.fit_config()).as_params()
        self.assertNotIn("purity_threshold", fit_params)
        self.assertNotIn("m_r", fit_params)
        with self.assertRaisesRegex(ValueError, "Unknown algorithm keys: m_r"):
            config = self.fit_config()
            config["algorithm"]["m_r"] = 0.8
            ReconstructionConfig.from_mapping(config)

    def test_reconstruction_config_round_trips_and_derives(self):
        cfg = ReconstructionConfig.from_mapping(self.fit_config())
        self.assertEqual(ReconstructionConfig.from_mapping(cfg.to_mapping()), cfg)
        self.assertAlmostEqual(cfg.bin_size_z, 0.5)
        self.assertAlmostEqual(cfg.bin_size_t, 0.025)
        self.assertAlmostEqual(cfg.m_minTracks, cfg.minTracks)
        halved = replace(cfg, bins_z=400)
        self.assertAlmostEqual(halved.m_minTracks, 2 * cfg.minTracks)
        with self.assertRaisesRegex(ValueError, "bins.z must be a positive"):
            replace(cfg, bins_z=0)

    def test_cli_import_does_not_load_fit_or_plotting(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                'import sys,json; import src.cli; print(json.dumps([m for m in sys.modules if m.startswith(("src.fit", "src.analysis", "src.optimize", "matplotlib", "optuna"))]))',
            ],
            cwd=Path(__file__).parents[1],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(json.loads(result.stdout), [])


if __name__ == "__main__":
    unittest.main()
