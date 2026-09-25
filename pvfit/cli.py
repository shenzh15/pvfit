"""Command-line interface for reconstruction and independent HDF5 analysis."""

import argparse
import glob
import sys
from dataclasses import replace
from pathlib import Path

import yaml

from . import __version__
from .config_loader import (
    ANALYSIS_NAMES,
    AnalysisConfig,
    ReconstructionConfig,
    load_config,
)


def _positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _event_number(value):
    value = int(value)
    if value < -1:
        raise argparse.ArgumentTypeError(
            "must be -1 (all events) or a nonnegative event id"
        )
    return value


def _add_options(parser, command):
    parser.add_argument(
        "--config",
        "--config-file",
        "-c",
        dest="config_file",
        type=Path,
        default=None if command == "analyze" else Path("configs/reconstruction.yaml"),
        help="Analysis YAML for analyze (optional); reconstruction YAML for fit/run",
    )
    parser.add_argument(
        "--output-dir",
        "--output",
        "-o",
        type=Path,
        help="Output directory; for analyze, write plots/tables directly here",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Show tracebacks on failure"
    )
    if command == "run":
        parser.add_argument(
            "--analysis-config",
            type=Path,
            default=Path("configs/analysis.yaml"),
            help="Independent analysis YAML (default: configs/analysis.yaml)",
        )
    if command in ("fit", "run"):
        parser.add_argument(
            "--max-workers",
            "-w",
            type=_positive_int,
            help="Override reconstruction worker count",
        )
        parser.add_argument(
            "--event-number",
            "-e",
            type=_event_number,
            help="Reconstruct one event; -1 selects all events",
        )
    if command == "analyze":
        parser.add_argument(
            "--input", "-i", type=Path, help="Existing HDF5 reconstruction output"
        )
    else:
        parser.add_argument(
            "--input",
            "-i",
            type=Path,
            help="Input ROOT file or glob (overrides paths.input_root)",
        )
    if command != "fit":
        parser.add_argument(
            "--analyses",
            "-a",
            nargs="+",
            choices=ANALYSIS_NAMES,
            help="Analyses to run (overrides configuration)",
        )


def _add_optimize_options(parser):
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("configs/optimization.yaml"),
        help="Optimization YAML (default: configs/optimization.yaml)",
    )
    parser.add_argument("--study-name", help="Override study.name")
    parser.add_argument("--output-dir", type=Path, help="Override study.output_dir")
    parser.add_argument("--n-trials", type=int, help="Number of additional trials")
    parser.add_argument("--n-jobs", type=int, help="Parallel trial jobs")
    parser.add_argument("--population-size", type=int, help="NSGA-II population size")
    parser.add_argument("--sampler-seed", type=int, help="NSGA-II sampler seed")
    parser.add_argument(
        "--trial-workers", type=int, help="Reconstruction workers per trial"
    )
    parser.add_argument(
        "--all-input-files",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use all matching ROOT inputs instead of the first file",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Export an existing study without new trials",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Show tracebacks on failure"
    )


def _add_toy_options(parser):
    parser.add_argument(
        "--profile",
        type=Path,
        default=None,
        help="Aggregate toy profile JSON (default: the one shipped with pvfit)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("input/toy_sample.root"),
        help="ROOT file to write; refuses to overwrite (default: input/toy_sample.root)",
    )
    parser.add_argument(
        "--events", type=_positive_int, default=10, help="Events to generate"
    )
    parser.add_argument("--seed", type=int, default=20260922, help="Random seed")
    parser.add_argument(
        "--with-histograms",
        action="store_true",
        help="Also store PVfit-made histogram and seed trees in the file",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("configs/reconstruction.yaml"),
        help="Reconstruction YAML giving the histogram binning for --with-histograms",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Show tracebacks on failure"
    )


def build_parser():
    parser = argparse.ArgumentParser(
        prog="pvfit",
        description="Reconstruct primary vertices and evaluate HDF5 reconstruction results.",
        allow_abbrev=False,
    )
    parser.add_argument("--version", action="version", version=f"PVFit {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("fit", "Reconstruct vertices from ROOT input"),
        ("analyze", "Analyze existing HDF5 results without running reconstruction"),
        ("run", "Run reconstruction followed by analysis"),
        ("optimize", "Run an NSGA-II parameter study over a reconstruction backend"),
        ("toy", "Generate a synthetic ROOT sample that needs no reference data"),
    ):
        command = commands.add_parser(
            name, help=help_text, description=help_text, allow_abbrev=False
        )
        if name == "optimize":
            _add_optimize_options(command)
        elif name == "toy":
            _add_toy_options(command)
        else:
            _add_options(command, name)
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def _prepare(args):
    config = load_config(args.config_file) if args.config_file else {}
    fit_options = analysis_options = None

    if args.command in ("fit", "run"):
        # Command-line values override the YAML; the dataclass validates both.
        overrides = {
            name: value
            for name, value in (
                ("input_root", args.input and str(args.input)),
                ("max_workers", args.max_workers),
                ("event_number", args.event_number),
                ("output_dir", args.output_dir and str(args.output_dir)),
            )
            if value is not None
        }
        cfg = replace(ReconstructionConfig.from_mapping(config), **overrides)
        inputs = sorted(glob.glob(str(Path(cfg.input_root).expanduser())))
        if not inputs:
            raise ValueError(f"No input ROOT files matched: {cfg.input_root}")
        fit_options = dict(
            input_files=inputs,
            track_tree_name=cfg.track_tree_name,
            mc_vertex_tree_name=cfg.mc_vertex_tree_name,
            params=cfg.as_params(),
            event_number=cfg.event_number,
            max_workers=cfg.max_workers,
            output_dir=str(Path(cfg.output_dir).expanduser()),
            hdf5_file=cfg.hdf5_file,
        )

    if args.command in ("analyze", "run"):
        analysis_config_path = (
            args.analysis_config if args.command == "run" else args.config_file
        )
        analysis_config = (
            load_config(analysis_config_path) if args.command == "run" else config
        )
        if analysis_config_path and "analysis" not in analysis_config:
            raise ValueError(
                "Analysis configuration requires an 'analysis' mapping; use a separate analysis YAML"
            )
        settings = AnalysisConfig.from_mapping(analysis_config)
        if args.analyses is not None:
            settings = replace(settings, analyses=tuple(args.analyses))
        if fit_options:
            input_hdf5 = Path(fit_options["output_dir"]) / fit_options["hdf5_file"]
            analysis_dir = Path(fit_options["output_dir"]) / "analysis"
        else:
            input_hdf5 = args.input or settings.input_hdf5
            if input_hdf5 is None:
                raise ValueError(
                    "analyze requires --input FILE.h5 or paths.input_hdf5 in the analysis YAML"
                )
            input_hdf5 = Path(input_hdf5).expanduser()
            if not input_hdf5.is_file():
                raise ValueError(f"HDF5 input does not exist: {input_hdf5}")
            analysis_dir = args.output_dir or settings.output_dir or "output/analysis"
        analysis_options = dict(
            hdf5_file=str(input_hdf5),
            output_dir=str(Path(analysis_dir).expanduser()),
            settings=settings,
        )
    return fit_options, analysis_options


def _run_fit(options):
    from .fit import run_fit

    return run_fit(**options)


def _run_analysis(options):
    from .analysis import run_analysis

    return run_analysis(**options)


def _run_toy(args):
    import json

    from .toy import default_profile, generate_sample

    profile = (
        json.loads(Path(args.profile).read_text())
        if args.profile
        else default_profile()
    )
    params = None
    if args.with_histograms:
        params = ReconstructionConfig.from_yaml(args.config).as_params()
    summary = generate_sample(profile, args.output, args.events, args.seed, params)
    print(json.dumps(summary, indent=2))


def _run_optimization(args):
    from .optimize.run import run_optimization

    overrides = {key: value for key, value in vars(args).items() if key != "config"}
    return run_optimization(args.config, overrides)


def main(argv=None):
    """Run the CLI; return 0 on success, 2 for bad inputs, or 1 on execution failure."""
    args = parse_args(argv)
    try:
        if args.command == "optimize":
            _run_optimization(args)
            return 0
        if args.command == "toy":
            _run_toy(args)
            return 0
        fit_options, analysis_options = _prepare(args)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        if args.debug:
            raise
        detail = (
            f"Missing configuration key: {exc}"
            if isinstance(exc, KeyError)
            else str(exc)
        )
        print(f"pvfit: error: {detail}", file=sys.stderr)
        return 2
    try:
        if fit_options:
            output = _run_fit(fit_options)
            print(f"Fitting complete: {output}")
            if analysis_options:
                analysis_options["hdf5_file"] = str(output)
        if analysis_options:
            output_dir = Path(analysis_options["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            done = output_dir / "analysis.done"
            done.unlink(missing_ok=True)
            _run_analysis(analysis_options)
            done.touch()
            print(f"Analysis complete: {output_dir}")
        return 0
    except KeyboardInterrupt:
        print("pvfit: interrupted", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        if args.debug:
            raise
        print(f"pvfit: error: {exc} (use --debug for a traceback)", file=sys.stderr)
        return 1
