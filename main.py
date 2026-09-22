#!/usr/bin/env python3
"""
main.py - PVfit main entry point (clean version)

Responsibilities:
  - load algorithm config
  - run fitting OR analysis
  - NEVER decide paths or glob files
"""

import argparse
import glob
import multiprocessing as mp
from pathlib import Path

from src.analysis import run_analysis

# ===============================
# Config
# ===============================
from src.config_loader import get_parameters, load_config, print_parameters

# ===============================
# FIT / ANALYSIS
# ===============================
from src.fit import run_fit


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser("PVfit")

    parser.add_argument(
        "--mode",
        "-m",
        choices=["fit", "analysis", "all"],
        required=True,
        help="Execution mode",
    )
    parser.add_argument(
        "--config-file",
        "-c",
        default="config.yaml",
        help="YAML configuration file",
    )
    # Optional overrides
    parser.add_argument(
        "--max-workers",
        "-w",
        type=int,
        default=None,
        help="Override number of parallel workers (default: None)",
    )
    parser.add_argument(
        "--event-number",
        "-e",
        type=int,
        default=None,
        help="Specific event number to process (-1 = all, default: None)",
    )
    parser.add_argument(
        "--analyses",
        "-a",
        nargs="+",
        default=None,
        help="Analyses to run in analysis mode (override config) (default: None)",
    )
    return parser.parse_args()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    args = parse_args()

    # Load config
    config = load_config(args.config_file)
    params = get_parameters(config)
    paths = config["paths"]
    workflow = config.get("workflow", {})
    print_parameters(params)

    output_hdf5 = f"{paths['output_dir']}/{paths['hdf5_file']}"

    # --------------------------------------------------------
    # FIT
    # --------------------------------------------------------
    if args.mode in ("fit", "all"):
        max_workers = (
            args.max_workers
            if args.max_workers is not None
            else workflow.get("max_workers", int(mp.cpu_count() / 2))
        )

        event_number = (
            args.event_number
            if args.event_number is not None
            else workflow.get("event_number", -1)
        )

        run_fit(
            input_files=glob.glob(paths["input_root"]),
            track_tree_name=paths["track_tree_name"],
            mc_vertex_tree_name=paths["mc_vertex_tree_name"],
            params=params,
            event_number=event_number,
            max_workers=max_workers,
            output_dir=paths["output_dir"],
            hdf5_file=paths["hdf5_file"],
        )

        print(f"\nFitting complete → {output_hdf5}")

        if args.mode == "fit":
            return

    # --------------------------------------------------------
    # ANALYSIS
    # --------------------------------------------------------
    if args.mode in ("analysis", "all"):
        analyses = (
            args.analyses
            if args.analyses is not None
            else workflow.get(
                "analyses",
                [
                    "efficiency_track",
                    "efficiency_geo",
                    "fake_rate_track",
                    "fake_rate_geo",
                    "merge_rate",
                    "split_rate",
                ],
            )
        )
        run_analysis(
            hdf5_file=output_hdf5,
            output_dir=f"{paths['output_dir']}/analysis",
            params=params,
            analyses=analyses,
        )

        done_file = Path(f"{paths['output_dir']}/analysis/analysis.done")
        done_file.touch()

        print(f"\nAnalysis complete → {done_file}")
        return


if __name__ == "__main__":
    main()
