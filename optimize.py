#!/usr/bin/env python3
"""
optimize.py — Optuna NSGA-II multi-objective parameter optimization for PVFit.

Thin CLI wrapper. Core logic lives in src/optimize/.

Usage:
    uv run python optimize.py --n-trials 200 --n-jobs 3 --trial-workers 20 \\
        --study-name pvfit_nsga2_v1

    uv run python optimize.py --n-trials 3 --n-jobs 1 --trial-workers 1 \\
        --study-name pvfit_nsga2_v1

    uv run python optimize.py --export-only --study-name pvfit_nsga2_v1
"""

import argparse
from pathlib import Path

import optuna
import yaml

from src.optimize.constants import SAMPLER_SEED
from src.optimize.export import export_results
from src.optimize.trial import get_study_dir, make_objective, resolve_trial_input_root

# ── Paths (CLI-level) ───────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
BASE_CONFIG = ROOT / "config.yaml"
OPTUNA_ROOT = ROOT / "output/optuna"


def parse_args():
    p = argparse.ArgumentParser(
        description="PVFit Optuna NSGA-II parameter optimization"
    )
    p.add_argument(
        "--study-name",
        default="pvfit_nsga2_v1",
        help="Optuna study name — must stay fixed across resumed runs (default: pvfit_nsga2_v1)",
    )
    p.add_argument("--n-trials", type=int, default=200, help="Number of trials to run")
    p.add_argument("--n-jobs", type=int, default=3, help="Parallel trial jobs")
    p.add_argument(
        "--population-size",
        type=int,
        default=50,
        help="NSGA-II population size (generation size). Must stay fixed within a study",
    )
    p.add_argument(
        "--trial-workers",
        type=int,
        default=20,
        help="workflow.max_workers per trial (total CPU ≈ n_jobs × trial_workers)",
    )
    p.add_argument(
        "--export-only",
        action="store_true",
        help="Re-export results from existing DB without running new trials",
    )
    p.add_argument(
        "--all-input-files",
        action="store_true",
        help="Use the full input_root glob from config.yaml instead of only the first matched ROOT file",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not args.export_only and args.n_trials < 1:
        raise ValueError("--n-trials must be >= 1 unless --export-only is used")
    if args.n_jobs < 1:
        raise ValueError("--n-jobs must be >= 1")
    if args.population_size < 2:
        raise ValueError("--population-size must be >= 2")
    if args.trial_workers < 1:
        raise ValueError("--trial-workers must be >= 1")

    OPTUNA_ROOT.mkdir(parents=True, exist_ok=True)
    study_dir = get_study_dir(args.study_name, OPTUNA_ROOT)
    study_dir.mkdir(parents=True, exist_ok=True)

    with open(BASE_CONFIG) as f:
        base_cfg = yaml.safe_load(f)
    trial_input_root = resolve_trial_input_root(
        base_cfg, use_first_input_file=not args.all_input_files, root=ROOT
    )

    db_path = study_dir / "optuna_study.db"
    storage = f"sqlite:///{db_path}"
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        study_name=args.study_name,
        sampler=optuna.samplers.NSGAIISampler(
            seed=SAMPLER_SEED,
            population_size=args.population_size,
        ),
        directions=["maximize", "minimize", "minimize", "minimize"],
        storage=storage,
        load_if_exists=True,
    )

    existing_population_size = study.user_attrs.get("population_size")
    if existing_population_size is None:
        study.set_user_attr("population_size", args.population_size)
    elif existing_population_size != args.population_size:
        raise ValueError(
            f"Study '{args.study_name}' was created with population_size="
            f"{existing_population_size}, but this run requested {args.population_size}."
        )

    existing_sampler_seed = study.user_attrs.get("sampler_seed")
    if existing_sampler_seed is None:
        study.set_user_attr("sampler_seed", SAMPLER_SEED)
    elif existing_sampler_seed != SAMPLER_SEED:
        raise ValueError(
            f"Study '{args.study_name}' uses sampler_seed={existing_sampler_seed}, "
            f"but this script is configured with {SAMPLER_SEED}."
        )

    if args.export_only:
        export_results(study, study_dir)
        return

    n_complete = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    )
    print(
        f"Study '{args.study_name}': {n_complete} completed trials already in DB.\n"
        f"Running {args.n_trials} more trial(s) "
        f"(n_jobs={args.n_jobs}, population_size={args.population_size}, "
        f"trial_workers={args.trial_workers}, "
        f"total CPU ≈ {args.n_jobs * args.trial_workers}).\n"
        f"Trial input_root: {trial_input_root}\n"
        f"Study output_dir: {study_dir}"
    )

    study.optimize(
        make_objective(base_cfg, args.trial_workers, trial_input_root, study_dir, ROOT),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        catch=(RuntimeError, FileNotFoundError, KeyError, ValueError, StopIteration),
    )

    export_results(study, study_dir)


if __name__ == "__main__":
    main()
