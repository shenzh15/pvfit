"""Run or export an NSGA-II study; the ``pvfit optimize`` command."""

from pathlib import Path

import optuna

from ..config_loader import AnalysisConfig, load_config
from .backends import get_backend
from .config import (
    check_study_configuration,
    implementation_fingerprint,
    input_fingerprint,
    load_optimization_settings,
)
from .export import export_results
from .trial import get_study_dir, make_objective, resolve_trial_input_root

DIRECTIONS = ["maximize", "minimize", "minimize", "minimize"]


def run_optimization(config_path, overrides):
    """Load settings, then run ``n_trials`` more trials or just export the study."""
    settings = load_optimization_settings(config_path, overrides)
    study_name = settings["study_name"]
    study_dir = get_study_dir(study_name, settings["output_dir"])
    db_path = study_dir / "optuna_study.db"
    storage = f"sqlite:///{db_path}"
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if overrides.get("export_only"):
        if not db_path.is_file():
            raise FileNotFoundError(f"Study database does not exist: {db_path}")
        export_results(
            optuna.load_study(study_name=study_name, storage=storage), study_dir
        )
        return

    backend = get_backend(settings["backend"])
    base_cfg = load_config(settings["reconstruction_config"])
    analysis_cfg = load_config(settings["analysis_config"])
    if "analysis" not in analysis_cfg:
        raise ValueError("Analysis configuration requires an 'analysis' mapping")
    backend.validate_config(base_cfg)
    AnalysisConfig.from_mapping(analysis_cfg)
    root = Path.cwd()
    trial_input_root = resolve_trial_input_root(
        backend.input_pattern(base_cfg),
        use_first_input_file=not settings["all_input_files"],
        root=root,
    )
    study_dir.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=study_name,
        sampler=optuna.samplers.NSGAIISampler(
            seed=settings["sampler_seed"], population_size=settings["population_size"]
        ),
        directions=DIRECTIONS,
        storage=storage,
        load_if_exists=True,
    )
    for key in ("population_size", "sampler_seed"):
        existing = study.user_attrs.get(key)
        if existing is not None and existing != settings[key]:
            raise ValueError(
                f"Study '{study_name}' uses {key}={existing}, but requested {settings[key]}"
            )
        study.set_user_attr(key, settings[key])
    check_study_configuration(
        study,
        {
            "backend": backend.name,
            "reconstruction": base_cfg,
            "analysis": analysis_cfg,
            "input_root": trial_input_root,
            "inputs": input_fingerprint(trial_input_root, root),
            "implementation": implementation_fingerprint(),
            "working_directory": str(root),
            "search_space": settings["search_space"],
        },
    )

    n_complete = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    print(
        f"Study '{study_name}' ({backend.name} backend): "
        f"{n_complete} completed trials already in DB.\n"
        f"Running {settings['n_trials']} more trial(s) "
        f"(n_jobs={settings['n_jobs']}, population_size={settings['population_size']}, "
        f"trial_workers={settings['trial_workers']}).\n"
        f"Trial input_root: {trial_input_root}\nStudy output_dir: {study_dir}"
    )
    study.optimize(
        make_objective(
            backend,
            base_cfg,
            settings["trial_workers"],
            trial_input_root,
            study_dir,
            root,
            analysis_cfg=analysis_cfg,
            search_space=settings["search_space"],
        ),
        n_trials=settings["n_trials"],
        n_jobs=settings["n_jobs"],
        catch=(RuntimeError, FileNotFoundError, KeyError, ValueError, StopIteration),
    )
    export_results(study, study_dir)
