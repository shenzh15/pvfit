"""Trial sampling, config generation, execution, metric parsing, and metadata."""

import copy
import csv
import glob
import json
import re
import subprocess
import traceback
from pathlib import Path

import optuna
import yaml

from .constants import (
    AUDIT_SPECS,
    OBJECTIVE_SPECS,
    PARAM_COLS,
    REF_BIN_SIZE_T,
    REF_BIN_SIZE_Z,
    REQUIRED_ANALYSES,
)


# ── Derived m_minTracks ──────────────────────────────────────────────────────
def compute_m_minTracks(
    min_tracks: float, bins_z: int, bins_t: int, cfg: dict
) -> float:
    """Replicate config_loader scaling: minTracks → effective m_minTracks."""
    zmin = cfg["algorithm"]["z_range"]["min"]
    zmax = cfg["algorithm"]["z_range"]["max"]
    tmin = cfg["algorithm"]["t_range"]["min"]
    tmax = cfg["algorithm"]["t_range"]["max"]
    bin_size_z = (zmax - zmin) / bins_z
    bin_size_t = (tmax - tmin) / bins_t
    return min_tracks * (bin_size_z * bin_size_t) / (REF_BIN_SIZE_Z * REF_BIN_SIZE_T)


def resolve_trial_input_root(
    base_cfg: dict, use_first_input_file: bool, root: Path
) -> str:
    """Resolve the input ROOT pattern for trials.

    By default we use the first matching ROOT file for faster exploratory scans.
    Pass --all-input-files to keep the full baseline pattern.
    """
    input_pattern = base_cfg["paths"]["input_root"]
    if not use_first_input_file:
        return input_pattern

    matches = sorted(glob.glob(str(root / input_pattern)))
    if not matches:
        raise FileNotFoundError(
            f"No ROOT files matched input_root pattern: {input_pattern}"
        )

    first_match = Path(matches[0]).resolve()
    try:
        return str(first_match.relative_to(root))
    except ValueError:
        return str(first_match)


def sanitize_study_name(study_name: str) -> str:
    """Convert study_name into a filesystem-safe directory name."""
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", study_name).strip("._")
    return sanitized or "study"


def get_study_dir(study_name: str, optuna_root: Path) -> Path:
    return optuna_root / sanitize_study_name(study_name)


def write_failure_metadata(
    trial_dir: Path,
    trial_number: int,
    stage: str,
    exc: Exception,
    sampled: dict,
    m_min: float,
):
    """Persist structured failure details for a failed trial."""
    failure = {
        "trial_number": trial_number,
        "stage": stage,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "sampled_params": sampled,
        "derived_m_minTracks": m_min,
        "trial_output_dir": str(trial_dir),
        "traceback": traceback.format_exc(),
    }
    (trial_dir / "failure.json").write_text(json.dumps(failure, indent=2))


def param_signature(params: dict) -> str:
    """Create a stable signature for exact-parameter duplicate detection."""
    compact = {key: params.get(key, "") for key in PARAM_COLS}
    return json.dumps(compact, sort_keys=True, separators=(",", ":"))


def generation_attr_key(study: optuna.Study) -> str:
    return f"{study.sampler.__class__.__name__}:generation"


def get_live_trial_generation(trial: optuna.Trial) -> int | None:
    frozen = trial.study._storage.get_trial(trial._trial_id)
    return frozen.system_attrs.get(generation_attr_key(trial.study))


def find_duplicate_completed_trial(
    study: optuna.Study,
    sampled: dict,
) -> optuna.trial.FrozenTrial | None:
    """Return a completed trial with identical params, if any."""
    target_sig = param_signature(sampled)
    for frozen in study.get_trials(
        deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)
    ):
        if param_signature(frozen.params) == target_sig:
            return frozen
    return None


def persist_metrics(
    trial_dir: Path,
    *,
    trial_number: int,
    sampled: dict,
    m_min: float,
    obj_values: dict[str, float],
    audit_values: dict[str, float],
    input_root: str,
    generation: int | None,
    param_sig: str,
    reused_from_trial: int | None,
):
    metrics = {
        "trial_number": trial_number,
        "sampled_params": sampled,
        "derived_m_minTracks": m_min,
        "objectives": obj_values,
        "audit_metrics": audit_values,
        "trial_output_dir": str(trial_dir),
        "trial_input_root": input_root,
        "generation": generation,
        "param_signature": param_sig,
        "reused_from_trial": reused_from_trial,
    }
    (trial_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


def write_reuse_artifacts(
    trial_dir: Path, duplicate_trial: optuna.trial.FrozenTrial
) -> None:
    """Leave breadcrumbs when a trial reuses an already-computed result."""
    (trial_dir / "stdout.log").write_text(
        f"Reused results from completed trial {duplicate_trial.number}.\n"
    )
    (trial_dir / "stderr.log").write_text("")
    (trial_dir / "exit_code.txt").write_text("reused")


# ── Parameter sampling ───────────────────────────────────────────────────────
def sample_params(trial: optuna.Trial) -> dict:
    return {
        "minTracks": trial.suggest_float("minTracks", 1.0, 3.0),
        "smear_window_size": trial.suggest_int("smear_window_size", 1, 4),
        "bins_z": trial.suggest_int("bins_z", 400, 1200, step=40),
        "bins_t": trial.suggest_int("bins_t", 42, 140, step=7),
        "max_chi2": trial.suggest_float("max_chi2", 4.0, 15.0, step=0.5),
        "nsigma_merge": trial.suggest_float("nsigma_merge", 3.0, 8.0, step=0.5),
        "max_track_bl_chi2": trial.suggest_float(
            "max_track_bl_chi2", 5.0, 20.0, step=1.0
        ),
        "pp_max_track_z0_err": trial.suggest_float(
            "pp_max_track_z0_err", 0.5, 3.0, step=0.25
        ),
    }


# ── Trial config writer ─────────────────────────────────────────────────────
def write_trial_config(
    base_cfg: dict,
    sampled: dict,
    trial_dir: Path,
    trial_workers: int,
    input_root: str,
) -> Path:
    """Deep-copy base config, apply all trial overrides, write config.yaml."""
    cfg = copy.deepcopy(base_cfg)
    alg = cfg["algorithm"]

    alg["minTracks"] = sampled["minTracks"]
    alg["smear_window_size"] = sampled["smear_window_size"]
    alg["bins"]["z"] = sampled["bins_z"]
    alg["bins"]["t"] = sampled["bins_t"]
    alg["max_chi2"] = sampled["max_chi2"]
    alg["nsigma_merge"] = sampled["nsigma_merge"]
    alg["max_track_bl_chi2"] = sampled["max_track_bl_chi2"]
    alg["pp_max_track_z0_err"] = sampled["pp_max_track_z0_err"]

    cfg["paths"]["output_dir"] = str(trial_dir)
    cfg["paths"]["input_root"] = input_root
    cfg["workflow"]["analyses"] = REQUIRED_ANALYSES
    cfg["workflow"]["max_workers"] = trial_workers

    trial_dir.mkdir(parents=True, exist_ok=True)
    config_path = trial_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return config_path


# ── CSV reader ───────────────────────────────────────────────────────────────
def read_csv_value(csv_path: Path, column: str) -> float:
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        row = next(reader)
    return float(row[column])


# ── Objective function factory ───────────────────────────────────────────────
def make_objective(
    base_cfg: dict, trial_workers: int, input_root: str, study_dir: Path, root: Path
):
    def objective(trial: optuna.Trial) -> tuple[float, float, float, float]:
        sampled = sample_params(trial)
        trial_dir = study_dir / f"trial_{trial.number}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        config_path = write_trial_config(
            base_cfg, sampled, trial_dir, trial_workers, input_root
        )

        # Compute derived m_minTracks and store as user attr for export
        m_min = compute_m_minTracks(
            sampled["minTracks"], sampled["bins_z"], sampled["bins_t"], base_cfg
        )
        generation = get_live_trial_generation(trial)
        param_sig = param_signature(sampled)
        trial.set_user_attr("derived_m_minTracks", m_min)
        trial.set_user_attr("trial_output_dir", str(trial_dir))
        trial.set_user_attr("trial_input_root", input_root)
        trial.set_user_attr("generation", generation)
        trial.set_user_attr("param_signature", param_sig)

        stdout_path = trial_dir / "stdout.log"
        stderr_path = trial_dir / "stderr.log"
        exit_code_path = trial_dir / "exit_code.txt"

        try:
            duplicate_trial = find_duplicate_completed_trial(trial.study, sampled)
            if duplicate_trial is not None:
                trial.set_user_attr("duplicate_of_trial_number", duplicate_trial.number)

                obj_values = {
                    export_col: val
                    for (_, _, export_col), val in zip(
                        OBJECTIVE_SPECS, duplicate_trial.values
                    )
                }
                audit_values = {
                    export_col: duplicate_trial.user_attrs.get(export_col, "")
                    for _, _, export_col in AUDIT_SPECS
                }
                for key, value in audit_values.items():
                    if value != "":
                        trial.set_user_attr(key, value)

                write_reuse_artifacts(trial_dir, duplicate_trial)
                persist_metrics(
                    trial_dir,
                    trial_number=trial.number,
                    sampled=sampled,
                    m_min=m_min,
                    obj_values=obj_values,
                    audit_values=audit_values,
                    input_root=input_root,
                    generation=generation,
                    param_sig=param_sig,
                    reused_from_trial=duplicate_trial.number,
                )

                print(
                    f"[Trial {trial.number}] reused Trial {duplicate_trial.number} "
                    f"(generation={generation})"
                )
                return tuple(duplicate_trial.values)

            # ── Run main.py ──────────────────────────────────────────────────
            with open(stdout_path, "w") as stdout_f, open(stderr_path, "w") as stderr_f:
                result = subprocess.run(
                    [
                        "uv",
                        "run",
                        "python",
                        "main.py",
                        "--mode",
                        "all",
                        "--config-file",
                        str(config_path),
                    ],
                    stdout=stdout_f,
                    stderr=stderr_f,
                    text=True,
                    cwd=root,
                )

            exit_code_path.write_text(str(result.returncode))

            if result.returncode != 0:
                tail_text = stderr_path.read_text(errors="replace")[-800:]
                raise RuntimeError(
                    f"Trial {trial.number}: main.py exited with rc={result.returncode}\n"
                    f"{tail_text}"
                )

            # ── Parse objectives and audit metrics ───────────────────────────
            analysis_dir = trial_dir / "analysis"
            obj_values: dict[str, float] = {}
            audit_values: dict[str, float] = {}

            for fname, col, export_col in OBJECTIVE_SPECS:
                csv_path = analysis_dir / fname
                if not csv_path.exists():
                    raise RuntimeError(
                        f"Trial {trial.number}: missing output {csv_path}"
                    )
                obj_values[export_col] = read_csv_value(csv_path, col)

            for fname, col, export_col in AUDIT_SPECS:
                csv_path = analysis_dir / fname
                if not csv_path.exists():
                    raise RuntimeError(
                        f"Trial {trial.number}: missing audit output {csv_path}"
                    )
                audit_values[export_col] = read_csv_value(csv_path, col)

            # Store audit metrics as user attrs so they survive to export
            for k, v in audit_values.items():
                trial.set_user_attr(k, v)

            persist_metrics(
                trial_dir,
                trial_number=trial.number,
                sampled=sampled,
                m_min=m_min,
                obj_values=obj_values,
                audit_values=audit_values,
                input_root=input_root,
                generation=generation,
                param_sig=param_sig,
                reused_from_trial=None,
            )

            eff = obj_values["efficiency_true_overall"]
            fake = obj_values["fake_rate_rec_overall"]
            merge = obj_values["merge_rate_true_overall"]
            split = obj_values["split_rate_rec_overall"]

            print(
                f"[Trial {trial.number}] "
                f"eff={eff:.4f}  fake={fake:.4f}  "
                f"merge={merge:.4f}  split={split:.4f}  "
                f"m_minTracks={m_min:.4f}"
            )
            return eff, fake, merge, split
        except (
            RuntimeError,
            FileNotFoundError,
            KeyError,
            ValueError,
            StopIteration,
        ) as exc:
            write_failure_metadata(
                trial_dir=trial_dir,
                trial_number=trial.number,
                stage="objective",
                exc=exc,
                sampled=sampled,
                m_min=m_min,
            )
            print(f"[Trial {trial.number}] FAILED: {exc}")
            raise

    return objective
