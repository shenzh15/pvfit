"""Export optimization results to CSV files."""

import csv
from pathlib import Path

import optuna

from .constants import AUDIT_SPECS, OBJECTIVE_SPECS


def trial_columns(trials) -> tuple[list[str], list[str]]:
    """Sampled parameter and backend-derived columns, in first-seen order."""
    params = {key: None for t in trials for key in t.params}
    derived = {
        key: None for t in trials for key in t.user_attrs if key.startswith("derived_")
    }
    return list(params), list(derived)


def _trial_to_row(
    t: optuna.trial.FrozenTrial, param_cols: list[str], derived_cols: list[str]
) -> dict:
    """Flatten a trial into the export row dict."""
    row: dict = {
        "trial_number": t.number,
        "state": t.state.name,
    }
    # Objectives (only present for COMPLETE trials)
    if t.values is not None:
        for (_, _, export_col), val in zip(OBJECTIVE_SPECS, t.values):
            row[export_col] = val
    else:
        for _, _, export_col in OBJECTIVE_SPECS:
            row[export_col] = ""

    # Audit metrics from user attrs (may be absent for failed trials)
    for _, _, export_col in AUDIT_SPECS:
        row[export_col] = t.user_attrs.get(export_col, "")

    # Sampled params
    for col in param_cols:
        row[col] = t.params.get(col, "")
    for col in derived_cols:
        row[col] = t.user_attrs.get(col, "")
    row["trial_output_dir"] = t.user_attrs.get("trial_output_dir", "")
    row["trial_input_root"] = t.user_attrs.get("trial_input_root", "")
    row["generation"] = t.user_attrs.get("generation", "")
    row["param_signature"] = t.user_attrs.get("param_signature", "")
    row["duplicate_of_trial_number"] = t.user_attrs.get("duplicate_of_trial_number", "")
    return row


def export_generation_summary(
    study: optuna.Study, header_rows: list[dict], study_dir: Path
) -> None:
    """Write per-generation completeness/uniqueness diagnostics."""
    population_size = study.user_attrs.get("population_size", "")
    generation_rows = [row for row in header_rows if row["state"] == "COMPLETE"]

    grouped: dict[int, list[dict]] = {}
    for row in generation_rows:
        generation = row["generation"]
        if generation in ("", None):
            continue
        grouped.setdefault(int(generation), []).append(row)

    summary_path = study_dir / "generation_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "generation",
                "population_size",
                "n_complete_trials",
                "n_unique_param_signatures",
                "is_full_population",
                "is_unique_population",
                "duplicate_trial_numbers",
            ],
        )
        writer.writeheader()

        for generation in sorted(grouped):
            rows = grouped[generation]
            trial_numbers = [int(r["trial_number"]) for r in rows]
            signatures = [r["param_signature"] for r in rows]
            unique_count = len(set(signatures))
            duplicate_numbers = [
                str(trial_num)
                for trial_num, sig in zip(trial_numbers, signatures, strict=True)
                if signatures.count(sig) > 1
            ]
            writer.writerow(
                {
                    "generation": generation,
                    "population_size": population_size,
                    "n_complete_trials": len(rows),
                    "n_unique_param_signatures": unique_count,
                    "is_full_population": (
                        len(rows) == population_size if population_size != "" else ""
                    ),
                    "is_unique_population": len(rows) == unique_count,
                    "duplicate_trial_numbers": ",".join(duplicate_numbers),
                }
            )
    print(f"Generation summary → {summary_path}")


def export_results(study: optuna.Study, study_dir: Path):
    """Export all trials and Pareto front to CSV files."""
    study_dir.mkdir(parents=True, exist_ok=True)

    all_trials = study.trials
    completed = [t for t in all_trials if t.state == optuna.trial.TrialState.COMPLETE]
    param_cols, derived_cols = trial_columns(all_trials)

    header = (
        ["trial_number", "state"]
        + [c for _, _, c in OBJECTIVE_SPECS]
        + [c for _, _, c in AUDIT_SPECS]
        + param_cols
        + derived_cols
        + [
            "trial_output_dir",
            "trial_input_root",
            "generation",
            "param_signature",
            "duplicate_of_trial_number",
        ]
    )
    all_rows: list[dict] = []

    # all_trials.csv — every trial (all states)
    all_path = study_dir / "all_trials.csv"
    with open(all_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for t in all_trials:
            row = _trial_to_row(t, param_cols, derived_cols)
            all_rows.append(row)
            writer.writerow(row)
    print(f"All trials ({len(all_trials)}, {len(completed)} complete) → {all_path}")

    # pareto_front.csv — best (non-dominated) completed trials
    pareto = study.best_trials
    pareto_path = study_dir / "pareto_front.csv"
    with open(pareto_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for t in pareto:
            writer.writerow(_trial_to_row(t, param_cols, derived_cols))
    print(f"Pareto front ({len(pareto)} trials) → {pareto_path}")

    # Console summary
    print("\n── Pareto Front ──")
    print(f"{'trial':>6}  {'eff':>7}  {'fake':>7}  {'merge':>7}  {'split':>7}")
    for t in sorted(pareto, key=lambda t: -(t.values[0] if t.values else 0)):
        e, f_, m, s = t.values
        print(f"{t.number:>6}  {e:>7.4f}  {f_:>7.4f}  {m:>7.4f}  {s:>7.4f}")

    export_generation_summary(study, all_rows, study_dir)
