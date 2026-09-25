"""Shared constants for optimization and analysis."""

# ── Analyses to run per trial (skip resolution/histogram/pull to save time) ───
REQUIRED_ANALYSES = [
    "efficiency_track",
    "fake_rate_track",
    "merge_rate",
    "split_rate",
]

# ── Objective CSV specs ──────────────────────────────────────────────────────
# Each tuple: (filename, column_name, export_column_name); every overall CSV
# has the same columns (see analysis/binned_rate.py), so the column is 'rate'.
OBJECTIVE_SPECS = [
    (
        "efficiency_overall_true_binned_pvfit_track_based.csv",
        "rate",
        "efficiency_true_overall",
    ),
    (
        "fake_rate_overall_rec_binned_pvfit_track_based.csv",
        "rate",
        "fake_rate_rec_overall",
    ),
    (
        "merge_rate_overall_true_binned_pvfit_track_based.csv",
        "rate",
        "merge_rate_true_overall",
    ),
    (
        "split_rate_overall_rec_binned_pvfit_track_based.csv",
        "rate",
        "split_rate_rec_overall",
    ),
]

# ── Audit (companion) metrics — not optimized, but exported for auditing ─────
AUDIT_SPECS = [
    (
        "merge_rate_overall_rec_binned_pvfit_track_based.csv",
        "rate",
        "merge_rate_rec_overall",
    ),
    (
        "split_rate_overall_true_binned_pvfit_track_based.csv",
        "rate",
        "split_rate_true_overall",
    ),
]

# ── Derived from OBJECTIVE_SPECS — used by both optimize.py and notebooks ────
OBJ_COLS = [spec[2] for spec in OBJECTIVE_SPECS]
OBJ_LABELS = [
    "Efficiency (true)",
    "Fake rate (rec)",
    "Merge rate (true)",
    "Split rate (rec)",
]
OBJ_DIRS = ["max", "min", "min", "min"]

# Baseline from output/baseline/analysis
BASELINE = {
    "efficiency_true_overall": 0.7966079359013217,
    "fake_rate_rec_overall": 0.015221333650444452,
    "merge_rate_true_overall": 0.03328808824110802,
    "split_rate_rec_overall": 0.003983708416327259,
}
