"""Reusable analysis functions for Optuna optimization results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .constants import OBJ_COLS, OBJ_DIRS


def compute_pareto_mask(
    df: pd.DataFrame,
    obj_cols: list[str] | None = None,
    obj_dirs: list[str] | None = None,
) -> pd.Series:
    """Return a boolean mask marking the non-dominated rows in df."""
    if obj_cols is None:
        obj_cols = OBJ_COLS
    if obj_dirs is None:
        obj_dirs = OBJ_DIRS

    values = df[obj_cols].to_numpy(dtype=float)
    maximize = [direction == "max" for direction in obj_dirs]
    is_pareto = np.ones(len(df), dtype=bool)

    for i in range(len(values)):
        if not is_pareto[i]:
            continue
        for j in range(len(values)):
            if i == j or not is_pareto[j]:
                continue

            better_or_equal = True
            strictly_better = False
            for k, is_max in enumerate(maximize):
                if is_max:
                    if values[j, k] < values[i, k]:
                        better_or_equal = False
                        break
                    if values[j, k] > values[i, k]:
                        strictly_better = True
                else:
                    if values[j, k] > values[i, k]:
                        better_or_equal = False
                        break
                    if values[j, k] < values[i, k]:
                        strictly_better = True

            if better_or_equal and strictly_better:
                is_pareto[i] = False
                break

    return pd.Series(is_pareto, index=df.index)


def load_study_data(
    study_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw trials, build a unique solution space, and recompute Pareto.

    Returns (df, df_unique, df_pareto_unique):
      - df: all COMPLETE trials, including duplicates
      - df_unique: one representative row per param_signature
      - df_pareto_unique: Pareto subset computed on df_unique only
    """
    study_dir = Path(study_dir)
    df_all = pd.read_csv(study_dir / "all_trials.csv")
    df = df_all[df_all["state"] == "COMPLETE"].copy()
    df = df.sort_values("trial_number").reset_index(drop=True)

    if "param_signature" in df.columns:
        df_unique = df.drop_duplicates(subset=["param_signature"], keep="first").copy()
        pareto_signatures: set[str] | None = None
    else:
        df_unique = df.copy()
        pareto_signatures = None

    df_unique = df_unique.reset_index(drop=True)
    df_unique["is_pareto"] = compute_pareto_mask(df_unique)
    df_unique["is_unique_representative"] = True

    representative_trials = set(df_unique["trial_number"])
    df["is_unique_representative"] = df["trial_number"].isin(representative_trials)

    if "param_signature" in df_unique.columns:
        pareto_signatures = set(
            df_unique.loc[df_unique["is_pareto"], "param_signature"]
        )
        df["is_pareto_unique"] = df["param_signature"].isin(pareto_signatures)
    else:
        pareto_trials = set(df_unique.loc[df_unique["is_pareto"], "trial_number"])
        df["is_pareto_unique"] = df["trial_number"].isin(pareto_trials)

    df_p = df_unique[df_unique["is_pareto"]].copy()
    return df, df_unique, df_p


def normalize_objectives(
    df: pd.DataFrame,
    obj_cols: list[str] | None = None,
    obj_dirs: list[str] | None = None,
    ref_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Normalize objectives to [0, 1] where 1 = best.

    Uses ref_df (or df itself) to determine min/max for scaling.
    """
    if obj_cols is None:
        obj_cols = OBJ_COLS
    if obj_dirs is None:
        obj_dirs = OBJ_DIRS
    if ref_df is None:
        ref_df = df

    result = pd.DataFrame(index=df.index)
    for col, direction in zip(obj_cols, obj_dirs, strict=True):
        vmin, vmax = ref_df[col].min(), ref_df[col].max()
        span = vmax - vmin
        if span == 0:
            result[col] = 0.5
        elif direction == "max":
            result[col] = (df[col] - vmin) / span
        else:
            result[col] = 1 - (df[col] - vmin) / span
    return result


def utopia_distance(
    df: pd.DataFrame,
    obj_cols: list[str] | None = None,
    obj_dirs: list[str] | None = None,
    ref_df: pd.DataFrame | None = None,
) -> pd.Series:
    """Compute Euclidean distance to utopia point (1,1,...,1) in normalized space."""
    normed = normalize_objectives(df, obj_cols, obj_dirs, ref_df)
    cols = obj_cols if obj_cols is not None else OBJ_COLS
    return np.sqrt(((normed[cols] - 1) ** 2).sum(axis=1))


def profile_scan(
    df: pd.DataFrame,
    param_col: str,
    obj_col: str,
    obj_dir: str = "max",
    n_bins: int = 15,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Profile scan: for each bin of param_col, find the best obj_col value.

    Returns (bin_centers, best_values, all_values_per_bin).
    """
    vals = df[param_col].values
    unique_vals = np.sort(df[param_col].unique())

    if len(unique_vals) <= n_bins:
        groups = df.groupby(param_col)
        bin_centers, best_vals, all_vals = [], [], []
        for name, group in sorted(groups):
            bin_centers.append(name)
            if obj_dir == "max":
                best_vals.append(group[obj_col].max())
            else:
                best_vals.append(group[obj_col].min())
            all_vals.append(group[obj_col].values)
        return np.array(bin_centers), np.array(best_vals), all_vals

    bins = np.linspace(vals.min(), vals.max(), n_bins + 1)
    bin_centers, best_vals, all_vals = [], [], []
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (vals >= bins[i]) & (vals < bins[i + 1])
        else:
            mask = (vals >= bins[i]) & (vals <= bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_centers.append((bins[i] + bins[i + 1]) / 2)
        subset = df.loc[mask, obj_col].values
        if obj_dir == "max":
            best_vals.append(subset.max())
        else:
            best_vals.append(subset.min())
        all_vals.append(subset)
    return np.array(bin_centers), np.array(best_vals), all_vals


def pareto_2d(
    df: pd.DataFrame,
    max_col: str,
    min_col: str,
) -> pd.DataFrame:
    """Find the 2D non-dominated front: maximize max_col, minimize min_col."""
    pts = df[[max_col, min_col]].values
    is_pareto = np.ones(len(pts), dtype=bool)
    for i in range(len(pts)):
        if not is_pareto[i]:
            continue
        for j in range(len(pts)):
            if i == j or not is_pareto[j]:
                continue
            # j dominates i if j is at least as good on both and strictly better on one
            if pts[j, 0] >= pts[i, 0] and pts[j, 1] <= pts[i, 1]:
                if pts[j, 0] > pts[i, 0] or pts[j, 1] < pts[i, 1]:
                    is_pareto[i] = False
                    break
    return df[is_pareto].copy()
