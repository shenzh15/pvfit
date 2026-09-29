# Parameter optimization

[Back to README](../README.md)

`pvfit optimize` runs a multi-objective scan with Optuna's NSGA-II sampler over
four objectives:

| Objective | Direction | Denominator |
| --- | --- | --- |
| Efficiency | maximize | true vertices |
| Fake rate | minimize | rec vertices |
| Merge rate | minimize | true vertices |
| Split rate | minimize | rec vertices |

[configs/optimization.yaml](../configs/optimization.yaml) defines the study
name/output, trial count, parallel jobs, sampler seed, population size, workers
per trial, and the search ranges.
The `execution.reconstruction_config` and `execution.analysis_config` references
are resolved relative to the optimization YAML's directory. Data and output
paths remain relative to the working directory. CLI options override YAML
settings.

## Search space

Eight parameters are varied together: `minTracks`, `smear_window_size`, `bins_z`,
`bins_t`, `max_chi2`, `nsigma_merge`, `max_track_bl_chi2`, `pp_max_track_z0_err`.
Their bounds, steps and types are in `search_space`.

## Input selection

The optimizer defaults to the **first file in sorted glob order** from the
reconstruction configuration's `paths.input_root`. To use every matching file,
set `execution.all_input_files: true` in the optimization YAML or pass
`--all-input-files`. This differs from `fit` and `run`, which read all matches.

## Commands

Set the reconstruction YAML's `paths.input_root` to an existing sample first;
the toy quick start's `--input` override does not change that YAML.

```bash
# use the study settings and search ranges from YAML
uv run pvfit optimize --config configs/optimization.yaml

# smoke test
uv run pvfit optimize --config configs/optimization.yaml \
  --n-trials 3 --n-jobs 1 --trial-workers 1 --study-name pvfit_smoke

# include all matching ROOT files in a separate study
uv run pvfit optimize --config configs/optimization.yaml \
  --all-input-files --study-name pvfit_all_inputs

# re-export the Pareto front from an existing study, no new trials
uv run pvfit optimize --config configs/optimization.yaml --export-only
```

## Resuming a study

The same study name continues the same SQLite study and keeps completed trials.
The study records the backend, reconstruction and analysis settings, search
space, working directory, selected input paths and each selected file's size
and modification time. It also records the installed package version and a
digest of the `pvfit` Python source and JSON files.

Changing these recorded values requires a new study name; the error names what
changed. With all inputs enabled, adding a matching file changes the fingerprint.
With the default first-file selection, only a change to the selected file or
its identity does so; adding a later-sorting file leaves the selection unchanged.
Changes to Python source include comments and docstrings. Markdown documentation
outside `pvfit/` does not affect the digest.

The sampler seed and population size must also stay fixed. You can change the
additional trial count or worker/job counts. Studies created before the stage
configuration split or input/code fingerprinting support export only.

> Optuna preserves the study state across a resume but **not** the NSGA-II
> sampler RNG state, so a resumed run can re-suggest a point it already sampled.
> `pvfit optimize` detects exact duplicate parameter signatures and reuses the
> stored objective values instead of re-running the pipeline. That reuse is safe
> only because the study's inputs and code are pinned as described above.

## Results

Results land in `output/optuna/<study_name>/`, with `all_trials.csv`,
`pareto_front.csv`, `generation_summary.csv` and a `trial_N/` directory per trial
holding the backend's configuration, `analysis.yaml`, logs, parsed `metrics.json`
and analysis CSVs. Each trial changes only the reconstruction parameters. The
analysis configuration retains its thresholds and runs the four required metric
modules.

Use [analysis_optuna.ipynb](../notebooks/analysis_optuna.ipynb) to inspect these
results. It builds on the helpers in `pvfit/optimize/analysis.py`. Notebook outputs
are stripped in the repository; run the cells yourself.

## Adding a backend

`execution.backend` names the reconstruction program each trial runs. A backend
(`pvfit/optimize/backends/`) turns the sampled parameters into that program's
configuration and produces an HDF5 file in the
[PVFit layout](data-format.md#hdf5-layout). The trial scores it with
`pvfit analyze`, so every backend uses the same evaluation code.
`pvfit` is the only backend today. Adding one (for example a GPU reconstruction)
means implementing the `Backend` methods in `backends/base.py` — the parameter
names in `search_space` and any derived values are the backend's to define —
and registering it in `backends/__init__.py`. A backend can also cap how many
trials run at once, which `n_jobs` must respect.
