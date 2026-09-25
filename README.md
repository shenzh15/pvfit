# PVFit

Primary vertex (PV) reconstruction and performance analysis in **4D** — the three
spatial coordinates plus track time — for LHCb Upgrade-II style studies.

PVFit provides a Python environment to develop 4D primary vertex reconstruction
algorithms and study their physics performance before porting them to GPUs.
It combines reconstruction, parameter optimization and performance analysis,
including efficiency, fake, merge and split rates, resolution, and pulls.

## How it works

The pipeline has two stages joined by a single HDF5 file:

1. **Fit** — read tracks and MC truth vertices from ROOT, fill a 2D `(z, t)`
   histogram of track-beamline intersections, find seeds as histogram peaks,
   fit each seed with an adaptive Tukey-weighted vertex fit, optionally merge
   close vertices, and write everything to HDF5.
2. **Analysis** — read that HDF5 and produce performance plots (PDF) and
   summary tables (CSV).

Splitting at HDF5 means a reconstruction pass is done once and any number of
analyses can be re-run against it without re-fitting.

## Requirements

- Python ≥ 3.12
- [`uv`](https://docs.astral.sh/uv/) for environment management

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
git clone https://github.com/OWNER/pvfit.git
cd pvfit
```

Install the project and its locked dependencies. This creates the `pvfit`
command in the project environment:

```bash
uv sync --locked
uv run pvfit --help
```

Alternatively, install the checkout with `python -m pip install .` in a virtual
environment and invoke `pvfit` directly. `python -m pvfit` is also supported.

## Input data

PVFit reads a ROOT file produced by an Allen monitoring job. **No sample data
ships with this repository** — you need to point `paths.input_root` at your own
file. `paths.input_root` accepts a glob, and multiple files are concatenated with
`event_number` offset so that event ids stay unique.

Two trees are required.

**Track tree** (default `velo_kalman_filter/monitor_tree`), one row per track:

| Branch                                                         | Meaning                                                    |
| -------------------------------------------------------------- | ---------------------------------------------------------- |
| `event_number`                                               | event id, used to group rows                               |
| `beamPOCA_x`, `beamPOCA_y`, `beamPOCA_z`, `beamPOCA_t` | point of closest approach to the beamline                  |
| `beamPOCA_tx`, `beamPOCA_ty`                               | track slopes at the POCA                                   |
| `c00`, `c11`, `c55`                                      | diagonal covariance entries for`x`, `y`, `t`         |
| `chi2`                                                       | track-beamline chi2, used by the beamline chi2 cut         |
| `p`, `pt`, `fromPV`                                      | kinematics / provenance, carried through to the output     |
| `mc_key`                                                     | MC particle key —**required** for truth association |
| `ovtx_x`, `ovtx_y`, `ovtx_z`, `ovtx_t`                 | MC origin vertex of the track                              |

**Truth vertex tree** (default `mc_pv_dumper/MCPV_effective_tree`), one row per
true PV:

| Branch                        | Meaning                                                          |
| ----------------------------- | ---------------------------------------------------------------- |
| `event_number`              | event id                                                         |
| `x`, `y`, `z`, `time` | true vertex position                                             |
| `effective_pv_key`          | unique key of the vertex within the event                        |
| `track_keys`                | variable-length array of the`mc_key`s belonging to this vertex |
| `n_valid_tracks`            | track count from the dumper, stored as`expected_ntracks`       |

### Truth association

A reconstructed track row belongs to a true vertex when its `mc_key` appears in
that vertex's `track_keys`, **within the same event**. Association is purely by
key; track and vertex coordinates never decide membership.

Consequences worth knowing:

- A true vertex enters the performance denominators only if at least **4** of its
  tracks survive the `z`/`t` acceptance selection. `n_valid_tracks` is recorded
  as `expected_ntracks` but is *not* substituted for this selected-track count.
- Missing `mc_key` / `effective_pv_key` / `track_keys` / `n_valid_tracks` raises
  an error. There is no silent fallback.
- A truth particle owned by two effective vertices is an error, not a warning.
- Truth association feeds **only** the truth overlays and the performance
  denominators. It never reaches the histogram, the seed finder or the vertex
  fit, so it cannot change reconstructed objects.

## Running

### Via Snakemake

```bash
uv run snakemake -n      # show the DAG
uv run snakemake -c 1    # run fit + analysis

# override the two stage configurations
uv run snakemake -c 1 --config \
  reconstruction_config=configs/reconstruction.yaml \
  analysis_config=configs/analysis.yaml
```

### Command-line interface

```bash
# reconstruct, analyze an existing result, or run both stages
uv run pvfit fit --config configs/reconstruction.yaml
uv run pvfit analyze --input output/baseline/fit_results.h5 \
  --config configs/analysis.yaml --output output/reanalysis
uv run pvfit run --config configs/reconstruction.yaml \
  --analysis-config configs/analysis.yaml

# analyze compatible HDF5 output without any reconstruction configuration
uv run pvfit analyze --input /path/to/results.h5 \
  --output output/external --analyses efficiency_track resolution

# one event, one worker
uv run pvfit fit --config configs/reconstruction.yaml --event-number 1 --max-workers 1

uv run pvfit analyze --help
uv run pvfit --version
```

`fit` reads ROOT and writes HDF5. `analyze` reads an existing HDF5 file without
loading or running the fitter. `run` executes both stages. External results must
follow the documented HDF5 layout; this command does not convert arbitrary ROOT
or HDF5 formats automatically.

`--output` / `--output-dir` selects the reconstruction output directory for `fit`
and `run` (analysis goes in its `analysis/` subdirectory). For `analyze`, it is the
actual destination for plots and tables. Standalone analysis accepts `--input`
or `paths.input_hdf5` in its own YAML; its optional `paths.output_dir` is the plot
destination. It does not derive input or settings from a reconstruction config.
Relative data/output paths and CLI config paths are relative to the working
directory. `--config-file` / `-c`, `-w`, `-e` and `-a` remain aliases.

Standalone analysis needs no config unless overriding its defaults. See
[`configs/analysis.yaml`](configs/analysis.yaml). The `analysis:` section holds
matching thresholds, selected analyses and `method_name` for output labels;
`--analyses` overrides that selection. `run` reads the reconstruction YAML via
`--config` and the analysis YAML via `--analysis-config`, checking both before
fitting. The defaults are `configs/reconstruction.yaml` and
`configs/analysis.yaml`, respectively.

Track-pull analysis includes all valid key-associated tracks without a residual
chi-square cut. It has no cut threshold to configure.

Malformed arguments or configuration return exit code 2; execution failures
return 1 and a concise error; successful runs return 0. Use `--debug` after the
subcommand to retain the full traceback. `analysis.done` is written only after
analysis succeeds.

`pvfit` and `python -m pvfit` call the same CLI implementation. The entry point
handles arguments and dispatch; reconstruction and analysis remain independently
callable Python modules.

## Configuration

Each stage owns its configuration:

| File                                                          | Contents                                                                           | Consumer                                             |
| ------------------------------------------------------------- | ---------------------------------------------------------------------------------- | ---------------------------------------------------- |
| [`configs/reconstruction.yaml`](configs/reconstruction.yaml) | ROOT input, reconstruction output, workers/events, histogram and fit parameters    | `pvfit fit`; reconstruction stage of `pvfit run` |
| [`configs/analysis.yaml`](configs/analysis.yaml)             | analysis selection, truth matching, plotting and resolution settings               | `pvfit analyze`; analysis stage of `pvfit run`   |
| [`configs/optimization.yaml`](configs/optimization.yaml)     | study settings, concurrency, search ranges and references to the two stage configs | `pvfit optimize --config ...`                      |

The former root `config.yaml` has been replaced by these files. Put analysis
settings under `analysis:` in the analysis file, not under reconstruction's
`algorithm:` or `workflow:`. No stage merges the two parameter dictionaries.

The following `paths`, `workflow` and `algorithm` sections belong to
`configs/reconstruction.yaml`.

### `paths`

| Key                     | Meaning                                      |
| ----------------------- | -------------------------------------------- |
| `input_root`          | input ROOT file or glob                      |
| `track_tree_name`     | track tree path inside the file              |
| `mc_vertex_tree_name` | truth vertex tree path inside the file       |
| `output_dir`          | output root; acts as the experiment id       |
| `hdf5_file`           | name of the HDF5 written under`output_dir` |

### `workflow`

| Key              | Meaning                         |
| ---------------- | ------------------------------- |
| `event_number` | event to process,`-1` for all |
| `max_workers`  | parallel worker processes       |

### `algorithm`

| Key                      | Default                                | Meaning                                                                 |
| ------------------------ | -------------------------------------- | ----------------------------------------------------------------------- |
| `minTracks`            | `2.5`                                | seed peak threshold —**see the scaling note below**              |
| `smear_window_size`    | `2`                                  | histogram smoothing window                                              |
| `beamline_chi2_cut`    | `true`                               | apply a track-beamline chi2 cut before filling                          |
| `max_track_bl_chi2`    | `10.0`                               | threshold for that cut                                                  |
| `z0_err_cut`           | `true`                               | apply a max`z0` error cut before filling                              |
| `pp_max_track_z0_err`  | `1.5`                                | threshold for that cut, in mm                                           |
| `z_range`, `t_range` | `[-200, 200]` mm, `[-0.6, 0.8]` ns | histogram extent; also the track acceptance                             |
| `bins`                 | `z: 800`, `t: 56`                  | histogram binning                                                       |
| `max_chi2`             | `10`                                 | Tukey weight threshold, and the track-vertex matching cut               |
| `is_adaptive_tukey`    | `true`                               | recompute chi2 each iteration instead of reusing the initial assignment |
| `drop_low_track_seeds` | `true`                               | drop seeds with < 4 tracks and reassign their tracks                    |
| `merge_vertices`       | `true`                               | merge close vertices after fitting                                      |
| `nsigma_merge`         | `5`                                  | merge when`chi2_zt < nsigma_merge²` in `(z, t)`                    |

> **`minTracks` is rescaled, and the value you write is not the value used.**
> The threshold is multiplied by the bin area relative to a reference binning of
> `bin_size_z = 0.5` mm and `bin_size_t = 0.025` ns (that is, `z: 800` bins over
> `[-200, 200]` and `t: 56` bins over `[-0.6, 0.8]`), so that a peak threshold
> stays physically comparable when the binning changes. At the reference binning
> the factor is exactly 1. See `pvfit/config_loader.py`.

Both files are read into a dataclass (`ReconstructionConfig`, `AnalysisConfig` in
`pvfit/config_loader.py`) that holds every default and derived value in one place.
A key the dataclass does not know is an error, not a silent no-op, so typos are
caught before anything runs.

### Analysis settings

These keys belong to `analysis:` in `configs/analysis.yaml`:

| Key                                                                                    | Default                                   | Meaning                                                                                                                                                        |
| -------------------------------------------------------------------------------------- | ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `analyses`                                                                           | list in the shipped YAML                  | analysis modules to run                                                                                                                                        |
| `method_name`                                                                        | `PVfit`                                 | labels and output filename prefix                                                                                                                              |
| `recall_threshold`, `purity_threshold`                                             | `0.7`, `0.7`                          | track-based matching thresholds                                                                                                                                |
| `z_scale`, `t_scale`, `m_r`, `nsigma`                                          | `1.5`, `0.06`, `0.8`, `3.0`       | geo (elliptical) matching in`(z, t)`                                                                                                                         |
| `plot_event_id`                                                                      | `0`                                     | event drawn by the`plot_histogram` analysis — must be an `event_number` that exists in the input, otherwise that analysis prints a warning and is skipped |
| `draw_fixed_ellipses`, `draw_fitted_ellipses`                                      | `false`, `true`                       | overlays on that plot                                                                                                                                          |
| `plot_min_ntracks`, `plot_max_ntracks`, `plot_bin_width_ntracks` | `4`, `200`, `5` | track-multiplicity binning of the performance plots |
| `x_residual_range`, `y_residual_range`, `z_residual_range`, `t_residual_range` | `0.1`, `0.1`, `0.7` mm; `0.06` ns | existing vertex-resolution selection                                                                                                                           |

### Matching methods

Reconstructed and true vertices are paired in one of two ways, and several
analyses ship in both flavours:

- **Track-based** — a match requires both recall and purity of shared tracks to
  exceed their thresholds.
- **Geo-based** — a match is an elliptical cut in `(z, t)`, with
  `sigma_z = z_scale * m_r / 2` and `sigma_t = t_scale * m_r / 2`.

## Output

```
output/<experiment>/
├── fit_results.h5
├── logs/
│   ├── fit.log
│   └── analysis.log
└── analysis/
    ├── analysis.done
    ├── <metric>_<true|rec>_binned_<method>.pdf / .csv   # one per rate result
    ├── <metric>_overall_<true|rec>_binned_<method>.csv
    ├── efficiency_comparison_geo_vs_reso_*.pdf / efficiency_ratio_geo_vs_reso_*.pdf
    ├── resolution_*.pdf
    ├── pull_analysis_input_track.pdf / .png
    ├── pull_analysis_input_track_summary.json
    ├── pull_analysis_fitted_vertex.pdf
    └── histogram_event_*.pdf
```

### HDF5 layout

```
fit_results.h5
├── @n_events
├── @source                        "PVFit" (producer label used by plots)
└── <event_id>/
    ├── @n_tracks, @n_true_vertices, @n_rec_vertices, @n_seedings, @truth_matching
    │
    ├── histogram [n_bins_z, n_bins_t] float64
    │   @z_min, @z_max, @t_min, @t_max, @n_bins_z, @n_bins_t
    │
    ├── seeds/
    │   └── z, t                   [n_seedings] float32  (all seeds, incl. dropped/merged)
    │
    ├── true_vertices/
    │   ├── vertex_id              [n_true] int32
    │   ├── effective_pv_key       [n_true] int64
    │   ├── x, y, z, t             [n_true] float32
    │   ├── expected_ntracks       [n_true] int32
    │   └── track_indices          [n_true] vlen<int32>
    │
    ├── rec_vertices/
    │   ├── x, y, z, t                     [n_rec] float32
    │   ├── x_err, y_err, z_err, t_err     [n_rec] float32
    │   ├── x_seed, y_seed, z_seed, t_seed [n_rec] float32
    │   └── track_indices                  [n_rec] vlen<int32>
    │
    └── tracks/
        ├── track_id               [n_tracks] int32
        ├── x, y, z, t             [n_tracks] float32
        ├── tx, ty                 [n_tracks] float32
        ├── backward               [n_tracks] bool
        ├── cov_xx, cov_yy, cov_tt [n_tracks] float32
        ├── chi2, p, pt            [n_tracks] float32
        ├── fromPV                 [n_tracks] bool
        ├── mc_key                 [n_tracks] int64
        └── ovtx_x, ovtx_y, ovtx_z, ovtx_t [n_tracks] float32
```

`track_indices` are positional indices into the `tracks/` arrays of the same
event, so truth and reconstruction can be compared on identical row ordering.

The `plot_histogram` analysis needs only `histogram` and its attributes; it draws
`seeds/`, `true_vertices/` and `rec_vertices/` (and `z_err`/`t_err` ellipses) when
present, so other producers can write just the parts they have. Files without
`seeds/` fall back to `rec_vertices/{z,t}_seed`.

## Analysis modules

| Analysis key         | Module                          | Description                                              |
| -------------------- | ------------------------------- | -------------------------------------------------------- |
| `efficiency_track` | `analyze_efficiency_track.py` | reconstruction efficiency, track-based matching          |
| `efficiency_geo`   | `analyze_efficiency_geo.py`   | reconstruction efficiency, fixed and resolution-scaled `(z, t)` ellipses |
| `fake_rate_track`  | `analyze_fake_rate_track.py`  | fake vertex rate, track-based matching                   |
| `fake_rate_geo`    | `analyze_fake_rate_geo.py`    | fake vertex rate, fixed and resolution-scaled `(z, t)` ellipses |
| `merge_rate`       | `analyze_merge_rate_track.py` | vertex merge rate                                        |
| `split_rate`       | `analyze_split_rate_track.py` | vertex split rate                                        |
| `merge_rate_geo`   | `analyze_merge_split_geo.py`  | merge rate, true vertices inside one rec ellipse (fixed and resolution-scaled) |
| `split_rate_geo`   | `analyze_merge_split_geo.py`  | split rate, one true vertex inside several rec ellipses (fixed and resolution-scaled) |
| `resolution`       | `analyze_resolution.py`       | position resolution vs track multiplicity                |
| `track_pull`       | `analyze_track_pull.py`       | input track pulls, track vs true vertex                  |
| `vertex_pull`      | `analyze_vertex_pull.py`      | fitted vertex pulls, rec vs true vertex                  |
| `plot_histogram`   | `plot_histogram.py`           | 2D`(z, t)` histogram with vertex overlays              |

The six rate analyses only decide which vertices count as successes and return
`RateResult` objects; `binned_rate.py` holds the shared bins, counters,
Clopper-Pearson errors and the writers `run_analysis` applies to each result.
Every rate is written the same way. `<metric>_<true|rec>_binned_<method>.csv`
has one row per track-multiplicity bin with `bin_start, bin_end, bin_center,
rate, error_lower, error_upper, denominator, numerator`; the matching
`<metric>_overall_<true|rec>_binned_<method>.csv` has one row with `numerator,
denominator, rate, error_lower, error_upper`, the matching parameters and
`method_name`. `<method>` is `<method_name>_Track_Based`, `_Geo_Based` or
`_Reso_Based` after the matching used.

## Input track residual diagnostics

The `track_pull` analysis transports each key-associated track from its POCA to
its matched true vertex's z plane before subtracting x, y, and t. Time transport
uses the forward/backward sign and beta = 1, as in the fitter. The denominators
remain the reported POCA uncertainties; these plots show **scaled residuals**,
not pulls based on a fully propagated covariance.

- `pull_analysis_input_track.pdf` / `.png`: all finite key-associated tracks
  with positive variances, without a residual chi-square cut.
- `pull_analysis_input_track_summary.json`: counts, means, standard deviations,
  and central 68% half-widths under `statistics`. Statistics include all valid
  associated tracks, including tails outside the plotting window; plot titles
  also report the overflow fraction.

The input still has the reconstruction's z/t acceptance and minimum truth-vertex
multiplicity requirements. A width near one does not establish uncertainty
calibration because the analysis uses the reported POCA uncertainties.

New HDF5 results include `tracks/tx`, `tracks/ty`, and `tracks/backward`. For older
files missing these fields, regenerate results before running this analysis:

```bash
uv run pvfit fit --config configs/reconstruction.yaml
uv run pvfit analyze --input output/baseline/fit_results.h5 \
  --config configs/analysis.yaml --analyses track_pull
```

## Parameter optimization

`pvfit optimize` runs a multi-objective scan with Optuna's NSGA-II sampler over
four objectives:

| Objective  | Direction | Denominator   |
| ---------- | --------- | ------------- |
| Efficiency | maximize  | true vertices |
| Fake rate  | minimize  | rec vertices  |
| Merge rate | minimize  | true vertices |
| Split rate | minimize  | rec vertices  |

`configs/optimization.yaml` defines the study name/output, trial count, parallel
jobs, sampler seed, population size, workers per trial, and the search ranges.
The `execution.reconstruction_config` and `execution.analysis_config` references
are resolved relative to the optimization YAML's directory. Data and output
paths remain relative to the working directory. CLI options override YAML
settings.

Eight parameters are varied together: `minTracks`, `smear_window_size`, `bins_z`,
`bins_t`, `max_chi2`, `nsigma_merge`, `max_track_bl_chi2`, `pp_max_track_z0_err`.
Their bounds, steps and types are in `search_space`.

`execution.backend` names the reconstruction program each trial runs. A backend
(`pvfit/optimize/backends/`) turns the sampled parameters into that program's
configuration and produces an HDF5 file in the PVFit layout; the trial then
scores it with `pvfit analyze`, so every backend is evaluated by the same code.
`pvfit` is the only backend today. Adding one (for example a GPU reconstruction)
means implementing the `Backend` methods in `backends/base.py` — the parameter
names in `search_space` and any derived values are the backend's to define —
and registering it in `backends/__init__.py`. A backend can also cap how many
trials run at once, which `n_jobs` must respect.

```bash
# use the study settings and search ranges from YAML
uv run pvfit optimize --config configs/optimization.yaml

# smoke test
uv run pvfit optimize --config configs/optimization.yaml \
  --n-trials 3 --n-jobs 1 --trial-workers 1 --study-name pvfit_smoke

# re-export the Pareto front from an existing study, no new trials
uv run pvfit optimize --config configs/optimization.yaml --export-only
```

Studies are resumable: the same study name continues the same SQLite study and
keeps completed trials. Backend, reconstruction, analysis, input and search-space
settings are recorded with the study; changing them requires a new study name. Studies
created before this configuration split can still be exported, but must not be
resumed for new trials without a new name. Changing the additional trial count
or worker/job counts is allowed.

> Optuna preserves the study state across a resume but **not** the NSGA-II
> sampler RNG state, so a resumed run can re-suggest a point it already sampled.
> `pvfit optimize` detects exact duplicate parameter signatures and reuses the
> stored objective values instead of re-running the pipeline.

Results land in `output/optuna/<study_name>/`, with `all_trials.csv`,
`pareto_front.csv`, `generation_summary.csv` and a `trial_N/` directory per trial
holding the backend's configuration, `analysis.yaml`, logs, parsed `metrics.json`
and analysis CSVs. Each trial changes only the reconstruction parameters. The
analysis configuration retains its thresholds and runs the four required metric
modules.

`notebooks/analysis_optuna.ipynb` is a notebook for inspecting those results; it builds on
the helpers in `pvfit/optimize/analysis.py`. Notebook outputs are stripped in the
repository — run the cells yourself.

## Layout

```
pvfit/
├── pyproject.toml           # package metadata and installed pvfit command
├── notebooks/               # optimization result inspection (analysis_optuna.ipynb)
├── configs/
│   ├── reconstruction.yaml # inputs, reconstruction settings and output
│   ├── analysis.yaml       # independent evaluation and plots
│   └── optimization.yaml   # study, search space and stage config references
├── Snakefile                # fit -> analysis workflow
├── pvfit/
│   ├── cli.py               # fit / analyze / run commands
│   ├── __main__.py          # python -m pvfit forwards to the same CLI
│   ├── config_loader.py     # config parsing and derived parameters
│   ├── fit/                 # ROOT -> seeding -> vertex fit -> HDF5
│   ├── analysis/            # HDF5 -> plots and tables
│   └── optimize/            # trials, backends/, export, post-hoc analysis
├── tests/
└── docs/derivations/        # notebooks behind the algorithm's math
```

## Derivations

`docs/derivations/` records the math the implementation rests on. These are
reference material, not part of the pipeline, and nothing in `pvfit/` imports them.

| File                                 | Topic                                                                |
| ------------------------------------ | -------------------------------------------------------------------- |
| `math_in_track_fit.ipynb`          | residuals and derivatives of the track-vertex chi2                   |
| `tukeyweight_chi2.ipynb`           | Tukey weight behaviour as a function of chi2                         |
| `zbl_uncertainty_derivation.ipynb` | uncertainty on the beamline`z` intersection, two equivalent routes |
| `Gaussian_cdf_approximation.ipynb` | the polynomial Gaussian-CDF approximation used in the seeding        |
| `Error_Of_RMS.md`                  | standard error of an RMS estimate,`sigma / sqrt(2n)`               |

Running them needs the dev dependency group (`sympy`, `ipykernel`):

```bash
uv sync --group dev
```

## Development

```bash
uv run python -m unittest discover -s tests -v   # tests
uv run ruff check .                              # lint
uv run ruff format .                             # format
```

## Acknowledgements

PVFit reads track and truth data from monitoring outputs of
[Allen](https://gitlab.cern.ch/lhcb/Allen), the LHCb GPU trigger framework
(© CERN for the benefit of the LHCb Collaboration, Apache-2.0).

## License

MIT — see [LICENSE](LICENSE).
