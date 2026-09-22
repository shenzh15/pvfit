# PVFit

Primary vertex (PV) reconstruction and performance analysis in **4D** — the three
spatial coordinates plus track time — for LHCb Upgrade-II style studies.

PVFit re-implements the beamline PV finder in Python so that the algorithm can be
inspected, modified and scanned parameter-by-parameter outside the production
framework, and pairs it with a performance suite (efficiency, fake / merge / split
rates, resolution, pulls) that runs on the same reconstructed output.

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

Dependencies are resolved from `uv.lock` on first `uv run`; there is no separate
install step.

## Input data

PVFit reads a ROOT file produced by an Allen monitoring job. **No sample data
ships with this repository** — you need to point `paths.input_root` at your own
file. `paths.input_root` accepts a glob, and multiple files are concatenated with
`event_number` offset so that event ids stay unique.

Two trees are required.

**Track tree** (default `velo_kalman_filter/monitor_tree`), one row per track:

| Branch | Meaning |
| --- | --- |
| `event_number` | event id, used to group rows |
| `beamPOCA_x`, `beamPOCA_y`, `beamPOCA_z`, `beamPOCA_t` | point of closest approach to the beamline |
| `beamPOCA_tx`, `beamPOCA_ty` | track slopes at the POCA |
| `c00`, `c11`, `c55` | diagonal covariance entries for `x`, `y`, `t` |
| `chi2` | track-beamline chi2, used by the beamline chi2 cut |
| `p`, `pt`, `fromPV` | kinematics / provenance, carried through to the output |
| `mc_key` | MC particle key — **required** for truth association |
| `ovtx_x`, `ovtx_y`, `ovtx_z`, `ovtx_t` | MC origin vertex of the track |

**Truth vertex tree** (default `mc_pv_dumper/MCPV_effective_tree`), one row per
true PV:

| Branch | Meaning |
| --- | --- |
| `event_number` | event id |
| `x`, `y`, `z`, `time` | true vertex position |
| `effective_pv_key` | unique key of the vertex within the event |
| `track_keys` | variable-length array of the `mc_key`s belonging to this vertex |
| `n_valid_tracks` | track count from the dumper, stored as `expected_ntracks` |

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
```

### Via the Python entry point

```bash
uv run python main.py --mode all      --config-file config.yaml
uv run python main.py --mode fit      --config-file config.yaml
uv run python main.py --mode analysis --config-file config.yaml

# run a subset of analyses
uv run python main.py --mode analysis -a efficiency_track resolution vertex_pull

# single event, single worker (useful when debugging)
uv run python main.py --mode fit -e 0 -w 1
```

`--max-workers` / `-w` and `--event-number` / `-e` override the corresponding
`workflow` entries in the config.

### Standalone histogram plot

Plot the stored Allen `(z, t)` histogram for one event with the truth overlay,
without running the pipeline:

```bash
uv run python plot_histograms_2d.py --input /path/to/sample.root \
  --event 1 --output-dir output/plots
```

## Configuration

Everything is driven by `config.yaml`.

### `paths`

| Key | Meaning |
| --- | --- |
| `input_root` | input ROOT file or glob |
| `track_tree_name` | track tree path inside the file |
| `mc_vertex_tree_name` | truth vertex tree path inside the file |
| `output_dir` | output root; acts as the experiment id |
| `hdf5_file` | name of the HDF5 written under `output_dir` |

### `workflow`

| Key | Meaning |
| --- | --- |
| `event_number` | event to process, `-1` for all |
| `max_workers` | parallel worker processes |
| `analyses` | list of analysis keys to run |

### `algorithm`

| Key | Default | Meaning |
| --- | --- | --- |
| `minTracks` | `2.5` | seed peak threshold — **see the scaling note below** |
| `smear_window_size` | `2` | histogram smoothing window |
| `beamline_chi2_cut` | `true` | apply a track-beamline chi2 cut before filling |
| `max_track_bl_chi2` | `10.0` | threshold for that cut |
| `z0_err_cut` | `true` | apply a max `z0` error cut before filling |
| `pp_max_track_z0_err` | `1.5` | threshold for that cut, in mm |
| `z_range`, `t_range` | `[-200, 200]` mm, `[-0.6, 0.8]` ns | histogram extent; also the track acceptance |
| `bins` | `z: 800`, `t: 56` | histogram binning |
| `max_chi2` | `10` | Tukey weight threshold, and the track-vertex matching cut |
| `is_adaptive_tukey` | `true` | recompute chi2 each iteration instead of reusing the initial assignment |
| `drop_low_track_seeds` | `true` | drop seeds with < 4 tracks and reassign their tracks |
| `merge_vertices` | `true` | merge close vertices after fitting |
| `nsigma_merge` | `5` | merge when `chi2_zt < nsigma_merge²` in `(z, t)` |
| `recall_threshold`, `purity_threshold` | `0.7`, `0.7` | track-based matching thresholds |
| `z_scale`, `t_scale`, `m_r`, `nsigma` | `1.5`, `0.06`, `0.8`, `3.0` | geo (elliptical) matching in `(z, t)` |
| `plot_event_id` | `0` | event drawn by the `plot_histogram` analysis — must be an `event_number` that exists in the input, otherwise that analysis prints a warning and is skipped |
| `draw_fixed_ellipses`, `draw_fitted_ellipses` | `false`, `true` | overlays on that plot |

> **`minTracks` is rescaled, and the value you write is not the value used.**
> The threshold is multiplied by the bin area relative to a reference binning of
> `bin_size_z = 0.5` mm and `bin_size_t = 0.025` ns (that is, `z: 800` bins over
> `[-200, 200]` and `t: 56` bins over `[-0.6, 0.8]`), so that a peak threshold
> stays physically comparable when the binning changes. At the reference binning
> the factor is exactly 1. See `src/config_loader.py`.

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
    ├── efficiency_true_binned_*.pdf   / .csv
    ├── fake_rate_rec_binned_*.pdf     / .csv
    ├── merge_rate_true_binned_*.pdf   / .csv
    ├── split_rate_rec_binned_*.pdf    / .csv
    ├── resolution_*.pdf
    ├── pull_analysis_input_track.pdf
    ├── pull_analysis_fitted_vertex.pdf
    └── histogram_event_*.pdf
```

### HDF5 layout

```
fit_results.h5
├── @n_events
└── <event_id>/
    ├── @n_tracks, @n_true_vertices, @n_rec_vertices, @n_seedings, @truth_matching
    │
    ├── histogram [n_bins_z, n_bins_t] float64
    │   @z_min, @z_max, @t_min, @t_max, @n_bins_z, @n_bins_t
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
        ├── cov_xx, cov_yy, cov_tt [n_tracks] float32
        ├── chi2, p, pt            [n_tracks] float32
        ├── fromPV                 [n_tracks] bool
        ├── mc_key                 [n_tracks] int64
        └── ovtx_x, ovtx_y, ovtx_z, ovtx_t [n_tracks] float32
```

`track_indices` are positional indices into the `tracks/` arrays of the same
event, so truth and reconstruction can be compared on identical row ordering.

## Analysis modules

| Analysis key | Module | Description |
| --- | --- | --- |
| `efficiency_track` | `analyze_efficiency_track.py` | reconstruction efficiency, track-based matching |
| `efficiency_geo` | `analyze_efficiency_geo.py` | reconstruction efficiency, elliptical `(z, t)` matching |
| `fake_rate_track` | `analyze_fake_rate_track.py` | fake vertex rate, track-based matching |
| `fake_rate_geo` | `analyze_fake_rate_geo.py` | fake vertex rate, elliptical matching |
| `merge_rate` | `analyze_merge_rate_track.py` | vertex merge rate |
| `split_rate` | `analyze_split_rate_track.py` | vertex split rate |
| `resolution` | `analyze_resolution.py` | position resolution vs track multiplicity |
| `track_pull` | `analyze_track_pull.py` | input track pulls, track vs true vertex |
| `vertex_pull` | `analyze_vertex_pull.py` | fitted vertex pulls, rec vs true vertex |
| `plot_histogram` | `plot_histogram.py` | 2D `(z, t)` histogram with vertex overlays |

## Parameter optimization

`optimize.py` runs a multi-objective scan with Optuna's NSGA-II sampler over four
objectives:

| Objective | Direction | Denominator |
| --- | --- | --- |
| Efficiency | maximize | true vertices |
| Fake rate | minimize | rec vertices |
| Merge rate | minimize | true vertices |
| Split rate | minimize | rec vertices |

Eight parameters are varied together: `minTracks`, `smear_window_size`, `bins_z`,
`bins_t`, `max_chi2`, `nsigma_merge`, `max_track_bl_chi2`, `pp_max_track_z0_err`.

```bash
# full run
uv run python optimize.py --n-trials 200 --n-jobs 3 --trial-workers 20 \
  --study-name pvfit_nsga2_v1

# smoke test
uv run python optimize.py --n-trials 3 --n-jobs 1 --trial-workers 1 \
  --study-name pvfit_nsga2_v1

# re-export the Pareto front from an existing study, no new trials
uv run python optimize.py --export-only --study-name pvfit_nsga2_v1
```

Studies are resumable: the same `--study-name` continues the same SQLite study
and keeps completed trials.

> Optuna preserves the study state across a resume but **not** the NSGA-II
> sampler RNG state, so a resumed run can re-suggest a point it already sampled.
> `optimize.py` detects exact duplicate parameter signatures and reuses the
> stored objective values instead of re-running the pipeline.

Results land in `output/optuna/<study_name>/`, with `all_trials.csv`,
`pareto_front.csv`, `generation_summary.csv` and a `trial_N/` directory per trial
holding its config, logs, parsed `metrics.json` and analysis CSVs.

`analysis_optuna.ipynb` is a notebook for inspecting those results; it builds on
the helpers in `src/optimize/analysis.py`. Notebook outputs are stripped in the
repository — run the cells yourself.

## Layout

```
pvfit/
├── main.py                  # fit / analysis entry point
├── optimize.py              # Optuna NSGA-II CLI
├── plot_histograms_2d.py    # standalone histogram plot
├── analysis_optuna.ipynb    # optimization result inspection
├── config.yaml              # all runtime configuration
├── Snakefile                # fit -> analysis workflow
├── src/
│   ├── config_loader.py     # config parsing and derived parameters
│   ├── fit/                 # ROOT -> seeding -> vertex fit -> HDF5
│   ├── analysis/            # HDF5 -> plots and tables
│   └── optimize/            # trial sampling, export, post-hoc analysis
├── tests/
└── docs/derivations/        # notebooks behind the algorithm's math
```

## Derivations

`docs/derivations/` records the math the implementation rests on. These are
reference material, not part of the pipeline, and nothing in `src/` imports them.

| File | Topic |
| --- | --- |
| `math_in_track_fit.ipynb` | residuals and derivatives of the track-vertex chi2 |
| `tukeyweight_chi2.ipynb` | Tukey weight behaviour as a function of chi2 |
| `zbl_uncertainty_derivation.ipynb` | uncertainty on the beamline `z` intersection, two equivalent routes |
| `Gaussian_cdf_approximation.ipynb` | the polynomial Gaussian-CDF approximation used in the seeding |
| `Error_Of_RMS.md` | standard error of an RMS estimate, `sigma / sqrt(2n)` |

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

The reconstruction follows the beamline PV algorithm developed for
[Allen](https://gitlab.cern.ch/lhcb/Allen), the LHCb GPU trigger framework
(© CERN for the benefit of the LHCb Collaboration, Apache-2.0). This repository
is an independent Python implementation and contains no Allen source code.

## License

MIT — see [LICENSE](LICENSE).
