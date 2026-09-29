# Usage and configuration

[Back to README](../README.md)

Run the examples from the project root. For your first run, use the
[toy quick start](../README.md#quick-start). The reconstruction YAML
expects `input/toy_sample.root`; set `paths.input_root` or pass `--input` to
`fit` or `run` for your own sample.

Alternatively, install the checkout with `python -m pip install .` in a virtual
environment and invoke `pvfit` directly. `python -m pvfit` uses the same CLI.
Wheels include the same default YAML templates. With no explicit config option,
`fit`, `run` and `optimize` prefer a local `configs/<stage>.yaml`, then use the
shipped template. An explicitly named missing file remains an error. Run
`pvfit init-config` to copy all three templates into `configs/` for editing;
`--output DIRECTORY` chooses another directory, and existing files are never
overwritten. The wheel supports the CLI; use the source archive or checkout for
Snakemake, notebooks and documentation.

## Command-line interface

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
follow the [HDF5 layout](data-format.md#hdf5-layout); this command does not
convert arbitrary ROOT or HDF5 formats automatically.

`--output` / `--output-dir` selects the reconstruction output directory for `fit`
and `run` (analysis goes in its `analysis/` subdirectory). For `analyze`, it is the
actual destination for plots and tables. Standalone analysis accepts `--input`
or `paths.input_hdf5` in its own YAML; its optional `paths.output_dir` is the plot
destination. It does not derive input paths from a reconstruction config.
The input-track chi-square threshold defaults to the value stored with the
reconstruction result (see [track diagnostics](analysis.md#input-track-residual-diagnostics)).
Relative data/output paths and CLI config paths are relative to the working
directory. `--config-file` / `-c`, `-w`, `-e` and `-a` remain aliases.

Standalone analysis needs no config unless overriding its defaults. See
[`configs/analysis.yaml`](../configs/analysis.yaml). The `analysis:` section holds
matching thresholds, selected analyses and `method_name` for output labels;
`--analyses` overrides that selection. `run` reads the reconstruction YAML via
`--config` and the analysis YAML via `--analysis-config`, checking both before
fitting. The defaults are `configs/reconstruction.yaml` and
`configs/analysis.yaml`, respectively.

For `fit`, `analyze` and `run`, malformed arguments or configuration return
exit code 2; execution failures return 1 and a concise error; successful runs
return 0. Use `--debug` after the
subcommand to retain the full traceback. `analysis.done` is written only after
analysis succeeds.

Reconstruction and analysis also remain independently callable Python modules.

## Snakemake

```bash
uv run snakemake -n      # show the DAG
uv run snakemake -c 1    # run fit + analysis

# override the two stage configurations
uv run snakemake -c 1 --config \
  reconstruction_config=configs/reconstruction.yaml \
  analysis_config=configs/analysis.yaml
```

Snakemake reads input paths from the reconstruction YAML. Its default input
matches `pvfit toy`; update the YAML for another sample. Snakemake writes `logs/fit.log` and
`logs/analysis.log` beneath the reconstruction output directory.

## Configuration

The configuration loader validates each value: numbers must be finite,
flags must be real YAML booleans (`true`/`false`, not the strings `"true"` or
`"false"` and not `0`/`1`), thresholds such as `recall_threshold` and
`purity_threshold` must lie in `[0, 1]`, and scales, windows and cuts must be
positive. A value outside these rules stops the run with a message naming the
key rather than silently producing an empty or distorted result.

Each stage owns its configuration:

| File | Contents | Consumer |
| --- | --- | --- |
| [`configs/reconstruction.yaml`](../configs/reconstruction.yaml) | ROOT input, reconstruction output, workers/events, histogram and fit parameters | `pvfit fit`; reconstruction stage of `pvfit run` |
| [`configs/analysis.yaml`](../configs/analysis.yaml) | analysis selection, truth matching, plotting and resolution settings | `pvfit analyze`; analysis stage of `pvfit run` |
| [`configs/optimization.yaml`](../configs/optimization.yaml) | study settings, concurrency, search ranges and references to the two stage configs | `pvfit optimize --config ...` |

The former root `config.yaml` has been replaced by these files. Put analysis
settings under `analysis:` in the analysis file, not under reconstruction's
`algorithm:` or `workflow:`. No stage merges the two parameter dictionaries.

The following `paths`, `workflow` and `algorithm` sections belong to
`configs/reconstruction.yaml`.

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
| `event_number` | internal event ID to process; `-1` selects all events |
| `max_workers` | parallel worker processes |

### `algorithm`

| Key | Shipped value | Meaning |
| --- | --- | --- |
| `minTracks` | `2.5` | seed peak threshold — **see the scaling note below** |
| `smear_window_size` | `2` | histogram smoothing window |
| `beamline_chi2_cut` | `true` | apply a track-beamline chi2 cut before filling |
| `max_track_bl_chi2` | `10.0` | threshold for that cut |
| `z0_err_cut` | `true` | apply a max `z0` error cut before filling |
| `pp_max_track_z0_err` | `1.5` | threshold for that cut, in mm |
| `z_range`, `t_range` | `[-200, 200]` mm, `[-0.8, 0.8]` ns | histogram extent; also the track acceptance |
| `bins` | `z: 800`, `t: 64` | histogram binning |
| `max_chi2` | `10` | Tukey weight threshold, and the track-vertex matching cut |
| `is_adaptive_tukey` | `true` | recompute chi2 each iteration instead of reusing the initial assignment |
| `drop_low_track_seeds` | `true` | drop seeds with < 4 tracks and reassign their tracks |
| `merge_vertices` | `true` | merge close vertices after fitting |
| `nsigma_merge` | `5` | merge when `chi2_zt < nsigma_merge²` in `(z, t)` |

> **PVFit scales `minTracks` with the histogram bin area.**
> It multiplies the configured threshold by the bin area relative to
> `bin_size_z = 0.5` mm and `bin_size_t = 0.025` ns (that is, `z: 800` bins over
> `[-200, 200]` and `t: 64` bins over `[-0.8, 0.8]`), so that a peak threshold
> stays physically comparable when the binning changes. At the reference binning
> the factor is exactly 1. See `pvfit/config_loader.py`.

`ReconstructionConfig` and `AnalysisConfig` in `pvfit/config_loader.py`
hold defaults and derived values. Both reject unknown keys before execution.

### Analysis settings

These keys belong to `analysis:` in `configs/analysis.yaml`:

| Key | Shipped value | Meaning |
| --- | --- | --- |
| `analyses` | list in the shipped YAML | analysis modules to run |
| `method_name` | `PVfit` | labels and output filename prefix |
| `recall_threshold`, `purity_threshold` | `0.7`, `0.7` | track-based matching thresholds |
| `track_pull_chi2` | `null` | follow reconstruction's recorded `max_chi2` (HDF5 metadata, or adjacent `reconstruction.yaml` for older results); a positive number explicitly overrides the input-track diagnostic cut |
| `z_scale`, `t_scale`, `m_r`, `nsigma` | `1.5`, `0.06`, `0.8`, `3.0` | geo (elliptical) matching in `(z, t)` |
| `plot_event_id` | `0` | internal event ID (HDF5 group name) to plot; a missing group skips the plot with a warning |
| `draw_fixed_ellipses`, `draw_fitted_ellipses` | `false`, `true` | overlays on that plot |
| `plot_min_ntracks`, `plot_max_ntracks`, `plot_bin_width_ntracks` | `4`, `200`, `5` | track-multiplicity binning of the performance plots |
| `efficiency_plot_max_ntracks` | `150` | horizontal display limit of all LHCb2 efficiency figures; keeps complete bins and does not change overall rates |
| `rate_plot_max_ntracks` | `150` | horizontal display limit of the LHCb2 fake/merge/split rate figures |
| `rate_plot_ymax` | `{}` | optional per-plot y maxima, e.g. `{merge_rate_geo_rec: 0.25}`; otherwise use central values without expanding for long error bars |
| `x_residual_range`, `y_residual_range`, `z_residual_range`, `t_residual_range` | `0.1`, `0.1`, `0.7` mm; `0.06` ns | half-width of the residual window the `resolution` analysis keeps per coordinate; the reported std and bias describe the core inside it |

Event selections use the [dense internal IDs](data-format.md#event-identity),
not the original ROOT event numbers. See [analysis definitions](analysis.md)
for matching criteria and the interpretation of residual plots.

## Run provenance

Each CLI `fit` and `analyze` stage (including the two stages of `run`) writes a
unique `provenance/<UTC timestamp>_<id>_<stage>/` directory under its output
directory. It contains `manifest.json` and an effective `reconstruction.yaml`
or `analysis.yaml`, including CLI overrides. Reruns retain previous records.
The manifest records the invocation and working directory, input paths/sizes/
modification times, package version and source SHA-256, Git commit and dirty
flag when available, Python/dependency versions, and completion or failure.
Input identities use file metadata, not content checksums of large ROOT files.
A hard process kill may leave the status `running`; this is not completion.

HDF5 files store the schema version, package/source identity, effective fitter
parameters and, for CLI fits, the provenance manifest path. Copy the provenance
directory with the results to retain the full record; absolute recorded paths
describe the original run and may no longer resolve after moving it. Analysis
does not modify its input HDF5. Direct Python calls retain HDF5 writer metadata;
the per-stage manifests are a CLI feature. Optimizer studies retain their
existing configuration and input/source fingerprint checks.

`uv.lock` fixes the tested environment. Package dependency ranges allow
compatible installations (including `mplhep>=1.1.2,<2`); use `uv sync --locked`
to reproduce the locked plotting stack.
