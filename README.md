# PVFit

PVFit is a Python environment for developing 4D primary vertex reconstruction
algorithms in three spatial coordinates and time, and evaluating their performance
for LHCb Upgrade-II style studies before porting them to GPUs.

## How it works

The pipeline has two stages joined by a single HDF5 file:

1. **Fit** — read tracks and MC truth vertices from ROOT, fill a 2D `(z, t)`
   histogram of track-beamline intersections, find seeds as histogram peaks,
   fit each seed with an adaptive Tukey-weighted vertex fit, optionally merge
   close vertices, and write the results to HDF5.
2. **Analysis** — read that HDF5 and produce performance plots and summary
   tables: efficiency, fake, merge and split rates, resolution, and pulls.

You can rerun analyses on an existing HDF5 file without repeating the fit.
`pvfit optimize` scans reconstruction parameters with Optuna.

## Installation

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).
Install uv if needed, then clone the project:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
git clone https://github.com/shenzh15/pvfit.git
cd pvfit
```

## Quick start

From the project root, install the locked dependencies, generate ten synthetic
events and run reconstruction plus analysis:

```bash
uv sync --locked
uv run pvfit toy --events 10
uv run pvfit run --config configs/reconstruction.yaml --input input/toy_sample.root \
  --output-dir output/toy --analysis-config configs/analysis.yaml --max-workers 2
```

The results are `output/toy/fit_results.h5` and plots and tables under
`output/toy/analysis/`. The toy generator refuses to overwrite an existing
sample; choose another `--output` and pass it to `run --input` when repeating it.
Toy results demonstrate the workflow; they are not detector performance.
See the [toy model and limitations](docs/toy.md).

## Using your own data

PVFit reads Allen monitoring ROOT files with a track tree and an MC truth vertex
tree; the defaults are `velo_kalman_filter/monitor_tree` and
`mc_pv_dumper/MCPV_effective_tree`.
Both trees must carry `run_number` and `event_number`, and truth association
requires track `mc_key` values and the truth vertex membership keys.
Pass your file or a quoted glob to `--input`, or set `paths.input_root` in
[configs/reconstruction.yaml](configs/reconstruction.yaml); adjust the independent
[analysis configuration](configs/analysis.yaml) for evaluation and plots.

## Documentation

- [Usage and configuration](docs/usage.md): CLI commands, Snakemake and YAML settings.
- [Synthetic example](docs/toy.md): generation, calibration and limitations.
- [Analysis](docs/analysis.md): matching, metric definitions, output files and track diagnostics.
- [Optimization](docs/optimization.md): scans, input selection, study resume and backends.
- [Derivations](docs/derivations/README.md): mathematical reference notes and notebooks.

### HDF5 layout

See [data formats](docs/data-format.md) for the complete ROOT input requirements,
event numbering, truth association and HDF5 schema for compatible producers.

## Development

```bash
uv sync --locked --group dev
uv run python -m unittest discover -s tests -v
uv run ruff check .
uv run ruff format .
```

The implementation lives in `pvfit/fit/`, `pvfit/analysis/`, `pvfit/toy/` and
`pvfit/optimize/`; `pvfit/cli.py` provides the command-line entry point.

## Acknowledgements

PVFit reads track and truth data from monitoring outputs of
[Allen](https://gitlab.cern.ch/lhcb/Allen), the LHCb GPU trigger framework
(© CERN for the benefit of the LHCb Collaboration, Apache-2.0).

## License

MIT — see [LICENSE](LICENSE).
