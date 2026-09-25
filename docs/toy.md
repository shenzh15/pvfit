# Synthetic example

Runs the whole pipeline without the reference ROOT sample or any private
filesystem. From the project root:

```bash
uv sync --locked
uv run pvfit toy --events 10                        # writes input/toy_sample.root
uv run pvfit run --config configs/reconstruction.yaml --input input/toy_sample.root \
  --output-dir output/toy --analysis-config configs/analysis.yaml   # writes output/toy/
```

`pvfit toy` refuses to overwrite an existing file: pass another `--output` and
give that path to `pvfit run --input`. `--seed` and
`--events` choose the sample; the default seed is `20260922` and event numbers
start at zero.

## Model

`pvfit/toy/profile.json` holds aggregate statistics fitted to a 100-event reference
sample by `pvfit.toy.calibrate_profile`: quantiles and correlations only, no
event records. All vertices, tracks and keys are generated afresh from the seed.

- A correlated Gaussian describes the luminous region in x, y, z and t.
- Negative-binomial counts describe vertices per event and tracks per vertex.
- Separate forward/backward and effective-PV/background populations describe
  momentum, slope magnitude, state errors, fit chi-square and transported
  scaled residuals, each split into six momentum bands (below 0.5, 0.5–1, 1–2,
  2–5, 5–10 and above 10 GeV) with its own Gaussian rank copula, so the
  momentum dependence of the residual width survives. Tabulated quantiles
  describe the continuous marginals; time errors keep their discrete
  resolution levels.
- Measured x/y/t are sampled at the track origin plane and transported along
  the same straight line to the closest approach to the beamline, with the
  fitter's forward/backward sign and speed of light.
- Track rows are shuffled and carry explicit event-local `mc_key`,
  `effective_pv_key` and `track_keys`. Background means absent from the
  effective-PV lists, not necessarily a fake detector track.

`toy_metadata` in the ROOT file identifies it as `synthetic_parametric_toy`
and records the model and seed.

## Scope and limitations

A demonstration and regression-testing sample, not detector simulation: no
hits, decay chains, material or full inter-track correlations; uniform
azimuth, equal x/y state uncertainties, a fixed small slope variance and no
slope-position covariance; residual tails clipped at each band's 0.1% and
99.9% quantiles. Toy efficiencies and resolutions must not be quoted as
detector performance.

The bundled profile uses schema version 2; other profiles fail explicitly. To
refit one from another reference sample, call
`pvfit.toy.calibrate_profile(path_to_reference_root)` and write the returned
dict as JSON; every nonempty momentum band needs at least ten tracks.
