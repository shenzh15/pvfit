# Changelog

## [0.1.0] - 2026-09-29

Initial versioned release of the public PVFit workflow.

- Reconstruct 4D primary vertices, analyze compatible HDF5 files independently,
  and run Optuna parameter studies.
- Identify input events by `(run_number, event_number)` and associate truth
  tracks by event-local keys.
- Provide a synthetic ROOT generator and a frozen synthetic reconstruction
  regression with exact membership checks and numerical tolerances.
- Report residual standard deviations, bias and statistical uncertainties;
  let input-track diagnostics inherit reconstruction's chi-square threshold.
- Use LHCb2 rate plots and export true/reconstructed multiplicity distributions
  separately from the efficiency numerators and denominators.
- Record effective CLI configurations, input identities, source and dependency
  versions, and stage status; mark newly written HDF5 files as schema version 1.
- Include default YAML configurations in wheels and provide `pvfit init-config`.
  Default reconstruction input is the generated `input/toy_sample.root`.
- Add CI for Python 3.12 and 3.13, distribution installation checks, and tagged
  GitHub releases with wheel and source archives.

The existing geo, track and reso matching/rate definitions and optimization
objectives remain unchanged. See [analysis documentation](docs/analysis.md)
for their definitions and sample-specific validation observations.
