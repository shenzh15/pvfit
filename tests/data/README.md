# Synthetic reconstruction regression

`toy_regression_input.json` contains **generated synthetic rows**, not recorded
simulation events. It freezes two small events (83 selected tracks in total)
from the bundled aggregate toy profile, seed `20260929`, with the multiplicity
overrides and event ordering recorded in the file. Freezing rows also avoids
making this reconstruction test depend on changes in NumPy's random sampler.
The reconstruction and analysis configurations are stored with the input.

`toy_regression_expected.json` was frozen using the pre-release source snapshot
at commit `f19f09e`, before adding packaging and provenance. Review of this
baseline found two reconstructed vertices in each event and 37 CSV outputs:
18 per-bin rate tables, their 18 overall tables, and one population table.
The test compares event identities, selected MC keys, truth/reconstructed track
memberships, seeds, fitted coordinates/errors, histogram sums and every CSV.
It runs with both one and two reconstruction workers.

Identities, memberships and counts must match exactly. Fitted values use
`rtol=2e-6, atol=1e-7` (stored HDF5 coordinates/errors are float32); rates and
confidence intervals use `rtol=1e-10, atol=1e-12`. The test does not compare
rendered files or serialize interpreter-specific objects. The hand-built rate
tests separately cover fake, merge and split cases absent from this tiny sample.

Update a baseline only after reviewing the numerical and membership changes.
Use `tests.data.toy_regression.reconstruct` and `summarize` to produce a candidate
in a temporary directory, then review its diff against the committed JSON.
Tests never regenerate or silently accept expected results.
