# Data formats

[Back to README](../README.md)

## ROOT input

PVFit reads ROOT files produced by Allen monitoring jobs or the
[toy generator](toy.md). Set `paths.input_root` or pass `--input` to `fit` or
`run`; a glob reads all matching files together. The optimizer has a separate
[input selection policy](optimization.md#input-selection).

### Event identity

Allen monitoring dumps preserve the run and event numbers from MDF ODIN.
Event numbers can repeat across runs, so an event is identified by the pair
`(run_number, event_number)`, and both trees **must** carry a `run_number`
branch; a dump without it is rejected. PVFit numbers the events densely in
`(run_number, event_number)` order across all input files, and that integer is
the id used everywhere downstream (`workflow.event_number`, `plot_event_id`, the
HDF5 group names). Each HDF5 event group records the original `run_number`,
`source_event_number` and `source_file` in its attributes. The same
`(run_number, event_number)` appearing in two input files is an error.

### ROOT trees

Both trees are required. Positions use mm, times use ns and momenta use MeV.

**Track tree** (default `velo_kalman_filter/monitor_tree`), one row per track:

| Branch | Meaning |
| --- | --- |
| `event_number` | event id, used to group rows |
| `run_number` | run id — **required**; events are keyed by (run, event) |
| `beamPOCA_x`, `beamPOCA_y`, `beamPOCA_z`, `beamPOCA_t` | point of closest approach to the beamline |
| `beamPOCA_tx`, `beamPOCA_ty` | track slopes at the POCA |
| `backward` | track direction flag used for time transport; required |
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
| `run_number` | run id — **required** |
| `x`, `y`, `z`, `time` | true vertex position |
| `effective_pv_key` | unique key of the vertex within the event |
| `track_keys` | variable-length array of the `mc_key`s belonging to this vertex |
| `n_valid_tracks` | track count from the dumper, stored as `expected_ntracks` |

### Truth association

A reconstructed track row belongs to a true vertex when its `mc_key` appears in
that vertex's `track_keys`, **within the same event**. Association is purely by
key; track and vertex coordinates never decide membership.

Selection and validation:

- A true vertex enters the performance denominators only if at least **4** of its
  tracks survive the `z`/`t` acceptance selection. `n_valid_tracks` is recorded
  as `expected_ntracks` but is *not* substituted for this selected-track count.
- Missing `mc_key` / `effective_pv_key` / `track_keys` / `n_valid_tracks` raises
  an error. There is no silent fallback.
- A truth particle owned by two effective vertices is an error, not a warning.
- Truth association supplies the truth overlays and performance evaluation.
  Reconstruction uses the selected track states independently of truth membership.

## HDF5 layout

```
fit_results.h5
├── @n_events
├── @source                        "PVFit" (producer label used by plots)
├── @reconstruction_max_chi2        algorithm.max_chi2 used for this reconstruction
└── <event_id>/
    ├── @n_tracks, @n_true_vertices, @n_rec_vertices, @n_seedings, @truth_matching
    ├── @run_number, @source_event_number, @source_file   (the dump's own identity)
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
