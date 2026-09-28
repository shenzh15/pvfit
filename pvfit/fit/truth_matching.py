"""Associate selected track rows with truth vertices by event-local MC keys."""

import numpy as np
import pandas as pd


def read_mcpv_dataframe(tree):
    """Keep vector branches (track_keys) as arrays, without exploding PV rows."""
    return pd.DataFrame(tree.arrays(library="np"))


def select_tracks(tracks, params):
    """Apply the reconstruction's z/t acceptance, preserving row order."""
    return tracks[
        tracks["beamPOCA_z"].between(params["zmin"], params["zmax"])
        & tracks["beamPOCA_t"].between(params["tmin"], params["tmax"])
    ]


def build_true_vertex_info(tracks, vertices, min_tracks=4):
    """Match mc_key membership in track_keys within a single event.

    Return positional indices into selected tracks. Duplicate reconstructed
    mc_keys remain distinct track rows. The minimum counts selected reconstructed
    rows, not the dump's n_valid_tracks. Input coordinates do not decide membership.
    """
    if "mc_key" not in tracks:
        raise ValueError("Truth matching requires monitor_tree.mc_key")
    required = {
        "x",
        "y",
        "z",
        "time",
        "effective_pv_key",
        "track_keys",
        "n_valid_tracks",
    }
    missing = required - set(vertices.columns)
    if missing:
        raise ValueError(
            f"Truth matching: missing effective PV fields {sorted(missing)}"
        )
    for frame in (tracks, vertices):
        if "event_number" not in frame:
            raise ValueError("Truth matching requires event_number in both inputs")
    events = set(tracks["event_number"]) | set(vertices["event_number"])
    if len(events) > 1:
        raise ValueError("Truth matching inputs must belong to the same single event")
    if "run_number" in tracks and "run_number" in vertices:
        runs = set(tracks["run_number"]) | set(vertices["run_number"])
        if len(runs) > 1:
            raise ValueError(
                "Truth matching inputs must belong to the same single event: "
                f"event_number {events.pop()} appears in runs {sorted(runs)}; "
                "key events by (run_number, event_number) first"
            )
    if vertices["effective_pv_key"].duplicated().any():
        raise ValueError("Duplicate effective_pv_key within one event")
    keys = tracks["mc_key"].to_numpy()
    # A truth particle must not belong to two effective vertices.
    owners = {}
    for vertex in vertices.itertuples(index=False):
        for key in vertex.track_keys:
            if key < 0:
                raise ValueError("Negative key in truth track_keys")
            previous = owners.setdefault(int(key), int(vertex.effective_pv_key))
            if previous != int(vertex.effective_pv_key):
                raise ValueError(f"mc_key {key} belongs to multiple truth vertices")
    result = []
    for position, (_, vertex) in enumerate(vertices.iterrows()):
        mask = (keys >= 0) & np.isin(keys, vertex["track_keys"])
        indices = np.flatnonzero(mask).tolist()
        if len(indices) < min_tracks:
            continue
        result.append(
            {
                "vertex_id": position,
                "effective_pv_key": int(vertex["effective_pv_key"]),
                "x": float(vertex["x"]),
                "y": float(vertex["y"]),
                "z": float(vertex["z"]),
                "t": float(vertex["time"]),
                "expected_ntracks": int(vertex["n_valid_tracks"]),
                "track_indices": indices,
            }
        )
    return result
