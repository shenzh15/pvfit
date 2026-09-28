"""Identify every event by ``(run_number, event_number)``.

Allen monitoring dumps preserve the run and event numbers from MDF ODIN.
Event numbers can repeat across runs, so PVFit identifies events across
input files by the pair ``(run_number, event_number)``. Both trees must
therefore carry a ``run_number`` branch; there is no fallback to
``event_number`` alone.

Downstream code (event selection, the HDF5 group names, the analysis) uses one
integer per event. That integer is the position of the event's ``(run_number,
event_number)`` pair in the sorted list of all pairs over all input files. The
original run and event numbers and the source file are kept alongside so the
output can always be traced back to the dump.
"""

import numpy as np
import pandas as pd

RUN_COLUMN = "run_number"
EVENT_COLUMN = "event_number"
SOURCE_EVENT_COLUMN = "source_event_number"
IDENTITY_COLUMNS = (RUN_COLUMN, EVENT_COLUMN)


def require_run_numbers(tracks, vertices):
    """Both trees must identify events by run and event number."""
    missing = [
        name
        for name, frame in (("track", tracks), ("truth", vertices))
        if RUN_COLUMN not in frame.columns or EVENT_COLUMN not in frame.columns
    ]
    if missing:
        noun = "trees lack" if len(missing) > 1 else "tree lacks"
        raise ValueError(
            f"{' and '.join(missing)} {noun} run_number/event_number; "
            "events are identified by (run_number, event_number) and a dump "
            "without run numbers cannot be read"
        )


def unify_event_identity(per_file):
    """Give the events of all input files one dense integer id each.

    ``per_file`` is a list of ``(tracks, vertices, source_file)``. Returns the
    concatenated ``(tracks, vertices, identity)``: both frames have their
    ``event_number`` replaced by the unified id and keep the dump's own value in
    ``source_event_number``; ``identity`` has one row per event with
    ``event_number``, ``run_number``, ``source_event_number`` and
    ``source_file``.

    The same ``(run_number, event_number)`` in two files is an error: it would
    be two different events with one name.
    """
    if not per_file:
        raise ValueError("No input data to identify events in")
    pairs = []
    for tracks, vertices, source_file in per_file:
        require_run_numbers(tracks, vertices)
        found = pd.concat(
            [
                tracks[list(IDENTITY_COLUMNS)],
                vertices[list(IDENTITY_COLUMNS)],
            ],
            ignore_index=True,
        ).drop_duplicates()
        pairs.append(found.assign(source_file=str(source_file)))
    pairs = pd.concat(pairs, ignore_index=True)
    clash = pairs[pairs.duplicated(list(IDENTITY_COLUMNS), keep=False)]
    if len(clash):
        first = clash.sort_values(list(IDENTITY_COLUMNS)).iloc[0]
        raise ValueError(
            f"(run_number, event_number) = ({first[RUN_COLUMN]}, "
            f"{first[EVENT_COLUMN]}) appears in more than one input file; "
            "each event must have a unique identity across all inputs"
        )
    pairs = pairs.sort_values(list(IDENTITY_COLUMNS)).reset_index(drop=True)
    identity = pd.DataFrame(
        {
            EVENT_COLUMN: np.arange(len(pairs), dtype=np.int64),
            RUN_COLUMN: pairs[RUN_COLUMN].to_numpy(np.int64),
            SOURCE_EVENT_COLUMN: pairs[EVENT_COLUMN].to_numpy(np.int64),
            "source_file": pairs["source_file"].to_numpy(),
        }
    )
    lookup = pd.MultiIndex.from_frame(pairs[list(IDENTITY_COLUMNS)])

    def remap(frame):
        frame = frame.copy()
        keys = pd.MultiIndex.from_frame(frame[list(IDENTITY_COLUMNS)])
        frame[SOURCE_EVENT_COLUMN] = frame[EVENT_COLUMN].to_numpy(np.int64)
        frame[EVENT_COLUMN] = lookup.get_indexer(keys).astype(np.int64)
        return frame

    tracks = pd.concat([remap(t) for t, _, _ in per_file], ignore_index=True)
    vertices = pd.concat([remap(v) for _, v, _ in per_file], ignore_index=True)
    return tracks, vertices, identity
