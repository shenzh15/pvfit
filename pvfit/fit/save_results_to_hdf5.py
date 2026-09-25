#!/usr/bin/env python3
"""Save PVfit results to HDF5 format for performance studies.

HDF5 structure (top-level groups):
    <output_file>.h5/
    ├── @n_events: int
    ├── @source: "PVFit"
    │
    ├── event_0/
    │   ├── @n_tracks: int
    │   ├── @n_true_vertices: int
    │   ├── @n_rec_vertices: int
    │   ├── @n_seedings: int
    │   │
    │   ├── histogram [n_bins_z, n_bins_t] float32
    │   │   @z_min, @z_max, @t_min, @t_max: float
    │   │   @n_bins_z, @n_bins_t: int
    │   │
    │   ├── seeds/
    │   │   └── z, t [n_seedings] float32 (all seeds, incl. dropped/merged)
    │   │
    │   ├── true_vertices/
    │   │   ├── x, y, z, t [n_true] float32
    │   │   ├── expected_ntracks [n_true] int32
    │   │   └── track_indices [n_true] vlen<int32>
    │   │
    │   ├── rec_vertices/
    │   │   ├── x, y, z, t [n_rec] float32
    │   │   ├── x_err, y_err, z_err, t_err [n_rec] float32
    │   │   ├── x_seed, y_seed, z_seed, t_seed [n_rec] float32
    │   │   └── track_indices [n_rec] vlen<int32>
    │   │
    │   └── tracks/
    │       ├── x, y, z, t [n_tracks] float32
    │       ├── tx, ty [n_tracks] float32; backward [n_tracks] bool
    │       ├── cov_xx, cov_yy, cov_tt [n_tracks] float32
    │       └── ovtx_x, ovtx_y, ovtx_z, ovtx_t [n_tracks] float32
"""

from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def _vlen_int32(rows):
    """1D object array of int32 arrays for a vlen dataset.

    np.array(rows, dtype=object) builds a 2D array when all rows have the same
    length (e.g. a single vertex), which h5py cannot write as vlen.
    """
    out = np.empty(len(rows), dtype=object)
    for i, row in enumerate(rows):
        out[i] = np.asarray(row, dtype=np.int32)
    return out


def save_event_to_hdf5(event_result, event_data_selected, output_file, params):
    """Save a single event's results to HDF5 file with dynamic array sizes.

    Args:
        event_result: Event processing results from process_event()
        event_data_selected: Selected track data for the event (pandas DataFrame)
        output_file: Path to output HDF5 file
        params: Algorithm parameters (for histogram metadata)
    """
    event_number = event_result["event_number"]
    vertex_true_info = event_result["vertex_true_info"]
    vertex_rec_info = event_result["vertex_rec_info"]
    histogram = event_result["histogram"]
    n_seedings = event_result["n_seedings"]

    n_tracks = len(event_data_selected)
    n_true_vertices = len(vertex_true_info)
    n_rec_vertices = len(vertex_rec_info)

    with h5py.File(output_file, "a") as f:
        # Create event group
        event_group_name = str(event_number)
        if event_group_name in f:
            del f[event_group_name]
        event_group = f.create_group(event_group_name)

        # Store event-level metadata
        event_group.attrs["n_tracks"] = n_tracks
        event_group.attrs["n_true_vertices"] = n_true_vertices
        event_group.attrs["n_rec_vertices"] = n_rec_vertices
        event_group.attrs["n_seedings"] = n_seedings
        event_group.attrs["truth_matching"] = "key"

        # ============================================================
        # Histogram
        # ============================================================
        ds = event_group.create_dataset(
            "histogram", data=histogram, compression="gzip", compression_opts=4
        )
        ds.attrs["z_min"] = params["zmin"]
        ds.attrs["z_max"] = params["zmax"]
        ds.attrs["t_min"] = params["tmin"]
        ds.attrs["t_max"] = params["tmax"]
        ds.attrs["n_bins_z"] = params["Nbins_z"]
        ds.attrs["n_bins_t"] = params["Nbins_t"]

        # ============================================================
        # Seeds (every histogram peak, before seeds are dropped/merged)
        # ============================================================
        seeds = np.asarray(event_result["seeds"], dtype=np.float32).reshape(-1, 2)
        seeds_group = event_group.create_group("seeds")
        seeds_group.create_dataset("z", data=seeds[:, 0])
        seeds_group.create_dataset("t", data=seeds[:, 1])

        # ============================================================
        # True Vertices
        # ============================================================
        true_group = event_group.create_group("true_vertices")
        true_group.create_dataset(
            "effective_pv_key",
            data=np.array(
                [v["effective_pv_key"] for v in vertex_true_info], dtype=np.int64
            ),
        )

        # Initialize arrays
        true_x = np.zeros(n_true_vertices, dtype=np.float32)
        true_y = np.zeros(n_true_vertices, dtype=np.float32)
        true_z = np.zeros(n_true_vertices, dtype=np.float32)
        true_t = np.zeros(n_true_vertices, dtype=np.float32)
        true_expected_ntracks = np.zeros(n_true_vertices, dtype=np.int32)
        true_vertex_id = np.zeros(n_true_vertices, dtype=np.int32)
        true_track_indices = []

        # Fill data
        for i, vertex in enumerate(vertex_true_info):
            true_x[i] = vertex["x"]
            true_y[i] = vertex["y"]
            true_z[i] = vertex["z"]
            true_t[i] = vertex["t"]
            true_expected_ntracks[i] = vertex["expected_ntracks"]
            true_vertex_id[i] = vertex["vertex_id"]
            true_track_indices.append(np.array(vertex["track_indices"], dtype=np.int32))

        # Save datasets
        true_group.create_dataset(
            "x", data=true_x, compression="gzip", compression_opts=4
        )
        true_group.create_dataset(
            "y", data=true_y, compression="gzip", compression_opts=4
        )
        true_group.create_dataset(
            "z", data=true_z, compression="gzip", compression_opts=4
        )
        true_group.create_dataset(
            "t", data=true_t, compression="gzip", compression_opts=4
        )
        true_group.create_dataset(
            "expected_ntracks",
            data=true_expected_ntracks,
            compression="gzip",
            compression_opts=4,
        )
        true_group.create_dataset(
            "vertex_id", data=true_vertex_id, compression="gzip", compression_opts=4
        )
        vlen_dtype = h5py.vlen_dtype(np.int32)
        true_group.create_dataset(
            "track_indices",
            data=_vlen_int32(true_track_indices),
            dtype=vlen_dtype,
            compression="gzip",
            compression_opts=4,
        )

        # ============================================================
        # Reconstructed Vertices
        # ============================================================
        rec_group = event_group.create_group("rec_vertices")

        # Initialize arrays
        rec_x = np.zeros(n_rec_vertices, dtype=np.float32)
        rec_y = np.zeros(n_rec_vertices, dtype=np.float32)
        rec_z = np.zeros(n_rec_vertices, dtype=np.float32)
        rec_t = np.zeros(n_rec_vertices, dtype=np.float32)
        rec_x_err = np.zeros(n_rec_vertices, dtype=np.float32)
        rec_y_err = np.zeros(n_rec_vertices, dtype=np.float32)
        rec_z_err = np.zeros(n_rec_vertices, dtype=np.float32)
        rec_t_err = np.zeros(n_rec_vertices, dtype=np.float32)
        rec_x_seed = np.zeros(n_rec_vertices, dtype=np.float32)
        rec_y_seed = np.zeros(n_rec_vertices, dtype=np.float32)
        rec_z_seed = np.zeros(n_rec_vertices, dtype=np.float32)
        rec_t_seed = np.zeros(n_rec_vertices, dtype=np.float32)
        rec_track_indices = []

        # Fill data
        for i, vertex in enumerate(vertex_rec_info):
            # Fitted positions
            rec_x[i] = vertex.get("Params_rec_x", vertex.get("rec_x", 0.0))
            rec_y[i] = vertex.get("Params_rec_y", vertex.get("rec_y", 0.0))
            rec_z[i] = vertex.get("Params_rec_z", vertex["rec_z"])
            rec_t[i] = vertex.get("Params_rec_t", vertex["rec_t"])
            # Fitting errors (sqrt of covariance diagonal)
            rec_x_err[i] = np.sqrt(vertex.get("cov_00", 0.0))
            rec_y_err[i] = np.sqrt(vertex.get("cov_11", 0.0))
            rec_z_err[i] = np.sqrt(vertex.get("cov_22", 0.0))
            rec_t_err[i] = np.sqrt(vertex.get("cov_33", 0.0))
            # Seeding initial positions
            rec_x_seed[i] = vertex.get("rec_x", 0.0)
            rec_y_seed[i] = vertex.get("rec_y", 0.0)
            rec_z_seed[i] = vertex["rec_z"]
            rec_t_seed[i] = vertex["rec_t"]
            # Track indices
            rec_track_indices.append(np.array(vertex["track_index"], dtype=np.int32))

        # Save datasets
        rec_group.create_dataset(
            "x", data=rec_x, compression="gzip", compression_opts=4
        )
        rec_group.create_dataset(
            "y", data=rec_y, compression="gzip", compression_opts=4
        )
        rec_group.create_dataset(
            "z", data=rec_z, compression="gzip", compression_opts=4
        )
        rec_group.create_dataset(
            "t", data=rec_t, compression="gzip", compression_opts=4
        )
        rec_group.create_dataset(
            "x_err", data=rec_x_err, compression="gzip", compression_opts=4
        )
        rec_group.create_dataset(
            "y_err", data=rec_y_err, compression="gzip", compression_opts=4
        )
        rec_group.create_dataset(
            "z_err", data=rec_z_err, compression="gzip", compression_opts=4
        )
        rec_group.create_dataset(
            "t_err", data=rec_t_err, compression="gzip", compression_opts=4
        )
        rec_group.create_dataset(
            "x_seed", data=rec_x_seed, compression="gzip", compression_opts=4
        )
        rec_group.create_dataset(
            "y_seed", data=rec_y_seed, compression="gzip", compression_opts=4
        )
        rec_group.create_dataset(
            "z_seed", data=rec_z_seed, compression="gzip", compression_opts=4
        )
        rec_group.create_dataset(
            "t_seed", data=rec_t_seed, compression="gzip", compression_opts=4
        )
        vlen_dtype = h5py.vlen_dtype(np.int32)
        rec_group.create_dataset(
            "track_indices",
            data=_vlen_int32(rec_track_indices),
            dtype=vlen_dtype,
            compression="gzip",
            compression_opts=4,
        )

        # ============================================================
        # Tracks (for input track pull analysis)
        # ============================================================
        tracks_group = event_group.create_group("tracks")
        tracks_group.create_dataset(
            "mc_key", data=event_data_selected["mc_key"].to_numpy(dtype=np.int64)
        )

        # Preserve the direction for transport to a truth vertex's z plane.
        for name in ("tx", "ty"):
            tracks_group.create_dataset(
                name,
                data=event_data_selected[f"beamPOCA_{name}"].to_numpy(dtype=np.float32),
                compression="gzip",
                compression_opts=4,
            )
        tracks_group.create_dataset(
            "backward",
            data=event_data_selected["backward"].to_numpy(dtype=np.bool_),
            compression="gzip",
            compression_opts=4,
        )

        # Extract track data from event_data_selected
        track_x = np.array(event_data_selected["beamPOCA_x"].values, dtype=np.float32)
        track_y = np.array(event_data_selected["beamPOCA_y"].values, dtype=np.float32)
        track_z = np.array(event_data_selected["beamPOCA_z"].values, dtype=np.float32)
        track_t = np.array(event_data_selected["beamPOCA_t"].values, dtype=np.float32)
        track_cov_xx = np.array(event_data_selected["c00"].values, dtype=np.float32)
        track_cov_yy = np.array(event_data_selected["c11"].values, dtype=np.float32)
        track_cov_tt = np.array(event_data_selected["c55"].values, dtype=np.float32)
        track_ovtx_x = np.array(event_data_selected["ovtx_x"].values, dtype=np.float32)
        track_ovtx_y = np.array(event_data_selected["ovtx_y"].values, dtype=np.float32)
        track_ovtx_z = np.array(event_data_selected["ovtx_z"].values, dtype=np.float32)
        track_ovtx_t = np.array(event_data_selected["ovtx_t"].values, dtype=np.float32)
        track_p = np.array(event_data_selected["p"].values, dtype=np.float32)
        track_pt = np.array(event_data_selected["pt"].values, dtype=np.float32)
        track_chi2 = np.array(event_data_selected["chi2"].values, dtype=np.float32)
        fromPV = np.array(event_data_selected["fromPV"].values, dtype=np.bool_)
        track_id = np.arange(n_tracks, dtype=np.int32)

        # Save datasets
        tracks_group.create_dataset(
            "x", data=track_x, compression="gzip", compression_opts=4
        )
        tracks_group.create_dataset(
            "y", data=track_y, compression="gzip", compression_opts=4
        )
        tracks_group.create_dataset(
            "z", data=track_z, compression="gzip", compression_opts=4
        )
        tracks_group.create_dataset(
            "t", data=track_t, compression="gzip", compression_opts=4
        )
        tracks_group.create_dataset(
            "cov_xx", data=track_cov_xx, compression="gzip", compression_opts=4
        )
        tracks_group.create_dataset(
            "cov_yy", data=track_cov_yy, compression="gzip", compression_opts=4
        )
        tracks_group.create_dataset(
            "cov_tt", data=track_cov_tt, compression="gzip", compression_opts=4
        )
        tracks_group.create_dataset(
            "ovtx_x", data=track_ovtx_x, compression="gzip", compression_opts=4
        )
        tracks_group.create_dataset(
            "ovtx_y", data=track_ovtx_y, compression="gzip", compression_opts=4
        )
        tracks_group.create_dataset(
            "ovtx_z", data=track_ovtx_z, compression="gzip", compression_opts=4
        )
        tracks_group.create_dataset(
            "ovtx_t", data=track_ovtx_t, compression="gzip", compression_opts=4
        )
        tracks_group.create_dataset(
            "p", data=track_p, compression="gzip", compression_opts=4
        )
        tracks_group.create_dataset(
            "pt", data=track_pt, compression="gzip", compression_opts=4
        )
        tracks_group.create_dataset(
            "chi2", data=track_chi2, compression="gzip", compression_opts=4
        )
        tracks_group.create_dataset(
            "fromPV", data=fromPV, compression="gzip", compression_opts=4
        )
        tracks_group.create_dataset(
            "track_id", data=track_id, compression="gzip", compression_opts=4
        )


def save_all_events_to_hdf5(fit_results, output_file, params):
    """Save all events' results to HDF5 file with dynamic sizes per event.

    Args:
        fit_results: List of event processing results from process_event()
        output_file: Path to output HDF5 file
        params: Algorithm parameters (for histogram metadata)
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving {len(fit_results)} events to HDF5 file: {output_file}")

    # Clear existing file
    with h5py.File(output_file, "w") as f:
        f.attrs["n_events"] = len(fit_results)
        f.attrs["source"] = "PVFit"

    # Save each event
    for event_result in tqdm(fit_results, desc="Saving events to HDF5"):
        event_data_selected = event_result["event_data_selected"]
        save_event_to_hdf5(event_result, event_data_selected, output_file, params)

    print(f"Successfully saved {len(fit_results)} events to {output_file}")
