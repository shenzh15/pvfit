"""Configuration for reconstruction and independent analysis.

Each stage has one frozen dataclass. Its fields carry the defaults, its
``from_mapping`` is the only code that knows the YAML layout and validates it,
and ``to_mapping`` writes that layout back. Everything else asks the dataclass.
"""

import math
import multiprocessing as mp
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

ANALYSIS_NAMES = (
    "efficiency_track",
    "efficiency_geo",
    "fake_rate_track",
    "fake_rate_geo",
    "merge_rate",
    "split_rate",
    "merge_rate_geo",
    "split_rate_geo",
    "resolution",
    "plot_histogram",
    "track_pull",
    "vertex_pull",
)
# Analyses run when neither the YAML nor the command line selects any.
DEFAULT_PERFORMANCE_ANALYSES = (
    "efficiency_track",
    "efficiency_geo",
    "fake_rate_track",
    "fake_rate_geo",
    "merge_rate",
    "split_rate",
)

# Reference bin sizes: z ∈ [-200, 200] / 800 and t ∈ [-0.8, 0.8] / 64.
REF_BIN_SIZE_Z = 0.5  # mm
REF_BIN_SIZE_T = 0.025  # ns


def load_config(config_file):
    with open(config_file) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_file}")
    return config


def _section(config, name):
    section = config.get(name, {})
    if not isinstance(section, dict):
        raise ValueError(f"Configuration section '{name}' must be a mapping")
    return section


def _reject_unknown(section, name, known):
    unknown = sorted(set(section) - set(known))
    if unknown:
        raise ValueError(f"Unknown {name} keys: {', '.join(unknown)}")


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value):
    """A finite int or float; NaN and infinities are not numbers here."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _is_bool(value):
    return isinstance(value, bool)


def _check_bools(config, names, section):
    """Flags must be real booleans: the string "false" would count as true."""
    for name in names:
        if not _is_bool(getattr(config, name)):
            raise ValueError(f"{section}.{name} must be true or false")


def _check_positive(config, names, section):
    for name in names:
        value = getattr(config, name)
        if not _is_number(value) or value <= 0:
            raise ValueError(f"{section}.{name} must be a positive finite number")


def _check_path(value, name):
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{name} must be a nonempty path")


def _default_workers():
    return max(1, mp.cpu_count() // 2)


@dataclass(frozen=True)
class ReconstructionConfig:
    """Settings of ``pvfit fit``: the YAML's paths, workflow and algorithm."""

    # paths (data paths are relative to the working directory)
    input_root: str
    track_tree_name: str
    mc_vertex_tree_name: str
    output_dir: str
    hdf5_file: str = "fit_results.h5"

    # workflow
    event_number: int = -1  # -1: all events
    max_workers: int = field(default_factory=_default_workers)

    # algorithm: track cuts (the YAML spells this one minTracks)
    minTracks: float = 2.5
    smear_window_size: int = 2
    beamline_chi2_cut: bool = False
    max_track_bl_chi2: float = 10.0
    z0_err_cut: bool = False
    pp_max_track_z0_err: float = 1.5  # mm

    # algorithm: histogram
    z_min: float = -200.0
    z_max: float = 200.0
    t_min: float = -0.8
    t_max: float = 0.8
    bins_z: int = 800
    bins_t: int = 64

    # algorithm: vertex fitting and merging
    max_chi2: float = 10.0
    is_adaptive_tukey: bool = True
    drop_low_track_seeds: bool = True
    merge_vertices: bool = False
    nsigma_merge: float = 5.0

    _PATH_KEYS = (
        "input_root",
        "track_tree_name",
        "mc_vertex_tree_name",
        "output_dir",
        "hdf5_file",
    )
    _WORKFLOW_KEYS = ("event_number", "max_workers")
    _ALGORITHM_KEYS = (
        "minTracks", "smear_window_size", "beamline_chi2_cut", "max_track_bl_chi2",
        "z0_err_cut", "pp_max_track_z0_err", "z_range", "t_range", "bins", "max_chi2",
        "is_adaptive_tukey", "drop_low_track_seeds", "merge_vertices", "nsigma_merge",
    )  # fmt: skip

    def __post_init__(self):
        for name in (
            "input_root",
            "track_tree_name",
            "mc_vertex_tree_name",
            "output_dir",
        ):
            _check_path(getattr(self, name), f"paths.{name}")
        filename = Path(self.hdf5_file)
        if (
            not str(self.hdf5_file).strip()
            or filename.is_absolute()
            or len(filename.parts) != 1
            or filename.name in (".", "..")
        ):
            raise ValueError("paths.hdf5_file must be a filename, not a directory path")
        if not _is_int(self.max_workers) or self.max_workers < 1:
            raise ValueError("workflow.max_workers must be a positive integer")
        if not _is_int(self.event_number) or self.event_number < -1:
            raise ValueError(
                "workflow.event_number must be -1 or a nonnegative integer"
            )
        for axis in ("z", "t"):
            bins = getattr(self, f"bins_{axis}")
            if not _is_int(bins) or bins < 1:
                raise ValueError(f"algorithm.bins.{axis} must be a positive integer")
            low, high = getattr(self, f"{axis}_min"), getattr(self, f"{axis}_max")
            if not (_is_number(low) and _is_number(high) and low < high):
                raise ValueError(f"algorithm.{axis}_range requires min < max")
        if not _is_int(self.smear_window_size) or self.smear_window_size < 0:
            raise ValueError(
                "algorithm.smear_window_size must be a nonnegative integer"
            )
        _check_positive(
            self,
            (
                "minTracks",
                "max_track_bl_chi2",
                "pp_max_track_z0_err",
                "max_chi2",
                "nsigma_merge",
            ),
            "algorithm",
        )
        _check_bools(
            self,
            (
                "beamline_chi2_cut",
                "z0_err_cut",
                "is_adaptive_tukey",
                "drop_low_track_seeds",
                "merge_vertices",
            ),
            "algorithm",
        )

    @classmethod
    def from_mapping(cls, config):
        """Build from a parsed reconstruction YAML; missing sections use defaults."""
        paths = _section(config, "paths")
        workflow = _section(config, "workflow")
        algorithm = _section(config, "algorithm")
        _reject_unknown(paths, "paths", cls._PATH_KEYS)
        _reject_unknown(workflow, "workflow", cls._WORKFLOW_KEYS)
        _reject_unknown(algorithm, "algorithm", cls._ALGORITHM_KEYS)

        values = {}
        for name in (
            "input_root",
            "track_tree_name",
            "mc_vertex_tree_name",
            "output_dir",
        ):
            if name not in paths:
                raise KeyError(f"paths.{name}")
            values[name] = paths[name]
        if "hdf5_file" in paths:
            values["hdf5_file"] = paths["hdf5_file"]
        values.update(workflow)
        for name in cls._ALGORITHM_KEYS:
            if name not in algorithm or name in ("z_range", "t_range", "bins"):
                continue
            values[name] = algorithm[name]
        for axis in ("z", "t"):
            limits = algorithm.get(f"{axis}_range", {})
            if not isinstance(limits, dict):
                raise ValueError(f"algorithm.{axis}_range must be a mapping")
            for bound in ("min", "max"):
                if bound in limits:
                    values[f"{axis}_{bound}"] = limits[bound]
        bins = algorithm.get("bins", {})
        if not isinstance(bins, dict):
            raise ValueError("algorithm.bins must be a mapping")
        for axis in ("z", "t"):
            if axis in bins:
                values[f"bins_{axis}"] = bins[axis]
        return cls(**values)

    @classmethod
    def from_yaml(cls, path):
        return cls.from_mapping(load_config(path))

    def to_mapping(self):
        """The YAML layout ``from_mapping`` reads."""
        d = asdict(self)
        return {
            "paths": {name: d[name] for name in self._PATH_KEYS},
            "workflow": {name: d[name] for name in self._WORKFLOW_KEYS},
            "algorithm": {
                **{
                    name: d[name]
                    for name in self._ALGORITHM_KEYS
                    if name not in ("z_range", "t_range", "bins")
                },
                "z_range": {"min": d["z_min"], "max": d["z_max"]},
                "t_range": {"min": d["t_min"], "max": d["t_max"]},
                "bins": {"z": d["bins_z"], "t": d["bins_t"]},
            },
        }

    @property
    def bin_size_z(self):
        return (self.z_max - self.z_min) / self.bins_z

    @property
    def bin_size_t(self):
        return (self.t_max - self.t_min) / self.bins_t

    @property
    def m_minTracks(self):
        """minTracks scaled by bin area relative to the reference binning.

        Keeps the seeding threshold physically consistent across binnings.
        """
        return (
            self.minTracks
            * (self.bin_size_z * self.bin_size_t)
            / (REF_BIN_SIZE_Z * REF_BIN_SIZE_T)
        )

    def as_params(self):
        """The parameter dict the reconstruction modules read."""
        return {
            "m_minTracks": self.m_minTracks,
            "smear_window_size": self.smear_window_size,
            "zmin": self.z_min,
            "zmax": self.z_max,
            "tmin": self.t_min,
            "tmax": self.t_max,
            "Nbins_z": self.bins_z,
            "Nbins_t": self.bins_t,
            "bin_size_z": self.bin_size_z,
            "bin_size_t": self.bin_size_t,
            "max_chi2": self.max_chi2,
            "is_adaptive_tukey": self.is_adaptive_tukey,
            "drop_low_track_seeds": self.drop_low_track_seeds,
            "merge_vertices": self.merge_vertices,
            "nsigma_merge": self.nsigma_merge,
            "beamline_chi2_cut": self.beamline_chi2_cut,
            "max_track_bl_chi2": self.max_track_bl_chi2,
            "z0_err_cut": self.z0_err_cut,
            "pp_max_track_z0_err": self.pp_max_track_z0_err,
        }


@dataclass(frozen=True)
class AnalysisConfig:
    """Settings of ``pvfit analyze``: the YAML's analysis and paths sections."""

    analyses: tuple = DEFAULT_PERFORMANCE_ANALYSES
    method_name: str = "PVfit"

    # reconstructed-to-true vertex matching
    recall_threshold: float = 0.7
    purity_threshold: float = 0.7
    m_r: float = 0.8
    nsigma: float = 3.0
    z_scale: float = 1.5
    t_scale: float = 0.06

    # None follows the threshold recorded by reconstruction; a number overrides it.
    track_pull_chi2: float | None = None

    # vertex-resolution selection windows
    x_residual_range: float = 0.1  # mm
    y_residual_range: float = 0.1  # mm
    z_residual_range: float = 0.7  # mm
    t_residual_range: float = 0.06  # ns

    # binning in track multiplicity for the performance plots
    plot_min_ntracks: int = 4
    plot_max_ntracks: int = 200
    plot_bin_width_ntracks: int = 5

    # 2D histogram plot
    plot_event_id: int = 0
    draw_fixed_ellipses: bool = False
    draw_fitted_ellipses: bool = False

    # optional paths (relative to the working directory)
    input_hdf5: str | None = None
    output_dir: str | None = None

    _PATH_KEYS = ("input_hdf5", "output_dir")

    def __post_init__(self):
        analyses = (
            tuple(self.analyses) if isinstance(self.analyses, (list, tuple)) else None
        )
        if not analyses:
            raise ValueError("analyses must be a nonempty list of analysis names")
        unknown = [name for name in analyses if name not in ANALYSIS_NAMES]
        if unknown:
            raise ValueError(
                f"Unknown analyses: {unknown}; choose from {', '.join(ANALYSIS_NAMES)}"
            )
        object.__setattr__(self, "analyses", analyses)
        if not isinstance(self.method_name, str) or not self.method_name.strip():
            raise ValueError("analysis.method_name must be a nonempty string")
        for name in ("recall_threshold", "purity_threshold"):
            value = getattr(self, name)
            if not _is_number(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"analysis.{name} must be a number between 0 and 1")
        _check_positive(
            self,
            (
                "m_r",
                "nsigma",
                "z_scale",
                "t_scale",
                "x_residual_range",
                "y_residual_range",
                "z_residual_range",
                "t_residual_range",
            ),
            "analysis",
        )
        if self.track_pull_chi2 is not None:
            _check_positive(self, ("track_pull_chi2",), "analysis")
        _check_bools(self, ("draw_fixed_ellipses", "draw_fitted_ellipses"), "analysis")
        for name in ("plot_min_ntracks", "plot_max_ntracks", "plot_bin_width_ntracks"):
            if not _is_int(getattr(self, name)) or getattr(self, name) < 1:
                raise ValueError(f"analysis.{name} must be a positive integer")
        if not _is_int(self.plot_event_id) or self.plot_event_id < 0:
            raise ValueError("analysis.plot_event_id must be a nonnegative integer")
        if self.plot_min_ntracks > self.plot_max_ntracks:
            raise ValueError(
                "analysis.plot_min_ntracks must not exceed plot_max_ntracks"
            )
        for name in self._PATH_KEYS:
            if getattr(self, name) is not None:
                _check_path(getattr(self, name), f"paths.{name}")

    @classmethod
    def analysis_keys(cls):
        return tuple(f.name for f in fields(cls) if f.name not in cls._PATH_KEYS)

    @classmethod
    def from_mapping(cls, config):
        """Build from a parsed analysis YAML; missing keys use defaults."""
        section = _section(config, "analysis")
        paths = _section(config, "paths")
        _reject_unknown(section, "analysis", cls.analysis_keys())
        _reject_unknown(paths, "paths", cls._PATH_KEYS)
        return cls(**section, **paths)

    @classmethod
    def from_yaml(cls, path):
        return cls.from_mapping(load_config(path))

    def to_mapping(self):
        d = asdict(self)
        d["analyses"] = list(d["analyses"])
        mapping = {"analysis": {name: d[name] for name in self.analysis_keys()}}
        paths = {name: d[name] for name in self._PATH_KEYS if d[name] is not None}
        if paths:
            mapping["paths"] = paths
        return mapping
