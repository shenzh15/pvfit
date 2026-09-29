"""Reproducibility records for CLI stages, without scanning large input contents."""

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from uuid import uuid4

import yaml

from . import __version__

HDF5_SCHEMA_VERSION = 1
DEPENDENCIES = (
    "numpy",
    "scipy",
    "pandas",
    "h5py",
    "uproot",
    "awkward",
    "matplotlib",
    "mplhep",
    "pyyaml",
    "optuna",
)


def implementation_fingerprint():
    """Identify the imported package, including dirty source and bundled data."""
    package = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*")):
        if path.suffix in (".py", ".json", ".yaml") and "__pycache__" not in path.parts:
            digest.update(path.relative_to(package).as_posix().encode())
            digest.update(path.read_bytes())
    return {"pvfit_version": __version__, "source_sha256": digest.hexdigest()}


def code_identity():
    result = implementation_fingerprint()
    root = Path(__file__).resolve().parent.parent
    # Do not attribute an installed wheel to an unrelated enclosing repository.
    if not (root / ".git").exists():
        return result
    try:

        def git(*args):
            return subprocess.check_output(
                ["git", "-C", str(root), *args],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
            ).strip()

        result["git_commit"] = git("rev-parse", "HEAD")
        result["git_dirty"] = bool(
            git("status", "--porcelain", "--untracked-files=normal")
        )
    except (OSError, subprocess.SubprocessError):
        pass
    return result


def file_identity(path):
    """Stat identity, not a content checksum; avoids rereading multi-GB ROOT files."""
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _now():
    return datetime.now(UTC).isoformat()


class StageRecord:
    """Keep a unique effective-config snapshot and success/failure record per stage."""

    def __init__(self, output_dir, stage, configuration, inputs, argv):
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        self.directory = (
            Path(output_dir).expanduser().resolve()
            / "provenance"
            / f"{stamp}_{uuid4().hex[:8]}_{stage}"
        )
        self.path = self.directory / "manifest.json"
        self.configuration = configuration
        self.stage = stage
        versions = {}
        for name in DEPENDENCIES:
            try:
                versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                versions[name] = None
        self.data = {
            "manifest_version": 1,
            "stage": stage,
            "status": "running",
            "started_at": _now(),
            "cwd": str(Path.cwd()),
            "argv": list(argv),
            "implementation": code_identity(),
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "dependencies": versions,
            },
            "inputs": [file_identity(p) for p in inputs],
            "configuration": configuration,
            "outputs": [],
        }

    def _write(self):
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.data, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.path)

    def __enter__(self):
        self.directory.mkdir(parents=True)
        name = "reconstruction.yaml" if self.stage == "fit" else "analysis.yaml"
        (self.directory / name).write_text(
            yaml.safe_dump(self.configuration, sort_keys=False)
        )
        self._write()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.data["finished_at"] = _now()
        self.data["status"] = "completed" if exc is None else "failed"
        if exc is not None:
            self.data["error"] = {"type": exc_type.__name__, "message": str(exc)}
        self._write()
        return False
