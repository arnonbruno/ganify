"""Reproducible run manifests, hashes, environment capture, and timings."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Union

import pandas as pd


PathLike = Union[str, Path]


def hash_bytes(values: bytes) -> str:
    """Return a SHA-256 hex digest."""

    return hashlib.sha256(values).hexdigest()


def hash_json(values: object) -> str:
    """Hash canonical JSON-serializable values."""

    payload = json.dumps(
        values, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hash_bytes(payload)


def hash_dataframe(frame: pd.DataFrame) -> str:
    """Hash schema, row order, values, and dtypes deterministically."""

    header = {
        "columns": [str(column) for column in frame.columns],
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "index_name": str(frame.index.name),
    }
    csv = frame.to_csv(
        index=True,
        lineterminator="\n",
        float_format="%.17g",
        na_rep="__GANIFY_NA__",
    )
    return hash_bytes(
        (json.dumps(header, sort_keys=True) + "\n" + csv).encode("utf-8")
    )


def _package_versions() -> Dict[str, str]:
    names = (
        "ganify",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "tensorflow",
    )
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _git_environment(repository: Optional[PathLike]) -> Dict[str, object]:
    if repository is None:
        return {}
    root = Path(repository)

    def run(*arguments: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *arguments],
                cwd=str(root),
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=10,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain=v1", "--untracked-files=all")
    try:
        diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        diff = b""
    return {
        "commit": commit or None,
        "dirty": bool(status),
        "status_sha256": hash_bytes(status.encode("utf-8")),
        "tracked_diff_sha256": hash_bytes(diff),
    }


def capture_environment(repository: Optional[PathLike] = None) -> Dict[str, object]:
    """Capture stable software/hardware metadata without importing TensorFlow."""

    environment: Dict[str, object] = {
        "python": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": _package_versions(),
        "determinism": {
            key: os.environ.get(key)
            for key in (
                "PYTHONHASHSEED",
                "TF_DETERMINISTIC_OPS",
                "TF_CUDNN_DETERMINISTIC",
                "CUDA_VISIBLE_DEVICES",
            )
        },
    }
    git = _git_environment(repository)
    if git:
        environment["git"] = git
    return environment


@dataclass
class RunManifest:
    """Complete provenance record for one model/dataset/split run."""

    run_id: str
    dataset_id: str
    model_name: str
    split_hash: str
    seeds: Dict[str, int]
    config: Dict[str, Any]
    hashes: Dict[str, str] = field(default_factory=dict)
    timings_seconds: Dict[str, float] = field(default_factory=dict)
    environment: Dict[str, object] = field(default_factory=dict)
    created_at_utc: str = ""
    status: str = "created"
    error: Optional[str] = None

    @classmethod
    def create(
        cls,
        *,
        dataset_id: str,
        model_name: str,
        split_hash: str,
        seeds: Mapping[str, int],
        config: Mapping[str, Any],
        repository: Optional[PathLike] = None,
        hashes: Optional[Mapping[str, str]] = None,
    ) -> "RunManifest":
        """Create a manifest with a deterministic ID and captured environment."""

        normalized_seeds = {str(key): int(value) for key, value in seeds.items()}
        normalized_config = dict(config)
        identity = {
            "dataset_id": dataset_id,
            "model_name": model_name,
            "split_hash": split_hash,
            "seeds": normalized_seeds,
            "config": normalized_config,
        }
        return cls(
            run_id=hash_json(identity)[:16],
            dataset_id=str(dataset_id),
            model_name=str(model_name),
            split_hash=str(split_hash),
            seeds=normalized_seeds,
            config=normalized_config,
            hashes=dict(hashes or {}),
            environment=capture_environment(repository),
            created_at_utc=datetime.now(timezone.utc).isoformat(),
        )

    def record_hash(self, name: str, value: str) -> None:
        """Record a named SHA-256 digest."""

        if not name:
            raise ValueError("hash name must not be empty")
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
            raise ValueError("hash value must be a SHA-256 hex digest")
        self.hashes[name] = value.lower()

    def record_timing(self, name: str, seconds: float) -> None:
        """Record one non-negative wall-clock duration."""

        if seconds < 0.0:
            raise ValueError("timing must be non-negative")
        self.timings_seconds[name] = float(seconds)

    def as_dict(self) -> Dict[str, object]:
        """Return JSON-friendly manifest values."""

        return asdict(self)

    def save(self, path: PathLike) -> Path:
        """Atomically persist the manifest as sorted JSON."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
        return destination

    @classmethod
    def load(cls, path: PathLike) -> "RunManifest":
        """Load a previously saved run manifest."""

        values = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**values)


class RunTimer(AbstractContextManager):
    """Context manager that writes elapsed time into a run manifest."""

    def __init__(self, manifest: RunManifest, name: str) -> None:
        self.manifest = manifest
        self.name = name
        self.started = 0.0

    def __enter__(self) -> "RunTimer":
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.manifest.record_timing(
            self.name, max(0.0, time.perf_counter() - self.started)
        )
        return None


__all__ = [
    "RunManifest",
    "RunTimer",
    "capture_environment",
    "hash_bytes",
    "hash_dataframe",
    "hash_json",
]
