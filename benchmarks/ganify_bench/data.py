"""Pinned dataset metadata and explicit offline loading behavior."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import pandas as pd

from ._config import PathLike, load_config


DEFAULT_CATALOG = (
    Path(__file__).resolve().parents[1] / "manifests" / "datasets" / "catalog.yaml"
)


class DatasetUnavailableError(FileNotFoundError):
    """Raised when a manifest exists but its data is unavailable offline."""


@dataclass(frozen=True)
class DatasetManifest:
    """Pinned metadata needed to acquire and evaluate one benchmark dataset."""

    dataset_id: str
    name: str
    version: str
    tiers: Tuple[str, ...]
    task: str
    target: Optional[str]
    source_url: str
    license: str
    local_path: str
    file_format: str = "csv"
    sha256: Optional[str] = None
    rows: Optional[int] = None
    columns: Optional[int] = None
    stressors: Tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""
    download: str = "deferred"

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "DatasetManifest":
        """Validate and construct a manifest from YAML/JSON values."""

        required = {
            "id",
            "name",
            "version",
            "tiers",
            "task",
            "source_url",
            "license",
            "local_path",
        }
        missing = sorted(required.difference(values))
        if missing:
            raise ValueError("dataset manifest is missing fields: %r" % missing)
        task = str(values["task"]).lower()
        if task not in {
            "binary_classification",
            "multiclass_classification",
            "regression",
            "structural",
            "controlled",
        }:
            raise ValueError("unsupported dataset task %r" % task)
        tiers = tuple(str(item) for item in values["tiers"])
        if not tiers:
            raise ValueError("dataset %r must belong to a suite tier" % values["id"])
        return cls(
            dataset_id=str(values["id"]),
            name=str(values["name"]),
            version=str(values["version"]),
            tiers=tiers,
            task=task,
            target=(
                None
                if values.get("target") is None
                else str(values.get("target"))
            ),
            source_url=str(values["source_url"]),
            license=str(values["license"]),
            local_path=str(values["local_path"]),
            file_format=str(values.get("format", "csv")).lower(),
            sha256=(
                None if values.get("sha256") in (None, "") else str(values["sha256"])
            ),
            rows=(None if values.get("rows") is None else int(values["rows"])),
            columns=(
                None if values.get("columns") is None else int(values["columns"])
            ),
            stressors=tuple(str(item) for item in values.get("stressors", [])),
            notes=str(values.get("notes", "")),
            download=str(values.get("download", "deferred")),
        )

    def as_dict(self) -> Dict[str, Any]:
        """Return JSON-friendly manifest values."""

        return {
            "id": self.dataset_id,
            "name": self.name,
            "version": self.version,
            "tiers": list(self.tiers),
            "task": self.task,
            "target": self.target,
            "source_url": self.source_url,
            "license": self.license,
            "local_path": self.local_path,
            "format": self.file_format,
            "sha256": self.sha256,
            "rows": self.rows,
            "columns": self.columns,
            "stressors": list(self.stressors),
            "notes": self.notes,
            "download": self.download,
        }


def load_dataset_catalog(
    path: PathLike = DEFAULT_CATALOG,
) -> Dict[str, DatasetManifest]:
    """Load all pinned dataset manifests, rejecting duplicate IDs."""

    values = load_config(path)
    entries = values.get("datasets")
    if not isinstance(entries, list) or not entries:
        raise ValueError("dataset catalog must contain a non-empty 'datasets' list")
    catalog: Dict[str, DatasetManifest] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("every dataset catalog entry must be a mapping")
        manifest = DatasetManifest.from_mapping(entry)
        if manifest.dataset_id in catalog:
            raise ValueError("duplicate dataset id %r" % manifest.dataset_id)
        catalog[manifest.dataset_id] = manifest
    return catalog


def get_dataset_manifest(
    dataset_id: str, path: PathLike = DEFAULT_CATALOG
) -> DatasetManifest:
    """Return one dataset manifest by stable ID."""

    catalog = load_dataset_catalog(path)
    if dataset_id not in catalog:
        raise KeyError(
            "unknown dataset %r; available IDs: %s"
            % (dataset_id, ", ".join(sorted(catalog)))
        )
    return catalog[dataset_id]


def sha256_file(path: PathLike, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it entirely into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _missing_dataset_message(
    manifest: DatasetManifest, destination: Path
) -> str:
    checksum = (
        " Verify SHA-256 %s." % manifest.sha256 if manifest.sha256 else ""
    )
    return (
        "Dataset %r is registered but unavailable offline at %s. "
        "Automatic download is %s; obtain it from %s and place the extracted "
        "file at that path.%s"
        % (
            manifest.dataset_id,
            destination,
            manifest.download,
            manifest.source_url,
            checksum,
        )
    )


def load_dataset(
    manifest_or_id: Union[DatasetManifest, str],
    *,
    data_root: PathLike,
    catalog_path: PathLike = DEFAULT_CATALOG,
    verify_hash: bool = True,
) -> pd.DataFrame:
    """Load one local dataset or raise an actionable deferred-download error."""

    manifest = (
        get_dataset_manifest(manifest_or_id, catalog_path)
        if isinstance(manifest_or_id, str)
        else manifest_or_id
    )
    root = Path(data_root)
    destination = root / manifest.local_path
    if not destination.is_file():
        raise DatasetUnavailableError(_missing_dataset_message(manifest, destination))
    if verify_hash and manifest.sha256:
        observed = sha256_file(destination)
        if observed.lower() != manifest.sha256.lower():
            raise ValueError(
                "SHA-256 mismatch for %r: expected %s, observed %s"
                % (manifest.dataset_id, manifest.sha256, observed)
            )
    if manifest.file_format == "csv":
        return pd.read_csv(destination)
    if manifest.file_format in {"parquet", "pq"}:
        try:
            return pd.read_parquet(destination)
        except ImportError as error:
            raise ImportError(
                "Parquet dataset %r needs an installed pandas parquet engine"
                % manifest.dataset_id
            ) from error
    if manifest.file_format in {"jsonl", "ndjson"}:
        return pd.read_json(destination, lines=True)
    raise ValueError(
        "unsupported format %r for dataset %r"
        % (manifest.file_format, manifest.dataset_id)
    )


__all__ = [
    "DEFAULT_CATALOG",
    "DatasetManifest",
    "DatasetUnavailableError",
    "get_dataset_manifest",
    "load_dataset",
    "load_dataset_catalog",
    "sha256_file",
]
