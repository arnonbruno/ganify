"""Suite manifests tying datasets to deterministic seed budgets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from ._config import PathLike, load_config
from .data import DEFAULT_CATALOG, DatasetManifest, load_dataset_catalog


DEFAULT_SUITE_DIRECTORY = Path(__file__).resolve().parents[1] / "configs" / "suites"


@dataclass(frozen=True)
class BenchmarkSuite:
    """Resolved benchmark suite configuration."""

    name: str
    version: str
    datasets: Tuple[DatasetManifest, ...]
    split_seeds: Tuple[int, ...]
    model_seeds: Tuple[int, ...]
    sample_seeds: Tuple[int, ...]
    max_rows: Optional[int]
    description: str

    def as_dict(self) -> Dict[str, Any]:
        """Return JSON-friendly suite values."""

        return {
            "name": self.name,
            "version": self.version,
            "datasets": [item.dataset_id for item in self.datasets],
            "split_seeds": list(self.split_seeds),
            "model_seeds": list(self.model_seeds),
            "sample_seeds": list(self.sample_seeds),
            "max_rows": self.max_rows,
            "description": self.description,
        }


def _suite_path(name_or_path: PathLike) -> Path:
    candidate = Path(name_or_path)
    if candidate.is_file():
        return candidate
    name = str(name_or_path)
    if name.endswith((".yaml", ".yml", ".json")):
        path = DEFAULT_SUITE_DIRECTORY / name
    else:
        path = DEFAULT_SUITE_DIRECTORY / ("%s.yaml" % name)
    return path


def _seed_tuple(values: object, name: str) -> Tuple[int, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError("%s must be a non-empty list" % name)
    seeds = tuple(int(value) for value in values)
    if len(set(seeds)) != len(seeds):
        raise ValueError("%s contains duplicate seeds" % name)
    return seeds


def load_suite(
    name_or_path: PathLike,
    *,
    catalog_path: PathLike = DEFAULT_CATALOG,
) -> BenchmarkSuite:
    """Load a suite and resolve every dataset against the pinned catalog."""

    values = load_config(_suite_path(name_or_path))
    ids = values.get("datasets")
    if not isinstance(ids, list) or not ids:
        raise ValueError("suite must contain a non-empty datasets list")
    if len(set(ids)) != len(ids):
        raise ValueError("suite contains duplicate dataset IDs")
    catalog = load_dataset_catalog(catalog_path)
    unknown = [dataset_id for dataset_id in ids if dataset_id not in catalog]
    if unknown:
        raise ValueError("suite references unknown datasets: %r" % unknown)
    name = str(values.get("name", Path(name_or_path).stem))
    max_rows = values.get("max_rows")
    return BenchmarkSuite(
        name=name,
        version=str(values.get("version", "unversioned")),
        datasets=tuple(catalog[dataset_id] for dataset_id in ids),
        split_seeds=_seed_tuple(values.get("split_seeds", [0]), "split_seeds"),
        model_seeds=_seed_tuple(values.get("model_seeds", [0]), "model_seeds"),
        sample_seeds=_seed_tuple(values.get("sample_seeds", [0]), "sample_seeds"),
        max_rows=None if max_rows is None else int(max_rows),
        description=str(values.get("description", "")),
    )


__all__ = ["BenchmarkSuite", "DEFAULT_SUITE_DIRECTORY", "load_suite"]
