"""Shared protocol, hashing, splitting, and capability primitives."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


PathLike = Union[str, os.PathLike]
PROTOCOL_VERSION = "1"
COMMON_LANES = ("numeric_regression", "multiclass_classwise")


class VersionComparisonError(RuntimeError):
    """Base error for version-comparison protocol failures."""


class VersionRefusalError(VersionComparisonError):
    """Raised when an isolated source imports a different version."""


class CapabilityError(VersionComparisonError):
    """Raised when a protocol requests an unsupported adapter capability."""


class CacheIntegrityError(VersionComparisonError):
    """Raised when a completed cache cell no longer matches its manifest."""


class HistoricalArtifactError(FileNotFoundError, VersionComparisonError):
    """Raised when historical evidence needed for rescoring is absent."""


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible values in a stable, strict form."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    )


def hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def hash_json(value: Any) -> str:
    return hash_bytes(canonical_json(value).encode("utf-8"))


def hash_file(path: PathLike) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def hash_dataframe(frame: pd.DataFrame) -> str:
    """Hash ordered values, index, labels, and dtypes exactly enough for cache keys."""

    metadata = {
        "columns": [repr(column) for column in frame.columns],
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "index_name": repr(frame.index.name),
    }
    payload = frame.to_csv(
        index=True,
        lineterminator="\n",
        float_format="%.17g",
        na_rep="__GANIFY_VERSION_COMPARISON_NA__",
    )
    return hash_bytes((canonical_json(metadata) + "\n" + payload).encode("utf-8"))


def hash_source_root(source_root: PathLike) -> str:
    """Hash the importable GANify source tree, including relative file names."""

    root = Path(source_root).expanduser().resolve()
    package = root / "ganify"
    if not (package / "__init__.py").is_file():
        raise FileNotFoundError(
            "source_root must contain ganify/__init__.py: %s" % root
        )
    files = []
    for path in package.rglob("*"):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        files.append(path)
    for name in ("setup.py", "pyproject.toml", "setup.cfg"):
        candidate = root / name
        if candidate.is_file():
            files.append(candidate)
    digest = hashlib.sha256()
    for path in sorted(set(files), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def atomic_write_json(path: PathLike, value: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def load_json(path: PathLike) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def git_provenance(source_root: PathLike) -> Dict[str, Any]:
    """Return commit/dirty metadata when a source root is a Git worktree."""

    root = Path(source_root).expanduser().resolve()

    def run(*arguments: str) -> Optional[str]:
        try:
            output = subprocess.check_output(
                ["git", *arguments],
                cwd=str(root),
                stderr=subprocess.DEVNULL,
                timeout=10,
                text=True,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            return None
        return output or None

    top = run("rev-parse", "--show-toplevel")
    if top is None or Path(top).resolve() != root:
        return {"commit": None, "dirty": None, "status_sha256": None}
    status = run("status", "--porcelain=v1", "--untracked-files=all") or ""
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status_sha256": hash_bytes(status.encode("utf-8")),
    }


def _canonical_scalar(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        value = value.isoformat()
    return canonical_json({"type": type(value).__name__, "value": value})


def _split_score(row_id: Any, seed: int, stratum: str) -> str:
    payload = "%d\0%s\0%s" % (int(seed), stratum, _canonical_scalar(row_id))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _split_hash(
    train: Sequence[Any], validation: Sequence[Any], test: Sequence[Any]
) -> str:
    return hash_json(
        {
            "train": sorted(_canonical_scalar(value) for value in train),
            "validation": sorted(
                _canonical_scalar(value) for value in validation
            ),
            "test": sorted(_canonical_scalar(value) for value in test),
        }
    )


@dataclass(frozen=True)
class FixedSplit:
    """Fixed row-ID partitions used by every compared source."""

    train: Tuple[Any, ...]
    validation: Tuple[Any, ...]
    test: Tuple[Any, ...]
    seed: int
    split_hash: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "train": list(self.train),
            "validation": list(self.validation),
            "test": list(self.test),
            "seed": self.seed,
            "split_hash": self.split_hash,
        }

    def assert_valid(self) -> None:
        partitions = [
            {_canonical_scalar(value) for value in values}
            for values in (self.train, self.validation, self.test)
        ]
        if partitions[0] & partitions[1] or partitions[0] & partitions[2]:
            raise ValueError("train row IDs overlap another partition")
        if partitions[1] & partitions[2]:
            raise ValueError("validation and test row IDs overlap")
        observed = _split_hash(self.train, self.validation, self.test)
        if observed != self.split_hash:
            raise ValueError(
                "split hash mismatch: expected %s, observed %s"
                % (self.split_hash, observed)
            )


def deterministic_row_split(
    row_ids: Iterable[Any],
    *,
    seed: int,
    test_size: float = 0.2,
    validation_size: float = 0.0,
    stratify: Optional[Iterable[Any]] = None,
) -> FixedSplit:
    """Make an order-independent split from immutable unique row IDs."""

    ids = list(row_ids)
    if len(ids) < 3:
        raise ValueError("at least three row IDs are required")
    if not 0.0 <= float(test_size) < 1.0:
        raise ValueError("test_size must be in [0, 1)")
    if not 0.0 <= float(validation_size) < 1.0:
        raise ValueError("validation_size must be in [0, 1)")
    if float(test_size) + float(validation_size) >= 1.0:
        raise ValueError("test_size + validation_size must be below 1")
    tokens = [_canonical_scalar(value) for value in ids]
    if len(tokens) != len(set(tokens)):
        raise ValueError("row IDs must be unique")

    if stratify is None:
        strata = ["__all__"] * len(ids)
    else:
        raw = list(stratify)
        if len(raw) != len(ids):
            raise ValueError("stratify must align with row IDs")
        strata = [_canonical_scalar(value) for value in raw]

    groups: Dict[str, List[Any]] = {}
    for row_id, stratum in zip(ids, strata):
        groups.setdefault(stratum, []).append(row_id)

    train: List[Any] = []
    validation: List[Any] = []
    test: List[Any] = []
    for stratum in sorted(groups):
        ordered = sorted(
            groups[stratum],
            key=lambda value: _split_score(value, int(seed), stratum),
        )
        count = len(ordered)
        n_test = int(round(count * float(test_size)))
        n_validation = int(round(count * float(validation_size)))
        if n_test + n_validation >= count:
            overflow = n_test + n_validation - count + 1
            reduce_validation = min(overflow, n_validation)
            n_validation -= reduce_validation
            n_test -= overflow - reduce_validation
        test.extend(ordered[:n_test])
        validation.extend(ordered[n_test : n_test + n_validation])
        train.extend(ordered[n_test + n_validation :])

    key = _canonical_scalar
    result = FixedSplit(
        train=tuple(sorted(train, key=key)),
        validation=tuple(sorted(validation, key=key)),
        test=tuple(sorted(test, key=key)),
        seed=int(seed),
        split_hash=_split_hash(train, validation, test),
    )
    result.assert_valid()
    if not result.train or not result.test:
        raise ValueError("split must produce non-empty train and test partitions")
    return result


def apply_fixed_split(
    frame: pd.DataFrame, split: FixedSplit, *, row_id_column: Any
) -> Dict[str, pd.DataFrame]:
    if row_id_column not in frame.columns:
        raise ValueError("row ID column %r is missing" % (row_id_column,))
    if bool(frame[row_id_column].duplicated().any()):
        raise ValueError("row ID column %r must be unique" % (row_id_column,))
    split.assert_valid()
    indexed = frame.set_index(row_id_column, drop=False)
    expected = {_canonical_scalar(value) for value in indexed.index}
    observed = {
        _canonical_scalar(value)
        for values in (split.train, split.validation, split.test)
        for value in values
    }
    if observed != expected:
        raise ValueError(
            "fixed split does not cover the dataframe exactly (missing=%d, extra=%d)"
            % (len(expected - observed), len(observed - expected))
        )
    return {
        "train": indexed.loc[list(split.train)].reset_index(drop=True),
        "validation": indexed.loc[list(split.validation)].reset_index(drop=True),
        "test": indexed.loc[list(split.test)].reset_index(drop=True),
    }


def allocate_class_counts(labels: Iterable[Any], n_rows: int) -> Tuple[Dict[str, Any], ...]:
    """Allocate exact proportional class counts with deterministic largest remainders."""

    count = int(n_rows)
    if count < 1:
        raise ValueError("n_rows must be positive")
    series = pd.Series(list(labels))
    if series.empty or bool(series.isna().any()):
        raise ValueError("class labels must be non-empty and non-missing")
    values: List[Any] = []
    frequencies: List[int] = []
    for value in series.tolist():
        token = _canonical_scalar(value)
        position = next(
            (
                index
                for index, existing in enumerate(values)
                if _canonical_scalar(existing) == token
            ),
            None,
        )
        if position is None:
            values.append(value.item() if isinstance(value, np.generic) else value)
            frequencies.append(1)
        else:
            frequencies[position] += 1
    order = sorted(range(len(values)), key=lambda index: _canonical_scalar(values[index]))
    values = [values[index] for index in order]
    frequencies = [frequencies[index] for index in order]
    ideal = np.asarray(frequencies, dtype=float) * count / float(sum(frequencies))
    allocated = np.floor(ideal).astype(int)
    remainder = count - int(allocated.sum())
    ranking = sorted(
        range(len(values)),
        key=lambda index: (-(ideal[index] - allocated[index]), _canonical_scalar(values[index])),
    )
    for index in ranking[:remainder]:
        allocated[index] += 1
    result = tuple(
        {"value": value, "count": int(class_count)}
        for value, class_count in zip(values, allocated.tolist())
        if class_count > 0
    )
    if sum(item["count"] for item in result) != count:
        raise RuntimeError("class allocation did not preserve requested row count")
    return result


def derived_seed(seed: int, *parts: Any) -> int:
    payload = canonical_json([int(seed), *parts]).encode("utf-8")
    # NumPy and TensorFlow both accept this positive 31-bit range.
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


@dataclass(frozen=True)
class AdapterProtocol:
    """Auditable implementation label and supported comparison capabilities."""

    name: str
    evidence_level: str
    source_policy: str
    lanes: Tuple[str, ...]
    implementation: str
    historical_version: Optional[str] = None
    aliases: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def require_lane(self, lane: str) -> None:
        normalized = normalize_lane(lane)
        if normalized not in self.lanes:
            raise CapabilityError(
                "%s does not support lane %r; supported lanes: %s"
                % (self.name, lane, ", ".join(self.lanes) or "none")
            )


ADAPTER_PROTOCOLS: Tuple[AdapterProtocol, ...] = (
    AdapterProtocol(
        name="v1.1_legacy_wgan",
        evidence_level="source_run",
        source_policy="exact_v1.1_source",
        lanes=COMMON_LANES,
        implementation="exact GANify 1.1 legacy fit_data(type='wgan')",
        historical_version="1.1.0",
        aliases=("v11", "v1_1", "legacy_v1.1"),
    ),
    AdapterProtocol(
        name="v1.2_historical_artifact",
        evidence_level="artifact_level",
        source_policy="artifact_only_no_source_claim",
        lanes=COMMON_LANES,
        implementation="rescore persisted v1.2 synthetic CSV artifacts only",
        historical_version="1.2.0",
        aliases=("v12_artifact", "v1_2_artifact"),
    ),
    AdapterProtocol(
        name="v1.2_recipe_reimplementation",
        evidence_level="recipe_reimplementation",
        source_policy="current_compatibility_api",
        lanes=COMMON_LANES,
        implementation="v1.2 recipe reimplemented through current Ganify.fit_data",
        historical_version="1.2.0",
        aliases=("v12_recipe", "v1_2_recipe"),
    ),
    AdapterProtocol(
        name="v2_numeric_compatibility",
        evidence_level="source_run",
        source_policy="exact_explicit_source",
        lanes=COMMON_LANES,
        implementation="current numeric compatibility fit_data API",
        aliases=("v2_numeric", "v2_compatibility"),
    ),
    AdapterProtocol(
        name="v2_numeric_recipe",
        evidence_level="source_run",
        source_policy="exact_explicit_source",
        lanes=COMMON_LANES,
        implementation="current numeric recipe fit_data API",
        aliases=("v2_recipe",),
    ),
    AdapterProtocol(
        name="v2_conditional_native",
        evidence_level="source_run",
        source_policy="exact_explicit_source",
        lanes=COMMON_LANES,
        implementation="current native conditional fit/sample API",
        aliases=("v2_conditional", "conditional_native"),
    ),
)

_ADAPTER_LOOKUP: Dict[str, AdapterProtocol] = {}
for _protocol in ADAPTER_PROTOCOLS:
    _ADAPTER_LOOKUP[_protocol.name] = _protocol
    for _alias in _protocol.aliases:
        _ADAPTER_LOOKUP[_alias] = _protocol


def get_adapter_protocol(name: str) -> AdapterProtocol:
    normalized = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in _ADAPTER_LOOKUP:
        raise KeyError(
            "unknown version adapter %r; available: %s"
            % (name, ", ".join(protocol.name for protocol in ADAPTER_PROTOCOLS))
        )
    return _ADAPTER_LOOKUP[normalized]


def normalize_lane(lane: str) -> str:
    normalized = str(lane).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "regression": "numeric_regression",
        "single_table_numeric_regression": "numeric_regression",
        "numeric": "numeric_regression",
        "classification": "multiclass_classwise",
        "multiclass": "multiclass_classwise",
        "classwise": "multiclass_classwise",
        "classwise_multiclass": "multiclass_classwise",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in COMMON_LANES:
        raise ValueError(
            "lane must be one of %s" % ", ".join(COMMON_LANES)
        )
    return normalized


def ensure_finite_numeric(frame: pd.DataFrame, *, name: str) -> None:
    if frame.empty or frame.shape[1] < 1:
        raise CapabilityError("%s must contain rows and columns" % name)
    non_numeric = [
        column
        for column in frame.columns
        if not pd.api.types.is_numeric_dtype(frame[column].dtype)
    ]
    if non_numeric:
        raise CapabilityError(
            "%s requires numeric columns; unsupported columns: %r"
            % (name, non_numeric)
        )
    values = frame.to_numpy(dtype=float)
    if not bool(np.isfinite(values).all()):
        raise CapabilityError("%s requires finite, non-missing values" % name)


def validate_seed_list(values: Any, name: str) -> Tuple[int, ...]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("%s must be a non-empty list" % name)
    seeds = tuple(int(value) for value in values)
    if len(set(seeds)) != len(seeds):
        raise ValueError("%s contains duplicate seeds" % name)
    return seeds


def resolve_path(value: PathLike, base: PathLike) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(base) / path
    return path.resolve()


def validate_protocol(protocol: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate the top-level JSON protocol without importing GANify."""

    values = dict(protocol)
    version = str(values.get("protocol_version", PROTOCOL_VERSION))
    if version != PROTOCOL_VERSION:
        raise ValueError(
            "unsupported protocol_version %r (expected %s)"
            % (version, PROTOCOL_VERSION)
        )
    datasets = values.get("datasets", [])
    if not isinstance(datasets, list):
        raise ValueError("protocol datasets must be a list")
    if not datasets and "historical_kuairand" not in values:
        raise ValueError(
            "protocol needs datasets or a historical_kuairand rescoring block"
        )
    models = values.get("models", values.get("sources", []))
    if not isinstance(models, list):
        raise ValueError("protocol models must be a list")
    if not models and "historical_kuairand" not in values:
        raise ValueError(
            "protocol needs models or a historical_kuairand rescoring block"
        )
    validate_seed_list(values.get("split_seeds", [42]), "split_seeds")
    validate_seed_list(
        values.get("fit_seeds", values.get("model_seeds", [42])),
        "fit_seeds",
    )
    validate_seed_list(values.get("sample_seeds", [42]), "sample_seeds")
    for dataset in datasets:
        if not isinstance(dataset, Mapping):
            raise TypeError("dataset entries must be mappings")
        for required in ("id", "path", "row_id", "lane"):
            if required not in dataset:
                raise ValueError("dataset entry is missing %r" % required)
        lane = normalize_lane(str(dataset["lane"]))
        if lane == "multiclass_classwise" and not dataset.get("target"):
            raise ValueError("multiclass_classwise datasets require target")
    for model in models:
        if not isinstance(model, Mapping):
            raise TypeError("model entries must be mappings")
        for required in ("adapter",):
            if required not in model:
                raise ValueError("model entry is missing %r" % required)
        adapter = get_adapter_protocol(str(model["adapter"]))
        if adapter.evidence_level == "artifact_level":
            if "artifact_path" not in model and "artifact_paths" not in model:
                raise ValueError(
                    "artifact-backed model %r needs artifact_path or "
                    "artifact_paths" % adapter.name
                )
        else:
            for required in ("source_root", "expected_version"):
                if required not in model:
                    raise ValueError(
                        "source-backed model %r is missing %r"
                        % (adapter.name, required)
                    )
    # This also proves that cache identities can be serialized strictly.
    canonical_json(values)
    values["protocol_version"] = version
    values["models"] = [dict(item) for item in models]
    return values


__all__ = [
    "ADAPTER_PROTOCOLS",
    "AdapterProtocol",
    "COMMON_LANES",
    "CacheIntegrityError",
    "CapabilityError",
    "FixedSplit",
    "HistoricalArtifactError",
    "PROTOCOL_VERSION",
    "VersionComparisonError",
    "VersionRefusalError",
    "allocate_class_counts",
    "apply_fixed_split",
    "atomic_write_json",
    "canonical_json",
    "derived_seed",
    "deterministic_row_split",
    "ensure_finite_numeric",
    "get_adapter_protocol",
    "git_provenance",
    "hash_bytes",
    "hash_dataframe",
    "hash_file",
    "hash_json",
    "hash_source_root",
    "load_json",
    "normalize_lane",
    "resolve_path",
    "validate_protocol",
    "validate_seed_list",
]
