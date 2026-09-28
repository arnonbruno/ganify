"""Content-addressed orchestration for reproducible version comparisons."""

from __future__ import annotations

import json
import os
import platform
import sys
import time
import traceback
from importlib import metadata as importlib_metadata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .aggregation import AggregationResult, aggregate_version_comparison
from .core import (
    CacheIntegrityError,
    CapabilityError,
    FixedSplit,
    apply_fixed_split,
    atomic_write_json,
    deterministic_row_split,
    get_adapter_protocol,
    git_provenance,
    hash_dataframe,
    hash_file,
    hash_json,
    hash_source_root,
    load_json,
    normalize_lane,
    resolve_path,
    validate_protocol,
    validate_seed_list,
)
from .evaluator import EvaluationResult, evaluate_stages
from .historical import HistoricalRescoreResult, rescore_historical_kuairand
from .isolation import WORKER_PATH, invoke_source_worker


PathLike = Union[str, os.PathLike]


@dataclass
class VersionComparisonRunResult:
    """Persisted comparison evidence and hierarchy-aware summaries."""

    manifests: pd.DataFrame
    metrics: pd.DataFrame
    privacy: pd.DataFrame
    reliability: pd.DataFrame
    samples: pd.DataFrame
    splits: List[Dict[str, Any]]
    protocol: Dict[str, Any]
    aggregation: Optional[AggregationResult] = None
    historical: Optional[HistoricalRescoreResult] = None

    def save(self, directory: PathLike) -> Dict[str, str]:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        paths = {
            "manifests": destination / "manifests.csv",
            "metrics": destination / "metrics.csv",
            "privacy": destination / "privacy.csv",
            "reliability": destination / "reliability.csv",
            "samples": destination / "samples.csv",
            "splits": destination / "splits.json",
            "protocol": destination / "resolved_protocol.json",
        }
        self.manifests.to_csv(paths["manifests"], index=False)
        self.metrics.to_csv(paths["metrics"], index=False)
        self.privacy.to_csv(paths["privacy"], index=False)
        self.reliability.to_csv(paths["reliability"], index=False)
        self.samples.to_csv(paths["samples"], index=False)
        atomic_write_json(paths["splits"], self.splits)
        atomic_write_json(paths["protocol"], self.protocol)
        if self.aggregation is not None:
            fit_path = destination / "fit_means.csv"
            summary_path = destination / "fit_summary.csv"
            delta_path = destination / "paired_deltas.csv"
            self.aggregation.fit_means.to_csv(fit_path, index=False)
            self.aggregation.summary.to_csv(summary_path, index=False)
            self.aggregation.paired_deltas.to_csv(delta_path, index=False)
            paths.update(
                {
                    "fit_means": fit_path,
                    "fit_summary": summary_path,
                    "paired_deltas": delta_path,
                }
            )
        return {name: str(path) for name, path in paths.items()}


def load_protocol(path: PathLike) -> Dict[str, Any]:
    """Load and validate a strict JSON comparison protocol."""

    protocol_path = Path(path).expanduser().resolve()
    if protocol_path.suffix.lower() != ".json":
        raise ValueError("version comparison protocol must be JSON")
    values = load_json(protocol_path)
    if not isinstance(values, Mapping):
        raise TypeError("version comparison protocol must contain a JSON object")
    return validate_protocol(values)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tooling_hash() -> str:
    files = sorted(Path(__file__).resolve().parent.glob("*.py"))
    return hash_json(
        [{"name": path.name, "sha256": hash_file(path)} for path in files]
    )


def _runtime_environment() -> Dict[str, Any]:
    versions = {}
    for name in (
        "ganify",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "tensorflow",
        "keras",
    ):
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": versions,
        "determinism": {
            name: os.environ.get(name)
            for name in (
                "PYTHONHASHSEED",
                "TF_DETERMINISTIC_OPS",
                "TF_CUDNN_DETERMINISTIC",
                "CUDA_VISIBLE_DEVICES",
            )
        },
    }


def _strict_csv(path: Path, options: Optional[Mapping[str, Any]]) -> pd.DataFrame:
    frame = pd.read_csv(path, **dict(options or {}))
    if frame.empty or frame.columns.has_duplicates:
        raise ValueError("dataset must contain rows and unique columns: %s" % path)
    return frame


def _model_label(model: Mapping[str, Any], adapter_name: str) -> str:
    return str(model.get("name", adapter_name))


def _version_label(
    model: Mapping[str, Any], adapter_name: str, expected_version: Optional[str]
) -> str:
    if model.get("version_label") is not None:
        return str(model["version_label"])
    if adapter_name == "v1.1_legacy_wgan":
        return "v1.1_exact_source"
    if adapter_name == "v1.2_recipe_reimplementation":
        return "v1.2_recipe_reimplementation"
    if adapter_name == "v1.2_historical_artifact":
        return "v1.2_historical_artifact"
    if adapter_name.startswith("v2_"):
        return adapter_name
    return str(expected_version or adapter_name)


def _sample_hashes(worker_result: Mapping[str, Any]) -> Iterable[Tuple[Path, str]]:
    for sample in worker_result.get("samples", []):
        for stage in sample.get("stages", []):
            yield Path(stage["path"]), str(stage["sha256"])


def _verify_cached_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("status") != "completed":
        return
    worker = manifest.get("worker", {})
    for path, expected in _sample_hashes(worker):
        if not path.is_file():
            raise CacheIntegrityError("cached sample is absent: %s" % path)
        observed = hash_file(path)
        if observed != expected:
            raise CacheIntegrityError(
                "cached sample hash mismatch for %s: expected %s, observed %s"
                % (path, expected, observed)
            )
    for output in manifest.get("evaluation_outputs", []):
        for item in output.get("files", {}).values():
            path = Path(item["path"])
            if not path.is_file():
                raise CacheIntegrityError(
                    "cached evaluation output is absent: %s" % path
                )
            if hash_file(path) != item["sha256"]:
                raise CacheIntegrityError(
                    "cached evaluation output hash mismatch: %s" % path
                )


def _read_evaluation_outputs(
    manifest: Mapping[str, Any]
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]:
    metrics: List[pd.DataFrame] = []
    privacy: List[pd.DataFrame] = []
    reliability: List[pd.DataFrame] = []
    for output in manifest.get("evaluation_outputs", []):
        files = output["files"]
        metrics.append(pd.read_csv(files["metrics"]["path"]))
        privacy.append(pd.read_csv(files["privacy"]["path"]))
        reliability.append(pd.read_csv(files["reliability"]["path"]))
    return metrics, privacy, reliability


def _annotate(
    frame: pd.DataFrame,
    *,
    dataset_id: str,
    lane: str,
    model_name: str,
    adapter: str,
    version: str,
    source_version: Optional[str],
    split_seed: int,
    fit_seed: int,
    sample_seed: int,
    split_hash: str,
    fit_id: str,
) -> pd.DataFrame:
    output = frame.copy()
    values = {
        "dataset_id": dataset_id,
        "lane": lane,
        "model_name": model_name,
        "adapter": adapter,
        "version": version,
        "source_version": source_version,
        "split_seed": int(split_seed),
        "fit_seed": int(fit_seed),
        "sample_seed": int(sample_seed),
        "split_hash": split_hash,
        "fit_id": fit_id,
    }
    for name, value in reversed(list(values.items())):
        output.insert(0, name, value)
    if "candidate" in output.columns:
        control = output["candidate"].astype(str).str.startswith("control:")
        output.loc[control, "version"] = output.loc[control, "candidate"]
        output.loc[control, "adapter"] = output.loc[control, "candidate"]
        output.loc[control, "model_name"] = output.loc[control, "candidate"]
    return output


def _aggregation_records(
    metrics: pd.DataFrame, reliability: pd.DataFrame
) -> pd.DataFrame:
    """Add failed/missing metric cells from reliability without fabricating values."""

    if metrics.empty:
        return metrics.copy()
    records = metrics.copy()
    records["success"] = np.isfinite(
        pd.to_numeric(records["value"], errors="coerce")
    )
    if reliability.empty or "run_success" not in reliability.columns:
        return records
    metric_ids = [
        name
        for name in ("stage", "pillar", "metric", "column", "detail")
        if name in records.columns
    ]
    identity = [
        name
        for name in (
            "dataset_id",
            "lane",
            "model_name",
            "adapter",
            "version",
            "source_version",
            "split_seed",
            "fit_seed",
            "sample_seed",
            "split_hash",
            "fit_id",
            "candidate",
        )
        if name in records.columns and name in reliability.columns
    ]
    failures = reliability.loc[
        ~reliability["run_success"].fillna(False).astype(bool)
    ]
    placeholders: List[Dict[str, Any]] = []
    for _, failure in failures.iterrows():
        adapter = str(failure.get("adapter", ""))
        if "historical_artifact" in adapter or adapter.startswith("control:"):
            continue
        universe = records
        for name in ("dataset_id", "lane"):
            if name in universe.columns and name in failure.index:
                universe = universe.loc[universe[name] == failure[name]]
        stage = failure.get("stage")
        if (
            "stage" in universe.columns
            and stage is not None
            and not pd.isna(stage)
        ):
            universe = universe.loc[universe["stage"] == stage]
        universe = universe.loc[
            ~universe.get(
                "adapter", pd.Series(index=universe.index, dtype=object)
            )
            .astype(str)
            .str.startswith("control:")
        ]
        for metric_values in universe.loc[:, metric_ids].drop_duplicates().to_dict(
            "records"
        ):
            row = {
                name: failure[name]
                for name in identity
                if name in failure.index
            }
            row.update(metric_values)
            row.update({"value": float("nan"), "success": False})
            placeholders.append(row)
    if not placeholders:
        return records
    additions = pd.DataFrame(placeholders)
    keys = identity + metric_ids
    existing_tokens = {
        tuple("__NA__" if pd.isna(value) else repr(value) for value in row)
        for row in records.loc[:, keys].itertuples(index=False, name=None)
    }
    keep = []
    for row in additions.loc[:, keys].itertuples(index=False, name=None):
        token = tuple("__NA__" if pd.isna(value) else repr(value) for value in row)
        keep.append(token not in existing_tokens)
    additions = additions.loc[keep]
    if additions.empty:
        return records
    return pd.concat([records, additions], ignore_index=True, sort=False)


class VersionComparisonRunner:
    """Run source-isolated fit cells with fail-closed content-addressed resume."""

    def __init__(
        self,
        protocol: Union[PathLike, Mapping[str, Any]],
        *,
        output_directory: Optional[PathLike] = None,
        resume: bool = True,
        retry_failures: bool = False,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        if isinstance(protocol, Mapping):
            self.protocol_path: Optional[Path] = None
            self.base_directory = Path.cwd()
            self.protocol = validate_protocol(protocol)
        else:
            self.protocol_path = Path(protocol).expanduser().resolve()
            self.base_directory = self.protocol_path.parent
            self.protocol = load_protocol(self.protocol_path)
        configured_output = self.protocol.get(
            "output_dir", "version_comparison_output"
        )
        self.output_directory = (
            resolve_path(configured_output, self.base_directory)
            if output_directory is None
            else resolve_path(output_directory, self.base_directory)
        )
        self.resume = bool(resume)
        self.retry_failures = bool(retry_failures)
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else self.protocol.get("timeout_seconds")
        )
        self.tooling_sha256 = _tooling_hash()
        self.evaluation_source_sha256 = hash_source_root(
            Path(__file__).resolve().parents[3]
        )
        self.runtime_environment = _runtime_environment()

    def _resolved_dataset(
        self, specification: Mapping[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        path = resolve_path(specification["path"], self.base_directory)
        if not path.is_file():
            raise FileNotFoundError(
                "dataset %r is unavailable at %s; no network fetch is performed"
                % (specification["id"], path)
            )
        frame = _strict_csv(path, specification.get("read_csv"))
        row_id = specification["row_id"]
        if row_id not in frame.columns:
            raise ValueError(
                "dataset %r is missing fixed row ID column %r"
                % (specification["id"], row_id)
            )
        if bool(frame[row_id].duplicated().any()):
            raise ValueError(
                "dataset %r row IDs are not unique" % specification["id"]
            )
        maximum = specification.get("max_rows", self.protocol.get("max_rows"))
        if maximum is not None and len(frame) > int(maximum):
            # Stable row-ID ordering makes deterministic truncation independent
            # of source CSV order.
            tokens = frame[row_id].map(
                lambda value: hash_json(
                    {"dataset": specification["id"], "row_id": value}
                )
            )
            frame = frame.loc[tokens.sort_values().index[: int(maximum)]].reset_index(
                drop=True
            )
        metadata = {
            "path": str(path),
            "file_sha256": hash_file(path),
            "dataframe_sha256": hash_dataframe(frame),
            "rows": int(len(frame)),
            "columns": [str(column) for column in frame.columns],
        }
        return frame, metadata

    def _split_dataset(
        self,
        frame: pd.DataFrame,
        specification: Mapping[str, Any],
        split_seed: int,
    ) -> Tuple[FixedSplit, Dict[str, pd.DataFrame], Dict[str, Any]]:
        lane = normalize_lane(str(specification["lane"]))
        target = specification.get("target")
        stratify = (
            frame[target]
            if lane == "multiclass_classwise"
            and bool(specification.get("stratify", True))
            else None
        )
        split = deterministic_row_split(
            frame[specification["row_id"]],
            seed=int(split_seed),
            test_size=float(specification.get("test_size", 0.2)),
            validation_size=float(specification.get("validation_size", 0.0)),
            stratify=stratify,
        )
        partitions = apply_fixed_split(
            frame, split, row_id_column=specification["row_id"]
        )
        drop_columns = [specification["row_id"]] + list(
            specification.get("drop_columns", [])
        )
        model_partitions = {
            name: values.drop(columns=drop_columns)
            for name, values in partitions.items()
        }
        manifest = {
            "dataset_id": str(specification["id"]),
            **split.as_dict(),
            "row_id_column": specification["row_id"],
            "lane": lane,
            "target": target,
            "partition_hashes": {
                name: hash_dataframe(values)
                for name, values in model_partitions.items()
            },
        }
        split_path = (
            self.output_directory
            / "splits"
            / ("%s_%s.json" % (specification["id"], split.split_hash[:16]))
        )
        atomic_write_json(split_path, manifest)
        manifest["path"] = str(split_path)
        return split, model_partitions, manifest

    def _cell_identity(
        self,
        *,
        dataset_spec: Mapping[str, Any],
        dataset_metadata: Mapping[str, Any],
        split: FixedSplit,
        train: pd.DataFrame,
        model: Mapping[str, Any],
        source_hash: str,
        adapter_name: str,
        fit_seed: int,
        sample_seeds: Sequence[int],
        sample_rows: int,
        stages: Any,
    ) -> Dict[str, Any]:
        return {
            "protocol_version": self.protocol["protocol_version"],
            "tooling_sha256": self.tooling_sha256,
            "evaluation_source_sha256": self.evaluation_source_sha256,
            "runtime_environment": self.runtime_environment,
            "worker_sha256": hash_file(WORKER_PATH),
            "dataset_id": str(dataset_spec["id"]),
            "dataset_protocol": dict(dataset_spec),
            "dataset_file_sha256": dataset_metadata["file_sha256"],
            "dataset_dataframe_sha256": dataset_metadata["dataframe_sha256"],
            "train_dataframe_sha256": hash_dataframe(train),
            "split_hash": split.split_hash,
            "split_seed": int(split.seed),
            "adapter": adapter_name,
            "model_name": _model_label(model, adapter_name),
            "model_config": dict(model.get("config", {})),
            "source_root": str(model.get("source_root", "")),
            "expected_version": model.get("expected_version"),
            "source_sha256": source_hash,
            "fit_seed": int(fit_seed),
            "sample_seeds": [int(seed) for seed in sample_seeds],
            "seeds": {
                "split": int(split.seed),
                "training": int(fit_seed),
                "model": int(fit_seed),
                "sample": [int(seed) for seed in sample_seeds],
            },
            "sample_rows": int(sample_rows),
            "lane": normalize_lane(str(dataset_spec["lane"])),
            "target": dataset_spec.get("target"),
            "stages": stages,
            "evaluation": dict(self.protocol.get("evaluation", {})),
        }

    def _failure_manifest(
        self,
        *,
        cell_id: str,
        identity: Mapping[str, Any],
        attempt: int,
        phase: str,
        error: Exception,
    ) -> Dict[str, Any]:
        return {
            "cell_id": cell_id,
            "identity": dict(identity),
            "status": "failed",
            "attempt": int(attempt),
            "failure_phase": phase,
            "failure_kind": (
                "capability" if isinstance(error, CapabilityError) else "error"
            ),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "finished_at_utc": _now(),
            "cache_hit": False,
        }

    def _cache_setup_failure(
        self,
        *,
        dataset_spec: Mapping[str, Any],
        dataset_metadata: Mapping[str, Any],
        split: FixedSplit,
        model: Mapping[str, Any],
        fit_seed: int,
        phase: str,
        error: Exception,
    ) -> Dict[str, Any]:
        """Persist pre-worker failures so resume never retries them silently."""

        adapter = get_adapter_protocol(str(model["adapter"]))
        identity = {
            "protocol_version": self.protocol["protocol_version"],
            "tooling_sha256": self.tooling_sha256,
            "evaluation_source_sha256": self.evaluation_source_sha256,
            "runtime_environment": self.runtime_environment,
            "dataset_id": str(dataset_spec["id"]),
            "dataset_protocol": dict(dataset_spec),
            "dataset_file_sha256": dataset_metadata["file_sha256"],
            "dataset_dataframe_sha256": dataset_metadata["dataframe_sha256"],
            "split_hash": split.split_hash,
            "split_seed": int(split.seed),
            "adapter": adapter.name,
            "model_name": _model_label(model, adapter.name),
            "model": dict(model),
            "expected_version": model.get("expected_version"),
            "source_sha256": None,
            "fit_seed": int(fit_seed),
            "failure_boundary": phase,
        }
        cell_id = hash_json(identity)
        cell_directory = self.output_directory / "cells" / cell_id
        manifest_path = cell_directory / "manifest.json"
        if self.resume and manifest_path.is_file():
            cached = dict(load_json(manifest_path))
            if not self.retry_failures:
                cached["cache_hit"] = True
                cached["failure_reused"] = True
                return cached
            attempt = int(cached.get("attempt", 1)) + 1
        else:
            attempt = 1
        manifest = self._failure_manifest(
            cell_id=cell_id,
            identity=identity,
            attempt=attempt,
            phase=phase,
            error=error,
        )
        cell_directory.mkdir(parents=True, exist_ok=True)
        atomic_write_json(manifest_path, manifest)
        return manifest

    def _evaluate_worker(
        self,
        manifest: Dict[str, Any],
        train: pd.DataFrame,
        test: pd.DataFrame,
        dataset_spec: Mapping[str, Any],
        model: Mapping[str, Any],
        cell_directory: Path,
    ) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]:
        worker = manifest["worker"]
        adapter_name = str(worker["adapter"]["adapter"])
        model_name = _model_label(model, adapter_name)
        version = _version_label(
            model, adapter_name, str(model.get("expected_version"))
        )
        source_version = worker.get("provenance", {}).get("observed_version")
        evaluation_config = dict(self.protocol.get("evaluation", {}))
        categorical = dataset_spec.get(
            "categorical_columns",
            evaluation_config.get("categorical_columns"),
        )
        constraints = dataset_spec.get(
            "constraints", evaluation_config.get("constraints")
        )
        privacy = dict(evaluation_config.get("privacy", {}))
        include_controls = bool(evaluation_config.get("controls", True))
        task = dataset_spec.get("task")
        if task is None:
            task = (
                "regression"
                if normalize_lane(str(dataset_spec["lane"])) == "numeric_regression"
                else "classification"
            )
        target = dataset_spec.get("target")
        utility_target = target if target is not None else None
        utility_task = task if utility_target is not None else None

        metric_frames: List[pd.DataFrame] = []
        privacy_frames: List[pd.DataFrame] = []
        reliability_frames: List[pd.DataFrame] = []
        outputs = []
        for sample in worker.get("samples", []):
            sample_seed = int(sample["sample_seed"])
            stages = {
                str(record["stage"]): pd.read_csv(record["path"])
                for record in sample["stages"]
            }
            result = evaluate_stages(
                train,
                test,
                stages,
                candidate=model_name,
                categorical_columns=categorical,
                constraints=constraints,
                target_column=utility_target,
                task=utility_task,
                random_state=sample_seed,
                c2st_folds=int(evaluation_config.get("c2st_folds", 5)),
                privacy=privacy,
                include_controls=include_controls,
            )
            fit_id = manifest["cell_id"]
            annotated_metrics = _annotate(
                result.metrics,
                dataset_id=str(dataset_spec["id"]),
                lane=normalize_lane(str(dataset_spec["lane"])),
                model_name=model_name,
                adapter=adapter_name,
                version=version,
                source_version=source_version,
                split_seed=int(manifest["identity"]["split_seed"]),
                fit_seed=int(manifest["identity"]["fit_seed"]),
                sample_seed=sample_seed,
                split_hash=str(manifest["identity"]["split_hash"]),
                fit_id=fit_id,
            )
            annotated_privacy = _annotate(
                result.privacy,
                dataset_id=str(dataset_spec["id"]),
                lane=normalize_lane(str(dataset_spec["lane"])),
                model_name=model_name,
                adapter=adapter_name,
                version=version,
                source_version=source_version,
                split_seed=int(manifest["identity"]["split_seed"]),
                fit_seed=int(manifest["identity"]["fit_seed"]),
                sample_seed=sample_seed,
                split_hash=str(manifest["identity"]["split_hash"]),
                fit_id=fit_id,
            )
            annotated_reliability = _annotate(
                result.reliability,
                dataset_id=str(dataset_spec["id"]),
                lane=normalize_lane(str(dataset_spec["lane"])),
                model_name=model_name,
                adapter=adapter_name,
                version=version,
                source_version=source_version,
                split_seed=int(manifest["identity"]["split_seed"]),
                fit_seed=int(manifest["identity"]["fit_seed"]),
                sample_seed=sample_seed,
                split_hash=str(manifest["identity"]["split_hash"]),
                fit_id=fit_id,
            )
            evaluation_directory = cell_directory / (
                "evaluation_%s" % sample_seed
            )
            evaluation_directory.mkdir(parents=True, exist_ok=True)
            file_paths = {
                "metrics": evaluation_directory / "metrics.csv",
                "privacy": evaluation_directory / "privacy.csv",
                "reliability": evaluation_directory / "reliability.csv",
            }
            annotated_metrics.to_csv(file_paths["metrics"], index=False)
            annotated_privacy.to_csv(file_paths["privacy"], index=False)
            annotated_reliability.to_csv(file_paths["reliability"], index=False)
            outputs.append(
                {
                    "sample_seed": sample_seed,
                    "files": {
                        name: {"path": str(path), "sha256": hash_file(path)}
                        for name, path in file_paths.items()
                    },
                }
            )
            metric_frames.append(annotated_metrics)
            privacy_frames.append(annotated_privacy)
            reliability_frames.append(annotated_reliability)
        manifest["evaluation_outputs"] = outputs
        return metric_frames, privacy_frames, reliability_frames

    def _run_source_cell(
        self,
        *,
        dataset_spec: Mapping[str, Any],
        dataset_metadata: Mapping[str, Any],
        split: FixedSplit,
        train: pd.DataFrame,
        test: pd.DataFrame,
        model: Mapping[str, Any],
        fit_seed: int,
        sample_seeds: Sequence[int],
        sample_rows: int,
        stages: Any,
    ) -> Tuple[
        Dict[str, Any],
        List[pd.DataFrame],
        List[pd.DataFrame],
        List[pd.DataFrame],
    ]:
        adapter = get_adapter_protocol(str(model["adapter"]))
        adapter.require_lane(str(dataset_spec["lane"]))
        if adapter.evidence_level == "artifact_level":
            raise CapabilityError(
                "%s must use artifact rescoring, not a source fit cell"
                % adapter.name
            )
        source_root = resolve_path(model["source_root"], self.base_directory)
        source_hash = hash_source_root(source_root)
        identity = self._cell_identity(
            dataset_spec=dataset_spec,
            dataset_metadata=dataset_metadata,
            split=split,
            train=train,
            model=model,
            source_hash=source_hash,
            adapter_name=adapter.name,
            fit_seed=fit_seed,
            sample_seeds=sample_seeds,
            sample_rows=sample_rows,
            stages=stages,
        )
        cell_id = hash_json(identity)
        cell_directory = self.output_directory / "cells" / cell_id
        manifest_path = cell_directory / "manifest.json"

        if self.resume and manifest_path.is_file():
            cached = dict(load_json(manifest_path))
            if cached.get("cell_id") != cell_id:
                raise CacheIntegrityError(
                    "cache cell identity does not match directory %s"
                    % cell_directory
                )
            if cached.get("status") == "completed":
                _verify_cached_manifest(cached)
                cached["cache_hit"] = True
                metrics, privacy, reliability = _read_evaluation_outputs(cached)
                return cached, metrics, privacy, reliability
            if not self.retry_failures:
                cached["cache_hit"] = True
                cached["failure_reused"] = True
                return cached, [], [], []
            attempt = int(cached.get("attempt", 1)) + 1
            prior_failures = list(cached.get("prior_failures", [])) + [
                {
                    "attempt": cached.get("attempt"),
                    "error_type": cached.get("error_type"),
                    "error_message": cached.get("error_message"),
                    "failure_phase": cached.get("failure_phase"),
                }
            ]
        else:
            attempt = 1
            prior_failures = []

        cell_directory.mkdir(parents=True, exist_ok=True)
        train_path = cell_directory / "train.csv"
        train.to_csv(train_path, index=False, float_format="%.17g")
        request = {
            "action": "fit_sample",
            "source_root": str(source_root),
            "expected_version": str(model["expected_version"]),
            "adapter": adapter.name,
            "lane": normalize_lane(str(dataset_spec["lane"])),
            "target": dataset_spec.get("target"),
            "fit_seed": int(fit_seed),
            "training_seed": int(fit_seed),
            "model_seed": int(fit_seed),
            "sample_seeds": [int(seed) for seed in sample_seeds],
            "sample_rows": int(sample_rows),
            "config": dict(model.get("config", {})),
            "stages": stages,
            "train_path": str(train_path),
            "output_dir": str(cell_directory / "samples"),
        }
        manifest: Dict[str, Any] = {
            "cell_id": cell_id,
            "identity": identity,
            "status": "running",
            "attempt": attempt,
            "prior_failures": prior_failures,
            "started_at_utc": _now(),
            "cache_hit": False,
            "source_git": git_provenance(source_root),
            "request": request,
        }
        atomic_write_json(manifest_path, manifest)
        worker = invoke_source_worker(
            request,
            work_directory=cell_directory,
            timeout_seconds=(
                None
                if self.timeout_seconds is None
                else float(self.timeout_seconds)
            ),
            raise_on_failure=False,
        )
        manifest["worker"] = worker
        if worker.get("status") != "completed":
            manifest.update(
                {
                    "status": "failed",
                    "failure_phase": worker.get("failure_phase", "worker"),
                    "failure_kind": worker.get("failure_kind", "error"),
                    "error_type": worker.get("error_type"),
                    "error_message": worker.get("error_message"),
                    "finished_at_utc": _now(),
                }
            )
            atomic_write_json(manifest_path, manifest)
            return manifest, [], [], []

        try:
            metric_frames, privacy_frames, reliability_frames = self._evaluate_worker(
                manifest,
                train,
                test,
                dataset_spec,
                model,
                cell_directory,
            )
        except Exception as error:
            manifest.update(
                {
                    "status": "failed",
                    "failure_phase": "evaluation",
                    "failure_kind": "error",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "traceback": traceback.format_exc(),
                    "finished_at_utc": _now(),
                }
            )
            atomic_write_json(manifest_path, manifest)
            return manifest, [], [], []
        manifest.update({"status": "completed", "finished_at_utc": _now()})
        atomic_write_json(manifest_path, manifest)
        return manifest, metric_frames, privacy_frames, reliability_frames

    def _run_artifact_cell(
        self,
        *,
        dataset_spec: Mapping[str, Any],
        dataset_metadata: Mapping[str, Any],
        split: FixedSplit,
        train: pd.DataFrame,
        test: pd.DataFrame,
        model: Mapping[str, Any],
    ) -> Tuple[
        Dict[str, Any],
        List[pd.DataFrame],
        List[pd.DataFrame],
        List[pd.DataFrame],
    ]:
        """Score one persisted v1.2 release without inventing source evidence."""

        adapter = get_adapter_protocol(str(model["adapter"]))
        if adapter.name != "v1.2_historical_artifact":
            raise CapabilityError(
                "only v1.2_historical_artifact is accepted by artifact cells"
            )
        adapter.require_lane(str(dataset_spec["lane"]))
        raw_paths = model.get("artifact_paths")
        if raw_paths is None:
            raw_paths = {"raw": model["artifact_path"]}
        if not isinstance(raw_paths, Mapping) or "raw" not in raw_paths:
            raise ValueError("artifact_paths must be a stage mapping containing raw")
        paths = {
            str(stage): resolve_path(path, self.base_directory)
            for stage, path in raw_paths.items()
        }
        for stage, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(
                    "historical artifact stage %r is absent: %s" % (stage, path)
                )
        artifact_hashes = {
            stage: hash_file(path) for stage, path in paths.items()
        }
        artifact_frames = {
            stage: pd.read_csv(path) for stage, path in paths.items()
        }
        row_counts = {len(frame) for frame in artifact_frames.values()}
        if len(row_counts) != 1:
            raise ValueError("historical artifact stages must have equal row counts")
        artifact_rows = next(iter(row_counts))
        artifact_source_hash = hash_json(artifact_hashes)
        artifact_seed = int(model.get("artifact_sample_seed", 42))
        identity = self._cell_identity(
            dataset_spec=dataset_spec,
            dataset_metadata=dataset_metadata,
            split=split,
            train=train,
            model=model,
            source_hash=artifact_source_hash,
            adapter_name=adapter.name,
            fit_seed=-1,
            sample_seeds=[artifact_seed],
            sample_rows=artifact_rows,
            stages={stage: {"artifact_path": str(path)} for stage, path in paths.items()},
        )
        identity.update(
            {
                "artifact_paths": {
                    stage: str(path) for stage, path in paths.items()
                },
                "artifact_sha256": artifact_hashes,
                "evidence_level": "artifact_level",
                "source_available": False,
            }
        )
        cell_id = hash_json(identity)
        cell_directory = self.output_directory / "cells" / cell_id
        manifest_path = cell_directory / "manifest.json"
        if self.resume and manifest_path.is_file():
            cached = dict(load_json(manifest_path))
            if cached.get("status") == "completed":
                _verify_cached_manifest(cached)
                cached["cache_hit"] = True
                metrics, privacy, reliability = _read_evaluation_outputs(cached)
                return cached, metrics, privacy, reliability
            if not self.retry_failures:
                cached["cache_hit"] = True
                cached["failure_reused"] = True
                return cached, [], [], []
            attempt = int(cached.get("attempt", 1)) + 1
        else:
            attempt = 1
        cell_directory.mkdir(parents=True, exist_ok=True)

        stage_records = []
        row_count = None
        class_counts = None
        for stage, path in paths.items():
            frame = artifact_frames[stage]
            if list(frame.columns) != list(train.columns):
                raise ValueError(
                    "historical artifact %s columns do not exactly match the "
                    "comparison table" % path
                )
            if row_count is None:
                row_count = len(frame)
            stage_records.append(
                {
                    "stage": stage,
                    "path": str(path),
                    "sha256": artifact_hashes[stage],
                    "dataframe_sha256": hash_dataframe(frame),
                    "rows": int(len(frame)),
                    "columns": [str(column) for column in frame.columns],
                }
            )
            target = dataset_spec.get("target")
            if stage == "raw" and target is not None:
                counts = frame[target].value_counts(dropna=False, sort=False)
                class_counts = [
                    {"value": value, "count": int(count)}
                    for value, count in counts.items()
                ]
        worker = {
            "status": "completed",
            "action": "artifact_rescore",
            "adapter": {"adapter": adapter.name, **adapter.as_dict()},
            "lane": normalize_lane(str(dataset_spec["lane"])),
            "target": dataset_spec.get("target"),
            "fit_seed": None,
            "sample_seeds": [artifact_seed],
            "sample_rows": int(row_count or 0),
            "resolved_config": {"artifact_paths": identity["artifact_paths"]},
            "timings_seconds": {"fit": 0.0, "sampling_total": 0.0},
            "provenance": {
                "evidence_level": "artifact_level",
                "source_available": False,
                "historical_version": str(
                    model.get("historical_version", "1.2.0")
                ),
                "observed_version": None,
                "artifact_sha256": artifact_hashes,
            },
            "samples": [
                {
                    "sample_seed": artifact_seed,
                    "sample_seconds": 0.0,
                    "class_counts": class_counts,
                    "stages": stage_records,
                }
            ],
        }
        manifest: Dict[str, Any] = {
            "cell_id": cell_id,
            "identity": identity,
            "status": "running",
            "attempt": attempt,
            "started_at_utc": _now(),
            "cache_hit": False,
            "worker": worker,
            "request": {
                "action": "artifact_rescore",
                "artifact_paths": identity["artifact_paths"],
                "source_available": False,
            },
        }
        atomic_write_json(manifest_path, manifest)
        try:
            metric_frames, privacy_frames, reliability_frames = self._evaluate_worker(
                manifest,
                train,
                test,
                dataset_spec,
                model,
                cell_directory,
            )
        except Exception as error:
            manifest.update(
                {
                    "status": "failed",
                    "failure_phase": "evaluation",
                    "failure_kind": "error",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "traceback": traceback.format_exc(),
                    "finished_at_utc": _now(),
                }
            )
            atomic_write_json(manifest_path, manifest)
            return manifest, [], [], []
        manifest.update({"status": "completed", "finished_at_utc": _now()})
        atomic_write_json(manifest_path, manifest)
        return manifest, metric_frames, privacy_frames, reliability_frames

    @staticmethod
    def _manifest_row(manifest: Mapping[str, Any]) -> Dict[str, Any]:
        identity = manifest.get("identity", {})
        worker = manifest.get("worker", {})
        provenance = worker.get("provenance", {}) if isinstance(worker, Mapping) else {}
        timings = worker.get("timings_seconds", {}) if isinstance(worker, Mapping) else {}
        return {
            "cell_id": manifest.get("cell_id"),
            "status": manifest.get("status"),
            "attempt": manifest.get("attempt"),
            "cache_hit": manifest.get("cache_hit", False),
            "failure_reused": manifest.get("failure_reused", False),
            "dataset_id": identity.get("dataset_id"),
            "adapter": identity.get("adapter"),
            "model_name": identity.get("model_name"),
            "split_seed": identity.get("split_seed"),
            "fit_seed": identity.get("fit_seed"),
            "split_hash": identity.get("split_hash"),
            "source_sha256": identity.get("source_sha256"),
            "expected_version": identity.get("expected_version"),
            "observed_version": provenance.get("observed_version"),
            "worker_seconds": worker.get("worker_seconds")
            if isinstance(worker, Mapping)
            else None,
            "fit_seconds": timings.get("fit"),
            "sampling_seconds": timings.get("sampling_total"),
            "partial_fit_seconds": worker.get("partial_fit_seconds")
            if isinstance(worker, Mapping)
            else None,
            "partial_sample_seconds": worker.get("partial_sample_seconds")
            if isinstance(worker, Mapping)
            else None,
            "failure_phase": manifest.get("failure_phase"),
            "failure_kind": manifest.get("failure_kind"),
            "error_type": manifest.get("error_type"),
            "error_message": manifest.get("error_message"),
            "started_at_utc": manifest.get("started_at_utc"),
            "finished_at_utc": manifest.get("finished_at_utc"),
        }

    @staticmethod
    def _failed_reliability(
        manifest: Mapping[str, Any],
        model: Mapping[str, Any],
        dataset_spec: Mapping[str, Any],
        sample_seeds: Sequence[int],
    ) -> pd.DataFrame:
        identity = manifest.get("identity", {})
        adapter = str(identity.get("adapter", model.get("adapter")))
        version = _version_label(
            model, adapter, identity.get("expected_version")
        )
        rows = []
        for seed in sample_seeds:
            rows.append(
                {
                    "dataset_id": dataset_spec["id"],
                    "lane": normalize_lane(str(dataset_spec["lane"])),
                    "model_name": _model_label(model, adapter),
                    "adapter": adapter,
                    "version": version,
                    "source_version": None,
                    "split_seed": identity.get("split_seed"),
                    "fit_seed": identity.get("fit_seed"),
                    "sample_seed": int(seed),
                    "split_hash": identity.get("split_hash"),
                    "fit_id": manifest.get("cell_id"),
                    "candidate": _model_label(model, adapter),
                    "stage": None,
                    "evaluation_success": False,
                    "privacy_success": False,
                    "run_success": False,
                    "failure_phase": manifest.get("failure_phase"),
                    "error_type": manifest.get("error_type"),
                    "error_message": manifest.get("error_message"),
                    "evaluation_seconds": 0.0,
                    "privacy_seconds": 0.0,
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _sample_rows(
        manifest: Mapping[str, Any],
        model: Mapping[str, Any],
        dataset_spec: Mapping[str, Any],
    ) -> List[Dict[str, Any]]:
        identity = manifest.get("identity", {})
        rows = []
        worker = manifest.get("worker", {})
        for sample in worker.get("samples", []) if isinstance(worker, Mapping) else []:
            for stage in sample.get("stages", []):
                rows.append(
                    {
                        "cell_id": manifest.get("cell_id"),
                        "dataset_id": dataset_spec["id"],
                        "model_name": identity.get("model_name"),
                        "adapter": identity.get("adapter"),
                        "split_seed": identity.get("split_seed"),
                        "fit_seed": identity.get("fit_seed"),
                        "sample_seed": sample.get("sample_seed"),
                        "stage": stage.get("stage"),
                        "path": stage.get("path"),
                        "sha256": stage.get("sha256"),
                        "rows": stage.get("rows"),
                        "class_counts": json.dumps(
                            sample.get("class_counts"),
                            sort_keys=True,
                            default=str,
                        ),
                    }
                )
        return rows

    def _historical(self) -> Optional[HistoricalRescoreResult]:
        config = self.protocol.get("historical_kuairand")
        if config is None:
            return None
        if not isinstance(config, Mapping):
            raise TypeError("historical_kuairand must be a mapping")
        return rescore_historical_kuairand(
            v11_artifact_root=resolve_path(
                config["v11_artifact_root"], self.base_directory
            ),
            v12_artifact_root=resolve_path(
                config["v12_artifact_root"], self.base_directory
            ),
            real_video_path=resolve_path(
                config["real_video_path"], self.base_directory
            ),
            real_clicks_path=resolve_path(
                config["real_clicks_path"], self.base_directory
            ),
            output_directory=self.output_directory / "historical_kuairand",
            constraints=config.get("constraints"),
            privacy=config.get("privacy"),
            include_controls=bool(config.get("controls", False)),
        )

    def run(self) -> VersionComparisonRunResult:
        self.output_directory.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            self.output_directory / "resolved_protocol.json", self.protocol
        )
        split_seeds = validate_seed_list(
            self.protocol.get("split_seeds", [42]), "split_seeds"
        )
        fit_seeds = validate_seed_list(
            self.protocol.get(
                "fit_seeds", self.protocol.get("model_seeds", [42])
            ),
            "fit_seeds",
        )
        global_sample_seeds = validate_seed_list(
            self.protocol.get("sample_seeds", [42]), "sample_seeds"
        )
        models = list(self.protocol["models"])

        manifests: List[Dict[str, Any]] = []
        metric_frames: List[pd.DataFrame] = []
        privacy_frames: List[pd.DataFrame] = []
        reliability_frames: List[pd.DataFrame] = []
        sample_rows: List[Dict[str, Any]] = []
        split_manifests: List[Dict[str, Any]] = []

        for dataset_spec in self.protocol["datasets"]:
            frame, dataset_metadata = self._resolved_dataset(dataset_spec)
            for split_seed in split_seeds:
                split, partitions, split_manifest = self._split_dataset(
                    frame, dataset_spec, split_seed
                )
                split_manifests.append(split_manifest)
                train = partitions["train"]
                test = partitions["test"]
                for model in models:
                    adapter = get_adapter_protocol(str(model["adapter"]))
                    if adapter.evidence_level == "artifact_level":
                        artifact_seeds = [
                            int(model.get("artifact_sample_seed", 42))
                        ]
                        try:
                            (
                                manifest,
                                cell_metrics,
                                cell_privacy,
                                cell_reliability,
                            ) = self._run_artifact_cell(
                                dataset_spec=dataset_spec,
                                dataset_metadata=dataset_metadata,
                                split=split,
                                train=train,
                                test=test,
                                model=model,
                            )
                        except Exception as error:
                            manifest = self._cache_setup_failure(
                                dataset_spec=dataset_spec,
                                dataset_metadata=dataset_metadata,
                                split=split,
                                model=model,
                                fit_seed=-1,
                                phase="artifact_setup",
                                error=error,
                            )
                            cell_metrics = []
                            cell_privacy = []
                            cell_reliability = []
                        manifests.append(self._manifest_row(manifest))
                        metric_frames.extend(cell_metrics)
                        privacy_frames.extend(cell_privacy)
                        reliability_frames.extend(cell_reliability)
                        if manifest.get("status") != "completed":
                            reliability_frames.append(
                                self._failed_reliability(
                                    manifest,
                                    model,
                                    dataset_spec,
                                    artifact_seeds,
                                )
                            )
                        sample_rows.extend(
                            self._sample_rows(manifest, model, dataset_spec)
                        )
                        continue
                    model_sample_seeds = validate_seed_list(
                        model.get("sample_seeds", global_sample_seeds),
                        "model sample_seeds",
                    )
                    sample_count = int(
                        model.get(
                            "sample_rows",
                            dataset_spec.get(
                                "sample_rows",
                                self.protocol.get("sample_rows", len(train)),
                            ),
                        )
                    )
                    stages = model.get(
                        "stages",
                        dataset_spec.get(
                            "stages", self.protocol.get("stages", {"raw": {}})
                        ),
                    )
                    for fit_seed in fit_seeds:
                        try:
                            (
                                manifest,
                                cell_metrics,
                                cell_privacy,
                                cell_reliability,
                            ) = self._run_source_cell(
                                dataset_spec=dataset_spec,
                                dataset_metadata=dataset_metadata,
                                split=split,
                                train=train,
                                test=test,
                                model=model,
                                fit_seed=fit_seed,
                                sample_seeds=model_sample_seeds,
                                sample_rows=sample_count,
                                stages=stages,
                            )
                        except Exception as error:
                            manifest = self._cache_setup_failure(
                                dataset_spec=dataset_spec,
                                dataset_metadata=dataset_metadata,
                                split=split,
                                model=model,
                                fit_seed=fit_seed,
                                phase="cell_setup",
                                error=error,
                            )
                            cell_metrics = []
                            cell_privacy = []
                            cell_reliability = []
                        manifests.append(self._manifest_row(manifest))
                        metric_frames.extend(cell_metrics)
                        privacy_frames.extend(cell_privacy)
                        reliability_frames.extend(cell_reliability)
                        if manifest.get("status") != "completed":
                            reliability_frames.append(
                                self._failed_reliability(
                                    manifest,
                                    model,
                                    dataset_spec,
                                    model_sample_seeds,
                                )
                            )
                        sample_rows.extend(
                            self._sample_rows(manifest, model, dataset_spec)
                        )

        historical = self._historical()
        if historical is not None:
            metric_frames.append(historical.metrics)
            privacy_frames.append(historical.privacy)
            reliability_frames.append(historical.reliability)

        metrics = (
            pd.concat(metric_frames, ignore_index=True, sort=False)
            if metric_frames
            else pd.DataFrame()
        )
        privacy = (
            pd.concat(privacy_frames, ignore_index=True, sort=False)
            if privacy_frames
            else pd.DataFrame()
        )
        reliability = (
            pd.concat(reliability_frames, ignore_index=True, sort=False)
            if reliability_frames
            else pd.DataFrame()
        )
        manifest_frame = pd.DataFrame(manifests)
        sample_frame = pd.DataFrame(sample_rows)

        aggregation = None
        aggregation_config = dict(self.protocol.get("aggregation", {}))
        aggregation_metrics = _aggregation_records(metrics, reliability)
        aggregation_metrics = (
            aggregation_metrics.loc[
                ~aggregation_metrics.get(
                    "adapter",
                    pd.Series(index=aggregation_metrics.index, dtype=object),
                )
                .astype(str)
                .str.contains("historical_artifact", regex=False)
                & ~aggregation_metrics.get(
                    "adapter",
                    pd.Series(index=aggregation_metrics.index, dtype=object),
                )
                .astype(str)
                .str.startswith("control:")
                & pd.to_numeric(
                    aggregation_metrics.get(
                        "sample_seed",
                        pd.Series(index=aggregation_metrics.index, dtype=float),
                    ),
                    errors="coerce",
                ).notna()
            ].copy()
            if not aggregation_metrics.empty
            else aggregation_metrics
        )
        if (
            not aggregation_metrics.empty
            and {"metric", "value", "sample_seed", "version"}.issubset(
                aggregation_metrics.columns
            )
        ):
            aggregation = aggregate_version_comparison(
                aggregation_metrics,
                reference_version=aggregation_config.get("reference_version"),
                expected_sample_seeds=global_sample_seeds,
                directions=aggregation_config.get("directions"),
                confidence=float(aggregation_config.get("confidence", 0.95)),
                n_bootstrap=int(
                    aggregation_config.get("bootstrap_replicates", 1000)
                ),
                random_state=int(aggregation_config.get("random_state", 0)),
            )

        resolved_protocol = {
            **self.protocol,
            "_execution": {
                "output_directory": str(self.output_directory),
                "resume": self.resume,
                "retry_failures": self.retry_failures,
                "tooling_sha256": self.tooling_sha256,
                "evaluation_source_sha256": self.evaluation_source_sha256,
                "worker_sha256": hash_file(WORKER_PATH),
                "environment": self.runtime_environment,
                "created_at_utc": _now(),
            },
        }
        result = VersionComparisonRunResult(
            manifests=manifest_frame,
            metrics=metrics,
            privacy=privacy,
            reliability=reliability,
            samples=sample_frame,
            splits=split_manifests,
            protocol=resolved_protocol,
            aggregation=aggregation,
            historical=historical,
        )
        result.save(self.output_directory)
        return result


def run_protocol(
    protocol: Union[PathLike, Mapping[str, Any]],
    *,
    output_directory: Optional[PathLike] = None,
    resume: bool = True,
    retry_failures: bool = False,
    timeout_seconds: Optional[float] = None,
) -> VersionComparisonRunResult:
    return VersionComparisonRunner(
        protocol,
        output_directory=output_directory,
        resume=resume,
        retry_failures=retry_failures,
        timeout_seconds=timeout_seconds,
    ).run()


__all__ = [
    "VersionComparisonRunResult",
    "VersionComparisonRunner",
    "load_protocol",
    "run_protocol",
]

