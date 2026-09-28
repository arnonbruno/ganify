"""Fail-closed release orchestration and claim decisions.

This module keeps release evidence disaggregated.  It never computes a
weighted quality score, never retries a failed adapter, and never treats an
unavailable baseline as a skipped benchmark.
"""

from __future__ import annotations

import inspect
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd

from ._config import PathLike, load_config
from .adapters import (
    AdapterLike,
    FRONTIER_ADAPTER_NAMES,
    MANDATORY_CONTROL_NAMES,
    ModelAdapter,
    clone_adapter,
    create_adapter,
    normalize_adapter_name,
)
from .controlled import ControlledPanel, generate_controlled_panel
from .data import (
    get_dataset_manifest,
    load_dataset,
)
from .gate import GateReport, evaluate_gates
from .run import hash_dataframe, hash_json
from .splits import deterministic_split
from .suite import BenchmarkSuite, load_suite


DEFAULT_RELEASE_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "configs" / "releases"
)
RELEASE_PROTOCOL_VERSION = "1"

_METRIC_COLUMNS = [
    "run_id",
    "suite",
    "dataset_id",
    "model_name",
    "adapter_name",
    "split_seed",
    "model_seed",
    "sample_seed",
    "split_hash",
    "stage",
    "pillar",
    "metric",
    "column",
    "detail",
    "value",
]
_RELIABILITY_COLUMNS = [
    "run_id",
    "suite",
    "dataset_id",
    "model_name",
    "adapter_name",
    "split_seed",
    "model_seed",
    "sample_seed",
    "split_hash",
    "attempt",
    "fit_success",
    "sample_success",
    "evaluation_success",
    "privacy_success",
    "run_success",
    "failure_phase",
    "error_type",
    "error_message",
    "fit_seconds",
    "sample_seconds",
    "evaluation_seconds",
    "privacy_seconds",
]


def _strict_json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Mapping):
        return {
            str(key): _strict_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _release_path(name_or_path: PathLike) -> Path:
    candidate = Path(name_or_path)
    if candidate.is_file():
        return candidate
    name = str(name_or_path)
    if name.endswith((".yaml", ".yml", ".json")):
        return DEFAULT_RELEASE_DIRECTORY / name
    return DEFAULT_RELEASE_DIRECTORY / ("%s.yaml" % name)


def load_release_config(
    config: Union[PathLike, Mapping[str, Any]] = "vnext",
) -> Dict[str, Any]:
    """Load a JSON-compatible release protocol."""

    values = (
        dict(config)
        if isinstance(config, Mapping)
        else load_config(_release_path(config))
    )
    if not str(values.get("version", "")).strip():
        raise ValueError("release config needs a non-empty version")
    return values


def _seed_tuple(values: Any, name: str) -> Tuple[int, ...]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("%s must be a non-empty seed list" % name)
    seeds = tuple(int(value) for value in values)
    if len(set(seeds)) != len(seeds):
        raise ValueError("%s contains duplicate seeds" % name)
    return seeds


@dataclass(frozen=True)
class ReleaseSuitePlan:
    """Fully resolved datasets and seeds for one release suite."""

    name: str
    datasets: Tuple[str, ...]
    split_seeds: Tuple[int, ...]
    model_seeds: Tuple[int, ...]
    sample_seeds: Tuple[int, ...]
    max_rows: Optional[int] = None
    version: str = "unversioned"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("release suite name must not be empty")
        if not self.datasets or len(set(self.datasets)) != len(self.datasets):
            raise ValueError(
                "release suite %r needs unique datasets" % self.name
            )
        for name, values in (
            ("split_seeds", self.split_seeds),
            ("model_seeds", self.model_seeds),
            ("sample_seeds", self.sample_seeds),
        ):
            if not values or len(set(values)) != len(values):
                raise ValueError(
                    "release suite %r needs unique %s" % (self.name, name)
                )
        if self.max_rows is not None and int(self.max_rows) < 5:
            raise ValueError("release suite max_rows must be at least five")

    @classmethod
    def from_suite(cls, suite: BenchmarkSuite) -> "ReleaseSuitePlan":
        return cls(
            name=suite.name,
            datasets=tuple(item.dataset_id for item in suite.datasets),
            split_seeds=tuple(suite.split_seeds),
            model_seeds=tuple(suite.model_seeds),
            sample_seeds=tuple(suite.sample_seeds),
            max_rows=suite.max_rows,
            version=suite.version,
        )

    @classmethod
    def from_mapping(
        cls, name: str, values: Mapping[str, Any]
    ) -> "ReleaseSuitePlan":
        datasets = values.get("datasets")
        if not isinstance(datasets, (list, tuple)) or not datasets:
            raise ValueError(
                "release suite %r needs a non-empty datasets list" % name
            )
        maximum = values.get("max_rows")
        return cls(
            name=str(values.get("name", name)),
            datasets=tuple(str(value) for value in datasets),
            split_seeds=_seed_tuple(
                values.get("split_seeds", [0]), "split_seeds"
            ),
            model_seeds=_seed_tuple(
                values.get("model_seeds", [0]), "model_seeds"
            ),
            sample_seeds=_seed_tuple(
                values.get("sample_seeds", [0]), "sample_seeds"
            ),
            max_rows=None if maximum is None else int(maximum),
            version=str(values.get("version", "unversioned")),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "datasets": list(self.datasets),
            "split_seeds": list(self.split_seeds),
            "model_seeds": list(self.model_seeds),
            "sample_seeds": list(self.sample_seeds),
            "max_rows": self.max_rows,
        }


def _suite_plans(values: Any) -> Tuple[ReleaseSuitePlan, ...]:
    if isinstance(values, Mapping):
        plans = []
        for name, raw in values.items():
            if isinstance(raw, ReleaseSuitePlan):
                plan = raw
            elif isinstance(raw, BenchmarkSuite):
                plan = ReleaseSuitePlan.from_suite(raw)
            elif isinstance(raw, str):
                plan = ReleaseSuitePlan.from_suite(load_suite(raw))
            elif isinstance(raw, Mapping):
                plan = ReleaseSuitePlan.from_mapping(str(name), raw)
            else:
                raise TypeError("invalid release suite %r" % (name,))
            plans.append(plan)
    elif isinstance(values, (list, tuple)):
        plans = []
        for raw in values:
            if isinstance(raw, ReleaseSuitePlan):
                plan = raw
            elif isinstance(raw, BenchmarkSuite):
                plan = ReleaseSuitePlan.from_suite(raw)
            elif isinstance(raw, str):
                plan = ReleaseSuitePlan.from_suite(load_suite(raw))
            elif isinstance(raw, Mapping):
                name = str(raw.get("name", "")).strip()
                if not name:
                    raise ValueError("inline release suite needs a name")
                plan = ReleaseSuitePlan.from_mapping(name, raw)
            else:
                raise TypeError("invalid release suite entry")
            plans.append(plan)
    else:
        raise TypeError("release suites must be a mapping or sequence")
    names = [plan.name for plan in plans]
    if not plans or len(names) != len(set(names)):
        raise ValueError("release config needs unique suites")
    return tuple(plans)


@dataclass(frozen=True)
class ReleaseModelSpec:
    """One candidate, control, or external frontier protocol."""

    name: str
    adapter: str
    config: Mapping[str, Any] = field(default_factory=dict)
    role: str = "baseline"
    mandatory: bool = False

    @classmethod
    def from_value(
        cls, value: Union[str, Mapping[str, Any]]
    ) -> "ReleaseModelSpec":
        if isinstance(value, str):
            normalized = normalize_adapter_name(value)
            return cls(name=normalized, adapter=normalized)
        if not isinstance(value, Mapping):
            raise TypeError("release model specification must be a mapping")
        adapter = normalize_adapter_name(
            str(value.get("adapter", value.get("name", "")))
        )
        name = str(value.get("name", adapter)).strip()
        if not name or not adapter:
            raise ValueError("release model needs name and adapter")
        config = value.get("config", {})
        if not isinstance(config, Mapping):
            raise TypeError("release model config must be a mapping")
        try:
            json.dumps(config, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise TypeError(
                "release model config must be JSON-serializable"
            ) from error
        return cls(
            name=name,
            adapter=adapter,
            config=dict(config),
            role=str(value.get("role", "baseline")),
            mandatory=bool(value.get("mandatory", False)),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "adapter": self.adapter,
            "config": dict(self.config),
            "role": self.role,
            "mandatory": self.mandatory,
        }


def _model_specs(
    values: Any, adapter_overrides: Mapping[str, AdapterLike]
) -> Tuple[ReleaseModelSpec, ...]:
    if values in (None, []):
        if not adapter_overrides:
            raise ValueError("release config needs at least one model")
        raw_values: Sequence[Union[str, Mapping[str, Any]]] = [
            {
                "name": name,
                "adapter": getattr(adapter, "name", name),
            }
            for name, adapter in adapter_overrides.items()
        ]
    elif isinstance(values, Mapping):
        raw_values = [
            {
                "name": name,
                **(
                    dict(value)
                    if isinstance(value, Mapping)
                    else {"adapter": value}
                ),
            }
            for name, value in values.items()
        ]
    elif isinstance(values, (list, tuple)):
        raw_values = values
    else:
        raise TypeError("release models must be a mapping or sequence")
    specs = tuple(ReleaseModelSpec.from_value(value) for value in raw_values)
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("release config contains duplicate model names")
    return specs


@dataclass(frozen=True)
class ReleaseRunManifest:
    """Deterministic provenance for one intended seed cell."""

    run_id: str
    protocol_version: str
    release_version: str
    candidate: str
    suite: str
    suite_version: str
    dataset_id: str
    model_name: str
    adapter_name: str
    split_seed: int
    model_seed: int
    sample_seed: int
    split_hash: Optional[str]
    config: Mapping[str, Any]
    hashes: Mapping[str, str]
    status: str
    attempt: int
    failure_phase: Optional[str]
    error_type: Optional[str]
    error_message: Optional[str]
    sealed_test: bool
    fit_scope: str
    selection_scope: str
    test_scope: str
    test_unsealed_before_fit: bool

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReleaseRunResult:
    """All disaggregated release evidence."""

    manifests: pd.DataFrame
    metrics: pd.DataFrame
    reliability: pd.DataFrame
    privacy: pd.DataFrame
    constraints: pd.DataFrame
    protocol: Mapping[str, Any]

    def __post_init__(self) -> None:
        self.metrics.attrs["contains_blended_score"] = False
        self.privacy.attrs["contains_blended_score"] = False
        self.constraints.attrs["contains_blended_score"] = False

    @property
    def failures(self) -> pd.DataFrame:
        if self.manifests.empty or "status" not in self.manifests:
            return self.manifests.iloc[0:0].copy()
        return self.manifests.loc[
            self.manifests["status"] != "completed"
        ].reset_index(drop=True)

    @property
    def long_metrics(self) -> pd.DataFrame:
        return self.metrics

    @property
    def reliability_rows(self) -> pd.DataFrame:
        return self.reliability

    @property
    def privacy_rows(self) -> pd.DataFrame:
        return self.privacy

    @property
    def constraint_rows(self) -> pd.DataFrame:
        return self.constraints

    def manifest_records(self) -> List[Dict[str, Any]]:
        records = [
            _strict_json_value(record)
            for record in self.manifests.to_dict(orient="records")
        ]
        return sorted(records, key=lambda row: str(row["run_id"]))

    def manifest_json(self) -> str:
        return (
            json.dumps(
                self.manifest_records(),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )

    def save(self, directory: PathLike) -> Path:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "manifests.json").write_text(
            self.manifest_json(), encoding="utf-8"
        )
        (destination / "manifests.jsonl").write_text(
            "".join(
                json.dumps(
                    record,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
                for record in self.manifest_records()
            ),
            encoding="utf-8",
        )
        (destination / "protocol.json").write_text(
            json.dumps(
                _strict_json_value(dict(self.protocol)),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        self.metrics.to_csv(destination / "metrics.csv", index=False)
        self.metrics.to_csv(destination / "long_metrics.csv", index=False)
        self.reliability.to_csv(
            destination / "reliability.csv", index=False
        )
        self.reliability.to_csv(
            destination / "reliability_rows.csv", index=False
        )
        self.privacy.to_csv(destination / "privacy.csv", index=False)
        self.privacy.to_csv(destination / "privacy_rows.csv", index=False)
        self.constraints.to_csv(
            destination / "constraints.csv", index=False
        )
        self.constraints.to_csv(
            destination / "constraint_rows.csv", index=False
        )
        summary = {
            "runs": int(len(self.manifests)),
            "completed": int(
                (self.manifests.get("status") == "completed").sum()
            ),
            "failed": int(len(self.failures)),
            "contains_blended_score": False,
        }
        (destination / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return destination


@dataclass
class _DatasetContext:
    frame: pd.DataFrame
    target: Optional[str]
    task: Optional[str]
    id_column: Optional[str]
    categorical_columns: Optional[Tuple[str, ...]]
    constraints: Tuple[Any, ...]
    version: str


def _constraint_sequence(value: Any) -> Tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping) and (
        "type" in value or "kind" in value
    ):
        return (value,)
    if hasattr(value, "evaluate") and callable(value.evaluate):
        return (value,)
    return tuple(value)


def _task_name(task: Optional[str], target: Optional[str]) -> Optional[str]:
    if task is None:
        return None
    value = str(task).lower()
    if value in {"binary_classification", "multiclass_classification"}:
        return "classification"
    if value == "regression":
        return "regression"
    if value == "controlled" and target is not None:
        return "classification"
    return None


def _error_values(error: BaseException) -> Tuple[str, str]:
    message = " ".join(str(error).split())
    return type(error).__name__, message[:2000]


def _call_evaluator(
    evaluator: Callable[..., Any],
    *,
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    synthetic: pd.DataFrame,
    context: Mapping[str, Any],
) -> Any:
    signature = inspect.signature(evaluator)
    parameters = signature.parameters
    accepts_keywords = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    kwargs = {
        "real_train": real_train,
        "real_test": real_test,
        "synthetic": synthetic,
        "context": context,
    }
    if accepts_keywords or all(name in parameters for name in kwargs):
        return evaluator(**kwargs)
    if len(parameters) >= 4:
        return evaluator(real_train, real_test, synthetic, context)
    return evaluator(real_train, real_test, synthetic)


def _as_metric_frame(value: Any) -> pd.DataFrame:
    if value is None or (
        isinstance(value, (list, tuple)) and len(value) == 0
    ):
        return pd.DataFrame(
            columns=["stage", "pillar", "metric", "column", "detail", "value"]
        )
    if isinstance(value, pd.DataFrame):
        frame = value.copy()
    elif hasattr(value, "to_frame"):
        frame = value.to_frame()
    elif isinstance(value, Mapping):
        frame = pd.DataFrame([value])
    else:
        frame = pd.DataFrame(value)
    required = {"pillar", "metric", "value"}
    if not required.issubset(frame.columns):
        raise ValueError(
            "release evaluator rows require columns %r" % sorted(required)
        )
    if "stage" not in frame:
        frame["stage"] = "raw"
    if "column" not in frame:
        frame["column"] = None
    if "detail" not in frame:
        frame["detail"] = None
    frame["value"] = pd.to_numeric(frame["value"], errors="raise")
    if not bool(np.isfinite(frame["value"].to_numpy(dtype=float)).all()):
        raise ValueError("release evaluator emitted non-finite metric values")
    return frame.loc[
        :, ["stage", "pillar", "metric", "column", "detail", "value"]
    ].reset_index(drop=True)


def _constraint_metric_rows(
    synthetic: pd.DataFrame, constraints: Sequence[Any]
) -> pd.DataFrame:
    def evaluate(value: Any) -> Tuple[np.ndarray, str]:
        if hasattr(value, "evaluate") and callable(value.evaluate):
            mask = np.asarray(value.evaluate(synthetic), dtype=bool)
            label = str(
                getattr(
                    value,
                    "label",
                    getattr(value, "name", "constraint"),
                )
            )
            return mask, label
        if not isinstance(value, Mapping):
            raise TypeError(
                "release constraints need evaluate(frame) or a mapping"
            )
        kind = str(value.get("type", value.get("kind", ""))).lower()
        label = str(value.get("name", kind or "constraint"))
        if kind in {"range", "bound", "bounds"}:
            column = value["column"]
            numeric = pd.to_numeric(
                synthetic[column], errors="coerce"
            ).to_numpy(dtype=float)
            mask = np.isfinite(numeric)
            lower = value.get(
                "minimum", value.get("min", value.get("lower"))
            )
            upper = value.get(
                "maximum", value.get("max", value.get("upper"))
            )
            if lower is not None:
                mask &= numeric >= float(lower)
            if upper is not None:
                mask &= numeric <= float(upper)
            if bool(value.get("allow_missing", False)):
                mask |= synthetic[column].isna().to_numpy(dtype=bool)
            return mask, label
        if kind in {"inequality", "pair", "pair_inequality"}:
            left_name = value.get("left", value.get("small"))
            right_name = value.get("right", value.get("big"))
            left = pd.to_numeric(
                synthetic[left_name], errors="coerce"
            ).to_numpy(dtype=float)
            if right_name in synthetic:
                right = pd.to_numeric(
                    synthetic[right_name], errors="coerce"
                ).to_numpy(dtype=float)
            else:
                right = np.full(len(synthetic), float(right_name))
            operation = str(value.get("operator", "<="))
            tolerance = float(value.get("tolerance", 0.0))
            operations = {
                "<=": left <= right + tolerance,
                "<": left < right + tolerance,
                ">=": left + tolerance >= right,
                ">": left + tolerance > right,
                "==": np.abs(left - right) <= tolerance,
            }
            if operation not in operations:
                raise ValueError(
                    "unsupported release constraint operator %r" % operation
                )
            return (
                np.isfinite(left)
                & np.isfinite(right)
                & operations[operation],
                label,
            )
        if kind in {"allowed", "allowed_values", "domain"}:
            column = value["column"]
            allowed = value.get("values", value.get("allowed_values", ()))
            mask = synthetic[column].isin(list(allowed)).to_numpy(dtype=bool)
            if bool(value.get("allow_missing", False)):
                mask |= synthetic[column].isna().to_numpy(dtype=bool)
            return mask, label
        raise ValueError(
            "unsupported release constraint mapping type %r" % kind
        )

    rows: List[Dict[str, Any]] = []
    masks = []
    for index, constraint in enumerate(constraints):
        valid, label = evaluate(constraint)
        if valid.shape != (len(synthetic),):
            raise ValueError(
                "constraint %d emitted shape %r" % (index, valid.shape)
            )
        masks.append(valid)
        if label == "constraint":
            label = "constraint_%d" % index
        rows.extend(
            [
                {
                    "stage": "raw",
                    "pillar": "constraint",
                    "metric": "valid_rate",
                    "column": None,
                    "detail": label,
                    "value": float(np.mean(valid)),
                },
                {
                    "stage": "raw",
                    "pillar": "constraint",
                    "metric": "violation_rate",
                    "column": None,
                    "detail": label,
                    "value": float(1.0 - np.mean(valid)),
                },
            ]
        )
    if constraints:
        joint = np.logical_and.reduce(masks)
        rows.extend(
            [
                {
                    "stage": "raw",
                    "pillar": "constraint",
                    "metric": "valid_rate",
                    "column": None,
                    "detail": "__all__",
                    "value": float(np.mean(joint)),
                },
                {
                    "stage": "raw",
                    "pillar": "constraint",
                    "metric": "violation_rate",
                    "column": None,
                    "detail": "__all__",
                    "value": float(1.0 - np.mean(joint)),
                },
            ]
        )
    return _as_metric_frame(rows)


def _default_evaluator(
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    synthetic: pd.DataFrame,
    context: Mapping[str, Any],
) -> pd.DataFrame:
    """Dependency-free metrics used by the cheap/offline protocol.

    Full release jobs can inject :func:`ganify.evaluation.aggregate_table_report`
    (or a richer project evaluator).  The default intentionally stays usable
    when optional ML stacks are absent.
    """

    rows: List[Dict[str, Any]] = []
    configured_categorical = context.get("categorical_columns")
    categorical = (
        set(configured_categorical)
        if configured_categorical is not None
        else {
            column
            for column in real_train.columns
            if (
                pd.api.types.is_bool_dtype(real_train[column].dtype)
                or isinstance(
                    real_train[column].dtype, pd.CategoricalDtype
                )
                or not pd.api.types.is_numeric_dtype(
                    real_train[column].dtype
                )
            )
        }
    )

    def add(
        pillar: str,
        metric: str,
        value: float,
        *,
        column: Any = None,
        detail: Optional[str] = None,
    ) -> None:
        if math.isfinite(float(value)):
            rows.append(
                {
                    "stage": "raw",
                    "pillar": pillar,
                    "metric": metric,
                    "column": column,
                    "detail": detail,
                    "value": float(value),
                }
            )

    numeric_columns = []
    for column in real_test.columns:
        if column in categorical:
            left = (
                real_test[column]
                .astype(object)
                .where(real_test[column].notna(), "__GANIFY_NA__")
                .value_counts(normalize=True, dropna=False)
            )
            right = (
                synthetic[column]
                .astype(object)
                .where(synthetic[column].notna(), "__GANIFY_NA__")
                .value_counts(normalize=True, dropna=False)
            )
            support = left.index.union(right.index)
            total_variation = 0.5 * float(
                np.abs(
                    left.reindex(support, fill_value=0.0).to_numpy()
                    - right.reindex(support, fill_value=0.0).to_numpy()
                ).sum()
            )
            add(
                "marginal",
                "total_variation",
                total_variation,
                column=column,
                detail="categorical",
            )
            continue
        real_values = pd.to_numeric(
            real_test[column], errors="coerce"
        ).to_numpy(dtype=float)
        synthetic_values = pd.to_numeric(
            synthetic[column], errors="coerce"
        ).to_numpy(dtype=float)
        real_observed = np.sort(real_values[np.isfinite(real_values)])
        synthetic_observed = np.sort(
            synthetic_values[np.isfinite(synthetic_values)]
        )
        add(
            "marginal",
            "missing_rate_error",
            abs(
                float(np.mean(~np.isfinite(real_values)))
                - float(np.mean(~np.isfinite(synthetic_values)))
            ),
            column=column,
            detail="numeric",
        )
        if len(real_observed) and len(synthetic_observed):
            points = np.unique(
                np.concatenate([real_observed, synthetic_observed])
            )
            left_cdf = np.searchsorted(
                real_observed, points, side="right"
            ) / float(len(real_observed))
            right_cdf = np.searchsorted(
                synthetic_observed, points, side="right"
            ) / float(len(synthetic_observed))
            add(
                "marginal",
                "ks_statistic",
                float(np.max(np.abs(left_cdf - right_cdf))),
                column=column,
                detail="numeric",
            )
            quantiles = np.linspace(0.0, 1.0, 101)
            add(
                "marginal",
                "quantile_mae",
                float(
                    np.mean(
                        np.abs(
                            np.quantile(real_observed, quantiles)
                            - np.quantile(
                                synthetic_observed, quantiles
                            )
                        )
                    )
                ),
                column=column,
                detail="numeric",
            )
            numeric_columns.append(column)

    def dependence_loss(
        left: pd.DataFrame, right: pd.DataFrame
    ) -> Optional[float]:
        if len(numeric_columns) < 2:
            return None
        left_correlation = left.loc[:, numeric_columns].corr(
            method="spearman"
        ).to_numpy(dtype=float)
        right_correlation = right.loc[:, numeric_columns].corr(
            method="spearman"
        ).to_numpy(dtype=float)
        upper = np.triu(
            np.ones_like(left_correlation, dtype=bool), k=1
        )
        valid = (
            upper
            & np.isfinite(left_correlation)
            & np.isfinite(right_correlation)
        )
        if not bool(valid.any()):
            return None
        return float(
            np.mean(
                np.abs(
                    left_correlation[valid] - right_correlation[valid]
                )
            )
        )

    observed_loss = dependence_loss(real_test, synthetic)
    if observed_loss is not None:
        add("dependence", "spearman_mae", observed_loss)
        rng = np.random.default_rng(int(context["sample_seed"]))
        independent = real_test.copy()
        for column in numeric_columns:
            independent[column] = (
                real_test[column]
                .iloc[rng.permutation(len(real_test))]
                .to_numpy()
            )
        independent_loss = dependence_loss(real_test, independent)
        midpoint = len(real_train) // 2
        real_floor = (
            dependence_loss(
                real_train.iloc[:midpoint],
                real_train.iloc[midpoint : midpoint * 2],
            )
            if midpoint >= 2
            else None
        )
        if (
            independent_loss is not None
            and real_floor is not None
            and independent_loss > real_floor + 1e-12
        ):
            add(
                "dependence",
                "normalized_dependence_gap",
                (observed_loss - real_floor)
                / (independent_loss - real_floor),
            )
            add(
                "dependence_control",
                "real_floor_loss",
                real_floor,
            )
            add(
                "dependence_control",
                "independent_loss",
                independent_loss,
            )

    train_hashes = set(
        pd.util.hash_pandas_object(
            real_train, index=False, categorize=True
        ).astype("uint64")
    )
    synthetic_hashes = pd.util.hash_pandas_object(
        synthetic, index=False, categorize=True
    ).astype("uint64")
    add(
        "nearest_neighbor",
        "exact_match_rate",
        float(synthetic_hashes.isin(train_hashes).mean()),
    )
    return _as_metric_frame(rows)


class ReleaseRunner:
    """Execute every preregistered seed cell exactly once."""

    def __init__(
        self,
        config: Union[PathLike, Mapping[str, Any]] = "vnext",
        *,
        suites: Optional[Any] = None,
        adapters: Optional[Mapping[str, AdapterLike]] = None,
        data_root: Optional[PathLike] = None,
        evaluator: Optional[Callable[..., Any]] = None,
        privacy_evaluator: Optional[Callable[..., Any]] = None,
        constraints: Optional[Mapping[str, Iterable[Any]]] = None,
    ) -> None:
        self.config = load_release_config(config)
        self.adapter_overrides = dict(adapters or {})
        self.suites = _suite_plans(
            self.config.get("suites", []) if suites is None else suites
        )
        self.models = _model_specs(
            self.config.get("models"), self.adapter_overrides
        )
        self.data_root = (
            None if data_root is None else Path(data_root)
        )
        self.evaluator = _default_evaluator if evaluator is None else evaluator
        self.privacy_evaluator = privacy_evaluator
        self.constraints = {
            str(dataset): _constraint_sequence(values)
            for dataset, values in (constraints or {}).items()
        }
        self.test_size = float(self.config.get("test_size", 0.2))
        self.validation_size = float(
            self.config.get("validation_size", 0.1)
        )
        if (
            self.test_size <= 0.0
            or self.validation_size < 0.0
            or self.test_size + self.validation_size >= 1.0
        ):
            raise ValueError("release split fractions are invalid")
        self.sample_rows = self.config.get("sample_rows")
        if self.sample_rows is not None and int(self.sample_rows) < 2:
            raise ValueError("release sample_rows must be at least two")
        self.sealed_test = bool(self.config.get("sealed_test", True))

    def _protocol(
        self,
        suites: Sequence[ReleaseSuitePlan],
        models: Sequence[ReleaseModelSpec],
        *,
        mode: str,
    ) -> Dict[str, Any]:
        return {
            "protocol_version": RELEASE_PROTOCOL_VERSION,
            "release_version": str(self.config["version"]),
            "candidate": str(self.config.get("candidate", "candidate")),
            "mode": mode,
            "sealed_test": self.sealed_test,
            "suites": {
                suite.name: suite.as_dict() for suite in suites
            },
            "models": [model.as_dict() for model in models],
            "mandatory_baselines": list(
                self.config.get(
                    "mandatory_baselines", MANDATORY_CONTROL_NAMES
                )
            ),
            "frontier_baselines": list(
                self.config.get(
                    "frontier_baselines", FRONTIER_ADAPTER_NAMES
                )
            ),
            "contains_blended_score": False,
        }

    def _cheap_plan(
        self,
    ) -> Tuple[Tuple[ReleaseSuitePlan, ...], Tuple[ReleaseModelSpec, ...]]:
        rows = int(self.config.get("cheap_rows", 256))
        suite = ReleaseSuitePlan(
            name=str(self.config.get("cheap_suite_name", "smoke")),
            datasets=("controlled_panel",),
            split_seeds=(0,),
            model_seeds=(0,),
            sample_seeds=(0,),
            max_rows=rows,
            version="controlled-cheap-v1",
        )
        selected = self.config.get("cheap_models")
        if selected is None:
            desired = {
                str(self.config.get("candidate_model", "ganify_conditional")),
                *map(
                    str,
                    self.config.get(
                        "mandatory_baselines", MANDATORY_CONTROL_NAMES
                    ),
                ),
            }
        else:
            desired = {str(value) for value in selected}
        models = tuple(model for model in self.models if model.name in desired)
        if not models:
            models = self.models
        cheap_models = []
        for model in models:
            config = dict(model.config)
            if model.adapter == "ganify_conditional":
                config.update(
                    {
                        "epochs": 1,
                        "batch_size": min(
                            int(config.get("batch_size", 64)), 64
                        ),
                        "n_critic": 1,
                        "noise_dim": min(
                            int(config.get("noise_dim", 16)), 16
                        ),
                        "generator_dims": [16],
                        "critic_dims": [16],
                    }
                )
            cheap_models.append(
                ReleaseModelSpec(
                    name=model.name,
                    adapter=model.adapter,
                    config=config,
                    role=model.role,
                    mandatory=model.mandatory,
                )
            )
        return (suite,), tuple(cheap_models)

    def _adapter(self, model: ReleaseModelSpec) -> ModelAdapter:
        override = self.adapter_overrides.get(
            model.name, self.adapter_overrides.get(model.adapter)
        )
        if override is not None:
            return clone_adapter(override)
        return create_adapter(model.adapter, model.config)

    def _dataset_options(self, dataset_id: str) -> Mapping[str, Any]:
        options = self.config.get("dataset_options", {})
        if not isinstance(options, Mapping):
            raise TypeError("dataset_options must be a mapping")
        value = options.get(dataset_id, {})
        if not isinstance(value, Mapping):
            raise TypeError(
                "dataset_options[%r] must be a mapping" % dataset_id
            )
        return value

    def _provided_dataset(
        self, dataset_id: str, provided: Mapping[str, Any]
    ) -> Optional[_DatasetContext]:
        if dataset_id not in provided:
            return None
        value = provided[dataset_id]
        options = self._dataset_options(dataset_id)
        if isinstance(value, ControlledPanel):
            return _DatasetContext(
                frame=value.frame.copy(deep=True),
                target="target",
                task="classification",
                id_column="row_id",
                categorical_columns=None,
                constraints=tuple(value.constraints),
                version=str(value.truth.get("version", "controlled")),
            )
        if isinstance(value, pd.DataFrame):
            payload: Mapping[str, Any] = {"frame": value}
        elif isinstance(value, Mapping):
            payload = value
        else:
            raise TypeError(
                "provided dataset %r must be a DataFrame, ControlledPanel, "
                "or mapping" % dataset_id
            )
        frame = payload.get("frame", payload.get("data"))
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("provided dataset %r has no DataFrame" % dataset_id)
        target = payload.get("target", options.get("target"))
        task = payload.get("task", options.get("task"))
        categories = payload.get(
            "categorical_columns", options.get("categorical_columns")
        )
        raw_constraints = payload.get(
            "constraints",
            self.constraints.get(
                dataset_id, options.get("constraints", ())
            ),
        )
        return _DatasetContext(
            frame=frame.copy(deep=True),
            target=None if target is None else str(target),
            task=_task_name(
                None if task is None else str(task),
                None if target is None else str(target),
            ),
            id_column=(
                None
                if payload.get("id_column", options.get("id_column")) is None
                else str(payload.get("id_column", options.get("id_column")))
            ),
            categorical_columns=(
                None
                if categories is None
                else tuple(str(value) for value in categories)
            ),
            constraints=_constraint_sequence(raw_constraints),
            version=str(payload.get("version", "provided")),
        )

    def _load_dataset(
        self,
        dataset_id: str,
        *,
        provided: Mapping[str, Any],
        rows: Optional[int],
        cheap: bool,
    ) -> _DatasetContext:
        supplied = self._provided_dataset(dataset_id, provided)
        if supplied is not None:
            context = supplied
        elif dataset_id == "controlled_panel":
            panel_rows = (
                int(rows)
                if rows is not None
                else int(
                    self.config.get(
                        "cheap_rows" if cheap else "controlled_rows",
                        256 if cheap else 10_000,
                    )
                )
            )
            panel = generate_controlled_panel(rows=panel_rows)
            context = _DatasetContext(
                frame=panel.frame.copy(deep=True),
                target="target",
                task="classification",
                id_column="row_id",
                categorical_columns=None,
                constraints=tuple(panel.constraints),
                version=str(panel.truth["version"]),
            )
        else:
            if self.data_root is None:
                raise FileNotFoundError(
                    "dataset %r needs data_root; no network download is "
                    "performed by ReleaseRunner" % dataset_id
                )
            manifest = get_dataset_manifest(dataset_id)
            context = _DatasetContext(
                frame=load_dataset(manifest, data_root=self.data_root),
                target=manifest.target,
                task=_task_name(manifest.task, manifest.target),
                id_column=(
                    None
                    if self._dataset_options(dataset_id).get("id_column")
                    is None
                    else str(
                        self._dataset_options(dataset_id)["id_column"]
                    )
                ),
                categorical_columns=None,
                constraints=self.constraints.get(dataset_id, ()),
                version=manifest.version,
            )
        maximum = None if rows is None else int(rows)
        if maximum is not None and len(context.frame) > maximum:
            # Stable source-order subset. Source row identity and the resulting
            # dataframe hash are retained in every run manifest.
            context.frame = context.frame.iloc[:maximum].reset_index(drop=True)
        if len(context.frame) < 5:
            raise ValueError(
                "dataset %r needs at least five rows" % dataset_id
            )
        if context.id_column is not None and context.id_column not in context.frame:
            raise ValueError(
                "dataset %r id column %r is missing"
                % (dataset_id, context.id_column)
            )
        if context.target is not None and context.target not in context.frame:
            raise ValueError(
                "dataset %r target %r is missing"
                % (dataset_id, context.target)
            )
        return context

    @staticmethod
    def _partitions(
        frame: pd.DataFrame,
        *,
        seed: int,
        test_size: float,
        validation_size: float,
    ) -> Tuple[Any, Dict[str, pd.DataFrame]]:
        positions = list(range(len(frame)))
        split = deterministic_split(
            positions,
            seed=int(seed),
            test_size=test_size,
            validation_size=validation_size,
        )
        return split, {
            "train": frame.iloc[list(split.train)].reset_index(drop=True),
            "validation": frame.iloc[
                list(split.validation)
            ].reset_index(drop=True),
            "test": frame.iloc[list(split.test)].reset_index(drop=True),
        }

    def _manifest(
        self,
        *,
        suite: ReleaseSuitePlan,
        dataset_id: str,
        model: ReleaseModelSpec,
        split_seed: int,
        model_seed: int,
        sample_seed: int,
        split_hash: Optional[str],
        hashes: Mapping[str, str],
        status: str,
        phase: Optional[str],
        error: Optional[BaseException],
        resolved_config: Optional[Mapping[str, Any]] = None,
    ) -> ReleaseRunManifest:
        identity = {
            "protocol_version": RELEASE_PROTOCOL_VERSION,
            "release_version": str(self.config["version"]),
            "candidate": str(self.config.get("candidate", "candidate")),
            "suite": suite.name,
            "suite_version": suite.version,
            "dataset_id": dataset_id,
            "model_name": model.name,
            "adapter_name": model.adapter,
            "split_seed": int(split_seed),
            "model_seed": int(model_seed),
            "sample_seed": int(sample_seed),
        }
        error_type = None
        error_message = None
        if error is not None:
            error_type, error_message = _error_values(error)
        return ReleaseRunManifest(
            run_id=hash_json(identity)[:20],
            protocol_version=RELEASE_PROTOCOL_VERSION,
            release_version=str(self.config["version"]),
            candidate=str(self.config.get("candidate", "candidate")),
            suite=suite.name,
            suite_version=suite.version,
            dataset_id=dataset_id,
            model_name=model.name,
            adapter_name=model.adapter,
            split_seed=int(split_seed),
            model_seed=int(model_seed),
            sample_seed=int(sample_seed),
            split_hash=split_hash,
            config=dict(
                model.config if resolved_config is None else resolved_config
            ),
            hashes=dict(hashes),
            status=status,
            attempt=1,
            failure_phase=phase,
            error_type=error_type,
            error_message=error_message,
            sealed_test=self.sealed_test,
            fit_scope="train_only",
            selection_scope="validation_or_train_only",
            test_scope="evaluation_only",
            test_unsealed_before_fit=False,
        )

    @staticmethod
    def _identified_metrics(
        frame: pd.DataFrame,
        manifest: ReleaseRunManifest,
    ) -> pd.DataFrame:
        values = frame.copy()
        identifiers = {
            "run_id": manifest.run_id,
            "suite": manifest.suite,
            "dataset_id": manifest.dataset_id,
            "model_name": manifest.model_name,
            "adapter_name": manifest.adapter_name,
            "split_seed": manifest.split_seed,
            "model_seed": manifest.model_seed,
            "sample_seed": manifest.sample_seed,
            "split_hash": manifest.split_hash,
        }
        for name, value in reversed(list(identifiers.items())):
            values.insert(0, name, value)
        return values.loc[:, _METRIC_COLUMNS]

    @staticmethod
    def _reliability(
        manifest: ReleaseRunManifest,
        *,
        fit_success: bool,
        sample_success: bool,
        evaluation_success: bool,
        privacy_success: bool,
        timings: Mapping[str, float],
    ) -> Dict[str, Any]:
        return {
            "run_id": manifest.run_id,
            "suite": manifest.suite,
            "dataset_id": manifest.dataset_id,
            "model_name": manifest.model_name,
            "adapter_name": manifest.adapter_name,
            "split_seed": manifest.split_seed,
            "model_seed": manifest.model_seed,
            "sample_seed": manifest.sample_seed,
            "split_hash": manifest.split_hash,
            "attempt": 1,
            "fit_success": bool(fit_success),
            "sample_success": bool(sample_success),
            "evaluation_success": bool(evaluation_success),
            "privacy_success": bool(privacy_success),
            "run_success": manifest.status == "completed",
            "failure_phase": manifest.failure_phase,
            "error_type": manifest.error_type,
            "error_message": manifest.error_message,
            "fit_seconds": float(timings.get("fit", 0.0)),
            "sample_seconds": float(timings.get("sample", 0.0)),
            "evaluation_seconds": float(timings.get("evaluation", 0.0)),
            "privacy_seconds": float(timings.get("privacy", 0.0)),
        }

    @staticmethod
    def _reliability_metrics(
        reliability: Mapping[str, Any],
        manifest: ReleaseRunManifest,
    ) -> pd.DataFrame:
        rows = []
        for metric, key in (
            ("fit_success_rate", "fit_success"),
            ("sample_success_rate", "sample_success"),
            ("evaluation_success_rate", "evaluation_success"),
            ("privacy_success_rate", "privacy_success"),
            ("run_success_rate", "run_success"),
        ):
            rows.append(
                {
                    "stage": "release",
                    "pillar": "reliability",
                    "metric": metric,
                    "column": None,
                    "detail": manifest.failure_phase,
                    "value": float(bool(reliability[key])),
                }
            )
        return ReleaseRunner._identified_metrics(
            _as_metric_frame(rows), manifest
        )

    def _failed_cells(
        self,
        *,
        suite: ReleaseSuitePlan,
        dataset_id: str,
        model: ReleaseModelSpec,
        split_seed: int,
        split_hash: Optional[str],
        hashes: Mapping[str, str],
        phase: str,
        error: BaseException,
        manifests: List[Dict[str, Any]],
        reliability_rows: List[Dict[str, Any]],
        metric_frames: List[pd.DataFrame],
    ) -> None:
        for model_seed in suite.model_seeds:
            for sample_seed in suite.sample_seeds:
                manifest = self._manifest(
                    suite=suite,
                    dataset_id=dataset_id,
                    model=model,
                    split_seed=split_seed,
                    model_seed=model_seed,
                    sample_seed=sample_seed,
                    split_hash=split_hash,
                    hashes=hashes,
                    status="failed",
                    phase=phase,
                    error=error,
                )
                manifests.append(manifest.as_dict())
                row = self._reliability(
                    manifest,
                    fit_success=False,
                    sample_success=False,
                    evaluation_success=False,
                    privacy_success=False,
                    timings={},
                )
                reliability_rows.append(row)
                metric_frames.append(
                    self._reliability_metrics(row, manifest)
                )

    def run(
        self,
        datasets: Optional[Mapping[str, Any]] = None,
        *,
        cheap: bool = False,
        output_directory: Optional[PathLike] = None,
    ) -> ReleaseRunResult:
        """Run the matrix once; failures become evidence rows, never retries."""

        provided = dict(datasets or {})
        suites, models = (
            self._cheap_plan() if cheap else (self.suites, self.models)
        )
        protocol = self._protocol(
            suites, models, mode="controlled-cheap" if cheap else "release"
        )
        manifests: List[Dict[str, Any]] = []
        reliability_rows: List[Dict[str, Any]] = []
        metric_frames: List[pd.DataFrame] = []
        privacy_frames: List[pd.DataFrame] = []
        constraint_frames: List[pd.DataFrame] = []

        for suite in suites:
            for dataset_id in suite.datasets:
                try:
                    dataset = self._load_dataset(
                        dataset_id,
                        provided=provided,
                        rows=suite.max_rows,
                        cheap=cheap,
                    )
                    data_hash = hash_dataframe(dataset.frame)
                except Exception as error:
                    for split_seed in suite.split_seeds:
                        for model in models:
                            self._failed_cells(
                                suite=suite,
                                dataset_id=dataset_id,
                                model=model,
                                split_seed=split_seed,
                                split_hash=None,
                                hashes={},
                                phase="load",
                                error=error,
                                manifests=manifests,
                                reliability_rows=reliability_rows,
                                metric_frames=metric_frames,
                            )
                    continue

                drop_columns = (
                    []
                    if dataset.id_column is None
                    else [dataset.id_column]
                )
                model_frame = dataset.frame.drop(columns=drop_columns)
                for split_seed in suite.split_seeds:
                    try:
                        split, partitions = self._partitions(
                            model_frame,
                            seed=split_seed,
                            test_size=self.test_size,
                            validation_size=self.validation_size,
                        )
                        train = partitions["train"]
                        test = partitions["test"]
                        if len(test) < 2:
                            raise ValueError(
                                "release split produced fewer than two test rows"
                            )
                        base_hashes = {
                            "dataset_sha256": data_hash,
                            "train_sha256": hash_dataframe(train),
                            "validation_sha256": hash_dataframe(
                                partitions["validation"]
                            ),
                            "test_sha256": hash_dataframe(test),
                        }
                    except Exception as error:
                        for model in models:
                            self._failed_cells(
                                suite=suite,
                                dataset_id=dataset_id,
                                model=model,
                                split_seed=split_seed,
                                split_hash=None,
                                hashes={"dataset_sha256": data_hash},
                                phase="split",
                                error=error,
                                manifests=manifests,
                                reliability_rows=reliability_rows,
                                metric_frames=metric_frames,
                            )
                        continue

                    for model in models:
                        for model_seed in suite.model_seeds:
                            adapter: Optional[ModelAdapter] = None
                            fit_error: Optional[BaseException] = None
                            fit_seconds = 0.0
                            resolved_config: Mapping[str, Any] = model.config
                            started = time.perf_counter()
                            try:
                                adapter = self._adapter(model)
                                resolved_config = adapter.get_config()
                                adapter.fit(
                                    train.copy(deep=True),
                                    seed=model_seed,
                                    target=dataset.target,
                                )
                            except Exception as error:
                                fit_error = error
                            fit_seconds = max(
                                0.0, time.perf_counter() - started
                            )

                            for sample_seed in suite.sample_seeds:
                                hashes = dict(base_hashes)
                                timings = {"fit": fit_seconds}
                                if fit_error is not None or adapter is None:
                                    manifest = self._manifest(
                                        suite=suite,
                                        dataset_id=dataset_id,
                                        model=model,
                                        split_seed=split_seed,
                                        model_seed=model_seed,
                                        sample_seed=sample_seed,
                                        split_hash=split.split_hash,
                                        hashes=hashes,
                                        status="failed",
                                        phase="fit",
                                        error=fit_error,
                                        resolved_config=resolved_config,
                                    )
                                    manifests.append(manifest.as_dict())
                                    row = self._reliability(
                                        manifest,
                                        fit_success=False,
                                        sample_success=False,
                                        evaluation_success=False,
                                        privacy_success=False,
                                        timings=timings,
                                    )
                                    reliability_rows.append(row)
                                    metric_frames.append(
                                        self._reliability_metrics(
                                            row, manifest
                                        )
                                    )
                                    continue

                                sample_error: Optional[BaseException] = None
                                sample_started = time.perf_counter()
                                try:
                                    count = (
                                        len(test)
                                        if self.sample_rows is None
                                        else int(self.sample_rows)
                                    )
                                    synthetic = adapter.sample(
                                        count, seed=sample_seed
                                    )
                                    if not isinstance(
                                        synthetic, pd.DataFrame
                                    ):
                                        raise TypeError(
                                            "adapter sample() must return a "
                                            "DataFrame"
                                        )
                                    if len(synthetic) != count:
                                        raise ValueError(
                                            "adapter returned %d rows, expected %d"
                                            % (len(synthetic), count)
                                        )
                                    missing = [
                                        column
                                        for column in train.columns
                                        if column not in synthetic.columns
                                    ]
                                    extra = [
                                        column
                                        for column in synthetic.columns
                                        if column not in train.columns
                                    ]
                                    if missing or extra:
                                        raise ValueError(
                                            "adapter schema mismatch; missing=%r, "
                                            "extra=%r" % (missing, extra)
                                        )
                                    synthetic = synthetic.loc[
                                        :, train.columns
                                    ].reset_index(drop=True)
                                    hashes["synthetic_sha256"] = (
                                        hash_dataframe(synthetic)
                                    )
                                except Exception as error:
                                    sample_error = error
                                timings["sample"] = max(
                                    0.0,
                                    time.perf_counter() - sample_started,
                                )
                                if sample_error is not None:
                                    manifest = self._manifest(
                                        suite=suite,
                                        dataset_id=dataset_id,
                                        model=model,
                                        split_seed=split_seed,
                                        model_seed=model_seed,
                                        sample_seed=sample_seed,
                                        split_hash=split.split_hash,
                                        hashes=hashes,
                                        status="failed",
                                        phase="sample",
                                        error=sample_error,
                                        resolved_config=resolved_config,
                                    )
                                    manifests.append(manifest.as_dict())
                                    row = self._reliability(
                                        manifest,
                                        fit_success=True,
                                        sample_success=False,
                                        evaluation_success=False,
                                        privacy_success=False,
                                        timings=timings,
                                    )
                                    reliability_rows.append(row)
                                    metric_frames.append(
                                        self._reliability_metrics(
                                            row, manifest
                                        )
                                    )
                                    continue

                                context = {
                                    "suite": suite.name,
                                    "dataset_id": dataset_id,
                                    "model_name": model.name,
                                    "adapter_name": model.adapter,
                                    "split_seed": split_seed,
                                    "model_seed": model_seed,
                                    "sample_seed": sample_seed,
                                    "split_hash": split.split_hash,
                                    "target": dataset.target,
                                    "task": dataset.task,
                                    "categorical_columns": (
                                        dataset.categorical_columns
                                    ),
                                    "constraints": dataset.constraints,
                                    "sealed_test": self.sealed_test,
                                }
                                evaluation_error: Optional[BaseException] = None
                                evaluation_started = time.perf_counter()
                                try:
                                    ordinary = _as_metric_frame(
                                        _call_evaluator(
                                            self.evaluator,
                                            real_train=train,
                                            real_test=test,
                                            synthetic=synthetic,
                                            context=context,
                                        )
                                    )
                                    constraint_metrics = (
                                        _constraint_metric_rows(
                                            synthetic,
                                            dataset.constraints,
                                        )
                                    )
                                    if not constraint_metrics.empty:
                                        ordinary = ordinary.loc[
                                            ordinary["pillar"] != "constraint"
                                        ]
                                        ordinary = pd.concat(
                                            [ordinary, constraint_metrics],
                                            ignore_index=True,
                                            sort=False,
                                        )
                                except Exception as error:
                                    evaluation_error = error
                                    ordinary = _as_metric_frame(None)
                                    constraint_metrics = _as_metric_frame(None)
                                timings["evaluation"] = max(
                                    0.0,
                                    time.perf_counter()
                                    - evaluation_started,
                                )

                                privacy_error: Optional[BaseException] = None
                                privacy_started = time.perf_counter()
                                try:
                                    if self.privacy_evaluator is not None:
                                        privacy_metrics = _as_metric_frame(
                                            _call_evaluator(
                                                self.privacy_evaluator,
                                                real_train=train,
                                                real_test=test,
                                                synthetic=synthetic,
                                                context=context,
                                            )
                                        )
                                    else:
                                        report = getattr(
                                            adapter,
                                            "privacy_report",
                                            None,
                                        )
                                        if report is None:
                                            privacy_metrics = _as_metric_frame(
                                                None
                                            )
                                        else:
                                            from ganify.privacy import (
                                                privacy_gate_records,
                                            )

                                            privacy_metrics = _as_metric_frame(
                                                privacy_gate_records(
                                                    None,
                                                    report,
                                                    stage="raw",
                                                )
                                            )
                                except Exception as error:
                                    privacy_error = error
                                    privacy_metrics = _as_metric_frame(None)
                                timings["privacy"] = max(
                                    0.0,
                                    time.perf_counter() - privacy_started,
                                )
                                phase = (
                                    "evaluation"
                                    if evaluation_error is not None
                                    else (
                                        "privacy"
                                        if privacy_error is not None
                                        else None
                                    )
                                )
                                failure = (
                                    evaluation_error
                                    if evaluation_error is not None
                                    else privacy_error
                                )
                                manifest = self._manifest(
                                    suite=suite,
                                    dataset_id=dataset_id,
                                    model=model,
                                    split_seed=split_seed,
                                    model_seed=model_seed,
                                    sample_seed=sample_seed,
                                    split_hash=split.split_hash,
                                    hashes=hashes,
                                    status=(
                                        "completed"
                                        if failure is None
                                        else "failed"
                                    ),
                                    phase=phase,
                                    error=failure,
                                    resolved_config=resolved_config,
                                )
                                manifests.append(manifest.as_dict())
                                reliability = self._reliability(
                                    manifest,
                                    fit_success=True,
                                    sample_success=True,
                                    evaluation_success=(
                                        evaluation_error is None
                                    ),
                                    privacy_success=(
                                        privacy_error is None
                                    ),
                                    timings=timings,
                                )
                                reliability_rows.append(reliability)
                                metric_frames.append(
                                    self._reliability_metrics(
                                        reliability, manifest
                                    )
                                )
                                if not ordinary.empty:
                                    identified = self._identified_metrics(
                                        ordinary, manifest
                                    )
                                    metric_frames.append(identified)
                                if not constraint_metrics.empty:
                                    constraint_frames.append(
                                        self._identified_metrics(
                                            constraint_metrics, manifest
                                        )
                                    )
                                if not privacy_metrics.empty:
                                    identified_privacy = (
                                        self._identified_metrics(
                                            privacy_metrics, manifest
                                        )
                                    )
                                    privacy_frames.append(
                                        identified_privacy
                                    )
                                    metric_frames.append(
                                        identified_privacy
                                    )

        manifests_frame = pd.DataFrame(manifests)
        if not manifests_frame.empty:
            manifests_frame = manifests_frame.sort_values(
                [
                    "suite",
                    "dataset_id",
                    "model_name",
                    "split_seed",
                    "model_seed",
                    "sample_seed",
                ],
                kind="stable",
            ).reset_index(drop=True)
        metrics = (
            pd.concat(metric_frames, ignore_index=True, sort=False)
            if metric_frames
            else pd.DataFrame(columns=_METRIC_COLUMNS)
        )
        metrics = metrics.loc[:, _METRIC_COLUMNS]
        metrics.attrs["contains_blended_score"] = False
        reliability = pd.DataFrame(
            reliability_rows, columns=_RELIABILITY_COLUMNS
        )
        privacy = (
            pd.concat(privacy_frames, ignore_index=True, sort=False)
            if privacy_frames
            else pd.DataFrame(columns=_METRIC_COLUMNS)
        ).loc[:, _METRIC_COLUMNS]
        constraint_frame = (
            pd.concat(constraint_frames, ignore_index=True, sort=False)
            if constraint_frames
            else pd.DataFrame(columns=_METRIC_COLUMNS)
        ).loc[:, _METRIC_COLUMNS]
        result = ReleaseRunResult(
            manifests=manifests_frame,
            metrics=metrics,
            reliability=reliability,
            privacy=privacy,
            constraints=constraint_frame,
            protocol=protocol,
        )
        if output_directory is not None:
            result.save(output_directory)
        return result

    def run_controlled_panel(
        self,
        *,
        rows: Optional[int] = None,
        output_directory: Optional[PathLike] = None,
    ) -> ReleaseRunResult:
        """Run the cheap, network-free controlled panel."""

        if rows is None:
            return self.run(cheap=True, output_directory=output_directory)
        panel = generate_controlled_panel(rows=int(rows))
        return self.run(
            {"controlled_panel": panel},
            cheap=True,
            output_directory=output_directory,
        )


@dataclass(frozen=True)
class StatisticalEvidence:
    rule_id: str
    passed: bool
    metric: str
    baseline: Optional[str]
    pairs: int
    estimate: Optional[float]
    ci_lower: Optional[float]
    ci_upper: Optional[float]
    message: str


@dataclass(frozen=True)
class ClaimEvaluation:
    claim_id: str
    claim: str
    scope: str
    priority: int
    passed: bool
    reasons: Tuple[str, ...]
    statistical_evidence: Tuple[StatisticalEvidence, ...]

    def as_dict(self) -> Dict[str, Any]:
        values = asdict(self)
        values["statistical_evidence"] = [
            asdict(item) for item in self.statistical_evidence
        ]
        return values


@dataclass(frozen=True)
class ReleaseDecisionReport:
    """Strongest supportable preregistered claim, or explicit no-claim."""

    approved: bool
    claim_id: Optional[str]
    claim: Optional[str]
    scope: str
    reasons: Tuple[str, ...]
    claims: Tuple[ClaimEvaluation, ...]
    ordinary_gate: Optional[GateReport]
    privacy_gate: Optional[Any]
    contains_blended_score: bool = False

    @property
    def no_claim(self) -> bool:
        return not self.approved

    def as_dict(self) -> Dict[str, Any]:
        return {
            "approved": self.approved,
            "claim_id": self.claim_id,
            "claim": self.claim,
            "scope": self.scope,
            "reasons": list(self.reasons),
            "claims": [claim.as_dict() for claim in self.claims],
            "ordinary_gate": (
                None
                if self.ordinary_gate is None
                else self.ordinary_gate.as_dict()
            ),
            "privacy_gate": (
                None
                if self.privacy_gate is None
                else self.privacy_gate.as_dict()
            ),
            "contains_blended_score": False,
        }

    def save(self, path: PathLike) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return destination


def _gate_config(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    config = dict(value) if isinstance(value, Mapping) else load_config(value)
    rules = config.get("rules")
    if isinstance(rules, list):
        normalized = []
        for raw in rules:
            if not isinstance(raw, Mapping):
                normalized.append(raw)
                continue
            rule = dict(raw)
            filters = dict(rule.get("filters", {}))
            for key in (
                "suite",
                "adapter_name",
                "split_seed",
                "model_seed",
                "sample_seed",
                "run_id",
            ):
                if key in rule:
                    filters[key] = rule.pop(key)
            if filters:
                rule["filters"] = filters
            normalized.append(rule)
        config["rules"] = normalized
    return config


def _formal_dp_eligible(report: Optional[Mapping[str, Any]]) -> bool:
    """Validate the complete DP boundary without importing TensorFlow."""

    if not isinstance(report, Mapping):
        return False
    accountant = report.get("accountant", {})
    boundary = report.get("privacy_boundary", {})
    if not isinstance(accountant, Mapping) or not isinstance(boundary, Mapping):
        return False
    epsilon = accountant.get("epsilon")
    try:
        finite_epsilon = epsilon is not None and math.isfinite(float(epsilon))
    except (TypeError, ValueError):
        finite_epsilon = False
    return bool(
        report.get("enabled", False)
        and report.get("mechanism_applied", False)
        and report.get("scope") == "end_to_end"
        and accountant.get("accountant_valid", False)
        and int(accountant.get("steps", 0)) > 0
        and finite_epsilon
        and boundary.get("preprocessing_accounted", False)
        and boundary.get("conditional_frequencies_accounted", False)
        and boundary.get("dataset_size_public", False)
        and boundary.get("noise_randomness_accounted", False)
        and not boundary.get("noise_seed_persisted", True)
        and boundary.get("delta_smaller_than_inverse_dataset", False)
    )


def _evaluate_privacy_release_gates(
    records: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    dp_report: Optional[Mapping[str, Any]],
) -> GateReport:
    """Apply privacy rules while refusing numeric-only formal-DP claims."""

    frame = records.copy()
    rules = config.get("rules", ())
    formal_requested = any(
        isinstance(rule, Mapping) and str(rule.get("metric")) == "formal_dp"
        for rule in rules
    )
    if formal_requested:
        if "metric" in frame:
            frame = frame.loc[
                frame["metric"].astype(str) != "formal_dp"
            ].copy()
        formal = pd.DataFrame(
            [
                {
                    "stage": "raw",
                    "pillar": "privacy_dp",
                    "metric": "formal_dp",
                    "value": float(_formal_dp_eligible(dp_report)),
                    "detail": (
                        "missing_or_unverified"
                        if dp_report is None
                        else str(dp_report.get("scope", "unknown"))
                    ),
                }
            ]
        )
        frame = pd.concat([frame, formal], ignore_index=True, sort=False)
    return evaluate_gates(frame, config)


def _claim_list(config: Mapping[str, Any]) -> Tuple[Mapping[str, Any], ...]:
    claims = config.get("claims", ())
    if not isinstance(claims, (list, tuple)):
        raise TypeError("release claims must be a sequence")
    normalized = []
    ids = set()
    for index, raw in enumerate(claims):
        if not isinstance(raw, Mapping):
            raise TypeError("release claim %d must be a mapping" % index)
        claim_id = str(raw.get("id", "")).strip()
        text = str(raw.get("claim", "")).strip()
        if not claim_id or not text:
            raise ValueError("release claims need id and claim text")
        if claim_id in ids:
            raise ValueError("duplicate release claim id %r" % claim_id)
        ids.add(claim_id)
        normalized.append(dict(raw))
    return tuple(
        sorted(
            normalized,
            key=lambda value: int(value.get("priority", 0)),
            reverse=True,
        )
    )


def _filtered(frame: pd.DataFrame, filters: Mapping[str, Any]) -> pd.DataFrame:
    selected = frame
    for key, expected in filters.items():
        if key not in selected:
            return selected.iloc[0:0]
        selected = selected.loc[
            selected[key].isin(expected)
            if isinstance(expected, (list, tuple, set))
            else selected[key] == expected
        ]
    return selected


def _paired_interval(
    values: np.ndarray,
    *,
    statistic: str,
    confidence: float,
    replicates: int,
    seed: int,
) -> Tuple[float, float, float]:
    reducer = np.mean if statistic == "mean" else np.median
    estimate = float(reducer(values))
    if len(values) == 1 or replicates == 0:
        return estimate, estimate, estimate
    rng = np.random.default_rng(seed)
    samples = rng.choice(
        values, size=(replicates, len(values)), replace=True
    )
    bootstrapped = reducer(samples, axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        estimate,
        float(np.quantile(bootstrapped, alpha)),
        float(np.quantile(bootstrapped, 1.0 - alpha)),
    )


class ReleaseDecision:
    """Evaluate coverage, gates, sealing, and preregistered evidence."""

    def __init__(
        self, config: Union[PathLike, Mapping[str, Any]] = "vnext"
    ) -> None:
        self.config = load_release_config(config)
        self.claims = _claim_list(self.config)

    @staticmethod
    def _frames(
        result: Optional[ReleaseRunResult],
        *,
        manifests: Optional[pd.DataFrame],
        metrics: Optional[pd.DataFrame],
        privacy: Optional[pd.DataFrame],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Mapping[str, Any]]:
        def frame(value: Any) -> pd.DataFrame:
            return (
                value.copy()
                if isinstance(value, pd.DataFrame)
                else pd.DataFrame(value)
            )

        if result is not None:
            return (
                frame(result.manifests),
                frame(result.metrics),
                frame(result.privacy),
                result.protocol,
            )
        if manifests is None or metrics is None:
            raise ValueError(
                "ReleaseDecision needs a ReleaseRunResult or manifests/metrics"
            )
        return (
            frame(manifests),
            frame(metrics),
            (
                pd.DataFrame(columns=frame(metrics).columns)
                if privacy is None
                else frame(privacy)
            ),
            {},
        )

    def _plans(
        self, protocol: Mapping[str, Any]
    ) -> Dict[str, ReleaseSuitePlan]:
        raw = self.config.get("suites")
        if raw:
            return {plan.name: plan for plan in _suite_plans(raw)}
        protocol_suites = protocol.get("suites", {})
        return {
            plan.name: plan for plan in _suite_plans(protocol_suites)
        }

    @staticmethod
    def _sealed(
        manifests: pd.DataFrame, suites: Sequence[str]
    ) -> Tuple[bool, str]:
        selected = _filtered(manifests, {"suite": list(suites)})
        columns = {
            "sealed_test",
            "fit_scope",
            "test_scope",
            "test_unsealed_before_fit",
        }
        if selected.empty or not columns.issubset(selected.columns):
            return False, "sealed-test discipline is not fully recorded"
        valid = (
            selected["sealed_test"].astype(bool)
            & selected["fit_scope"].eq("train_only")
            & selected["test_scope"].eq("evaluation_only")
            & ~selected["test_unsealed_before_fit"].astype(bool)
        )
        if not bool(valid.all()):
            return False, "one or more runs violated sealed-test discipline"
        return True, "sealed-test discipline verified"

    @staticmethod
    def _coverage(
        manifests: pd.DataFrame,
        *,
        plans: Mapping[str, ReleaseSuitePlan],
        suites: Sequence[str],
        models: Sequence[str],
    ) -> Tuple[bool, List[str]]:
        reasons = []
        required_columns = {
            "suite",
            "dataset_id",
            "model_name",
            "split_seed",
            "model_seed",
            "sample_seed",
            "status",
        }
        if not required_columns.issubset(manifests.columns):
            return False, ["run manifests lack exact seed coverage fields"]
        for suite_name in suites:
            if suite_name not in plans:
                reasons.append(
                    "required suite %r has no preregistered plan" % suite_name
                )
                continue
            plan = plans[suite_name]
            expected_seeds = {
                (split_seed, model_seed, sample_seed)
                for split_seed in plan.split_seeds
                for model_seed in plan.model_seeds
                for sample_seed in plan.sample_seeds
            }
            for dataset_id in plan.datasets:
                for model_name in models:
                    selected = _filtered(
                        manifests,
                        {
                            "suite": suite_name,
                            "dataset_id": dataset_id,
                            "model_name": model_name,
                        },
                    )
                    observed = [
                        (
                            int(row.split_seed),
                            int(row.model_seed),
                            int(row.sample_seed),
                        )
                        for row in selected.itertuples(index=False)
                    ]
                    if len(observed) != len(set(observed)):
                        reasons.append(
                            "%s/%s/%s has duplicate seed cells"
                            % (suite_name, dataset_id, model_name)
                        )
                        continue
                    if set(observed) != expected_seeds:
                        reasons.append(
                            "%s/%s/%s seed coverage is not exact"
                            % (suite_name, dataset_id, model_name)
                        )
                        continue
                    if not bool(selected["status"].eq("completed").all()):
                        reasons.append(
                            "%s/%s/%s has failed or incomplete runs"
                            % (suite_name, dataset_id, model_name)
                        )
        return not reasons, reasons

    @staticmethod
    def _no_blended_score(metrics: pd.DataFrame) -> Tuple[bool, str]:
        if bool(metrics.attrs.get("contains_blended_score", False)):
            return False, "metrics declare a blended score"
        if "quality_score" in metrics.columns:
            return False, "metrics contain a quality_score column"
        if "metric" in metrics:
            forbidden = metrics["metric"].astype(str).str.lower().isin(
                {"quality_score", "overall_score", "blended_score"}
            )
            if bool(forbidden.any()):
                return False, "metrics contain a forbidden blended score"
        return True, "metrics remain disaggregated"

    def _statistical_evidence(
        self,
        metrics: pd.DataFrame,
        *,
        claim: Mapping[str, Any],
        candidate: str,
    ) -> Tuple[StatisticalEvidence, ...]:
        raw = claim.get("statistical_evidence")
        if isinstance(raw, Mapping):
            rules = raw.get("comparisons", ())
            default_confidence = float(raw.get("confidence", 0.95))
            default_replicates = int(raw.get("bootstrap_replicates", 500))
        else:
            rules = raw or ()
            default_confidence = 0.95
            default_replicates = 500
        if not isinstance(rules, (list, tuple)):
            raise TypeError("statistical evidence comparisons must be a list")
        results = []
        for index, rule in enumerate(rules):
            if not isinstance(rule, Mapping):
                raise TypeError("statistical evidence rule must be a mapping")
            rule_id = str(rule.get("id", "evidence_%d" % index))
            metric = str(rule["metric"])
            filters = dict(rule.get("filters", {}))
            filters["metric"] = metric
            selected = _filtered(metrics, filters)
            baseline = rule.get("baseline")
            minimum_pairs = int(
                rule.get("minimum_pairs", rule.get("minimum_count", 2))
            )
            if baseline is None:
                count = int(
                    len(
                        _filtered(
                            selected, {"model_name": candidate}
                        )
                    )
                )
                passed = count >= minimum_pairs
                results.append(
                    StatisticalEvidence(
                        rule_id=rule_id,
                        passed=passed,
                        metric=metric,
                        baseline=None,
                        pairs=count,
                        estimate=None,
                        ci_lower=None,
                        ci_upper=None,
                        message=(
                            "minimum observations met"
                            if passed
                            else "insufficient preregistered observations"
                        ),
                    )
                )
                continue
            baseline = str(baseline)
            candidate_rows = _filtered(
                selected, {"model_name": candidate}
            )
            baseline_rows = _filtered(
                selected, {"model_name": baseline}
            )
            preferred = list(
                rule.get(
                    "pair_keys",
                    [
                        "suite",
                        "dataset_id",
                        "split_seed",
                        "model_seed",
                        "sample_seed",
                        "stage",
                        "pillar",
                        "metric",
                    ],
                )
            )
            keys = [
                key
                for key in preferred
                if key in candidate_rows.columns
                and key in baseline_rows.columns
            ]
            if not keys:
                paired = pd.DataFrame()
            else:
                left = (
                    candidate_rows.groupby(
                        keys, dropna=False, sort=True
                    )["value"]
                    .mean()
                    .rename("candidate")
                    .reset_index()
                )
                right = (
                    baseline_rows.groupby(
                        keys, dropna=False, sort=True
                    )["value"]
                    .mean()
                    .rename("baseline")
                    .reset_index()
                )
                paired = left.merge(
                    right, on=keys, how="inner", validate="one_to_one"
                )
            if len(paired):
                candidate_values = pd.to_numeric(
                    paired["candidate"], errors="coerce"
                ).to_numpy(dtype=float)
                baseline_values = pd.to_numeric(
                    paired["baseline"], errors="coerce"
                ).to_numpy(dtype=float)
                direction = str(rule.get("direction", "higher")).lower()
                if direction == "higher":
                    effects = candidate_values - baseline_values
                elif direction == "lower":
                    effects = baseline_values - candidate_values
                else:
                    raise ValueError(
                        "statistical direction must be higher or lower"
                    )
                effects = effects[np.isfinite(effects)]
            else:
                effects = np.empty(0, dtype=float)
            independence_failures = []
            for key, option in (
                ("split_seed", "minimum_split_seeds"),
                ("model_seed", "minimum_model_seeds"),
                ("sample_seed", "minimum_sample_seeds"),
                ("dataset_id", "minimum_datasets"),
            ):
                if option not in rule:
                    continue
                observed = (
                    int(paired[key].nunique(dropna=False))
                    if key in paired
                    else 0
                )
                required = int(rule[option])
                if observed < required:
                    independence_failures.append(
                        "%s=%d<%d" % (key, observed, required)
                    )
            if len(effects) < minimum_pairs or independence_failures:
                results.append(
                    StatisticalEvidence(
                        rule_id=rule_id,
                        passed=False,
                        metric=metric,
                        baseline=baseline,
                        pairs=int(len(effects)),
                        estimate=None,
                        ci_lower=None,
                        ci_upper=None,
                        message=(
                            "insufficient paired seed evidence"
                            + (
                                ": " + ", ".join(independence_failures)
                                if independence_failures
                                else ""
                            )
                        ),
                    )
                )
                continue
            confidence = float(
                rule.get("confidence", default_confidence)
            )
            replicates = int(
                rule.get("bootstrap_replicates", default_replicates)
            )
            statistic = str(rule.get("statistic", "median")).lower()
            if statistic not in {"mean", "median"}:
                raise ValueError("evidence statistic must be mean or median")
            estimate, lower, upper = _paired_interval(
                effects,
                statistic=statistic,
                confidence=confidence,
                replicates=replicates,
                seed=int(hash_json({"claim": claim["id"], "rule": rule_id})[:8], 16),
            )
            minimum_effect = float(rule.get("minimum_effect", 0.0))
            passed = bool(lower > minimum_effect)
            results.append(
                StatisticalEvidence(
                    rule_id=rule_id,
                    passed=passed,
                    metric=metric,
                    baseline=baseline,
                    pairs=int(len(effects)),
                    estimate=estimate,
                    ci_lower=lower,
                    ci_upper=upper,
                    message=(
                        "confidence interval supports the preregistered effect"
                        if passed
                        else "confidence interval does not clear the "
                        "preregistered effect"
                    ),
                )
            )
        return tuple(results)

    def evaluate(
        self,
        result: Optional[ReleaseRunResult] = None,
        *,
        manifests: Optional[pd.DataFrame] = None,
        metrics: Optional[pd.DataFrame] = None,
        privacy: Optional[pd.DataFrame] = None,
        dp_report: Optional[Mapping[str, Any]] = None,
    ) -> ReleaseDecisionReport:
        """Return the strongest fully supported preregistered claim."""

        manifest_frame, metric_frame, privacy_frame, protocol = self._frames(
            result,
            manifests=manifests,
            metrics=metrics,
            privacy=privacy,
        )
        plans = self._plans(protocol)
        candidate = str(
            self.config.get(
                "candidate_model",
                self.config.get("candidate", "ganify_conditional"),
            )
        )
        disaggregated, disaggregated_reason = self._no_blended_score(
            metric_frame
        )
        evaluations = []
        selected_claim: Optional[ClaimEvaluation] = None
        selected_ordinary: Optional[GateReport] = None
        selected_privacy: Optional[Any] = None
        evaluated_ordinary: Optional[GateReport] = None
        evaluated_privacy: Optional[Any] = None
        stronger_failures: List[str] = []

        for claim in self.claims:
            claim_id = str(claim["id"])
            scope = str(claim.get("scope", "narrow")).lower()
            reasons: List[str] = []
            required_suites = [
                str(value)
                for value in claim.get("required_suites", ())
            ]
            required_models = [
                str(value)
                for value in claim.get("required_models", (candidate,))
            ]
            mandatory = [
                str(value)
                for value in claim.get(
                    "mandatory_baselines",
                    self.config.get("mandatory_baselines", ()),
                )
            ]
            for value in mandatory:
                if value not in required_models:
                    required_models.append(value)

            is_broad = scope in {
                "broad_sota",
                "broad-sota",
                "sota",
            }
            if is_broad:
                for suite_name in ("smoke", "core", "stress"):
                    if suite_name not in required_suites:
                        required_suites.append(suite_name)
                frontier = [
                    str(value)
                    for value in claim.get(
                        "frontier_baselines",
                        self.config.get("frontier_baselines", ()),
                    )
                ]
                if not frontier:
                    reasons.append(
                        "broad SOTA requires named frontier baselines"
                    )
                for value in frontier:
                    if value not in required_models:
                        required_models.append(value)

            missing_plans = [
                suite for suite in required_suites if suite not in plans
            ]
            if missing_plans:
                reasons.append(
                    "required suite plans are missing: %s"
                    % ", ".join(sorted(missing_plans))
                )
            coverage, coverage_reasons = self._coverage(
                manifest_frame,
                plans=plans,
                suites=required_suites,
                models=required_models,
            )
            if not coverage:
                reasons.extend(coverage_reasons)
            sealed, sealed_reason = self._sealed(
                manifest_frame, required_suites
            )
            if not sealed:
                reasons.append(sealed_reason)
            if not disaggregated:
                reasons.append(disaggregated_reason)

            ordinary_config = _gate_config(
                claim.get(
                    "ordinary_gates",
                    self.config.get("ordinary_gates"),
                )
            )
            ordinary_report: Optional[GateReport] = None
            if ordinary_config is None:
                reasons.append("ordinary release gates are not preregistered")
            else:
                try:
                    ordinary_report = evaluate_gates(
                        metric_frame, ordinary_config
                    )
                except Exception as error:
                    reasons.append(
                        "ordinary gates could not be evaluated: %s"
                        % _error_values(error)[1]
                    )
                else:
                    if evaluated_ordinary is None:
                        evaluated_ordinary = ordinary_report
                    if not ordinary_report.passed:
                        failed = [
                            rule.rule_id
                            for rule in ordinary_report.rules
                            if not rule.passed
                        ]
                        reasons.append(
                            "ordinary gates failed: %s"
                            % ", ".join(failed)
                        )

            privacy_config = _gate_config(
                claim.get(
                    "privacy_gates",
                    self.config.get("privacy_gates"),
                )
            )
            privacy_report = None
            requires_privacy = bool(
                claim.get("requires_privacy", True)
            )
            if privacy_config is None:
                if requires_privacy:
                    reasons.append(
                        "privacy release gates are not preregistered"
                    )
            else:
                try:
                    candidate_privacy = _filtered(
                        privacy_frame, {"model_name": candidate}
                    )
                    privacy_report = _evaluate_privacy_release_gates(
                        candidate_privacy,
                        privacy_config,
                        dp_report=dp_report,
                    )
                except Exception as error:
                    reasons.append(
                        "privacy gates could not be evaluated: %s"
                        % _error_values(error)[1]
                    )
                else:
                    if evaluated_privacy is None:
                        evaluated_privacy = privacy_report
                    if not privacy_report.passed:
                        failed = [
                            rule.rule_id
                            for rule in privacy_report.rules
                            if not rule.passed
                        ]
                        reasons.append(
                            "privacy gates failed: %s"
                            % ", ".join(failed)
                        )

            evidence = self._statistical_evidence(
                metric_frame, claim=claim, candidate=candidate
            )
            evidence_required = bool(
                claim.get("requires_statistical_evidence", True)
            )
            if evidence_required and not evidence:
                reasons.append(
                    "no preregistered statistical comparison was evaluated"
                )
            failed_evidence = [
                value.rule_id for value in evidence if not value.passed
            ]
            if failed_evidence:
                reasons.append(
                    "statistical evidence failed: %s"
                    % ", ".join(failed_evidence)
                )

            evaluation = ClaimEvaluation(
                claim_id=claim_id,
                claim=str(claim["claim"]),
                scope=scope,
                priority=int(claim.get("priority", 0)),
                passed=not reasons,
                reasons=tuple(reasons),
                statistical_evidence=evidence,
            )
            evaluations.append(evaluation)
            if selected_claim is None and evaluation.passed:
                selected_claim = evaluation
                selected_ordinary = ordinary_report
                selected_privacy = privacy_report
            elif selected_claim is None:
                stronger_failures.extend(
                    "%s: %s" % (claim_id, reason) for reason in reasons
                )

        if selected_claim is None:
            reasons = tuple(
                stronger_failures
                or ["no preregistered claim has complete supporting evidence"]
            )
            return ReleaseDecisionReport(
                approved=False,
                claim_id=None,
                claim=None,
                scope="no_claim",
                reasons=reasons,
                claims=tuple(evaluations),
                ordinary_gate=evaluated_ordinary,
                privacy_gate=evaluated_privacy,
            )
        selected_reasons = [
            "selected the strongest fully supported preregistered claim"
        ]
        selected_reasons.extend(
            "stronger claim denied: %s" % reason
            for reason in stronger_failures
        )
        return ReleaseDecisionReport(
            approved=True,
            claim_id=selected_claim.claim_id,
            claim=selected_claim.claim,
            scope=selected_claim.scope,
            reasons=tuple(selected_reasons),
            claims=tuple(evaluations),
            ordinary_gate=selected_ordinary,
            privacy_gate=selected_privacy,
        )

    decide = evaluate


__all__ = [
    "ClaimEvaluation",
    "DEFAULT_RELEASE_DIRECTORY",
    "RELEASE_PROTOCOL_VERSION",
    "ReleaseDecision",
    "ReleaseDecisionReport",
    "ReleaseModelSpec",
    "ReleaseRunManifest",
    "ReleaseRunResult",
    "ReleaseRunner",
    "ReleaseSuitePlan",
    "StatisticalEvidence",
    "load_release_config",
]
