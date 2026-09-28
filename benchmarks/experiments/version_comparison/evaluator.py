"""Current-version evaluation for isolated-source samples and controls."""

from __future__ import annotations

import importlib
import json
import math
import sys
import time
import types
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .core import atomic_write_json, derived_seed


METRIC_COLUMNS = [
    "candidate",
    "stage",
    "pillar",
    "metric",
    "column",
    "detail",
    "value",
]
RELIABILITY_COLUMNS = [
    "candidate",
    "stage",
    "evaluation_success",
    "privacy_success",
    "run_success",
    "failure_phase",
    "error_type",
    "error_message",
    "evaluation_seconds",
    "privacy_seconds",
]


@contextmanager
def _ganify_subpackage(name: str) -> Any:
    """Import evaluation/privacy without requiring optional TensorFlow.

    GANify's evaluation and privacy packages are NumPy/pandas/sklearn-only,
    while the public top-level package imports the training engine eagerly.
    On evaluator-only hosts without TensorFlow, install a package namespace
    pointing at this working tree and import the requested real subpackage.
    """

    qualified = "ganify.%s" % name
    try:
        module = importlib.import_module(qualified)
    except ModuleNotFoundError as error:
        if error.name != "tensorflow":
            raise
    else:
        yield module
        return

    previous = {
        module_name: module
        for module_name, module in list(sys.modules.items())
        if module_name == "ganify" or module_name.startswith("ganify.")
    }
    restore_previous = "ganify" in previous
    for module_name in previous:
        del sys.modules[module_name]
    try:
        package_root = Path(__file__).resolve().parents[3] / "ganify"
        package = types.ModuleType("ganify")
        package.__file__ = str(package_root / "__init__.py")
        package.__package__ = "ganify"
        package.__path__ = [str(package_root)]
        sys.modules["ganify"] = package
        yield importlib.import_module(qualified)
    finally:
        for module_name in list(sys.modules):
            if module_name == "ganify" or module_name.startswith("ganify."):
                del sys.modules[module_name]
        if restore_previous:
            sys.modules.update(previous)


@dataclass
class EvaluationResult:
    """Disaggregated metrics, privacy evidence, and retained failures."""

    metrics: pd.DataFrame
    privacy: pd.DataFrame
    reliability: pd.DataFrame
    controls: Dict[str, pd.DataFrame]
    metadata: Dict[str, Any]

    def __post_init__(self) -> None:
        self.metrics.attrs["contains_blended_score"] = False
        self.privacy.attrs["contains_blended_score"] = False

    def save(self, directory: Path) -> Dict[str, str]:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        paths = {
            "metrics": destination / "metrics.csv",
            "privacy": destination / "privacy.csv",
            "reliability": destination / "reliability.csv",
            "metadata": destination / "evaluation.json",
        }
        self.metrics.to_csv(paths["metrics"], index=False)
        self.privacy.to_csv(paths["privacy"], index=False)
        self.reliability.to_csv(paths["reliability"], index=False)
        atomic_write_json(paths["metadata"], self.metadata)
        return {name: str(path) for name, path in paths.items()}


def bounded_subset(
    frame: pd.DataFrame, *, max_rows: Optional[int], random_state: int
) -> pd.DataFrame:
    """Select a deterministic, bounded audit subset without replacement."""

    if max_rows is None or len(frame) <= int(max_rows):
        return frame.reset_index(drop=True).copy()
    if int(max_rows) < 2:
        raise ValueError("audit max_rows must be at least two")
    rng = np.random.default_rng(int(random_state))
    indices = np.sort(rng.choice(len(frame), size=int(max_rows), replace=False))
    return frame.iloc[indices].reset_index(drop=True)


def build_controls(
    real_train: pd.DataFrame,
    *,
    n_rows: int,
    random_state: int,
) -> Dict[str, pd.DataFrame]:
    """Fit the mandatory bootstrap, independent, and Gaussian-copula controls."""

    from benchmarks.ganify_bench.adapters import (
        BootstrapAdapter,
        GaussianCopulaAdapter,
        IndependentMarginalsAdapter,
    )

    controls = {}
    specifications = (
        ("bootstrap", BootstrapAdapter()),
        ("independent", IndependentMarginalsAdapter()),
        ("gaussian_copula", GaussianCopulaAdapter()),
    )
    for index, (name, adapter) in enumerate(specifications):
        fit_seed = derived_seed(random_state, "control-fit", name, index)
        sample_seed = derived_seed(random_state, "control-sample", name, index)
        controls[name] = (
            adapter.fit(real_train, seed=fit_seed)
            .sample(int(n_rows), seed=sample_seed)
            .reset_index(drop=True)
        )
    return controls


def _privacy_options(values: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    options = {
        "enabled": True,
        "max_rows": 1000,
        "bootstrap_replicates": 20,
        "n_singling_attacks": 50,
        "sensitive_columns": (),
        "quasi_identifier_columns": None,
    }
    if values is not None:
        options.update(dict(values))
    return options


def _run_privacy(
    train: pd.DataFrame,
    test: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    candidate: str,
    stage: str,
    categorical_columns: Optional[Iterable[Any]],
    random_state: int,
    options: Mapping[str, Any],
) -> pd.DataFrame:
    with _ganify_subpackage("privacy") as privacy_api:
        maximum = options.get("max_rows")
        bounded_train = bounded_subset(
            train,
            max_rows=None if maximum is None else int(maximum),
            random_state=derived_seed(random_state, candidate, stage, "privacy-train"),
        )
        bounded_test = bounded_subset(
            test,
            max_rows=None if maximum is None else int(maximum),
            random_state=derived_seed(random_state, candidate, stage, "privacy-test"),
        )
        bounded_synthetic = bounded_subset(
            synthetic,
            max_rows=None if maximum is None else int(maximum),
            random_state=derived_seed(random_state, candidate, stage, "privacy-synthetic"),
        )
        report = privacy_api.audit_privacy(
            bounded_train,
            bounded_test,
            bounded_synthetic,
            sensitive_columns=tuple(options.get("sensitive_columns", ())),
            quasi_identifier_columns=options.get("quasi_identifier_columns"),
            categorical_columns=categorical_columns,
            membership_method=str(options.get("membership_method", "knn")),
            membership_k=int(options.get("membership_k", 1)),
            n_singling_attacks=int(options.get("n_singling_attacks", 50)),
            singling_predicate_size=int(options.get("singling_predicate_size", 3)),
            linkability_columns_a=options.get("linkability_columns_a"),
            linkability_columns_b=options.get("linkability_columns_b"),
            bootstrap_replicates=int(options.get("bootstrap_replicates", 20)),
            confidence=float(options.get("confidence", 0.95)),
            random_state=derived_seed(random_state, candidate, stage, "privacy"),
        )
    frame = report.to_frame(stage=stage)
    frame.insert(0, "candidate", candidate)
    return frame.loc[
        :, ["candidate", "stage", "pillar", "metric", "column", "detail", "value"]
    ]


def _empty_metrics() -> pd.DataFrame:
    return pd.DataFrame(columns=METRIC_COLUMNS)


def _aggregate_table_report(*args: Any, **kwargs: Any) -> pd.DataFrame:
    # Keep the lightweight namespace installed for the whole call because the
    # report resolves optional GANify subpackages lazily.
    with _ganify_subpackage("evaluation") as evaluation_api:
        return evaluation_api.aggregate_table_report(*args, **kwargs)


def _strict_frame(frame: Any, columns: Sequence[Any], name: str) -> pd.DataFrame:
    output = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
    if list(output.columns) != list(columns):
        raise ValueError(
            "%s columns must exactly match real training columns" % name
        )
    if output.empty:
        raise ValueError("%s must not be empty" % name)
    return output.reset_index(drop=True)


def evaluate_stages(
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    stages: Mapping[str, pd.DataFrame],
    *,
    candidate: str = "candidate",
    categorical_columns: Optional[Iterable[Any]] = None,
    constraints: Optional[Iterable[Mapping[str, Any]]] = None,
    target_column: Optional[Any] = None,
    task: Optional[str] = None,
    random_state: int = 0,
    c2st_folds: int = 5,
    privacy: Optional[Mapping[str, Any]] = None,
    include_controls: bool = True,
) -> EvaluationResult:
    """Evaluate each raw/calibrated/projected stage without pooling stages.

    Every fidelity and utility metric is delegated to ``ganify.evaluation``;
    every privacy row comes from ``ganify.privacy.audit_privacy``.  A failed
    stage remains in ``reliability`` and does not erase successful stages.
    """

    train = real_train.reset_index(drop=True).copy()
    test = _strict_frame(real_test, train.columns, "real_test")
    if not stages:
        raise ValueError("at least one generation stage is required")
    stage_frames = {
        str(name): _strict_frame(values, train.columns, "stage %r" % name)
        for name, values in stages.items()
    }
    if len(stage_frames) != len(stages):
        raise ValueError("generation stage names must be unique")
    if target_column is not None and target_column not in train.columns:
        raise ValueError("target column %r is missing" % (target_column,))
    if (target_column is None) != (task is None):
        raise ValueError("target_column and task must be provided together")
    normalized_task = None
    if task is not None:
        normalized_task = str(task).lower()
        if normalized_task in {"multiclass", "binary", "classification"}:
            normalized_task = "classification"
        elif normalized_task in {"numeric_regression", "regression"}:
            normalized_task = "regression"
        else:
            raise ValueError("task must be classification or regression")

    privacy_config = _privacy_options(privacy)
    metric_frames: List[pd.DataFrame] = []
    privacy_frames: List[pd.DataFrame] = []
    reliability_rows: List[Dict[str, Any]] = []
    control_frames: Dict[str, pd.DataFrame] = {}
    candidates: List[Tuple[str, str, pd.DataFrame]] = [
        (str(candidate), stage, frame)
        for stage, frame in stage_frames.items()
    ]

    control_failures: List[Tuple[str, Exception]] = []
    if include_controls:
        control_rows = max(len(frame) for frame in stage_frames.values())
        try:
            control_frames = build_controls(
                train, n_rows=control_rows, random_state=int(random_state)
            )
        except Exception as error:
            control_failures.append(("all", error))
        else:
            candidates.extend(
                ("control:%s" % name, "raw", frame)
                for name, frame in control_frames.items()
            )

    for name, stage, synthetic in candidates:
        evaluation_started = time.perf_counter()
        evaluation_success = False
        privacy_success: Optional[bool] = None
        failure_phase = None
        error_type = None
        error_message = None
        try:
            report = _aggregate_table_report(
                train,
                test,
                {stage: synthetic},
                categorical_columns=categorical_columns,
                constraints=constraints,
                target_column=target_column,
                task=normalized_task,
                random_state=derived_seed(random_state, name, stage, "evaluation"),
                c2st_folds=int(c2st_folds),
            )
            report.insert(0, "candidate", name)
            metric_frames.append(report.loc[:, METRIC_COLUMNS])
            evaluation_success = True
        except Exception as error:
            failure_phase = "evaluation"
            error_type = type(error).__name__
            error_message = str(error)
        evaluation_seconds = max(0.0, time.perf_counter() - evaluation_started)

        privacy_seconds = 0.0
        if bool(privacy_config.get("enabled", True)):
            privacy_started = time.perf_counter()
            try:
                privacy_frame = _run_privacy(
                    train,
                    test,
                    synthetic,
                    candidate=name,
                    stage=stage,
                    categorical_columns=categorical_columns,
                    random_state=int(random_state),
                    options=privacy_config,
                )
                privacy_frames.append(privacy_frame)
                metric_frames.append(privacy_frame.copy())
                privacy_success = True
            except Exception as error:
                privacy_success = False
                if failure_phase is None:
                    failure_phase = "privacy"
                    error_type = type(error).__name__
                    error_message = str(error)
                else:
                    error_message = "%s; privacy %s: %s" % (
                        error_message,
                        type(error).__name__,
                        error,
                    )
            privacy_seconds = max(0.0, time.perf_counter() - privacy_started)

        run_success = evaluation_success and (
            privacy_success is not False
        )
        reliability_rows.append(
            {
                "candidate": name,
                "stage": stage,
                "evaluation_success": evaluation_success,
                "privacy_success": privacy_success,
                "run_success": run_success,
                "failure_phase": failure_phase,
                "error_type": error_type,
                "error_message": error_message,
                "evaluation_seconds": evaluation_seconds,
                "privacy_seconds": privacy_seconds,
            }
        )

    for name, error in control_failures:
        reliability_rows.append(
            {
                "candidate": "control:%s" % name,
                "stage": "raw",
                "evaluation_success": False,
                "privacy_success": False
                if bool(privacy_config.get("enabled", True))
                else None,
                "run_success": False,
                "failure_phase": "control",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "evaluation_seconds": 0.0,
                "privacy_seconds": 0.0,
            }
        )

    metrics = (
        pd.concat(metric_frames, ignore_index=True, sort=False)
        if metric_frames
        else _empty_metrics()
    )
    privacy_frame = (
        pd.concat(privacy_frames, ignore_index=True, sort=False)
        if privacy_frames
        else _empty_metrics()
    )
    reliability = pd.DataFrame(reliability_rows, columns=RELIABILITY_COLUMNS)
    metadata = {
        "candidate": str(candidate),
        "stage_order": list(stage_frames),
        "controls": sorted(control_frames),
        "controls_required": ["bootstrap", "independent", "gaussian_copula"],
        "privacy": {
            key: value
            for key, value in privacy_config.items()
            if key not in {"canaries", "canary_decoys"}
        },
        "contains_blended_score": False,
        "evaluation_api": "ganify.evaluation.aggregate_table_report",
        "privacy_api": "ganify.privacy.audit_privacy",
    }
    return EvaluationResult(
        metrics=metrics,
        privacy=privacy_frame,
        reliability=reliability,
        controls=control_frames,
        metadata=metadata,
    )


__all__ = [
    "EvaluationResult",
    "METRIC_COLUMNS",
    "RELIABILITY_COLUMNS",
    "bounded_subset",
    "build_controls",
    "evaluate_stages",
]

