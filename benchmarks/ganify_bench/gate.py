"""Transparent pass/fail release gates over individual benchmark metrics."""

from __future__ import annotations

import operator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ._config import PathLike, load_config


@dataclass(frozen=True)
class GateRuleResult:
    """Observed value and decision for one preregistered gate rule."""

    rule_id: str
    passed: bool
    metric: str
    observed: Optional[float]
    operator: str
    threshold: float
    count: int
    message: str


@dataclass(frozen=True)
class GateReport:
    """Collection of gate decisions; no quality score is calculated."""

    passed: bool
    rules: Tuple[GateRuleResult, ...]
    config_version: str

    def as_dict(self) -> Dict[str, object]:
        """Return JSON-friendly gate results."""

        return {
            "passed": self.passed,
            "config_version": self.config_version,
            "rules": [asdict(rule) for rule in self.rules],
        }

    def to_frame(self) -> pd.DataFrame:
        """Return one auditable dataframe row per gate."""

        return pd.DataFrame([asdict(rule) for rule in self.rules])


def _config_values(config: Union[PathLike, Mapping[str, Any]]) -> Dict[str, Any]:
    if isinstance(config, (str, Path)):
        return load_config(config)
    return dict(config)


def _reduce(values: np.ndarray, name: str, quantile: Optional[float]) -> float:
    if name == "median":
        return float(np.median(values))
    if name == "mean":
        return float(np.mean(values))
    if name == "minimum":
        return float(np.min(values))
    if name == "maximum":
        return float(np.max(values))
    if name == "quantile":
        if quantile is None or not (0.0 <= quantile <= 1.0):
            raise ValueError("quantile aggregation needs quantile in [0, 1]")
        return float(np.quantile(values, quantile))
    raise ValueError("unsupported gate aggregation %r" % name)


def _compare(observed: float, operation: str, threshold: float) -> bool:
    functions = {
        "<": operator.lt,
        "<=": operator.le,
        ">": operator.gt,
        ">=": operator.ge,
        "==": lambda left, right: bool(np.isclose(left, right)),
        "!=": lambda left, right: not bool(np.isclose(left, right)),
    }
    if operation not in functions:
        raise ValueError("unsupported gate operator %r" % operation)
    return bool(functions[operation](observed, threshold))


def evaluate_gates(
    records: Union[pd.DataFrame, Iterable[Mapping[str, object]]],
    config: Union[PathLike, Mapping[str, Any]],
) -> GateReport:
    """Evaluate preregistered rules against long-form metric records.

    Rule filters can target any record column (for example ``stage: raw``).
    Missing required metrics fail closed. Optional missing rules pass with a
    clear message. Every required rule must pass for the report to pass.
    """

    frame = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    if "metric" not in frame.columns or "value" not in frame.columns:
        raise ValueError("gate records require 'metric' and 'value' columns")
    values = _config_values(config)
    raw_rules = values.get("rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        raise ValueError("gate config must contain a non-empty 'rules' list")
    decisions = []
    seen_ids = set()
    for index, raw in enumerate(raw_rules):
        if not isinstance(raw, Mapping):
            raise ValueError("gate rule %d must be a mapping" % index)
        rule_id = str(raw.get("id", "rule_%d" % index))
        if rule_id in seen_ids:
            raise ValueError("duplicate gate rule id %r" % rule_id)
        seen_ids.add(rule_id)
        metric = str(raw["metric"])
        operation = str(raw.get("operator", "<="))
        threshold = float(raw["threshold"])
        aggregation = str(raw.get("aggregation", "median")).lower()
        required = bool(raw.get("required", True))
        minimum_count = int(raw.get("minimum_count", 1))
        if minimum_count < 1:
            raise ValueError("minimum_count must be positive")

        selected = frame.loc[frame["metric"].astype(str) == metric]
        filters = dict(raw.get("filters", {}))
        for key in ("stage", "pillar", "dataset_id", "model_name", "detail", "column"):
            if key in raw:
                filters[key] = raw[key]
        for key, expected in filters.items():
            if key not in selected.columns:
                selected = selected.iloc[0:0]
                break
            if isinstance(expected, list):
                selected = selected.loc[selected[key].isin(expected)]
            else:
                selected = selected.loc[selected[key] == expected]
        numeric = pd.to_numeric(selected["value"], errors="coerce").to_numpy(
            dtype=float
        )
        numeric = numeric[np.isfinite(numeric)]
        if len(numeric) < minimum_count:
            passed = not required
            message = (
                "missing required observations: found %d, need %d"
                % (len(numeric), minimum_count)
                if required
                else "optional metric unavailable"
            )
            decisions.append(
                GateRuleResult(
                    rule_id=rule_id,
                    passed=passed,
                    metric=metric,
                    observed=None,
                    operator=operation,
                    threshold=threshold,
                    count=int(len(numeric)),
                    message=message,
                )
            )
            continue
        observed = _reduce(
            numeric,
            aggregation,
            None if raw.get("quantile") is None else float(raw["quantile"]),
        )
        passed = _compare(observed, operation, threshold)
        decisions.append(
            GateRuleResult(
                rule_id=rule_id,
                passed=passed,
                metric=metric,
                observed=observed,
                operator=operation,
                threshold=threshold,
                count=int(len(numeric)),
                message=(
                    "passed" if passed else "observed value violates threshold"
                ),
            )
        )
    required_results = [
        decision
        for decision, raw in zip(decisions, raw_rules)
        if bool(raw.get("required", True))
    ]
    return GateReport(
        passed=all(decision.passed for decision in required_results),
        rules=tuple(decisions),
        config_version=str(values.get("version", "unversioned")),
    )


__all__ = ["GateReport", "GateRuleResult", "evaluate_gates"]
