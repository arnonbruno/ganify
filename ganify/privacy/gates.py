"""Fail-closed privacy release gates."""

from __future__ import annotations

import json
import operator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PrivacyGateDecision:
    rule_id: str
    metric: str
    passed: bool
    observed: Optional[float]
    threshold: float
    operator: str
    count: int
    message: str


@dataclass(frozen=True)
class PrivacyGateReport:
    passed: bool
    rules: Tuple[PrivacyGateDecision, ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "rules": [asdict(rule) for rule in self.rules],
        }

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(rule) for rule in self.rules])


def formal_dp_eligible(report: Optional[Mapping[str, Any]]) -> bool:
    """Return true only for an accounted, end-to-end DP boundary."""

    if not isinstance(report, Mapping):
        return False
    accountant = report.get("accountant", {})
    boundary = report.get("privacy_boundary", {})
    if not isinstance(accountant, Mapping) or not isinstance(boundary, Mapping):
        return False
    epsilon = accountant.get("epsilon")
    return bool(
        report.get("enabled", False)
        and report.get("mechanism_applied", False)
        and report.get("scope") == "end_to_end"
        and accountant.get("accountant_valid", False)
        and int(accountant.get("steps", 0)) > 0
        and epsilon is not None
        and np.isfinite(float(epsilon))
        and boundary.get("preprocessing_accounted", False)
        and boundary.get("conditional_frequencies_accounted", False)
        and boundary.get("dataset_size_public", False)
        and boundary.get("noise_randomness_accounted", False)
        and not boundary.get("noise_seed_persisted", True)
        and boundary.get("delta_smaller_than_inverse_dataset", False)
    )


formal_dp_gate = formal_dp_eligible


def privacy_gate_records(
    attack_report: Optional[Any] = None,
    dp_report: Optional[Mapping[str, Any]] = None,
    *,
    stage: str = "synthetic",
) -> pd.DataFrame:
    """Build gate records without permitting an unverified formal-DP claim."""

    frames: List[pd.DataFrame] = []
    if attack_report is not None:
        if isinstance(attack_report, pd.DataFrame):
            frame = attack_report.copy()
        elif hasattr(attack_report, "to_frame"):
            try:
                frame = attack_report.to_frame(stage=stage)
            except TypeError:
                frame = attack_report.to_frame()
        else:
            frame = pd.DataFrame(attack_report)
        frames.append(frame)
    rows: List[Dict[str, Any]] = []
    if dp_report is not None:
        accountant = dp_report.get("accountant", {})
        boundary = dp_report.get("privacy_boundary", {})
        rows.append(
            {
                "stage": stage,
                "pillar": "privacy_dp",
                "metric": "formal_dp",
                "value": float(formal_dp_eligible(dp_report)),
                "detail": str(dp_report.get("scope", "unknown")),
            }
        )
        if isinstance(accountant, Mapping):
            rows.append(
                {
                    "stage": stage,
                    "pillar": "privacy_dp",
                    "metric": "dp_accountant_valid",
                    "value": float(
                        bool(accountant.get("accountant_valid", False))
                    ),
                    "detail": accountant.get("accountant"),
                }
            )
            epsilon = accountant.get("epsilon")
            delta = accountant.get("delta")
            if epsilon is not None and np.isfinite(float(epsilon)):
                rows.append(
                    {
                        "stage": stage,
                        "pillar": "privacy_dp",
                        "metric": "dp_epsilon",
                        "value": float(epsilon),
                        "detail": "delta=%s" % delta,
                    }
                )
        if isinstance(boundary, Mapping):
            for metric, key in (
                (
                    "dp_preprocessing_accounted",
                    "preprocessing_accounted",
                ),
                (
                    "dp_conditional_frequencies_accounted",
                    "conditional_frequencies_accounted",
                ),
                ("dp_dataset_size_public", "dataset_size_public"),
                (
                    "dp_noise_randomness_accounted",
                    "noise_randomness_accounted",
                ),
                (
                    "dp_delta_boundary_valid",
                    "delta_smaller_than_inverse_dataset",
                ),
            ):
                rows.append(
                    {
                        "stage": stage,
                        "pillar": "privacy_dp",
                        "metric": metric,
                        "value": float(bool(boundary.get(key, False))),
                        "detail": key,
                    }
                )
    if rows:
        frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame(
            columns=["stage", "pillar", "metric", "value", "detail"]
        )
    return pd.concat(frames, ignore_index=True, sort=False)


def _load_config(
    config: Union[str, Path, Mapping[str, Any]]
) -> Dict[str, Any]:
    if isinstance(config, Mapping):
        return dict(config)
    path = Path(config)
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ValueError(
                "gate config is not JSON-compatible YAML and PyYAML is absent"
            ) from exc
        value = yaml.safe_load(text)
    if not isinstance(value, Mapping):
        raise ValueError("gate config root must be a mapping")
    return dict(value)


def _compare(left: float, operation: str, right: float) -> bool:
    functions = {
        "<": operator.lt,
        "<=": operator.le,
        ">": operator.gt,
        ">=": operator.ge,
        "==": lambda a, b: bool(np.isclose(a, b)),
        "!=": lambda a, b: not bool(np.isclose(a, b)),
    }
    if operation not in functions:
        raise ValueError("unsupported gate operator %r" % operation)
    return bool(functions[operation](left, right))


def evaluate_privacy_gates(
    records: Any,
    config: Union[str, Path, Mapping[str, Any]],
    *,
    dp_report: Optional[Mapping[str, Any]] = None,
    stage: str = "synthetic",
) -> PrivacyGateReport:
    """Evaluate attack and formal-DP rules; missing required metrics fail."""

    frame = privacy_gate_records(records, dp_report, stage=stage)
    values = _load_config(config)
    rules = values.get("rules")
    if not isinstance(rules, list) or not rules:
        raise ValueError("gate config must contain a non-empty rules list")
    decisions: List[PrivacyGateDecision] = []
    seen = set()
    for index, raw in enumerate(rules):
        if not isinstance(raw, Mapping):
            raise ValueError("gate rule %d must be a mapping" % index)
        rule_id = str(raw.get("id", "rule_%d" % index))
        if rule_id in seen:
            raise ValueError("duplicate gate rule id %r" % rule_id)
        seen.add(rule_id)
        metric = str(raw["metric"])
        selected = frame.loc[
            frame.get("metric", pd.Series(dtype=object)).astype(str) == metric
        ]
        # A caller-provided numeric row can never establish formal DP.
        if metric == "formal_dp":
            selected = privacy_gate_records(
                None, dp_report, stage=stage
            )
            selected = selected.loc[selected["metric"] == "formal_dp"]
        filters = dict(raw.get("filters", {}))
        for key in ("stage", "pillar", "detail", "column"):
            if key in raw:
                filters[key] = raw[key]
        for key, expected in filters.items():
            if key not in selected:
                selected = selected.iloc[0:0]
                break
            selected = selected.loc[
                selected[key].isin(expected)
                if isinstance(expected, list)
                else selected[key] == expected
            ]
        numeric = (
            pd.to_numeric(selected.get("value"), errors="coerce")
            .to_numpy(dtype=np.float64)
            if "value" in selected
            else np.empty(0, dtype=np.float64)
        )
        numeric = numeric[np.isfinite(numeric)]
        minimum_count = int(raw.get("minimum_count", 1))
        required = bool(raw.get("required", True))
        operation = str(raw.get("operator", "<="))
        threshold = float(raw["threshold"])
        if len(numeric) < minimum_count:
            decisions.append(
                PrivacyGateDecision(
                    rule_id=rule_id,
                    metric=metric,
                    passed=not required,
                    observed=None,
                    threshold=threshold,
                    operator=operation,
                    count=len(numeric),
                    message=(
                        "missing required metric (fail closed)"
                        if required
                        else "optional metric unavailable"
                    ),
                )
            )
            continue
        aggregation = str(raw.get("aggregation", "maximum")).lower()
        reducers = {
            "maximum": np.max,
            "minimum": np.min,
            "mean": np.mean,
            "median": np.median,
        }
        if aggregation not in reducers:
            raise ValueError(
                "unsupported privacy gate aggregation %r" % aggregation
            )
        observed = float(reducers[aggregation](numeric))
        passed = _compare(observed, operation, threshold)
        decisions.append(
            PrivacyGateDecision(
                rule_id=rule_id,
                metric=metric,
                passed=passed,
                observed=observed,
                threshold=threshold,
                operator=operation,
                count=len(numeric),
                message=(
                    "passed"
                    if passed
                    else "observed value violates threshold"
                ),
            )
        )
    required_results = [
        decision
        for decision, raw in zip(decisions, rules)
        if bool(raw.get("required", True))
    ]
    return PrivacyGateReport(
        passed=all(result.passed for result in required_results),
        rules=tuple(decisions),
    )


__all__ = [
    "PrivacyGateDecision",
    "PrivacyGateReport",
    "evaluate_privacy_gates",
    "formal_dp_eligible",
    "formal_dp_gate",
    "privacy_gate_records",
]
