"""Inspection and lower-fidelity export helpers for privacy-sensitive state."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ganify.schema import encode_json_value

from ._utils import as_frame, infer_categorical, value_key


@dataclass(frozen=True)
class ArtifactFinding:
    category: str
    severity: str
    location: str
    evidence: str
    explanation: str

    def as_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ArtifactRiskReport:
    findings: Tuple[ArtifactFinding, ...]
    empirical_quantile_state_detected: bool
    category_state_detected: bool
    training_record_state_detected: bool
    model_weights_detected: bool
    model_artifacts_more_sensitive_than_samples: bool
    safe_for_public_release: bool
    summary: str
    limitations: Tuple[str, ...]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _finding(
    findings: List[ArtifactFinding],
    seen: set,
    *,
    category: str,
    severity: str,
    location: str,
    evidence: str,
    explanation: str,
) -> None:
    key = (category, location, evidence)
    if key in seen:
        return
    seen.add(key)
    findings.append(
        ArtifactFinding(
            category=category,
            severity=severity,
            location=location,
            evidence=evidence,
            explanation=explanation,
        )
    )


def _inspect_mapping(
    value: Any,
    *,
    location: str,
    findings: List[ArtifactFinding],
    seen: set,
) -> None:
    if isinstance(value, Mapping):
        transform_type = value.get("type")
        if transform_type in {
            "empirical_copula",
            "discrete_copula",
            "mode_aware",
        }:
            _finding(
                findings,
                seen,
                category="empirical_quantile_state",
                severity="high",
                location=location,
                evidence="numeric transform type=%s" % transform_type,
                explanation=(
                    "Fitted knots, support, centers, or scales are statistics "
                    "of private training values and can enable stronger attacks "
                    "than sample-only access."
                ),
            )
        for key, nested in value.items():
            child = "%s.%s" % (location, key) if location else str(key)
            canonical = str(key).lower()
            if canonical in {
                "vocabulary",
                "categories",
                "target_values",
                "labels",
            }:
                _finding(
                    findings,
                    seen,
                    category="category_state",
                    severity="high",
                    location=child,
                    evidence="fitted category/support state",
                    explanation=(
                        "Observed categories can reveal rare or unique private "
                        "values even when no generated row contains them."
                    ),
                )
            if canonical in {
                "output_matrix",
                "row_indices",
                "condition_defaults",
                "parameter_condition_defaults",
            }:
                _finding(
                    findings,
                    seen,
                    category="training_record_state",
                    severity="critical",
                    location=child,
                    evidence="training-derived row-level state",
                    explanation=(
                        "This state can contain encoded records, membership "
                        "indices, or literal values from private rows."
                    ),
                )
            if canonical in {
                "values",
                "coordinates",
                "centers",
                "scales",
                "weights",
                "counts",
            } and transform_type is not None:
                _finding(
                    findings,
                    seen,
                    category="empirical_quantile_state",
                    severity="high",
                    location=child,
                    evidence="fitted numeric support/statistics",
                    explanation=(
                        "Exact empirical transform state should be treated as "
                        "private model state, not as a harmless configuration."
                    ),
                )
            _inspect_mapping(
                nested,
                location=child,
                findings=findings,
                seen=seen,
            )
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _inspect_mapping(
                nested,
                location="%s[%d]" % (location, index),
                findings=findings,
                seen=seen,
            )


def _load_artifact_payload(
    artifact: Any,
) -> Tuple[List[Tuple[str, Any]], List[Path]]:
    payloads: List[Tuple[str, Any]] = []
    files: List[Path] = []
    if isinstance(artifact, (str, Path)):
        source = Path(artifact)
        if not source.exists():
            raise FileNotFoundError("artifact does not exist: %s" % source)
        files = sorted(
            [source] if source.is_file() else [path for path in source.rglob("*") if path.is_file()]
        )
        for path in files:
            if path.suffix.lower() == ".json":
                try:
                    payloads.append(
                        (
                            str(path),
                            json.loads(path.read_text(encoding="utf-8")),
                        )
                    )
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    payloads.append((str(path), {}))
    elif isinstance(artifact, Mapping):
        payloads.append(("artifact", artifact))
    elif hasattr(artifact, "to_dict"):
        payloads.append(
            (
                type(artifact).__name__,
                artifact.to_dict(),
            )
        )
    else:
        for name in ("transformer_", "sampler_"):
            nested = getattr(artifact, name, None)
            if nested is not None and hasattr(nested, "to_dict"):
                payloads.append(
                    ("%s.%s" % (type(artifact).__name__, name), nested.to_dict())
                )
        if not payloads:
            raise TypeError(
                "artifact must be a path, mapping, persisted-state object, or "
                "fitted engine"
            )
    return payloads, files


def inspect_artifact_risk(artifact: Any) -> ArtifactRiskReport:
    """Detect training-derived state that is riskier than generated samples."""

    payloads, files = _load_artifact_payload(artifact)
    findings: List[ArtifactFinding] = []
    seen: set = set()
    for location, payload in payloads:
        _inspect_mapping(
            payload,
            location=location,
            findings=findings,
            seen=seen,
        )
    for path in files:
        suffix = path.suffix.lower()
        name = path.name.lower()
        if suffix in {".h5", ".keras", ".ckpt", ".pb"} or "weights" in name:
            _finding(
                findings,
                seen,
                category="model_weights",
                severity="high",
                location=str(path),
                evidence="trained model parameters",
                explanation=(
                    "White-box parameter access supports attacks unavailable "
                    "against a finite synthetic sample."
                ),
            )
        if suffix == ".npz":
            _finding(
                findings,
                seen,
                category="binary_fitted_state",
                severity="high",
                location=str(path),
                evidence="opaque NumPy state archive",
                explanation=(
                    "Binary fitted state may contain exact supports, moments, "
                    "or row-level values and requires a schema-aware review."
                ),
            )
    categories = {finding.category for finding in findings}
    sensitive = bool(findings)
    return ArtifactRiskReport(
        findings=tuple(findings),
        empirical_quantile_state_detected=(
            "empirical_quantile_state" in categories
        ),
        category_state_detected="category_state" in categories,
        training_record_state_detected=(
            "training_record_state" in categories
        ),
        model_weights_detected="model_weights" in categories,
        model_artifacts_more_sensitive_than_samples=True,
        safe_for_public_release=not sensitive,
        summary=(
            "Model artifacts are more sensitive than finite synthetic samples "
            "because they can expose fitted supports, categories, row-level "
            "sampler state, and white-box parameters. Treat them as private."
            if sensitive
            else
            "No known high-risk fitted state was detected, but absence of a "
            "finding is not a release guarantee."
        ),
        limitations=(
            "Inspection is structural and cannot prove that an opaque binary "
            "artifact contains no private information.",
            "A safe public release still requires attack testing, access "
            "controls, composition review, and an explicit privacy policy.",
        ),
    )


artifact_risk_report = inspect_artifact_risk


def _quantized_probabilities(
    counts: np.ndarray, smoothing: float, quantum: float
) -> List[float]:
    probabilities = (
        counts.astype(np.float64) + float(smoothing)
    ) / float(np.sum(counts) + float(smoothing) * len(counts))
    units = int(round(1.0 / float(quantum)))
    if units < 1 or not np.isclose(units * float(quantum), 1.0):
        raise ValueError("probability_quantum must be the reciprocal of an integer")
    raw = probabilities * units
    allocated = np.floor(raw).astype(np.int64)
    remaining = units - int(np.sum(allocated))
    if remaining > 0:
        order = np.argsort(-(raw - allocated), kind="mergesort")
        allocated[order[:remaining]] += 1
    return (allocated.astype(np.float64) / float(units)).tolist()


def export_smoothed_marginals(
    frame: Any,
    destination: Optional[Union[str, Path]] = None,
    *,
    categorical_columns: Optional[Iterable[Any]] = None,
    category_domains: Optional[Mapping[Any, Sequence[Any]]] = None,
    numeric_bounds: Optional[Mapping[Any, Tuple[float, float]]] = None,
    bins: int = 16,
    smoothing: float = 1.0,
    numeric_quantum: float = 0.01,
    probability_quantum: float = 0.001,
    minimum_category_count: int = 2,
) -> Dict[str, Any]:
    """Export coarse, smoothed marginals without exact empirical CDF knots.

    This reduces artifact fidelity; it is **not** differential privacy. Bounds
    and category domains are marked empirical unless supplied by the caller.
    """

    table = as_frame(frame, name="frame")
    if isinstance(bins, bool) or int(bins) < 2:
        raise ValueError("bins must be at least two")
    if not math.isfinite(float(smoothing)) or float(smoothing) <= 0.0:
        raise ValueError("smoothing must be finite and positive")
    if not math.isfinite(float(numeric_quantum)) or float(numeric_quantum) <= 0:
        raise ValueError("numeric_quantum must be finite and positive")
    if isinstance(minimum_category_count, bool) or int(minimum_category_count) < 1:
        raise ValueError("minimum_category_count must be positive")
    categorical = set(infer_categorical(table, categorical_columns))
    domains = {} if category_domains is None else dict(category_domains)
    bounds = {} if numeric_bounds is None else dict(numeric_bounds)
    columns: List[Dict[str, Any]] = []
    empirical_bounds: List[Any] = []
    empirical_domains: List[Any] = []
    for column in table.columns:
        missing_count = int(table[column].isna().sum())
        if column in categorical:
            observed = table[column].dropna().tolist()
            counts_by_key: Dict[Tuple[str, str], int] = {}
            representative: Dict[Tuple[str, str], Any] = {}
            for raw in observed:
                key = value_key(raw)
                representative.setdefault(key, raw)
                counts_by_key[key] = counts_by_key.get(key, 0) + 1
            if column in domains:
                labels = list(domains[column])
            else:
                empirical_domains.append(column)
                labels = [
                    representative[key]
                    for key in sorted(representative)
                    if counts_by_key[key] >= int(minimum_category_count)
                ]
            label_keys = [value_key(label) for label in labels]
            category_counts = [
                int(counts_by_key.get(key, 0)) for key in label_keys
            ]
            suppressed = sum(counts_by_key.values()) - sum(category_counts)
            if suppressed > 0:
                labels.append("__OTHER__")
                category_counts.append(int(suppressed))
            labels.append(None)
            category_counts.append(missing_count)
            columns.append(
                {
                    "name": encode_json_value(column),
                    "kind": "categorical",
                    "labels": [encode_json_value(value) for value in labels],
                    "probabilities": _quantized_probabilities(
                        np.asarray(category_counts, dtype=np.int64),
                        float(smoothing),
                        float(probability_quantum),
                    ),
                    "domain": (
                        "public_or_fixed"
                        if column in domains
                        else "empirical_coarsened"
                    ),
                }
            )
            continue
        numeric = pd.to_numeric(
            table[column], errors="coerce"
        ).to_numpy(dtype=np.float64)
        finite = numeric[np.isfinite(numeric)]
        if column in bounds:
            lower, upper = map(float, bounds[column])
            boundary = "public_or_fixed"
        else:
            empirical_bounds.append(column)
            if len(finite):
                lower = (
                    math.floor(float(np.min(finite)) / numeric_quantum)
                    * numeric_quantum
                )
                upper = (
                    math.ceil(float(np.max(finite)) / numeric_quantum)
                    * numeric_quantum
                )
            else:
                lower, upper = 0.0, numeric_quantum
            boundary = "empirical_quantized"
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            if lower == upper and math.isfinite(lower):
                upper = lower + numeric_quantum
            else:
                raise ValueError("invalid numeric bounds for %r" % (column,))
        edges = np.linspace(lower, upper, int(bins) + 1)
        edges = np.round(edges / numeric_quantum) * numeric_quantum
        edges = np.unique(edges)
        if len(edges) < 2:
            edges = np.asarray([lower, upper], dtype=np.float64)
        clipped = np.clip(finite, edges[0], edges[-1])
        histogram, _ = np.histogram(clipped, bins=edges)
        histogram = np.concatenate(
            [histogram.astype(np.int64), np.asarray([missing_count])]
        )
        columns.append(
            {
                "name": encode_json_value(column),
                "kind": "numeric",
                "edges": [float(value) for value in edges],
                "probabilities": _quantized_probabilities(
                    histogram,
                    float(smoothing),
                    float(probability_quantum),
                ),
                "last_probability_is_missing": True,
                "bounds": boundary,
            }
        )
    output: Dict[str, Any] = {
        "format": "ganify-smoothed-quantized-marginals",
        "version": 1,
        "columns": columns,
        "parameters": {
            "bins": int(bins),
            "smoothing": float(smoothing),
            "numeric_quantum": float(numeric_quantum),
            "probability_quantum": float(probability_quantum),
            "minimum_category_count": int(minimum_category_count),
        },
        "privacy": {
            "differentially_private": False,
            "empirical_bound_columns": [
                encode_json_value(value) for value in empirical_bounds
            ],
            "empirical_domain_columns": [
                encode_json_value(value) for value in empirical_domains
            ],
            "warning": (
                "Smoothing and quantization reduce fidelity but do not provide "
                "differential privacy. Empirical bounds/domains remain "
                "training-derived."
            ),
        },
    }
    if destination is not None:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(output, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
    return output


export_smoothed_quantized_marginals = export_smoothed_marginals


__all__ = [
    "ArtifactFinding",
    "ArtifactRiskReport",
    "artifact_risk_report",
    "export_smoothed_marginals",
    "export_smoothed_quantized_marginals",
    "inspect_artifact_risk",
]
