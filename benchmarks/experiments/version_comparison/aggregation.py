"""Hierarchical sample-seed, fit-level, and paired-version aggregation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


Records = Union[pd.DataFrame, Iterable[Mapping[str, Any]]]
METRIC_ID_COLUMNS = ("stage", "pillar", "metric", "column", "detail")
FIT_ID_COLUMNS = ("split_seed", "fit_seed", "model_seed", "fit_id")
ENTITY_COLUMNS = ("dataset_id", "lane", "version", "adapter", "model_name", "candidate")


@dataclass
class AggregationResult:
    """All hierarchy levels needed to audit a version comparison."""

    fit_means: pd.DataFrame
    summary: pd.DataFrame
    paired_deltas: pd.DataFrame

    def __post_init__(self) -> None:
        self.fit_means.attrs["aggregation_unit"] = "fitted_model"
        self.summary.attrs["aggregation_unit"] = "fitted_model"
        self.paired_deltas.attrs["aggregation_unit"] = "paired_fitted_model"
        for frame in (self.fit_means, self.summary, self.paired_deltas):
            frame.attrs["columns_are_independent_units"] = False
            frame.attrs["sample_seeds_are_independent_units"] = False


def _frame(records: Records) -> pd.DataFrame:
    frame = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    required = {"metric", "value", "sample_seed"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("metric records are missing columns: %r" % missing)
    if frame.empty:
        raise ValueError("metric records must not be empty")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame


def _present(frame: pd.DataFrame, names: Sequence[str]) -> List[str]:
    return [name for name in names if name in frame.columns]


def _row_success(frame: pd.DataFrame) -> pd.Series:
    finite = np.isfinite(frame["value"].to_numpy(dtype=float))
    if "success" in frame.columns:
        explicit = frame["success"].fillna(False).astype(bool).to_numpy()
    elif "run_success" in frame.columns:
        explicit = frame["run_success"].fillna(False).astype(bool).to_numpy()
    elif "status" in frame.columns:
        explicit = frame["status"].astype(str).eq("completed").to_numpy()
    else:
        explicit = np.ones(len(frame), dtype=bool)
    return pd.Series(finite & explicit, index=frame.index)


def _seed_set(values: Optional[Sequence[int]], observed: pd.Series) -> Tuple[int, ...]:
    if values is None:
        result = tuple(sorted(int(value) for value in observed.unique()))
    else:
        result = tuple(int(value) for value in values)
    if not result or len(result) != len(set(result)):
        raise ValueError("expected_sample_seeds must be non-empty and unique")
    return result


def average_sample_seeds(
    records: Records,
    *,
    expected_sample_seeds: Optional[Sequence[int]] = None,
    fit_columns: Optional[Sequence[str]] = None,
    metric_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Average sample seeds inside each fitted model and metric/column cell.

    Failed or missing sample seeds make ``fit_success`` false.  A diagnostic
    mean is retained, but failed fits are excluded from fit-level summaries.
    """

    frame = _frame(records)
    frame["_row_success"] = _row_success(frame)
    expected = _seed_set(expected_sample_seeds, frame["sample_seed"])
    expected_set = set(expected)
    unexpected = sorted(
        set(int(value) for value in frame["sample_seed"]) - expected_set
    )
    if unexpected:
        raise ValueError("records contain unexpected sample seeds: %r" % unexpected)

    if fit_columns is None:
        fits = _present(frame, ENTITY_COLUMNS + FIT_ID_COLUMNS)
    else:
        fits = list(fit_columns)
    if not any(name in fits for name in FIT_ID_COLUMNS):
        raise ValueError(
            "fit_columns must identify fitted models with one of %r"
            % (FIT_ID_COLUMNS,)
        )
    if metric_columns is None:
        metrics = _present(frame, METRIC_ID_COLUMNS)
    else:
        metrics = list(metric_columns)
    if "metric" not in metrics:
        metrics.append("metric")
    groups = fits + [name for name in metrics if name not in fits]
    missing = sorted(set(groups).difference(frame.columns))
    if missing:
        raise ValueError("unknown aggregation columns: %r" % missing)

    duplicate_columns = groups + ["sample_seed"]
    duplicates = frame.duplicated(duplicate_columns, keep=False)
    if bool(duplicates.any()):
        example = frame.loc[duplicates, duplicate_columns].iloc[0].to_dict()
        raise ValueError(
            "multiple rows occupy one fit/sample/metric/column cell: %r" % example
        )

    rows: List[Dict[str, Any]] = []
    for key, group in frame.groupby(groups, dropna=False, sort=True):
        key_values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(groups, key_values))
        observed = {int(value) for value in group["sample_seed"]}
        successful = group.loc[group["_row_success"]]
        successful_seeds = {
            int(value) for value in successful["sample_seed"].tolist()
        }
        failed_or_missing = expected_set - successful_seeds
        values = successful["value"].to_numpy(dtype=float)
        row.update(
            {
                "value": float(np.mean(values)) if len(values) else float("nan"),
                "sample_mean": float(np.mean(values))
                if len(values)
                else float("nan"),
                "expected_sample_seeds": len(expected),
                "observed_sample_seeds": len(observed),
                "successful_sample_seeds": len(successful_seeds),
                "failed_sample_seeds": len(failed_or_missing),
                "fit_success": not failed_or_missing
                and observed == expected_set
                and len(values) == len(expected),
                "sample_seed_values": list(expected),
            }
        )
        rows.append(row)
    output = pd.DataFrame(rows)
    output.attrs["aggregation_unit"] = "fitted_model"
    output.attrs["sample_seed_reduction"] = "arithmetic_mean_within_fit"
    output.attrs["columns_are_independent_units"] = False
    return output


def _bootstrap_median(
    values: np.ndarray,
    *,
    confidence: float,
    n_bootstrap: int,
    seed: int,
) -> Tuple[float, float]:
    if len(values) == 1 or n_bootstrap == 0:
        value = float(np.median(values))
        return value, value
    rng = np.random.default_rng(int(seed))
    medians = np.empty(int(n_bootstrap), dtype=float)
    for index in range(int(n_bootstrap)):
        medians[index] = np.median(
            rng.choice(values, size=len(values), replace=True)
        )
    alpha = (1.0 - float(confidence)) / 2.0
    return (
        float(np.quantile(medians, alpha)),
        float(np.quantile(medians, 1.0 - alpha)),
    )


def _group_seed(random_state: int, key: Any) -> int:
    payload = json.dumps(
        [int(random_state), key],
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


def _direction(
    metric: str,
    pillar: Optional[str],
    directions: Optional[Mapping[str, str]],
) -> str:
    if directions and metric in directions:
        value = str(directions[metric]).lower()
        if value in {"higher", "higher_is_better", "maximize", "max"}:
            return "higher"
        if value in {"lower", "lower_is_better", "minimize", "min"}:
            return "lower"
        raise ValueError("invalid direction %r for metric %r" % (value, metric))
    higher = {
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "r2",
        "relative_primary_utility",
        "valid_rate",
        "support_recall",
        "authenticity",
        "coverage",
    }
    if metric in higher or str(metric).startswith("relative_"):
        return "higher"
    return "lower"


def summarize_fits(
    fit_means: pd.DataFrame,
    *,
    version_column: str = "version",
    group_columns: Optional[Sequence[str]] = None,
    directions: Optional[Mapping[str, str]] = None,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    random_state: int = 0,
) -> pd.DataFrame:
    """Report fit-level median/IQR/bootstrap CI/worst value and failures."""

    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if int(n_bootstrap) < 0:
        raise ValueError("n_bootstrap must be non-negative")
    required = {"metric", "value", "fit_success"}
    missing = sorted(required.difference(fit_means.columns))
    if missing:
        raise ValueError("fit means are missing columns: %r" % missing)
    if version_column not in fit_means.columns:
        raise ValueError("fit means need version column %r" % version_column)

    if group_columns is None:
        preferred = (
            "dataset_id",
            "lane",
            version_column,
            "adapter",
            "model_name",
            "candidate",
            *METRIC_ID_COLUMNS,
        )
        groups = _present(fit_means, preferred)
    else:
        groups = list(group_columns)
    if version_column not in groups:
        groups.append(version_column)
    if "metric" not in groups:
        groups.append("metric")
    groups = list(dict.fromkeys(groups))

    rows: List[Dict[str, Any]] = []
    for key, group in fit_means.groupby(groups, dropna=False, sort=True):
        key_values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(groups, key_values))
        successful = group.loc[
            group["fit_success"].astype(bool)
            & np.isfinite(group["value"].to_numpy(dtype=float))
        ]
        values = successful["value"].to_numpy(dtype=float)
        total = int(len(group))
        failed = total - int(len(successful))
        direction = _direction(
            str(row["metric"]),
            None if "pillar" not in row else str(row["pillar"]),
            directions,
        )
        if len(values):
            q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
            lower, upper = _bootstrap_median(
                values,
                confidence=float(confidence),
                n_bootstrap=int(n_bootstrap),
                seed=_group_seed(random_state, key_values),
            )
            worst = np.min(values) if direction == "higher" else np.max(values)
            mean = np.mean(values)
        else:
            q25 = median = q75 = lower = upper = worst = mean = float("nan")
        row.update(
            {
                "fit_count": total,
                "successful_fits": int(len(successful)),
                "failed_fits": failed,
                "failure_rate": failed / float(total) if total else float("nan"),
                "mean": float(mean),
                "median": float(median),
                "q25": float(q25),
                "q75": float(q75),
                "iqr": float(q75 - q25),
                "ci95_lower": float(lower),
                "ci95_upper": float(upper),
                "worst": float(worst),
                "direction": direction,
            }
        )
        rows.append(row)
    output = pd.DataFrame(rows)
    output.attrs["aggregation_unit"] = "fitted_model"
    output.attrs["confidence"] = float(confidence)
    output.attrs["columns_are_independent_units"] = False
    return output


def paired_version_deltas(
    fit_means: pd.DataFrame,
    *,
    reference_version: Any,
    version_column: str = "version",
    pair_columns: Optional[Sequence[str]] = None,
    metric_columns: Optional[Sequence[str]] = None,
    directions: Optional[Mapping[str, str]] = None,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    random_state: int = 0,
) -> pd.DataFrame:
    """Compute candidate-minus-reference deltas on matched fitted models."""

    if version_column not in fit_means.columns:
        raise ValueError("fit means need version column %r" % version_column)
    versions = list(pd.unique(fit_means[version_column]))
    if reference_version not in versions:
        raise ValueError("reference version %r is absent" % reference_version)
    if pair_columns is None:
        pairs = _present(
            fit_means,
            ("dataset_id", "lane", "split_seed", "fit_seed", "model_seed"),
        )
    else:
        pairs = list(pair_columns)
    if not any(name in pairs for name in ("fit_seed", "model_seed", "split_seed")):
        raise ValueError("paired deltas require fit-identifying seed columns")
    if metric_columns is None:
        metrics = _present(fit_means, METRIC_ID_COLUMNS)
    else:
        metrics = list(metric_columns)
    if "metric" not in metrics:
        metrics.append("metric")
    keys = list(dict.fromkeys(pairs + metrics))

    reference = fit_means.loc[
        fit_means[version_column] == reference_version,
        keys + ["value", "fit_success"],
    ].rename(
        columns={"value": "reference_value", "fit_success": "reference_success"}
    )
    if bool(reference.duplicated(keys, keep=False).any()):
        raise ValueError(
            "reference version has multiple fits for one pairing key; "
            "include additional pair_columns"
        )

    rows: List[Dict[str, Any]] = []
    for version in versions:
        if version == reference_version:
            continue
        candidate = fit_means.loc[
            fit_means[version_column] == version,
            keys + ["value", "fit_success"],
        ].rename(
            columns={"value": "candidate_value", "fit_success": "candidate_success"}
        )
        if bool(candidate.duplicated(keys, keep=False).any()):
            raise ValueError(
                "version %r has multiple fits for one pairing key; include "
                "additional pair_columns" % version
            )
        merged = reference.merge(candidate, on=keys, how="outer", indicator=True)
        merged[version_column] = version
        merged["reference_version"] = reference_version
        merged["paired_success"] = (
            merged["_merge"].eq("both")
            & merged["reference_success"].fillna(False).astype(bool)
            & merged["candidate_success"].fillna(False).astype(bool)
            & np.isfinite(
                pd.to_numeric(merged["reference_value"], errors="coerce")
            )
            & np.isfinite(
                pd.to_numeric(merged["candidate_value"], errors="coerce")
            )
        )
        merged["delta"] = (
            pd.to_numeric(merged["candidate_value"], errors="coerce")
            - pd.to_numeric(merged["reference_value"], errors="coerce")
        )

        summary_groups = [
            name
            for name in keys
            if name not in {"split_seed", "fit_seed", "model_seed", "fit_id"}
        ]
        for key, group in merged.groupby(summary_groups, dropna=False, sort=True):
            key_values = key if isinstance(key, tuple) else (key,)
            row = dict(zip(summary_groups, key_values))
            successful = group.loc[group["paired_success"]]
            values = successful["delta"].to_numpy(dtype=float)
            total = int(len(group))
            failed = total - int(len(successful))
            direction = _direction(
                str(row["metric"]),
                None if "pillar" not in row else str(row["pillar"]),
                directions,
            )
            if len(values):
                q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
                lower, upper = _bootstrap_median(
                    values,
                    confidence=float(confidence),
                    n_bootstrap=int(n_bootstrap),
                    seed=_group_seed(
                        random_state, [reference_version, version, *key_values]
                    ),
                )
                worst = np.min(values) if direction == "higher" else np.max(values)
            else:
                q25 = median = q75 = lower = upper = worst = float("nan")
            row.update(
                {
                    version_column: version,
                    "reference_version": reference_version,
                    "pair_count": total,
                    "successful_pairs": int(len(successful)),
                    "failed_or_missing_pairs": failed,
                    "failure_rate": failed / float(total)
                    if total
                    else float("nan"),
                    "median_delta": float(median),
                    "q25_delta": float(q25),
                    "q75_delta": float(q75),
                    "iqr_delta": float(q75 - q25),
                    "ci95_lower_delta": float(lower),
                    "ci95_upper_delta": float(upper),
                    "worst_delta": float(worst),
                    "direction": direction,
                    "delta_definition": "candidate_minus_reference",
                }
            )
            rows.append(row)
    output = pd.DataFrame(rows)
    output.attrs["aggregation_unit"] = "paired_fitted_model"
    output.attrs["reference_version"] = reference_version
    output.attrs["columns_are_independent_units"] = False
    return output


def aggregate_version_comparison(
    records: Records,
    *,
    reference_version: Optional[Any] = None,
    expected_sample_seeds: Optional[Sequence[int]] = None,
    directions: Optional[Mapping[str, str]] = None,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    random_state: int = 0,
) -> AggregationResult:
    """Run the complete non-flattening aggregation hierarchy."""

    fit_means = average_sample_seeds(
        records, expected_sample_seeds=expected_sample_seeds
    )
    summary = summarize_fits(
        fit_means,
        directions=directions,
        confidence=confidence,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    if reference_version is None:
        paired = pd.DataFrame()
    else:
        paired = paired_version_deltas(
            fit_means,
            reference_version=reference_version,
            directions=directions,
            confidence=confidence,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
    return AggregationResult(fit_means, summary, paired)


# Concise aliases for protocol consumers.
aggregate_fit_levels = summarize_fits
aggregate_comparison = aggregate_version_comparison


__all__ = [
    "AggregationResult",
    "aggregate_comparison",
    "aggregate_fit_levels",
    "aggregate_version_comparison",
    "average_sample_seeds",
    "paired_version_deltas",
    "summarize_fits",
]

