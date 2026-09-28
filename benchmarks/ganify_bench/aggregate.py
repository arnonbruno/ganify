"""Seed-aware metric aggregation that never blends metric pillars."""

from __future__ import annotations

from typing import Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd


MetricRecords = Union[pd.DataFrame, Iterable[Mapping[str, object]]]


def _frame(records: MetricRecords) -> pd.DataFrame:
    values = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    if values.empty:
        raise ValueError("metric records must not be empty")
    if "metric" not in values.columns or "value" not in values.columns:
        raise ValueError("metric records require 'metric' and 'value' columns")
    values["value"] = pd.to_numeric(values["value"], errors="coerce")
    return values


def _bootstrap_median_interval(
    values: np.ndarray,
    *,
    confidence: float,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple:
    if len(values) == 1 or n_bootstrap == 0:
        value = float(np.median(values))
        return value, value
    medians = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        medians[index] = np.median(sample)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(medians, alpha)),
        float(np.quantile(medians, 1.0 - alpha)),
    )


def aggregate_runs(
    records: MetricRecords,
    *,
    group_columns: Optional[Sequence[str]] = None,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    random_state: int = 0,
) -> pd.DataFrame:
    """Aggregate repeated runs while retaining dataset/model/stage/metric keys.

    Confidence intervals use a deterministic bootstrap of the median. The
    function returns one row per metric and never computes a weighted overall
    quality score.
    """

    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be in (0, 1)")
    if n_bootstrap < 0:
        raise ValueError("n_bootstrap must be non-negative")
    frame = _frame(records)
    if group_columns is None:
        preferred = ["dataset_id", "model_name", "stage", "pillar", "metric", "column", "detail"]
        groups = [column for column in preferred if column in frame.columns]
    else:
        groups = list(group_columns)
    if "metric" not in groups:
        groups.append("metric")
    missing = [column for column in groups if column not in frame.columns]
    if missing:
        raise ValueError("unknown aggregation columns: %r" % missing)

    rng = np.random.default_rng(random_state)
    rows: List[dict] = []
    grouped = frame.groupby(groups, dropna=False, sort=True)
    for key, group in grouped:
        values = group["value"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        key_values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(groups, key_values))
        if len(values) == 0:
            row.update(
                {
                    "count": 0,
                    "mean": float("nan"),
                    "median": float("nan"),
                    "std": float("nan"),
                    "minimum": float("nan"),
                    "maximum": float("nan"),
                    "ci_lower": float("nan"),
                    "ci_upper": float("nan"),
                }
            )
        else:
            lower, upper = _bootstrap_median_interval(
                values,
                confidence=confidence,
                n_bootstrap=n_bootstrap,
                rng=rng,
            )
            row.update(
                {
                    "count": int(len(values)),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": (
                        float(np.std(values, ddof=1))
                        if len(values) > 1
                        else 0.0
                    ),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                    "ci_lower": lower,
                    "ci_upper": upper,
                }
            )
        rows.append(row)
    output = pd.DataFrame(rows)
    output.attrs["contains_blended_score"] = False
    output.attrs["confidence"] = confidence
    return output


__all__ = ["MetricRecords", "aggregate_runs"]
