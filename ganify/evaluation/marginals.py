"""Univariate fidelity metrics for numeric and categorical columns."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ._utils import aligned_frames, finite_numeric, infer_categorical_columns


@dataclass(frozen=True)
class NumericMarginalMetrics:
    """Numeric marginal errors; every error is lower-is-better."""

    ks_statistic: float
    wasserstein_distance: float
    normalized_wasserstein: float
    zero_rate_real: float
    zero_rate_synthetic: float
    zero_rate_error: float
    lower_tail_mass_error: float
    upper_tail_mass_error: float
    tail_mass_error: float
    missing_rate_real: float
    missing_rate_synthetic: float
    missing_rate_error: float

    def as_dict(self) -> Dict[str, float]:
        """Return a JSON-friendly metric mapping."""

        return asdict(self)


@dataclass(frozen=True)
class CategoricalMarginalMetrics:
    """Categorical marginal errors and support diagnostics."""

    total_variation: float
    support_recall: float
    unseen_synthetic_rate: float
    missing_rate_real: float
    missing_rate_synthetic: float
    missing_rate_error: float

    def as_dict(self) -> Dict[str, float]:
        """Return a JSON-friendly metric mapping."""

        return asdict(self)


def _empirical_ks(left: np.ndarray, right: np.ndarray) -> float:
    """Compute the exact two-sample KS statistic without p-value gaming."""

    points = np.unique(np.concatenate([left, right]))
    left_cdf = np.searchsorted(np.sort(left), points, side="right") / float(len(left))
    right_cdf = np.searchsorted(np.sort(right), points, side="right") / float(
        len(right)
    )
    return float(np.max(np.abs(left_cdf - right_cdf)))


def _wasserstein_1d(left: np.ndarray, right: np.ndarray) -> float:
    """Compute one-dimensional empirical Wasserstein-1 distance exactly."""

    left = np.sort(left)
    right = np.sort(right)
    points = np.sort(np.concatenate([left, right]))
    if len(points) < 2:
        return 0.0
    deltas = np.diff(points)
    left_cdf = np.searchsorted(left, points[:-1], side="right") / float(len(left))
    right_cdf = np.searchsorted(right, points[:-1], side="right") / float(len(right))
    return float(np.sum(np.abs(left_cdf - right_cdf) * deltas))


def _robust_scale(values: np.ndarray) -> float:
    """Choose a stable real-data scale for normalized Wasserstein distance."""

    q25, q75 = np.quantile(values, [0.25, 0.75])
    scale = float(q75 - q25)
    if scale <= 0.0:
        scale = float(np.std(values))
    if scale <= 0.0:
        scale = float(np.max(values) - np.min(values))
    return scale if scale > 0.0 else 1.0


def numeric_marginal_metrics(
    real: object,
    synthetic: object,
    *,
    tail_quantiles: Tuple[float, float] = (0.01, 0.99),
    zero_tolerance: float = 0.0,
) -> NumericMarginalMetrics:
    """Evaluate one numeric marginal.

    Tail thresholds and normalization are estimated only from ``real``.
    Missingness is reported separately and omitted from distributional
    calculations.
    """

    lower_q, upper_q = tail_quantiles
    if not (0.0 < lower_q < upper_q < 1.0):
        raise ValueError("tail_quantiles must satisfy 0 < lower < upper < 1")
    if zero_tolerance < 0:
        raise ValueError("zero_tolerance must be non-negative")

    real_series = pd.to_numeric(pd.Series(real), errors="coerce")
    synthetic_series = pd.to_numeric(pd.Series(synthetic), errors="coerce")
    real_missing = float(real_series.isna().mean())
    synthetic_missing = float(synthetic_series.isna().mean())
    real_values = finite_numeric(real_series, name="real")
    synthetic_values = finite_numeric(synthetic_series, name="synthetic")
    if len(real_values) == 0 or len(synthetic_values) == 0:
        raise ValueError("numeric marginals need at least one finite value per table")

    lower, upper = np.quantile(real_values, [lower_q, upper_q])
    lower_real = float(np.mean(real_values <= lower))
    lower_synthetic = float(np.mean(synthetic_values <= lower))
    upper_real = float(np.mean(real_values >= upper))
    upper_synthetic = float(np.mean(synthetic_values >= upper))
    lower_error = abs(lower_synthetic - lower_real)
    upper_error = abs(upper_synthetic - upper_real)
    real_zero = float(np.mean(np.abs(real_values) <= zero_tolerance))
    synthetic_zero = float(np.mean(np.abs(synthetic_values) <= zero_tolerance))
    wasserstein = _wasserstein_1d(real_values, synthetic_values)

    return NumericMarginalMetrics(
        ks_statistic=_empirical_ks(real_values, synthetic_values),
        wasserstein_distance=wasserstein,
        normalized_wasserstein=wasserstein / _robust_scale(real_values),
        zero_rate_real=real_zero,
        zero_rate_synthetic=synthetic_zero,
        zero_rate_error=abs(synthetic_zero - real_zero),
        lower_tail_mass_error=lower_error,
        upper_tail_mass_error=upper_error,
        tail_mass_error=(lower_error + upper_error) / 2.0,
        missing_rate_real=real_missing,
        missing_rate_synthetic=synthetic_missing,
        missing_rate_error=abs(synthetic_missing - real_missing),
    )


def _category_key(value: object) -> object:
    """Use one collision-resistant marker for missing category values."""

    return ("__ganify_missing__",) if pd.isna(value) else (type(value).__name__, value)


def categorical_marginal_metrics(
    real: object, synthetic: object
) -> CategoricalMarginalMetrics:
    """Evaluate one categorical marginal with total-variation distance."""

    real_series = pd.Series(real)
    synthetic_series = pd.Series(synthetic)
    if len(real_series) == 0 or len(synthetic_series) == 0:
        raise ValueError("categorical marginals need at least one row per table")
    real_keys = real_series.map(_category_key)
    synthetic_keys = synthetic_series.map(_category_key)
    real_prob = real_keys.value_counts(dropna=False, normalize=True)
    synthetic_prob = synthetic_keys.value_counts(dropna=False, normalize=True)
    support = real_prob.index.union(synthetic_prob.index)
    total_variation = 0.5 * float(
        np.abs(
            real_prob.reindex(support, fill_value=0.0)
            - synthetic_prob.reindex(support, fill_value=0.0)
        ).sum()
    )
    real_support = set(real_prob.index)
    synthetic_support = set(synthetic_prob.index)
    support_recall = (
        len(real_support.intersection(synthetic_support)) / float(len(real_support))
        if real_support
        else 1.0
    )
    unseen = float(synthetic_keys.map(lambda item: item not in real_support).mean())
    real_missing = float(real_series.isna().mean())
    synthetic_missing = float(synthetic_series.isna().mean())
    return CategoricalMarginalMetrics(
        total_variation=total_variation,
        support_recall=support_recall,
        unseen_synthetic_rate=unseen,
        missing_rate_real=real_missing,
        missing_rate_synthetic=synthetic_missing,
        missing_rate_error=abs(synthetic_missing - real_missing),
    )


def categorical_total_variation(real: object, synthetic: object) -> float:
    """Return categorical total-variation distance in ``[0, 1]``."""

    return categorical_marginal_metrics(real, synthetic).total_variation


def marginal_metrics(
    real: object,
    synthetic: object,
    *,
    categorical_columns: Optional[Iterable[object]] = None,
    tail_quantiles: Tuple[float, float] = (0.01, 0.99),
    zero_tolerance: float = 0.0,
) -> pd.DataFrame:
    """Return one typed metric row per column.

    The output is deliberately not reduced to a scalar. Numeric and
    categorical columns retain their native diagnostics so one excellent
    marginal cannot hide a failed column or a failed dependence metric.
    """

    real_frame, synthetic_frame = aligned_frames(real, synthetic)
    categorical = set(
        infer_categorical_columns(real_frame, categorical_columns)
    )
    rows = []
    for column in real_frame.columns:
        if column in categorical:
            values = categorical_marginal_metrics(
                real_frame[column], synthetic_frame[column]
            ).as_dict()
            kind = "categorical"
        else:
            values = numeric_marginal_metrics(
                real_frame[column],
                synthetic_frame[column],
                tail_quantiles=tail_quantiles,
                zero_tolerance=zero_tolerance,
            ).as_dict()
            kind = "numeric"
        rows.append({"column": column, "kind": kind, **values})
    return pd.DataFrame(rows)


__all__ = [
    "CategoricalMarginalMetrics",
    "NumericMarginalMetrics",
    "categorical_marginal_metrics",
    "categorical_total_variation",
    "marginal_metrics",
    "numeric_marginal_metrics",
]
