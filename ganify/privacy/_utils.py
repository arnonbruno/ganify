"""Internal, dependency-light helpers for privacy audits."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Estimate:
    """A point estimate with an optional non-parametric bootstrap interval."""

    value: float
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    standard_error: Optional[float] = None
    confidence: float = 0.95
    bootstrap_replicates: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def as_frame(value: Any, *, name: str) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        frame = value.copy()
    else:
        frame = pd.DataFrame(value)
    if frame.columns.has_duplicates:
        raise ValueError("%s contains duplicate column names" % name)
    if len(frame) < 1 or len(frame.columns) < 1:
        raise ValueError("%s must contain rows and columns" % name)
    return frame.reset_index(drop=True)


def aligned_frames(
    reference: Any, value: Any, *, reference_name: str, value_name: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    left = as_frame(reference, name=reference_name)
    right = as_frame(value, name=value_name)
    missing = [column for column in left.columns if column not in right.columns]
    extra = [column for column in right.columns if column not in left.columns]
    if missing or extra:
        raise ValueError(
            "%s columns do not match %s; missing=%r, extra=%r"
            % (value_name, reference_name, missing, extra)
        )
    return left, right.loc[:, left.columns].reset_index(drop=True)


def infer_categorical(
    frame: pd.DataFrame,
    categorical_columns: Optional[Iterable[Any]] = None,
) -> Tuple[Any, ...]:
    if categorical_columns is not None:
        columns = tuple(categorical_columns)
        unknown = [column for column in columns if column not in frame.columns]
        if unknown:
            raise ValueError("unknown categorical columns: %r" % unknown)
        return columns
    return tuple(
        column
        for column in frame.columns
        if not (
            pd.api.types.is_numeric_dtype(frame[column].dtype)
            and not pd.api.types.is_bool_dtype(frame[column].dtype)
        )
    )


def _percentile(values: np.ndarray, probability: float) -> float:
    try:
        return float(np.quantile(values, probability, method="linear"))
    except TypeError:  # NumPy < 1.22
        return float(np.quantile(values, probability, interpolation="linear"))


def bootstrap_estimate(
    point: float,
    statistic: Callable[[np.random.Generator], float],
    *,
    replicates: int,
    confidence: float,
    random_state: int,
) -> Estimate:
    if isinstance(replicates, bool) or int(replicates) < 0:
        raise ValueError("bootstrap_replicates must be non-negative")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must be strictly between zero and one")
    replicates = int(replicates)
    if replicates == 0:
        return Estimate(float(point), confidence=float(confidence))
    rng = np.random.default_rng(int(random_state))
    values = np.asarray(
        [float(statistic(rng)) for _ in range(replicates)], dtype=np.float64
    )
    values = values[np.isfinite(values)]
    if not len(values):
        return Estimate(
            float(point),
            confidence=float(confidence),
            bootstrap_replicates=replicates,
        )
    alpha = (1.0 - float(confidence)) / 2.0
    return Estimate(
        value=float(point),
        ci_low=_percentile(values, alpha),
        ci_high=_percentile(values, 1.0 - alpha),
        standard_error=(
            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        ),
        confidence=float(confidence),
        bootstrap_replicates=replicates,
    )


def proportion_estimate(
    outcomes: Sequence[float],
    *,
    replicates: int,
    confidence: float,
    random_state: int,
) -> Estimate:
    values = np.asarray(outcomes, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("proportion outcomes must be a non-empty vector")
    point = float(np.mean(values))

    def statistic(rng: np.random.Generator) -> float:
        indices = rng.integers(0, len(values), size=len(values))
        return float(np.mean(values[indices]))

    return bootstrap_estimate(
        point,
        statistic,
        replicates=replicates,
        confidence=confidence,
        random_state=random_state,
    )


class MixedTypeDistance:
    """Gower-like mixed-type distance with robust numeric scaling.

    Scaling is fitted only on the supplied real audit reference. Synthetic
    values therefore cannot make an attack look weaker by inflating a scale.
    """

    def __init__(
        self,
        reference: pd.DataFrame,
        *,
        categorical_columns: Optional[Iterable[Any]] = None,
    ) -> None:
        self.columns = tuple(reference.columns)
        self.categorical = infer_categorical(reference, categorical_columns)
        categorical = set(self.categorical)
        self.numeric = tuple(
            column for column in self.columns if column not in categorical
        )
        self.scales: Dict[Any, float] = {}
        self.centers: Dict[Any, float] = {}
        for column in self.numeric:
            values = pd.to_numeric(
                reference[column], errors="coerce"
            ).to_numpy(dtype=np.float64)
            finite = values[np.isfinite(values)]
            if not len(finite):
                center, scale = 0.0, 1.0
            else:
                center = float(np.median(finite))
                q25 = _percentile(finite, 0.25)
                q75 = _percentile(finite, 0.75)
                scale = q75 - q25
                if not math.isfinite(scale) or scale <= 0.0:
                    scale = float(np.std(finite))
                if not math.isfinite(scale) or scale <= 0.0:
                    scale = float(np.ptp(finite))
                if not math.isfinite(scale) or scale <= 0.0:
                    scale = 1.0
            self.centers[column] = center
            self.scales[column] = scale

    def _validate(self, frame: pd.DataFrame, name: str) -> pd.DataFrame:
        missing = [column for column in self.columns if column not in frame]
        extra = [column for column in frame if column not in self.columns]
        if missing or extra:
            raise ValueError(
                "%s columns do not match distance reference; missing=%r, "
                "extra=%r" % (name, missing, extra)
            )
        return frame.loc[:, self.columns].reset_index(drop=True)

    def pairwise(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        *,
        chunk_size: int = 512,
    ) -> np.ndarray:
        left = self._validate(left, "left")
        right = self._validate(right, "right")
        if isinstance(chunk_size, bool) or int(chunk_size) < 1:
            raise ValueError("chunk_size must be positive")
        output = np.empty((len(left), len(right)), dtype=np.float64)
        width = max(1, len(self.columns))
        for start in range(0, len(left), int(chunk_size)):
            stop = min(start + int(chunk_size), len(left))
            total = np.zeros((stop - start, len(right)), dtype=np.float64)
            for column in self.numeric:
                left_raw = pd.to_numeric(
                    left[column].iloc[start:stop], errors="coerce"
                ).to_numpy(dtype=np.float64)
                right_raw = pd.to_numeric(
                    right[column], errors="coerce"
                ).to_numpy(dtype=np.float64)
                left_missing = ~np.isfinite(left_raw)
                right_missing = ~np.isfinite(right_raw)
                left_value = np.where(
                    left_missing, self.centers[column], left_raw
                )
                right_value = np.where(
                    right_missing, self.centers[column], right_raw
                )
                difference = (
                    left_value[:, None] - right_value[None, :]
                ) / self.scales[column]
                contribution = np.minimum(np.square(difference), 100.0)
                both_missing = left_missing[:, None] & right_missing[None, :]
                one_missing = left_missing[:, None] ^ right_missing[None, :]
                contribution[both_missing] = 0.0
                contribution[one_missing] = 1.0
                total += contribution
            for column in self.categorical:
                left_raw = left[column].iloc[start:stop].to_numpy(dtype=object)
                right_raw = right[column].to_numpy(dtype=object)
                lookup: Dict[Tuple[str, str], int] = {}

                def code(raw: Any) -> int:
                    key = value_key(raw)
                    if key not in lookup:
                        lookup[key] = len(lookup)
                    return lookup[key]

                left_codes = np.asarray(
                    [code(raw) for raw in left_raw], dtype=np.int64
                )
                right_codes = np.asarray(
                    [code(raw) for raw in right_raw], dtype=np.int64
                )
                equal = left_codes[:, None] == right_codes[None, :]
                total += (~equal).astype(np.float64)
            output[start:stop] = np.sqrt(total / float(width))
        return output


def value_key(value: Any) -> Tuple[str, str]:
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return ("missing", "")
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and value == 0.0:
        value = 0.0
    return (type(value).__name__, repr(value))


def row_keys(frame: pd.DataFrame) -> Tuple[Tuple[Tuple[str, str], ...], ...]:
    return tuple(
        tuple(value_key(value) for value in row)
        for row in frame.itertuples(index=False, name=None)
    )


__all__ = [
    "Estimate",
    "MixedTypeDistance",
    "aligned_frames",
    "as_frame",
    "bootstrap_estimate",
    "infer_categorical",
    "proportion_estimate",
    "row_keys",
]
