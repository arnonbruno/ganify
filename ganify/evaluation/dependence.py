"""Dependence metrics that cannot be improved by marginal rank remapping."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from ._utils import aligned_frames


@dataclass(frozen=True)
class DependenceMatrixErrors:
    """Pairwise matrix errors; all reported errors are lower-is-better."""

    pearson_mae: float
    pearson_rmse: float
    pearson_max_error: float
    spearman_mae: float
    spearman_rmse: float
    spearman_max_error: float
    n_pairs: int

    def as_dict(self) -> Dict[str, float]:
        """Return a JSON-friendly metric mapping."""

        values = asdict(self)
        return {key: float(value) for key, value in values.items()}


@dataclass(frozen=True)
class NormalizedDependenceGap:
    """Dependence loss normalized between real and independent controls."""

    metric: str
    candidate_loss: float
    real_floor_loss: float
    independent_loss: float
    normalized_gap: float

    def as_dict(self) -> Dict[str, object]:
        """Return a JSON-friendly metric mapping."""

        return asdict(self)


def _numeric_pair(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: Optional[Iterable[object]],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[object]]:
    selected = list(columns) if columns is not None else [
        column
        for column in real.columns
        if pd.api.types.is_numeric_dtype(real[column].dtype)
    ]
    if not selected:
        raise ValueError("dependence metrics require at least two numeric columns")
    unknown = [column for column in selected if column not in real.columns]
    if unknown:
        raise ValueError("unknown dependence columns: %r" % unknown)
    if len(selected) < 2:
        raise ValueError("dependence metrics require at least two numeric columns")

    output = []
    for frame, name in ((real, "real"), (synthetic, "synthetic")):
        numeric = frame.loc[:, selected].apply(pd.to_numeric, errors="coerce")
        if bool(numeric.isna().any().any()):
            # Pairwise deletion can produce incomparable matrices. Median
            # imputation is deterministic and is performed within each table.
            numeric = numeric.fillna(numeric.median(axis=0))
        if bool(numeric.isna().any().any()):
            raise ValueError("%s dependence columns contain no finite values" % name)
        values = numeric.to_numpy(dtype=float)
        if not bool(np.isfinite(values).all()):
            raise ValueError("%s dependence columns contain non-finite values" % name)
        output.append(numeric)
    return output[0], output[1], selected


def rank_transform(frame: pd.DataFrame) -> pd.DataFrame:
    """Map every column to deterministic fractional mid-ranks in ``(0, 1)``."""

    n_rows = len(frame)
    if n_rows == 0:
        raise ValueError("rank_transform requires at least one row")
    # ``rank(pct=True)`` uses rank / n and places the largest value at 1.
    # Mid-ranks below avoid exact endpoints and are stable under monotone maps.
    ranks = frame.rank(axis=0, method="average", na_option="keep")
    return (ranks - 0.5) / float(n_rows)


def _safe_corr(frame: pd.DataFrame, method: str) -> np.ndarray:
    matrix = frame.corr(method=method).to_numpy(dtype=float)
    # A constant column has undefined correlation. Treating its off-diagonal
    # entries as zero compares the absence of measured dependence consistently.
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def _matrix_errors(left: np.ndarray, right: np.ndarray) -> Tuple[float, float, float]:
    mask = np.triu(np.ones(left.shape, dtype=bool), k=1)
    errors = np.abs(left[mask] - right[mask])
    if len(errors) == 0:
        raise ValueError("dependence metrics require at least one column pair")
    return (
        float(np.mean(errors)),
        float(np.sqrt(np.mean(np.square(errors)))),
        float(np.max(errors)),
    )


def dependence_matrices(
    data: object,
    *,
    columns: Optional[Iterable[object]] = None,
    rank_space: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Return Pearson and Spearman matrices with stable column labels.

    ``rank_space=True`` is the anti-gaming default: Pearson is calculated
    after independent empirical-rank transforms. Spearman is also reported
    explicitly for auditability, even though it is mathematically equivalent
    to Pearson-on-ranks in the absence of tie-handling differences.
    """

    frame = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    if len(frame) == 0:
        raise ValueError("data must contain at least one row")
    selected = list(columns) if columns is not None else [
        column
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column].dtype)
    ]
    if len(selected) < 2:
        raise ValueError("dependence matrices require at least two numeric columns")
    numeric = frame.loc[:, selected].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.fillna(numeric.median(axis=0))
    if bool(numeric.isna().any().any()):
        raise ValueError("dependence columns contain no finite values")
    pearson_input = rank_transform(numeric) if rank_space else numeric
    pearson = pd.DataFrame(
        _safe_corr(pearson_input, "pearson"), index=selected, columns=selected
    )
    spearman = pd.DataFrame(
        _safe_corr(numeric, "spearman"), index=selected, columns=selected
    )
    return {"pearson": pearson, "spearman": spearman}


def dependence_matrix_errors(
    real: object,
    synthetic: object,
    *,
    columns: Optional[Iterable[object]] = None,
    rank_space: bool = True,
) -> DependenceMatrixErrors:
    """Compare real and synthetic Pearson/Spearman dependence matrices."""

    real_frame, synthetic_frame = aligned_frames(real, synthetic)
    real_numeric, synthetic_numeric, selected = _numeric_pair(
        real_frame, synthetic_frame, columns
    )
    real_matrices = dependence_matrices(
        real_numeric, columns=selected, rank_space=rank_space
    )
    synthetic_matrices = dependence_matrices(
        synthetic_numeric, columns=selected, rank_space=rank_space
    )
    pearson = _matrix_errors(
        real_matrices["pearson"].to_numpy(),
        synthetic_matrices["pearson"].to_numpy(),
    )
    spearman = _matrix_errors(
        real_matrices["spearman"].to_numpy(),
        synthetic_matrices["spearman"].to_numpy(),
    )
    n_pairs = len(selected) * (len(selected) - 1) // 2
    return DependenceMatrixErrors(
        pearson_mae=pearson[0],
        pearson_rmse=pearson[1],
        pearson_max_error=pearson[2],
        spearman_mae=spearman[0],
        spearman_rmse=spearman[1],
        spearman_max_error=spearman[2],
        n_pairs=n_pairs,
    )


def rank_dependence_metrics(
    real: object,
    synthetic: object,
    *,
    columns: Optional[Iterable[object]] = None,
) -> Dict[str, float]:
    """Return rank-space matrix errors as a plain mapping."""

    return dependence_matrix_errors(
        real, synthetic, columns=columns, rank_space=True
    ).as_dict()


def normalized_dependence_gap(
    real_train: object,
    real_test: object,
    synthetic: object,
    *,
    columns: Optional[Iterable[object]] = None,
    metric: str = "spearman_mae",
    random_state: int = 0,
) -> NormalizedDependenceGap:
    """Normalize candidate loss between real-sampling and independence controls.

    A value near zero is the real-train versus real-test floor; one is the
    independently permuted-column negative control. Values are not clipped.
    """

    from .controls import independently_permuted_columns

    train_frame, candidate = aligned_frames(real_train, synthetic)
    _, test_frame = aligned_frames(train_frame, real_test)
    independent = independently_permuted_columns(
        train_frame, random_state=random_state
    )
    candidate_errors = dependence_matrix_errors(
        test_frame, candidate, columns=columns, rank_space=True
    ).as_dict()
    floor_errors = dependence_matrix_errors(
        test_frame, train_frame, columns=columns, rank_space=True
    ).as_dict()
    independent_errors = dependence_matrix_errors(
        test_frame, independent, columns=columns, rank_space=True
    ).as_dict()
    if metric not in candidate_errors or metric == "n_pairs":
        available = sorted(key for key in candidate_errors if key != "n_pairs")
        raise ValueError(
            "metric must be one of %s" % ", ".join(available)
        )
    candidate_loss = float(candidate_errors[metric])
    floor_loss = float(floor_errors[metric])
    independent_loss = float(independent_errors[metric])
    denominator = independent_loss - floor_loss
    if denominator <= 1e-12:
        raise ValueError(
            "independent control does not separate from the real-data floor"
        )
    return NormalizedDependenceGap(
        metric=metric,
        candidate_loss=candidate_loss,
        real_floor_loss=floor_loss,
        independent_loss=independent_loss,
        normalized_gap=(candidate_loss - floor_loss) / denominator,
    )


__all__ = [
    "DependenceMatrixErrors",
    "NormalizedDependenceGap",
    "dependence_matrices",
    "dependence_matrix_errors",
    "normalized_dependence_gap",
    "rank_dependence_metrics",
    "rank_transform",
]
