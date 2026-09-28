"""Nearest-neighbor novelty, coverage, and authenticity diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors

from ._utils import aligned_frames, as_frame, dense_float, make_preprocessor


@dataclass(frozen=True)
class NearestNeighborMetrics:
    """Nearest-neighbor diagnostics in train-fitted encoded feature space."""

    novelty_mean: float
    novelty_median: float
    novelty_p05: float
    exact_match_rate: float
    authenticity: float
    coverage: float
    coverage_radius: float
    duplicate_rate: float

    def as_dict(self) -> Dict[str, float]:
        """Return a JSON-friendly metric mapping."""

        return asdict(self)


def _within_train_distances(train: np.ndarray) -> np.ndarray:
    if len(train) < 2:
        raise ValueError("nearest-neighbor metrics require at least two train rows")
    model = NearestNeighbors(n_neighbors=2, metric="euclidean")
    model.fit(train)
    distances, _ = model.kneighbors(train)
    return distances[:, 1]


def _duplicate_rate(values: np.ndarray, tolerance: float) -> float:
    if len(values) < 2:
        return 0.0
    model = NearestNeighbors(n_neighbors=2, metric="euclidean")
    model.fit(values)
    distances, _ = model.kneighbors(values)
    return float(np.mean(distances[:, 1] <= tolerance))


def nearest_neighbor_metrics(
    real_train: object,
    synthetic: object,
    *,
    real_holdout: Optional[object] = None,
    categorical_columns: Optional[Iterable[object]] = None,
    coverage_quantile: float = 0.95,
    exact_tolerance: float = 1e-12,
) -> NearestNeighborMetrics:
    """Evaluate novelty, held-out coverage, and sample authenticity.

    The mixed-type encoder and scale are fitted on real training rows only.
    Authenticity follows a conservative nearest-neighbor rule: a synthetic row
    is authentic when it is farther from its closest training row than that
    training row is from its own closest peer. Bootstrap copies therefore have
    authenticity zero and exact-match rate one.
    """

    if not (0.0 < coverage_quantile <= 1.0):
        raise ValueError("coverage_quantile must be in (0, 1]")
    if exact_tolerance < 0.0:
        raise ValueError("exact_tolerance must be non-negative")
    train_frame, synthetic_frame = aligned_frames(real_train, synthetic)
    holdout_frame = None
    if real_holdout is not None:
        _, holdout_frame = aligned_frames(train_frame, real_holdout)

    preprocessor, _, _ = make_preprocessor(train_frame, categorical_columns)
    train_encoded = dense_float(preprocessor.fit_transform(train_frame))
    synthetic_encoded = dense_float(preprocessor.transform(synthetic_frame))
    holdout_encoded = (
        dense_float(preprocessor.transform(holdout_frame))
        if holdout_frame is not None
        else None
    )

    train_peer_distance = _within_train_distances(train_encoded)
    train_index = NearestNeighbors(n_neighbors=1, metric="euclidean")
    train_index.fit(train_encoded)
    synthetic_distance, synthetic_neighbor = train_index.kneighbors(
        synthetic_encoded
    )
    synthetic_distance = synthetic_distance[:, 0]
    synthetic_neighbor = synthetic_neighbor[:, 0]
    exact_match_rate = float(np.mean(synthetic_distance <= exact_tolerance))
    authenticity = float(
        np.mean(
            synthetic_distance
            > train_peer_distance[synthetic_neighbor] + exact_tolerance
        )
    )
    radius = float(np.quantile(train_peer_distance, coverage_quantile))

    if holdout_encoded is None:
        coverage = float("nan")
    else:
        synthetic_index = NearestNeighbors(n_neighbors=1, metric="euclidean")
        synthetic_index.fit(synthetic_encoded)
        holdout_distance, _ = synthetic_index.kneighbors(holdout_encoded)
        coverage = float(np.mean(holdout_distance[:, 0] <= radius))

    return NearestNeighborMetrics(
        novelty_mean=float(np.mean(synthetic_distance)),
        novelty_median=float(np.median(synthetic_distance)),
        novelty_p05=float(np.quantile(synthetic_distance, 0.05)),
        exact_match_rate=exact_match_rate,
        authenticity=authenticity,
        coverage=coverage,
        coverage_radius=radius,
        duplicate_rate=_duplicate_rate(synthetic_encoded, exact_tolerance),
    )


__all__ = ["NearestNeighborMetrics", "nearest_neighbor_metrics"]
