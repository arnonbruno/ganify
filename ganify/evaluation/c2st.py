"""Classifier two-sample tests (C2ST) for real versus synthetic tables."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from ._utils import aligned_frames, make_preprocessor


@dataclass(frozen=True)
class C2STResult:
    """Cross-fitted C2ST result.

    An AUC near 0.5 means the chosen classifier cannot distinguish the tables;
    an AUC near 1.0 means it can. ``auc`` is orientation-corrected so sampling
    noise below 0.5 cannot misleadingly look better than chance.
    """

    auc: float
    raw_auc: float
    fold_aucs: Tuple[float, ...]
    n_real: int
    n_synthetic: int
    random_state: int

    def as_dict(self) -> Dict[str, object]:
        """Return a JSON-friendly result mapping."""

        return asdict(self)


def classifier_two_sample_test(
    real: object,
    synthetic: object,
    *,
    categorical_columns: Optional[Iterable[object]] = None,
    random_state: int = 0,
    folds: int = 5,
    max_samples_per_class: Optional[int] = 10000,
) -> C2STResult:
    """Run a deterministic, balanced logistic C2ST with out-of-fold scores."""

    real_frame, synthetic_frame = aligned_frames(real, synthetic)
    if folds < 2:
        raise ValueError("folds must be at least 2")
    rng = np.random.default_rng(random_state)
    n_per_class = min(len(real_frame), len(synthetic_frame))
    if max_samples_per_class is not None:
        if max_samples_per_class < 2:
            raise ValueError("max_samples_per_class must be at least 2")
        n_per_class = min(n_per_class, int(max_samples_per_class))
    if n_per_class < folds:
        raise ValueError("each table must have at least as many rows as folds")

    real_indices = rng.choice(len(real_frame), size=n_per_class, replace=False)
    synthetic_indices = rng.choice(
        len(synthetic_frame), size=n_per_class, replace=False
    )
    combined = real_frame.iloc[real_indices].copy()
    combined = combined.reset_index(drop=True)
    synthetic_sample = synthetic_frame.iloc[synthetic_indices].reset_index(drop=True)
    combined = pd.concat([combined, synthetic_sample], ignore_index=True)
    labels = np.concatenate(
        [np.zeros(n_per_class, dtype=int), np.ones(n_per_class, dtype=int)]
    )

    splitter = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=random_state
    )
    predictions = np.empty(len(labels), dtype=float)
    fold_scores = []
    for train_index, test_index in splitter.split(combined, labels):
        preprocessor, _, _ = make_preprocessor(
            combined.iloc[train_index], categorical_columns
        )
        classifier = Pipeline(
            [
                ("preprocess", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        max_iter=500,
                        solver="liblinear",
                        random_state=random_state,
                    ),
                ),
            ]
        )
        classifier.fit(combined.iloc[train_index], labels[train_index])
        scores = classifier.predict_proba(combined.iloc[test_index])[:, 1]
        predictions[test_index] = scores
        fold_scores.append(float(roc_auc_score(labels[test_index], scores)))

    raw_auc = float(roc_auc_score(labels, predictions))
    auc = max(raw_auc, 1.0 - raw_auc)
    return C2STResult(
        auc=auc,
        raw_auc=raw_auc,
        fold_aucs=tuple(fold_scores),
        n_real=n_per_class,
        n_synthetic=n_per_class,
        random_state=int(random_state),
    )


def c2st_auc(
    real: object,
    synthetic: object,
    *,
    categorical_columns: Optional[Iterable[object]] = None,
    random_state: int = 0,
    folds: int = 5,
    max_samples_per_class: Optional[int] = 10000,
) -> float:
    """Return only the orientation-corrected C2ST AUC."""

    return classifier_two_sample_test(
        real,
        synthetic,
        categorical_columns=categorical_columns,
        random_state=random_state,
        folds=folds,
        max_samples_per_class=max_samples_per_class,
    ).auc


__all__ = ["C2STResult", "c2st_auc", "classifier_two_sample_test"]
