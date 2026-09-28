"""Basic TSTR/TRTR downstream utility for classification and regression."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from ._utils import (
    aligned_frames,
    as_frame,
    infer_categorical_columns,
    make_preprocessor,
)


def _classification_metrics(
    model: Pipeline, features: pd.DataFrame, target: np.ndarray
) -> Dict[str, float]:
    prediction = model.predict(features)
    metrics = {
        "accuracy": float(accuracy_score(target, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "f1_macro": float(f1_score(target, prediction, average="macro")),
    }
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        classes = np.asarray(model.classes_)
        try:
            if probabilities.shape[1] == 2:
                positive = classes[1]
                binary_target = (target == positive).astype(int)
                metrics["roc_auc"] = float(
                    roc_auc_score(binary_target, probabilities[:, 1])
                )
            elif probabilities.shape[1] > 2:
                metrics["roc_auc_ovr_macro"] = float(
                    roc_auc_score(
                        target,
                        probabilities,
                        labels=classes,
                        multi_class="ovr",
                        average="macro",
                    )
                )
        except ValueError:
            # A held-out slice can omit a class. The core class-balanced metrics
            # remain valid and the unavailable AUC is made explicit.
            metrics["roc_auc"] = float("nan")
    return metrics


def _regression_metrics(
    model: Pipeline, features: pd.DataFrame, target: np.ndarray
) -> Dict[str, float]:
    prediction = np.asarray(model.predict(features), dtype=float)
    mse = float(mean_squared_error(target, prediction))
    return {
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(target, prediction)),
    }


def _model_pipeline(
    train_features: pd.DataFrame,
    estimator: BaseEstimator,
    categorical_columns: Optional[Iterable[object]],
) -> Pipeline:
    preprocessor, _, _ = make_preprocessor(train_features, categorical_columns)
    return Pipeline(
        [("preprocess", preprocessor), ("estimator", clone(estimator))]
    )


def _relative_score(
    trtr: Mapping[str, float],
    tstr: Mapping[str, float],
    dummy: Mapping[str, float],
    task: str,
) -> Dict[str, float]:
    output = {}
    for metric in sorted(set(trtr).intersection(tstr).intersection(dummy)):
        baseline = float(dummy[metric])
        real_score = float(trtr[metric])
        synthetic_score = float(tstr[metric])
        if not np.isfinite([baseline, real_score, synthetic_score]).all():
            output[metric] = float("nan")
            continue
        if task == "regression" and metric in {"mae", "rmse"}:
            denominator = baseline - real_score
            output[metric] = (
                (baseline - synthetic_score) / denominator
                if abs(denominator) > 1e-12
                else float("nan")
            )
        else:
            denominator = real_score - baseline
            output[metric] = (
                (synthetic_score - baseline) / denominator
                if abs(denominator) > 1e-12
                else float("nan")
            )
    return output


def evaluate_utility(
    real_train_features: object,
    real_train_target: object,
    real_test_features: object,
    real_test_target: object,
    synthetic_features: object,
    synthetic_target: object,
    *,
    task: str,
    categorical_columns: Optional[Iterable[object]] = None,
    estimator: Optional[BaseEstimator] = None,
    random_state: int = 0,
) -> Dict[str, object]:
    """Train on real and synthetic rows, then score the same real test set.

    The returned ``trtr`` and ``tstr`` sections remain separate. Relative skill
    is reported per downstream metric rather than as a blended quality score.
    Preprocessing is independently fitted inside each training pipeline and
    never uses the real test set.
    """

    real_train = as_frame(real_train_features, name="real_train_features")
    _, real_test = aligned_frames(real_train, real_test_features)
    _, synthetic = aligned_frames(real_train, synthetic_features)
    real_y = np.asarray(real_train_target)
    test_y = np.asarray(real_test_target)
    synthetic_y = np.asarray(synthetic_target)
    if real_y.ndim != 1 or len(real_y) != len(real_train):
        raise ValueError("real_train_target must be one-dimensional and row-aligned")
    if test_y.ndim != 1 or len(test_y) != len(real_test):
        raise ValueError("real_test_target must be one-dimensional and row-aligned")
    if synthetic_y.ndim != 1 or len(synthetic_y) != len(synthetic):
        raise ValueError("synthetic_target must be one-dimensional and row-aligned")
    normalized_task = task.lower()
    if normalized_task not in {"classification", "regression"}:
        raise ValueError("task must be 'classification' or 'regression'")

    resolved_categorical = infer_categorical_columns(
        real_train, categorical_columns
    )
    if estimator is None:
        if normalized_task == "classification":
            estimator = LogisticRegression(
                max_iter=500,
                solver="lbfgs",
                random_state=random_state,
            )
        else:
            estimator = Ridge(alpha=1.0)

    real_model = _model_pipeline(real_train, estimator, resolved_categorical)
    synthetic_model = _model_pipeline(
        synthetic, estimator, resolved_categorical
    )
    real_model.fit(real_train, real_y)
    synthetic_model.fit(synthetic, synthetic_y)

    if normalized_task == "classification":
        real_metrics = _classification_metrics(real_model, real_test, test_y)
        synthetic_metrics = _classification_metrics(
            synthetic_model, real_test, test_y
        )
        dummy_estimator: BaseEstimator = DummyClassifier(strategy="most_frequent")
    else:
        real_metrics = _regression_metrics(real_model, real_test, test_y)
        synthetic_metrics = _regression_metrics(synthetic_model, real_test, test_y)
        dummy_estimator = DummyRegressor(strategy="mean")

    dummy_model = _model_pipeline(
        real_train, dummy_estimator, resolved_categorical
    )
    dummy_model.fit(real_train, real_y)
    if normalized_task == "classification":
        dummy_metrics = _classification_metrics(dummy_model, real_test, test_y)
    else:
        dummy_metrics = _regression_metrics(dummy_model, real_test, test_y)

    return {
        "task": normalized_task,
        "trtr": real_metrics,
        "tstr": synthetic_metrics,
        "dummy": dummy_metrics,
        "relative_utility": _relative_score(
            real_metrics, synthetic_metrics, dummy_metrics, normalized_task
        ),
        "random_state": int(random_state),
    }


def classification_utility(
    real_train_features: object,
    real_train_target: object,
    real_test_features: object,
    real_test_target: object,
    synthetic_features: object,
    synthetic_target: object,
    **kwargs: object,
) -> Dict[str, object]:
    """Convenience wrapper for classification TSTR/TRTR."""

    return evaluate_utility(
        real_train_features,
        real_train_target,
        real_test_features,
        real_test_target,
        synthetic_features,
        synthetic_target,
        task="classification",
        **kwargs,
    )


def regression_utility(
    real_train_features: object,
    real_train_target: object,
    real_test_features: object,
    real_test_target: object,
    synthetic_features: object,
    synthetic_target: object,
    **kwargs: object,
) -> Dict[str, object]:
    """Convenience wrapper for regression TSTR/TRTR."""

    return evaluate_utility(
        real_train_features,
        real_train_target,
        real_test_features,
        real_test_target,
        synthetic_features,
        synthetic_target,
        task="regression",
        **kwargs,
    )


__all__ = [
    "classification_utility",
    "evaluate_utility",
    "regression_utility",
]
