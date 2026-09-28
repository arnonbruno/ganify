"""Attack-based privacy diagnostics for synthetic tabular data.

These attacks are empirical red-team tests, not proofs of privacy. In
particular, distance-to-closest-record (DCR) and nearest-neighbour distance
ratios (NNDR) are reported only as proximity diagnostics.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from ._utils import (
    Estimate,
    MixedTypeDistance,
    aligned_frames,
    as_frame,
    bootstrap_estimate,
    infer_categorical,
    proportion_estimate,
    row_keys,
    value_key,
)


PROXIMITY_LIMITATION = (
    "DCR and NNDR are proximity diagnostics, not privacy metrics or privacy "
    "guarantees. Their values depend on scaling, feature choice, population "
    "density, and the real-data comparison set."
)


@dataclass(frozen=True)
class ExactMatchAudit:
    exact_match_rate: Estimate
    holdout_exact_match_rate: Estimate
    train_only_exact_match_rate: Estimate
    collision_adjusted_exact_match_rate: float
    collision_adjusted_exact_match_rate_ci_high: Optional[float]
    unique_member_extraction_rate: float
    synthetic_rows: int
    exact_match_rows: int
    train_only_match_rows: int
    limitations: Tuple[str, ...]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProximityAudit:
    dcr_median: float
    dcr_p05: float
    dcr_mean: float
    nndr_median: Optional[float]
    nndr_p05: Optional[float]
    member_dcr_median: float
    nonmember_dcr_median: float
    zero_dcr_rate: float
    limitations: Tuple[str, ...]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MembershipInferenceAudit:
    method: str
    k: int
    roc_auc: Estimate
    tpr_at_1pct_fpr: Estimate
    member_score_mean: float
    nonmember_score_mean: float
    member_count: int
    nonmember_count: int
    limitations: Tuple[str, ...]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AttributeInferenceAudit:
    sensitive_column: Any
    kind: str
    metric: str
    attack_performance: Estimate
    baseline_performance: float
    advantage: Estimate
    evaluated_rows: int
    quasi_identifier_columns: Tuple[Any, ...]
    limitations: Tuple[str, ...]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SinglingOutAudit:
    attack_success_rate: Estimate
    control_success_rate: Estimate
    excess_success_rate: Estimate
    attacks: int
    predicate_size: int
    tolerance: float
    limitations: Tuple[str, ...]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LinkabilityAudit:
    attack_success_rate: Estimate
    permutation_control_rate: Estimate
    excess_success_rate: Estimate
    evaluated_rows: int
    auxiliary_columns_a: Tuple[Any, ...]
    auxiliary_columns_b: Tuple[Any, ...]
    limitations: Tuple[str, ...]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CanaryResult:
    index: int
    extracted: bool
    exact_count: int
    nearest_distance: float
    rank: float
    candidate_count: int
    exposure_bits: float

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CanaryAudit:
    extraction_rate: Estimate
    extracted_count: int
    canary_count: int
    maximum_exposure_bits: float
    mean_exposure_bits: float
    records: Tuple[CanaryResult, ...]
    limitations: Tuple[str, ...]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _difference_estimate(
    left: np.ndarray,
    right: np.ndarray,
    *,
    replicates: int,
    confidence: float,
    random_state: int,
) -> Estimate:
    if len(left) != len(right) or not len(left):
        raise ValueError("paired outcomes must have the same non-zero length")
    point = float(np.mean(left) - np.mean(right))

    def statistic(rng: np.random.Generator) -> float:
        indices = rng.integers(0, len(left), size=len(left))
        return float(np.mean(left[indices]) - np.mean(right[indices]))

    return bootstrap_estimate(
        point,
        statistic,
        replicates=replicates,
        confidence=confidence,
        random_state=random_state,
    )


def exact_match_audit(
    real_train: Any,
    real_holdout: Any,
    synthetic: Any,
    *,
    bootstrap_replicates: int = 200,
    confidence: float = 0.95,
    random_state: int = 0,
) -> ExactMatchAudit:
    """Measure exact extraction while adjusting for population collisions."""

    train, holdout = aligned_frames(
        real_train,
        real_holdout,
        reference_name="real_train",
        value_name="real_holdout",
    )
    _, generated = aligned_frames(
        train,
        synthetic,
        reference_name="real_train",
        value_name="synthetic",
    )
    train_keys = Counter(row_keys(train))
    holdout_keys = Counter(row_keys(holdout))
    generated_keys = row_keys(generated)
    train_match = np.asarray(
        [key in train_keys for key in generated_keys], dtype=np.float64
    )
    holdout_match = np.asarray(
        [key in holdout_keys for key in generated_keys], dtype=np.float64
    )
    train_only = train_match * (1.0 - holdout_match)
    exact = proportion_estimate(
        train_match,
        replicates=bootstrap_replicates,
        confidence=confidence,
        random_state=random_state + 11,
    )
    holdout_exact = proportion_estimate(
        holdout_match,
        replicates=bootstrap_replicates,
        confidence=confidence,
        random_state=random_state + 13,
    )
    train_only_exact = proportion_estimate(
        train_only,
        replicates=bootstrap_replicates,
        confidence=confidence,
        random_state=random_state + 17,
    )
    denominator = max(1.0 - holdout_exact.value, np.finfo(float).eps)
    collision_adjusted = max(
        0.0, (exact.value - holdout_exact.value) / denominator
    )

    def adjusted_statistic(rng: np.random.Generator) -> float:
        indices = rng.integers(
            0, len(train_match), size=len(train_match)
        )
        train_rate = float(np.mean(train_match[indices]))
        holdout_rate = float(np.mean(holdout_match[indices]))
        return max(
            0.0,
            (train_rate - holdout_rate)
            / max(1.0 - holdout_rate, np.finfo(float).eps),
        )

    adjusted = bootstrap_estimate(
        collision_adjusted,
        adjusted_statistic,
        replicates=bootstrap_replicates,
        confidence=confidence,
        random_state=random_state + 19,
    )
    matched_unique = {
        key
        for key in generated_keys
        if key in train_keys and key not in holdout_keys
    }
    unique_member_rate = float(len(matched_unique)) / float(
        max(1, len(train_keys))
    )
    return ExactMatchAudit(
        exact_match_rate=exact,
        holdout_exact_match_rate=holdout_exact,
        train_only_exact_match_rate=train_only_exact,
        collision_adjusted_exact_match_rate=float(
            min(1.0, collision_adjusted)
        ),
        collision_adjusted_exact_match_rate_ci_high=(
            None
            if adjusted.ci_high is None
            else float(min(1.0, adjusted.ci_high))
        ),
        unique_member_extraction_rate=unique_member_rate,
        synthetic_rows=len(generated),
        exact_match_rows=int(np.sum(train_match)),
        train_only_match_rows=int(np.sum(train_only)),
        limitations=(
            "Exact matching misses approximate memorization and can overstate "
            "risk on naturally duplicated or low-cardinality records.",
            "The collision adjustment uses the supplied holdout as a population "
            "collision control; a non-representative holdout weakens it.",
        ),
    )


def dcr_nndr_audit(
    real_train: Any,
    real_holdout: Any,
    synthetic: Any,
    *,
    categorical_columns: Optional[Iterable[Any]] = None,
) -> ProximityAudit:
    """Return DCR/NNDR proximity summaries without labelling them privacy."""

    train, holdout = aligned_frames(
        real_train,
        real_holdout,
        reference_name="real_train",
        value_name="real_holdout",
    )
    _, generated = aligned_frames(
        train,
        synthetic,
        reference_name="real_train",
        value_name="synthetic",
    )
    reference = pd.concat([train, holdout], ignore_index=True)
    distance = MixedTypeDistance(
        reference, categorical_columns=categorical_columns
    )
    generated_to_train = distance.pairwise(generated, train)
    ordered = np.sort(generated_to_train, axis=1)
    dcr = ordered[:, 0]
    nndr: Optional[np.ndarray]
    if len(train) >= 2:
        nndr = dcr / np.maximum(ordered[:, 1], np.finfo(float).eps)
    else:
        nndr = None
    member_dcr = np.min(distance.pairwise(train, generated), axis=1)
    nonmember_dcr = np.min(distance.pairwise(holdout, generated), axis=1)
    return ProximityAudit(
        dcr_median=float(np.median(dcr)),
        dcr_p05=float(np.quantile(dcr, 0.05)),
        dcr_mean=float(np.mean(dcr)),
        nndr_median=(
            None if nndr is None else float(np.median(nndr))
        ),
        nndr_p05=None if nndr is None else float(np.quantile(nndr, 0.05)),
        member_dcr_median=float(np.median(member_dcr)),
        nonmember_dcr_median=float(np.median(nonmember_dcr)),
        zero_dcr_rate=float(np.mean(dcr <= 1e-12)),
        limitations=(PROXIMITY_LIMITATION,),
    )


proximity_audit = dcr_nndr_audit


def _tpr_at_fpr(
    labels: np.ndarray, scores: np.ndarray, maximum_fpr: float = 0.01
) -> float:
    fpr, tpr, _ = roc_curve(
        labels, scores, pos_label=1, drop_intermediate=False
    )
    allowed = tpr[fpr <= maximum_fpr + 1e-15]
    return 0.0 if not len(allowed) else float(np.max(allowed))


def _membership_scores(
    member: pd.DataFrame,
    nonmember: pd.DataFrame,
    synthetic: pd.DataFrame,
    distance: MixedTypeDistance,
    *,
    method: str,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    candidates = pd.concat([member, nonmember], ignore_index=True)
    generated_distance = np.sort(
        distance.pairwise(candidates, synthetic), axis=1
    )
    selected_k = min(k, generated_distance.shape[1])
    if method == "knn":
        scores = -np.mean(generated_distance[:, :selected_k], axis=1)
    else:
        background_distance = distance.pairwise(candidates, candidates)
        np.fill_diagonal(background_distance, np.inf)
        background_distance.sort(axis=1)
        background_k = min(k, max(1, len(candidates) - 1))
        generated_radius = np.mean(
            generated_distance[:, :selected_k], axis=1
        )
        background_radius = np.mean(
            background_distance[:, :background_k], axis=1
        )
        epsilon = np.finfo(np.float64).eps
        effective_dimension = max(1, len(candidates.columns))
        scores = effective_dimension * np.log(
            (background_radius + epsilon) / (generated_radius + epsilon)
        )
    return scores[: len(member)], scores[len(member) :]


def membership_inference_attack(
    members: Any,
    nonmembers: Any,
    synthetic: Any,
    *,
    method: str = "knn",
    k: int = 1,
    categorical_columns: Optional[Iterable[Any]] = None,
    bootstrap_replicates: int = 200,
    confidence: float = 0.95,
    random_state: int = 0,
) -> MembershipInferenceAudit:
    """Run a sample-based membership inference attack.

    ``method="knn"`` uses closeness to generated records. ``"density_ratio"``
    compares synthetic kNN density with the pooled audit-population density.
    """

    member, nonmember = aligned_frames(
        members,
        nonmembers,
        reference_name="members",
        value_name="nonmembers",
    )
    _, generated = aligned_frames(
        member,
        synthetic,
        reference_name="members",
        value_name="synthetic",
    )
    canonical_method = str(method).strip().lower().replace("-", "_")
    aliases = {"nearest_neighbor": "knn", "density": "density_ratio"}
    canonical_method = aliases.get(canonical_method, canonical_method)
    if canonical_method not in {"knn", "density_ratio"}:
        raise ValueError("method must be 'knn' or 'density_ratio'")
    if isinstance(k, bool) or int(k) < 1:
        raise ValueError("k must be a positive integer")
    k = int(k)
    reference = pd.concat([member, nonmember], ignore_index=True)
    distance = MixedTypeDistance(
        reference, categorical_columns=categorical_columns
    )
    member_scores, nonmember_scores = _membership_scores(
        member,
        nonmember,
        generated,
        distance,
        method=canonical_method,
        k=k,
    )
    labels = np.concatenate(
        [
            np.ones(len(member_scores), dtype=np.int64),
            np.zeros(len(nonmember_scores), dtype=np.int64),
        ]
    )
    scores = np.concatenate([member_scores, nonmember_scores])
    auc_value = float(roc_auc_score(labels, scores))
    tpr_value = _tpr_at_fpr(labels, scores)

    def resampled_metric(
        rng: np.random.Generator, metric: str
    ) -> float:
        member_indices = rng.integers(
            0, len(member_scores), size=len(member_scores)
        )
        nonmember_indices = rng.integers(
            0, len(nonmember_scores), size=len(nonmember_scores)
        )
        sampled_scores = np.concatenate(
            [
                member_scores[member_indices],
                nonmember_scores[nonmember_indices],
            ]
        )
        sampled_labels = np.concatenate(
            [
                np.ones(len(member_indices), dtype=np.int64),
                np.zeros(len(nonmember_indices), dtype=np.int64),
            ]
        )
        if metric == "auc":
            return float(roc_auc_score(sampled_labels, sampled_scores))
        return _tpr_at_fpr(sampled_labels, sampled_scores)

    auc = bootstrap_estimate(
        auc_value,
        lambda rng: resampled_metric(rng, "auc"),
        replicates=bootstrap_replicates,
        confidence=confidence,
        random_state=random_state + 23,
    )
    tpr = bootstrap_estimate(
        tpr_value,
        lambda rng: resampled_metric(rng, "tpr"),
        replicates=bootstrap_replicates,
        confidence=confidence,
        random_state=random_state + 29,
    )
    return MembershipInferenceAudit(
        method=canonical_method,
        k=k,
        roc_auc=auc,
        tpr_at_1pct_fpr=tpr,
        member_score_mean=float(np.mean(member_scores)),
        nonmember_score_mean=float(np.mean(nonmember_scores)),
        member_count=len(member),
        nonmember_count=len(nonmember),
        limitations=(
            "This is a sample-only black-box attack. A near-chance result does "
            "not rule out stronger attacks, artifact access, side information, "
            "or attacks against subgroups.",
            "ROC-AUC measures ranking over the supplied balanced audit sets; "
            "TPR at 1% FPR is often uncertain for small holdouts.",
        ),
    )


def _categorical_prediction(
    values: Sequence[Any],
) -> Any:
    counts: Counter = Counter(value_key(value) for value in values)
    selected = min(
        counts,
        key=lambda key: (-counts[key], key),
    )
    for value in values:
        if value_key(value) == selected:
            return value
    raise RuntimeError("categorical vote failed")


def _equal_values(left: Any, right: Any) -> bool:
    return value_key(left) == value_key(right)


def attribute_inference_attack(
    real_evaluation: Any,
    synthetic: Any,
    sensitive_columns: Sequence[Any],
    *,
    quasi_identifier_columns: Optional[Sequence[Any]] = None,
    categorical_columns: Optional[Iterable[Any]] = None,
    k: int = 5,
    bootstrap_replicates: int = 200,
    confidence: float = 0.95,
    random_state: int = 0,
) -> Tuple[AttributeInferenceAudit, ...]:
    """Infer specified sensitive attributes from synthetic nearest neighbours."""

    evaluation = as_frame(real_evaluation, name="real_evaluation")
    _, generated = aligned_frames(
        evaluation,
        synthetic,
        reference_name="real_evaluation",
        value_name="synthetic",
    )
    sensitive = tuple(sensitive_columns)
    if not sensitive:
        raise ValueError("sensitive_columns must contain at least one column")
    unknown = [column for column in sensitive if column not in evaluation]
    if unknown:
        raise ValueError("unknown sensitive columns: %r" % unknown)
    if quasi_identifier_columns is None:
        quasi_identifiers = tuple(
            column for column in evaluation if column not in set(sensitive)
        )
    else:
        quasi_identifiers = tuple(quasi_identifier_columns)
    if not quasi_identifiers:
        raise ValueError("at least one quasi-identifier column is required")
    unknown_qi = [
        column for column in quasi_identifiers if column not in evaluation
    ]
    if unknown_qi:
        raise ValueError("unknown quasi-identifier columns: %r" % unknown_qi)
    if set(quasi_identifiers) & set(sensitive):
        raise ValueError("sensitive columns cannot also be quasi-identifiers")
    if isinstance(k, bool) or int(k) < 1:
        raise ValueError("k must be a positive integer")
    selected_k = min(int(k), len(generated))
    categorical = set(infer_categorical(evaluation, categorical_columns))
    qi_categorical = [
        column for column in quasi_identifiers if column in categorical
    ]
    distance = MixedTypeDistance(
        evaluation.loc[:, quasi_identifiers],
        categorical_columns=qi_categorical,
    )
    neighbors = np.argsort(
        distance.pairwise(
            evaluation.loc[:, quasi_identifiers],
            generated.loc[:, quasi_identifiers],
        ),
        axis=1,
        kind="mergesort",
    )[:, :selected_k]
    reports: List[AttributeInferenceAudit] = []
    for offset, column in enumerate(sensitive):
        actual = evaluation[column].to_numpy()
        neighbor_values = generated[column].to_numpy()[neighbors]
        is_categorical = column in categorical
        if is_categorical:
            predicted = np.asarray(
                [_categorical_prediction(row) for row in neighbor_values],
                dtype=object,
            )
            successes = np.asarray(
                [
                    float(_equal_values(left, right))
                    for left, right in zip(actual, predicted)
                ],
                dtype=np.float64,
            )
            counts = Counter(value_key(value) for value in actual)
            baseline = float(max(counts.values())) / float(len(actual))
            performance_value = float(np.mean(successes))
            denominator = max(1.0 - baseline, np.finfo(float).eps)
            advantage_value = (performance_value - baseline) / denominator
            performance = proportion_estimate(
                successes,
                replicates=bootstrap_replicates,
                confidence=confidence,
                random_state=random_state + 101 + offset,
            )

            def advantage_statistic(rng: np.random.Generator) -> float:
                indices = rng.integers(0, len(successes), size=len(successes))
                return (
                    float(np.mean(successes[indices])) - baseline
                ) / denominator

            advantage = bootstrap_estimate(
                advantage_value,
                advantage_statistic,
                replicates=bootstrap_replicates,
                confidence=confidence,
                random_state=random_state + 151 + offset,
            )
            metric = "accuracy"
        else:
            actual_numeric = pd.to_numeric(
                evaluation[column], errors="coerce"
            ).to_numpy(dtype=np.float64)
            generated_numeric = pd.to_numeric(
                generated[column], errors="coerce"
            ).to_numpy(dtype=np.float64)
            predicted_numeric = np.nanmean(
                generated_numeric[neighbors], axis=1
            )
            valid = np.isfinite(actual_numeric) & np.isfinite(predicted_numeric)
            if not bool(np.any(valid)):
                raise ValueError(
                    "sensitive numeric column %r has no finite audit rows"
                    % (column,)
                )
            actual_valid = actual_numeric[valid]
            predicted_valid = predicted_numeric[valid]
            squared_error = np.square(predicted_valid - actual_valid)
            baseline_center = float(np.mean(actual_valid))
            baseline_rmse = float(
                np.sqrt(np.mean(np.square(actual_valid - baseline_center)))
            )
            attack_rmse = float(np.sqrt(np.mean(squared_error)))
            denominator = max(baseline_rmse, np.finfo(float).eps)
            advantage_value = 1.0 - attack_rmse / denominator

            def rmse_statistic(rng: np.random.Generator) -> float:
                indices = rng.integers(
                    0, len(squared_error), size=len(squared_error)
                )
                return float(np.sqrt(np.mean(squared_error[indices])))

            performance = bootstrap_estimate(
                attack_rmse,
                rmse_statistic,
                replicates=bootstrap_replicates,
                confidence=confidence,
                random_state=random_state + 201 + offset,
            )

            def numeric_advantage(rng: np.random.Generator) -> float:
                indices = rng.integers(
                    0, len(squared_error), size=len(squared_error)
                )
                return 1.0 - (
                    float(np.sqrt(np.mean(squared_error[indices])))
                    / denominator
                )

            advantage = bootstrap_estimate(
                advantage_value,
                numeric_advantage,
                replicates=bootstrap_replicates,
                confidence=confidence,
                random_state=random_state + 251 + offset,
            )
            baseline = baseline_rmse
            metric = "rmse"
        reports.append(
            AttributeInferenceAudit(
                sensitive_column=column,
                kind="categorical" if is_categorical else "numeric",
                metric=metric,
                attack_performance=performance,
                baseline_performance=baseline,
                advantage=advantage,
                evaluated_rows=len(evaluation),
                quasi_identifier_columns=quasi_identifiers,
                limitations=(
                    "The attack uses only the selected quasi-identifiers and "
                    "synthetic nearest neighbours; other auxiliary information "
                    "may produce materially different risk.",
                    "The baseline is estimated on the audit population and is "
                    "not evidence that the sensitive attribute is safe.",
                ),
            )
        )
    return tuple(reports)


def singling_out_attack(
    real_train: Any,
    real_holdout: Any,
    synthetic: Any,
    *,
    categorical_columns: Optional[Iterable[Any]] = None,
    n_attacks: int = 100,
    predicate_size: int = 3,
    tolerance: float = 0.05,
    bootstrap_replicates: int = 200,
    confidence: float = 0.95,
    random_state: int = 0,
) -> SinglingOutAudit:
    """Approximate singling out with random synthetic-derived predicates."""

    train, holdout = aligned_frames(
        real_train,
        real_holdout,
        reference_name="real_train",
        value_name="real_holdout",
    )
    _, generated = aligned_frames(
        train,
        synthetic,
        reference_name="real_train",
        value_name="synthetic",
    )
    if isinstance(n_attacks, bool) or int(n_attacks) < 1:
        raise ValueError("n_attacks must be positive")
    if isinstance(predicate_size, bool) or int(predicate_size) < 1:
        raise ValueError("predicate_size must be positive")
    if not 0.0 <= float(tolerance) < 1.0:
        raise ValueError("tolerance must be in [0, 1)")
    predicate_size = min(int(predicate_size), len(train.columns))
    rng = np.random.default_rng(int(random_state))
    categorical = set(infer_categorical(train, categorical_columns))
    attack_success = np.zeros(int(n_attacks), dtype=np.float64)
    control_success = np.zeros(int(n_attacks), dtype=np.float64)
    for attack in range(int(n_attacks)):
        row = int(rng.integers(0, len(generated)))
        selected_indices = rng.choice(
            len(train.columns),
            size=predicate_size,
            replace=False,
        )
        selected = tuple(
            train.columns[int(index)] for index in selected_indices
        )
        selected_categorical = [
            column for column in selected if column in categorical
        ]
        reference = pd.concat(
            [train.loc[:, selected], holdout.loc[:, selected]],
            ignore_index=True,
        )
        metric = MixedTypeDistance(
            reference, categorical_columns=selected_categorical
        )
        predicate = generated.loc[[row], selected]
        train_distance = metric.pairwise(predicate, train.loc[:, selected])[0]
        holdout_distance = metric.pairwise(
            predicate, holdout.loc[:, selected]
        )[0]
        attack_success[attack] = float(
            np.count_nonzero(train_distance <= float(tolerance)) == 1
        )
        control_success[attack] = float(
            np.count_nonzero(holdout_distance <= float(tolerance)) == 1
        )
    return SinglingOutAudit(
        attack_success_rate=proportion_estimate(
            attack_success,
            replicates=bootstrap_replicates,
            confidence=confidence,
            random_state=random_state + 307,
        ),
        control_success_rate=proportion_estimate(
            control_success,
            replicates=bootstrap_replicates,
            confidence=confidence,
            random_state=random_state + 311,
        ),
        excess_success_rate=_difference_estimate(
            attack_success,
            control_success,
            replicates=bootstrap_replicates,
            confidence=confidence,
            random_state=random_state + 313,
        ),
        attacks=int(n_attacks),
        predicate_size=predicate_size,
        tolerance=float(tolerance),
        limitations=(
            "This is a synthetic-derived predicate approximation, not a "
            "complete implementation of every singling-out threat model.",
            "Success is sensitive to predicate width, numeric tolerance, and "
            "the size and representativeness of the control population.",
        ),
    )


def linkability_attack(
    real_evaluation: Any,
    synthetic: Any,
    *,
    auxiliary_columns_a: Optional[Sequence[Any]] = None,
    auxiliary_columns_b: Optional[Sequence[Any]] = None,
    categorical_columns: Optional[Iterable[Any]] = None,
    bootstrap_replicates: int = 200,
    confidence: float = 0.95,
    random_state: int = 0,
) -> LinkabilityAudit:
    """Approximate record linkage through two disjoint auxiliary views."""

    evaluation = as_frame(real_evaluation, name="real_evaluation")
    _, generated = aligned_frames(
        evaluation,
        synthetic,
        reference_name="real_evaluation",
        value_name="synthetic",
    )
    columns = tuple(evaluation.columns)
    if auxiliary_columns_a is None and auxiliary_columns_b is None:
        if len(columns) < 2:
            raise ValueError("linkability requires at least two columns")
        split = max(1, len(columns) // 2)
        view_a = columns[:split]
        view_b = columns[split:]
    elif auxiliary_columns_a is None or auxiliary_columns_b is None:
        raise ValueError("both auxiliary column sets must be provided")
    else:
        view_a = tuple(auxiliary_columns_a)
        view_b = tuple(auxiliary_columns_b)
    if not view_a or not view_b:
        raise ValueError("both auxiliary column sets must be non-empty")
    if set(view_a) & set(view_b):
        raise ValueError("auxiliary column sets must be disjoint")
    unknown = [
        column
        for column in view_a + view_b
        if column not in evaluation.columns
    ]
    if unknown:
        raise ValueError("unknown auxiliary columns: %r" % unknown)
    categorical = set(infer_categorical(evaluation, categorical_columns))

    def nearest(view: Tuple[Any, ...]) -> np.ndarray:
        view_categorical = [column for column in view if column in categorical]
        metric = MixedTypeDistance(
            evaluation.loc[:, view],
            categorical_columns=view_categorical,
        )
        values = metric.pairwise(
            evaluation.loc[:, view], generated.loc[:, view]
        )
        return np.argmin(values, axis=1)

    nearest_a = nearest(view_a)
    nearest_b = nearest(view_b)
    attack = (nearest_a == nearest_b).astype(np.float64)
    rng = np.random.default_rng(int(random_state) + 401)
    control = (
        nearest_a == nearest_b[rng.permutation(len(nearest_b))]
    ).astype(np.float64)
    return LinkabilityAudit(
        attack_success_rate=proportion_estimate(
            attack,
            replicates=bootstrap_replicates,
            confidence=confidence,
            random_state=random_state + 409,
        ),
        permutation_control_rate=proportion_estimate(
            control,
            replicates=bootstrap_replicates,
            confidence=confidence,
            random_state=random_state + 419,
        ),
        excess_success_rate=_difference_estimate(
            attack,
            control,
            replicates=bootstrap_replicates,
            confidence=confidence,
            random_state=random_state + 421,
        ),
        evaluated_rows=len(evaluation),
        auxiliary_columns_a=view_a,
        auxiliary_columns_b=view_b,
        limitations=(
            "This agreement-of-nearest-synthetic-anchors test is a "
            "linkability approximation, not proof that named identities can "
            "or cannot be linked.",
            "Results depend on the auxiliary views, distance metric, and "
            "available synthetic sample size.",
        ),
    )


def canary_extraction_audit(
    synthetic: Any,
    canaries: Any,
    *,
    decoys: Optional[Any] = None,
    categorical_columns: Optional[Iterable[Any]] = None,
    bootstrap_replicates: int = 200,
    confidence: float = 0.95,
    random_state: int = 0,
) -> CanaryAudit:
    """Test exact canary extraction and empirical candidate-set exposure."""

    canary_frame = as_frame(canaries, name="canaries")
    _, generated = aligned_frames(
        canary_frame,
        synthetic,
        reference_name="canaries",
        value_name="synthetic",
    )
    if decoys is None:
        candidates = canary_frame.copy()
    else:
        _, decoy_frame = aligned_frames(
            canary_frame,
            decoys,
            reference_name="canaries",
            value_name="decoys",
        )
        candidates = pd.concat(
            [canary_frame, decoy_frame], ignore_index=True
        )
    metric = MixedTypeDistance(
        candidates, categorical_columns=categorical_columns
    )
    candidate_distance = np.min(
        metric.pairwise(candidates, generated), axis=1
    )
    generated_counts = Counter(row_keys(generated))
    records: List[CanaryResult] = []
    extracted = np.zeros(len(canary_frame), dtype=np.float64)
    for index, key in enumerate(row_keys(canary_frame)):
        exact_count = int(generated_counts.get(key, 0))
        extracted[index] = float(exact_count > 0)
        distance_value = float(candidate_distance[index])
        lower = int(np.count_nonzero(candidate_distance < distance_value))
        tied = int(np.count_nonzero(candidate_distance == distance_value)) - 1
        rank = 1.0 + float(lower) + 0.5 * float(max(0, tied))
        exposure = math.log(float(len(candidates)), 2.0) - math.log(
            rank, 2.0
        )
        records.append(
            CanaryResult(
                index=index,
                extracted=bool(exact_count),
                exact_count=exact_count,
                nearest_distance=distance_value,
                rank=rank,
                candidate_count=len(candidates),
                exposure_bits=float(exposure),
            )
        )
    exposures = np.asarray(
        [record.exposure_bits for record in records], dtype=np.float64
    )
    return CanaryAudit(
        extraction_rate=proportion_estimate(
            extracted,
            replicates=bootstrap_replicates,
            confidence=confidence,
            random_state=random_state + 503,
        ),
        extracted_count=int(np.sum(extracted)),
        canary_count=len(canary_frame),
        maximum_exposure_bits=float(np.max(exposures)),
        mean_exposure_bits=float(np.mean(exposures)),
        records=tuple(records),
        limitations=(
            "Exposure is an empirical rank within the supplied candidate set, "
            "not a bound over an unspecified secret space.",
            "Failure to extract the tested canaries does not rule out "
            "memorization of other records or approximate extraction.",
        ),
    )


canary_audit = canary_extraction_audit


__all__ = [
    "AttributeInferenceAudit",
    "CanaryAudit",
    "CanaryResult",
    "ExactMatchAudit",
    "LinkabilityAudit",
    "MembershipInferenceAudit",
    "PROXIMITY_LIMITATION",
    "ProximityAudit",
    "SinglingOutAudit",
    "attribute_inference_attack",
    "canary_audit",
    "canary_extraction_audit",
    "dcr_nndr_audit",
    "exact_match_audit",
    "linkability_attack",
    "membership_inference_attack",
    "proximity_audit",
    "singling_out_attack",
]
