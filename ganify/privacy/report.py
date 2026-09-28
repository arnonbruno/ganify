"""Structured privacy red-team reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from .attacks import (
    AttributeInferenceAudit,
    CanaryAudit,
    ExactMatchAudit,
    LinkabilityAudit,
    MembershipInferenceAudit,
    ProximityAudit,
    SinglingOutAudit,
    attribute_inference_attack,
    canary_extraction_audit,
    dcr_nndr_audit,
    exact_match_audit,
    linkability_attack,
    membership_inference_attack,
    singling_out_attack,
)


@dataclass(frozen=True)
class PrivacyAuditReport:
    """A collection of empirical attacks with no blended privacy score."""

    exact_matches: ExactMatchAudit
    proximity: ProximityAudit
    membership_inference: MembershipInferenceAudit
    attribute_inference: Tuple[AttributeInferenceAudit, ...]
    singling_out: SinglingOutAudit
    linkability: Optional[LinkabilityAudit]
    canaries: Optional[CanaryAudit]
    confidence: float
    bootstrap_replicates: int
    limitations: Tuple[str, ...]

    def as_dict(self) -> Dict[str, Any]:
        output = asdict(self)
        output["contains_privacy_guarantee"] = False
        output["contains_blended_score"] = False
        return output

    def to_frame(self, *, stage: str = "synthetic") -> pd.DataFrame:
        """Return gate-compatible long-form point estimates."""

        rows: List[Dict[str, Any]] = []

        def add(
            pillar: str,
            metric: str,
            value: Any,
            *,
            column: Any = None,
            detail: Optional[str] = None,
        ) -> None:
            if value is None:
                return
            rows.append(
                {
                    "stage": stage,
                    "pillar": pillar,
                    "metric": metric,
                    "column": column,
                    "detail": detail,
                    "value": float(value),
                }
            )

        add(
            "privacy_attack",
            "exact_match_rate",
            self.exact_matches.exact_match_rate.value,
        )
        add(
            "privacy_attack",
            "exact_match_rate_ci_high",
            self.exact_matches.exact_match_rate.ci_high,
        )
        add(
            "privacy_attack",
            "holdout_exact_match_rate",
            self.exact_matches.holdout_exact_match_rate.value,
        )
        add(
            "privacy_attack",
            "train_only_exact_match_rate",
            self.exact_matches.train_only_exact_match_rate.value,
        )
        add(
            "privacy_attack",
            "collision_adjusted_exact_match_rate",
            self.exact_matches.collision_adjusted_exact_match_rate,
        )
        add(
            "privacy_attack",
            "collision_adjusted_exact_match_rate_ci_high",
            self.exact_matches.collision_adjusted_exact_match_rate_ci_high,
        )
        add(
            "privacy_attack",
            "unique_member_extraction_rate",
            self.exact_matches.unique_member_extraction_rate,
        )
        add(
            "privacy_proximity",
            "dcr_median",
            self.proximity.dcr_median,
            detail="diagnostic_not_privacy",
        )
        add(
            "privacy_proximity",
            "dcr_p05",
            self.proximity.dcr_p05,
            detail="diagnostic_not_privacy",
        )
        add(
            "privacy_proximity",
            "nndr_median",
            self.proximity.nndr_median,
            detail="diagnostic_not_privacy",
        )
        add(
            "privacy_attack",
            "membership_roc_auc",
            self.membership_inference.roc_auc.value,
            detail=self.membership_inference.method,
        )
        add(
            "privacy_attack",
            "membership_roc_auc_ci_high",
            self.membership_inference.roc_auc.ci_high,
            detail=self.membership_inference.method,
        )
        add(
            "privacy_attack",
            "membership_tpr_at_1pct_fpr",
            self.membership_inference.tpr_at_1pct_fpr.value,
            detail=self.membership_inference.method,
        )
        add(
            "privacy_attack",
            "membership_tpr_at_1pct_fpr_ci_high",
            self.membership_inference.tpr_at_1pct_fpr.ci_high,
            detail=self.membership_inference.method,
        )
        for attribute in self.attribute_inference:
            add(
                "privacy_attack",
                "attribute_inference_advantage",
                attribute.advantage.value,
                column=attribute.sensitive_column,
                detail=attribute.metric,
            )
            add(
                "privacy_attack",
                "attribute_inference_advantage_ci_high",
                attribute.advantage.ci_high,
                column=attribute.sensitive_column,
                detail=attribute.metric,
            )
            add(
                "privacy_attack",
                "attribute_inference_performance",
                attribute.attack_performance.value,
                column=attribute.sensitive_column,
                detail=attribute.metric,
            )
        add(
            "privacy_attack",
            "singling_out_excess_success_rate",
            self.singling_out.excess_success_rate.value,
            detail="approximation",
        )
        add(
            "privacy_attack",
            "singling_out_excess_success_rate_ci_high",
            self.singling_out.excess_success_rate.ci_high,
            detail="approximation",
        )
        if self.linkability is not None:
            add(
                "privacy_attack",
                "linkability_excess_success_rate",
                self.linkability.excess_success_rate.value,
                detail="approximation",
            )
            add(
                "privacy_attack",
                "linkability_excess_success_rate_ci_high",
                self.linkability.excess_success_rate.ci_high,
                detail="approximation",
            )
        if self.canaries is not None:
            add(
                "privacy_attack",
                "canary_extraction_rate",
                self.canaries.extraction_rate.value,
            )
            add(
                "privacy_attack",
                "canary_extraction_rate_ci_high",
                self.canaries.extraction_rate.ci_high,
            )
            add(
                "privacy_attack",
                "canary_maximum_exposure_bits",
                self.canaries.maximum_exposure_bits,
                detail="empirical_candidate_set",
            )
        frame = pd.DataFrame(
            rows,
            columns=[
                "stage",
                "pillar",
                "metric",
                "column",
                "detail",
                "value",
            ],
        )
        frame.attrs["contains_privacy_guarantee"] = False
        frame.attrs["contains_blended_score"] = False
        return frame


def audit_privacy(
    real_train: Any,
    real_holdout: Any,
    synthetic: Any,
    *,
    sensitive_columns: Sequence[Any] = (),
    quasi_identifier_columns: Optional[Sequence[Any]] = None,
    categorical_columns: Optional[Iterable[Any]] = None,
    membership_method: str = "knn",
    membership_k: int = 1,
    canaries: Optional[Any] = None,
    canary_decoys: Optional[Any] = None,
    n_singling_attacks: int = 100,
    singling_predicate_size: int = 3,
    linkability_columns_a: Optional[Sequence[Any]] = None,
    linkability_columns_b: Optional[Sequence[Any]] = None,
    bootstrap_replicates: int = 200,
    confidence: float = 0.95,
    random_state: int = 0,
) -> PrivacyAuditReport:
    """Run the standard attack battery against one synthetic release."""

    exact = exact_match_audit(
        real_train,
        real_holdout,
        synthetic,
        bootstrap_replicates=bootstrap_replicates,
        confidence=confidence,
        random_state=random_state,
    )
    proximity = dcr_nndr_audit(
        real_train,
        real_holdout,
        synthetic,
        categorical_columns=categorical_columns,
    )
    membership = membership_inference_attack(
        real_train,
        real_holdout,
        synthetic,
        method=membership_method,
        k=membership_k,
        categorical_columns=categorical_columns,
        bootstrap_replicates=bootstrap_replicates,
        confidence=confidence,
        random_state=random_state,
    )
    attributes = (
        attribute_inference_attack(
            real_holdout,
            synthetic,
            sensitive_columns,
            quasi_identifier_columns=quasi_identifier_columns,
            categorical_columns=categorical_columns,
            bootstrap_replicates=bootstrap_replicates,
            confidence=confidence,
            random_state=random_state,
        )
        if sensitive_columns
        else ()
    )
    singling = singling_out_attack(
        real_train,
        real_holdout,
        synthetic,
        categorical_columns=categorical_columns,
        n_attacks=n_singling_attacks,
        predicate_size=singling_predicate_size,
        bootstrap_replicates=bootstrap_replicates,
        confidence=confidence,
        random_state=random_state,
    )
    columns = list(pd.DataFrame(real_holdout).columns)
    linkability = (
        linkability_attack(
            real_holdout,
            synthetic,
            auxiliary_columns_a=linkability_columns_a,
            auxiliary_columns_b=linkability_columns_b,
            categorical_columns=categorical_columns,
            bootstrap_replicates=bootstrap_replicates,
            confidence=confidence,
            random_state=random_state,
        )
        if len(columns) >= 2
        else None
    )
    canary_report = (
        canary_extraction_audit(
            synthetic,
            canaries,
            decoys=canary_decoys,
            categorical_columns=categorical_columns,
            bootstrap_replicates=bootstrap_replicates,
            confidence=confidence,
            random_state=random_state,
        )
        if canaries is not None
        else None
    )
    return PrivacyAuditReport(
        exact_matches=exact,
        proximity=proximity,
        membership_inference=membership,
        attribute_inference=tuple(attributes),
        singling_out=singling,
        linkability=linkability,
        canaries=canary_report,
        confidence=float(confidence),
        bootstrap_replicates=int(bootstrap_replicates),
        limitations=(
            "Attack results are evidence about tested attacks and audit data, "
            "not a formal privacy guarantee.",
            "A passed threshold can be invalidated by stronger adversaries, "
            "artifact access, different auxiliary information, subgroup risk, "
            "or future releases composed with this one.",
            "No overall privacy score is computed; each attack and its "
            "uncertainty must be reviewed separately.",
        ),
    )


privacy_audit = audit_privacy


__all__ = ["PrivacyAuditReport", "audit_privacy", "privacy_audit"]
