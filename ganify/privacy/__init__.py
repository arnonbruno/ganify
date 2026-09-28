"""Privacy red-team audits, artifact inspection, and optional DP primitives.

Attack results and proximity diagnostics are empirical evidence only.
Differential-privacy claims are exposed separately and fail closed unless the
mechanism accountant and the complete preprocessing boundary are accounted.
"""

from .artifacts import (
    ArtifactFinding,
    ArtifactRiskReport,
    artifact_risk_report,
    export_smoothed_marginals,
    export_smoothed_quantized_marginals,
    inspect_artifact_risk,
)
from .attacks import (
    AttributeInferenceAudit,
    CanaryAudit,
    CanaryResult,
    ExactMatchAudit,
    LinkabilityAudit,
    MembershipInferenceAudit,
    ProximityAudit,
    SinglingOutAudit,
    attribute_inference_attack,
    canary_audit,
    canary_extraction_audit,
    dcr_nndr_audit,
    exact_match_audit,
    linkability_attack,
    membership_inference_attack,
    proximity_audit,
    singling_out_attack,
)
from .dp import (
    DPConfig,
    PrivacyAccountant,
    clip_and_noise_gradients,
    clip_gradients,
)
from .gates import (
    PrivacyGateDecision,
    PrivacyGateReport,
    evaluate_privacy_gates,
    formal_dp_eligible,
    formal_dp_gate,
    privacy_gate_records,
)
from .report import PrivacyAuditReport, audit_privacy, privacy_audit

__all__ = [
    "ArtifactFinding",
    "ArtifactRiskReport",
    "AttributeInferenceAudit",
    "CanaryAudit",
    "CanaryResult",
    "DPConfig",
    "ExactMatchAudit",
    "LinkabilityAudit",
    "MembershipInferenceAudit",
    "PrivacyAccountant",
    "PrivacyAuditReport",
    "PrivacyGateDecision",
    "PrivacyGateReport",
    "ProximityAudit",
    "SinglingOutAudit",
    "artifact_risk_report",
    "attribute_inference_attack",
    "audit_privacy",
    "canary_audit",
    "canary_extraction_audit",
    "clip_and_noise_gradients",
    "clip_gradients",
    "dcr_nndr_audit",
    "evaluate_privacy_gates",
    "exact_match_audit",
    "export_smoothed_marginals",
    "export_smoothed_quantized_marginals",
    "formal_dp_eligible",
    "formal_dp_gate",
    "inspect_artifact_risk",
    "linkability_attack",
    "membership_inference_attack",
    "privacy_audit",
    "privacy_gate_records",
    "proximity_audit",
    "singling_out_attack",
]
