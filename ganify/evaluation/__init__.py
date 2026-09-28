"""First-class, anti-gaming evaluation for synthetic tabular data.

The public APIs report separate metric pillars and generation stages. They
intentionally do not expose a blended "quality score".
"""

from .c2st import C2STResult, c2st_auc, classifier_two_sample_test
from .constraints import (
    AllowedValuesConstraint,
    CallableConstraint,
    Constraint,
    ConstraintLike,
    InequalityConstraint,
    RangeConstraint,
    constraint_from_spec,
    constraint_validity,
    evaluate_constraints,
)
from .controls import (
    bootstrap_control,
    independently_permuted_columns,
    permuted_columns_control,
)
from .dependence import (
    DependenceMatrixErrors,
    NormalizedDependenceGap,
    dependence_matrices,
    dependence_matrix_errors,
    normalized_dependence_gap,
    rank_dependence_metrics,
    rank_transform,
)
from .marginals import (
    CategoricalMarginalMetrics,
    NumericMarginalMetrics,
    categorical_marginal_metrics,
    categorical_total_variation,
    marginal_metrics,
    numeric_marginal_metrics,
)
from .neighbors import NearestNeighborMetrics, nearest_neighbor_metrics
from .report import (
    aggregate_stage_reports,
    aggregate_table_report,
    evaluate_table_stages,
)
from .utility import (
    classification_utility,
    evaluate_utility,
    regression_utility,
)

__all__ = [
    "AllowedValuesConstraint",
    "C2STResult",
    "CallableConstraint",
    "CategoricalMarginalMetrics",
    "Constraint",
    "ConstraintLike",
    "DependenceMatrixErrors",
    "InequalityConstraint",
    "NearestNeighborMetrics",
    "NormalizedDependenceGap",
    "NumericMarginalMetrics",
    "RangeConstraint",
    "aggregate_stage_reports",
    "aggregate_table_report",
    "bootstrap_control",
    "c2st_auc",
    "categorical_marginal_metrics",
    "categorical_total_variation",
    "classification_utility",
    "classifier_two_sample_test",
    "constraint_from_spec",
    "constraint_validity",
    "dependence_matrices",
    "dependence_matrix_errors",
    "evaluate_constraints",
    "evaluate_table_stages",
    "evaluate_utility",
    "independently_permuted_columns",
    "marginal_metrics",
    "nearest_neighbor_metrics",
    "normalized_dependence_gap",
    "numeric_marginal_metrics",
    "permuted_columns_control",
    "rank_dependence_metrics",
    "rank_transform",
    "regression_utility",
]
