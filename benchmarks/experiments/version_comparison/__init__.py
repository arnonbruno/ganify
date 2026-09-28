"""Reproducible, source-isolated GANify version comparison tooling."""

from .aggregation import (
    AggregationResult,
    aggregate_comparison,
    aggregate_fit_levels,
    aggregate_version_comparison,
    average_sample_seeds,
    paired_version_deltas,
    summarize_fits,
)
from .core import (
    ADAPTER_PROTOCOLS,
    AdapterProtocol,
    CacheIntegrityError,
    CapabilityError,
    FixedSplit,
    HistoricalArtifactError,
    VersionComparisonError,
    VersionRefusalError,
    allocate_class_counts,
    apply_fixed_split,
    deterministic_row_split,
    get_adapter_protocol,
    hash_source_root,
)
from .evaluator import (
    EvaluationResult,
    bounded_subset,
    build_controls,
    evaluate_stages,
)
from .historical import (
    HistoricalArtifact,
    HistoricalRescoreResult,
    discover_kuairand_artifacts,
    original_seed42_split,
    rescore_historical_kuairand,
)
from .isolation import invoke_source_worker, probe_source
from .runner import (
    VersionComparisonRunResult,
    VersionComparisonRunner,
    load_protocol,
    run_protocol,
)

__all__ = [
    "ADAPTER_PROTOCOLS",
    "AdapterProtocol",
    "AggregationResult",
    "CacheIntegrityError",
    "CapabilityError",
    "EvaluationResult",
    "FixedSplit",
    "HistoricalArtifact",
    "HistoricalArtifactError",
    "HistoricalRescoreResult",
    "VersionComparisonError",
    "VersionComparisonRunResult",
    "VersionComparisonRunner",
    "VersionRefusalError",
    "aggregate_comparison",
    "aggregate_fit_levels",
    "aggregate_version_comparison",
    "allocate_class_counts",
    "apply_fixed_split",
    "average_sample_seeds",
    "bounded_subset",
    "build_controls",
    "deterministic_row_split",
    "discover_kuairand_artifacts",
    "evaluate_stages",
    "get_adapter_protocol",
    "hash_source_root",
    "invoke_source_worker",
    "load_protocol",
    "original_seed42_split",
    "paired_version_deltas",
    "probe_source",
    "rescore_historical_kuairand",
    "run_protocol",
    "summarize_fits",
]

