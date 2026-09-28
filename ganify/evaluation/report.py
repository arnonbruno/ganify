"""Long-form table reports that preserve every generation stage."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ._utils import aligned_frames, as_frame, infer_categorical_columns
from .c2st import classifier_two_sample_test
from .constraints import ConstraintLike, constraint_validity
from .dependence import dependence_matrix_errors, normalized_dependence_gap
from .marginals import marginal_metrics
from .neighbors import nearest_neighbor_metrics
from .utility import evaluate_utility


def _metric_row(
    stage: str,
    pillar: str,
    metric: str,
    value: object,
    *,
    column: Optional[object] = None,
    detail: Optional[str] = None,
) -> Dict[str, object]:
    return {
        "stage": stage,
        "pillar": pillar,
        "metric": metric,
        "column": column,
        "detail": detail,
        "value": float(value),
    }


def _privacy_metric_rows(stage: str, payload: Any) -> List[Dict[str, object]]:
    """Normalize an explicit privacy report without running implicit attacks."""

    if isinstance(payload, Mapping) and (
        "attack_report" in payload or "dp_report" in payload
    ):
        from ganify.privacy import privacy_gate_records

        frame = privacy_gate_records(
            payload.get("attack_report"),
            payload.get("dp_report"),
            stage=stage,
        )
    elif isinstance(payload, Mapping) and {
        "scope",
        "accountant",
        "privacy_boundary",
    }.issubset(payload):
        from ganify.privacy import privacy_gate_records

        frame = privacy_gate_records(None, payload, stage=stage)
    elif isinstance(payload, pd.DataFrame):
        frame = payload.copy()
    elif hasattr(payload, "to_frame"):
        try:
            frame = payload.to_frame(stage=stage)
        except TypeError:
            frame = payload.to_frame()
    else:
        raise TypeError(
            "privacy report must be a report object, dataframe, DP report, "
            "or {'attack_report', 'dp_report'} mapping"
        )
    required = {"pillar", "metric", "value"}
    if not required.issubset(frame.columns):
        raise ValueError(
            "privacy report rows require columns %r" % sorted(required)
        )
    output: List[Dict[str, object]] = []
    for _, item in frame.iterrows():
        value = pd.to_numeric(
            pd.Series([item["value"]]), errors="coerce"
        ).iloc[0]
        if pd.isna(value):
            continue
        output.append(
            _metric_row(
                stage,
                str(item["pillar"]),
                str(item["metric"]),
                value,
                column=item.get("column"),
                detail=(
                    None
                    if pd.isna(item.get("detail"))
                    else str(item.get("detail"))
                ),
            )
        )
    return output


def aggregate_table_report(
    real_train: object,
    real_test: object,
    stages: Mapping[str, object],
    *,
    categorical_columns: Optional[Iterable[object]] = None,
    constraints: Optional[Iterable[ConstraintLike]] = None,
    target_column: Optional[object] = None,
    task: Optional[str] = None,
    random_state: int = 0,
    c2st_folds: int = 5,
    privacy_reports: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """Evaluate raw/calibrated/projected tables without combining their scores.

    Parameters
    ----------
    stages:
        An ordered mapping such as ``{"raw": raw, "calibrated": calibrated,
        "projected": projected}``. Each output row carries its stage name.
    target_column, task:
        When both are provided, add TSTR/TRTR classification or regression
        metrics using the target contained in each table.
    privacy_reports:
        Optional reports keyed by generation stage. Attacks are never run
        implicitly; callers must pass an explicit attack report, DP boundary
        report, or ``{"attack_report": ..., "dp_report": ...}`` bundle.

    Returns
    -------
    pandas.DataFrame
        Long-form rows with ``stage``, ``pillar``, ``metric``, ``column``,
        ``detail``, and ``value``. There is intentionally no overall or blended
        quality score.
    """

    train = as_frame(real_train, name="real_train")
    _, test = aligned_frames(train, real_test)
    if not stages:
        raise ValueError("at least one generation stage is required")
    stage_names = list(stages)
    if len(set(stage_names)) != len(stage_names):
        raise ValueError("generation stage names must be unique")
    categorical = infer_categorical_columns(train, categorical_columns)
    numeric = [column for column in train.columns if column not in set(categorical)]
    rows: List[Dict[str, object]] = []
    constraint_list = list(constraints) if constraints is not None else None
    privacy_by_stage = (
        {} if privacy_reports is None else dict(privacy_reports)
    )
    unknown_privacy_stages = [
        stage for stage in privacy_by_stage if stage not in stages
    ]
    if unknown_privacy_stages:
        raise ValueError(
            "privacy reports reference unknown stages: %r"
            % unknown_privacy_stages
        )

    for stage, values in stages.items():
        _, synthetic = aligned_frames(train, values)
        marginal = marginal_metrics(
            test, synthetic, categorical_columns=categorical
        )
        id_columns = {"column", "kind"}
        for _, marginal_row in marginal.iterrows():
            for metric, value in marginal_row.items():
                if metric in id_columns or pd.isna(value):
                    continue
                rows.append(
                    _metric_row(
                        stage,
                        "marginal",
                        str(metric),
                        value,
                        column=marginal_row["column"],
                        detail=str(marginal_row["kind"]),
                    )
                )

        if len(numeric) >= 2:
            dependence = dependence_matrix_errors(
                test, synthetic, columns=numeric, rank_space=True
            )
            for metric, value in dependence.as_dict().items():
                rows.append(
                    _metric_row(stage, "dependence", metric, value)
                )
            try:
                normalized = normalized_dependence_gap(
                    train,
                    test,
                    synthetic,
                    columns=numeric,
                    random_state=random_state,
                )
                rows.append(
                    _metric_row(
                        stage,
                        "dependence",
                        "normalized_dependence_gap",
                        normalized.normalized_gap,
                    )
                )
                rows.append(
                    _metric_row(
                        stage,
                        "dependence_control",
                        "real_floor_loss",
                        normalized.real_floor_loss,
                        detail=normalized.metric,
                    )
                )
                rows.append(
                    _metric_row(
                        stage,
                        "dependence_control",
                        "independent_loss",
                        normalized.independent_loss,
                        detail=normalized.metric,
                    )
                )
            except ValueError:
                # An actually independent real table can make the negative
                # control indistinguishable from the real floor. The raw matrix
                # errors remain valid; no normalized value is fabricated.
                pass

        folds = min(int(c2st_folds), len(test), len(synthetic))
        if folds >= 2:
            c2st = classifier_two_sample_test(
                test,
                synthetic,
                categorical_columns=categorical,
                random_state=random_state,
                folds=folds,
            )
            rows.append(_metric_row(stage, "c2st", "auc", c2st.auc))
            rows.append(_metric_row(stage, "c2st", "raw_auc", c2st.raw_auc))

        neighbors = nearest_neighbor_metrics(
            train,
            synthetic,
            real_holdout=test,
            categorical_columns=categorical,
        )
        for metric, value in neighbors.as_dict().items():
            rows.append(_metric_row(stage, "nearest_neighbor", metric, value))

        if constraint_list is not None:
            validity = constraint_validity(synthetic, constraint_list)
            for _, validity_row in validity.iterrows():
                rows.append(
                    _metric_row(
                        stage,
                        "constraint",
                        "valid_rate",
                        validity_row["valid_rate"],
                        detail=str(validity_row["constraint"]),
                    )
                )
                rows.append(
                    _metric_row(
                        stage,
                        "constraint",
                        "violation_rate",
                        validity_row["violation_rate"],
                        detail=str(validity_row["constraint"]),
                    )
                )

        if target_column is not None or task is not None:
            if target_column is None or task is None:
                raise ValueError(
                    "target_column and task must be provided together for utility"
                )
            if target_column not in train.columns:
                raise ValueError("target column %r is missing" % target_column)
            feature_columns = [
                column for column in train.columns if column != target_column
            ]
            utility_categorical = [
                column for column in categorical if column != target_column
            ]
            utility = evaluate_utility(
                train.loc[:, feature_columns],
                train[target_column],
                test.loc[:, feature_columns],
                test[target_column],
                synthetic.loc[:, feature_columns],
                synthetic[target_column],
                task=task,
                categorical_columns=utility_categorical,
                random_state=random_state,
            )
            for section in ("trtr", "tstr", "dummy", "relative_utility"):
                section_values = utility[section]
                if not isinstance(section_values, Mapping):
                    continue
                for metric, value in section_values.items():
                    rows.append(
                        _metric_row(
                            stage,
                            "utility",
                            str(metric),
                            value,
                            detail=section,
                        )
                    )
            primary_metric = (
                "balanced_accuracy"
                if str(task).lower() == "classification"
                else "r2"
            )
            relative_values = utility["relative_utility"]
            if (
                isinstance(relative_values, Mapping)
                and primary_metric in relative_values
            ):
                rows.append(
                    _metric_row(
                        stage,
                        "utility",
                        "relative_primary_utility",
                        relative_values[primary_metric],
                        detail=primary_metric,
                    )
                )

        if stage in privacy_by_stage:
            rows.extend(
                _privacy_metric_rows(stage, privacy_by_stage[stage])
            )

    result = pd.DataFrame(
        rows, columns=["stage", "pillar", "metric", "column", "detail", "value"]
    )
    result.attrs["stage_order"] = stage_names
    result.attrs["contains_blended_score"] = False
    return result


def aggregate_stage_reports(reports: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate long-form reports while retaining explicit stage columns."""

    if not reports:
        return pd.DataFrame(
            columns=["stage", "pillar", "metric", "column", "detail", "value"]
        )
    for report in reports:
        required = {"stage", "pillar", "metric", "value"}
        if not required.issubset(report.columns):
            raise ValueError("each report must include %r" % sorted(required))
    output = pd.concat(reports, ignore_index=True, sort=False)
    output.attrs["contains_blended_score"] = False
    return output


evaluate_table_stages = aggregate_table_report


__all__ = [
    "aggregate_stage_reports",
    "aggregate_table_report",
    "evaluate_table_stages",
]
