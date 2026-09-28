"""Reversible structural parameterizations for constrained tables."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ganify.schema import (
    ColumnType,
    TableSchema,
    decode_json_value,
    encode_json_value,
)
from ganify.preprocessing import TableTransformer

from .core import (
    BoundConstraint,
    ConstraintLike,
    ConstraintSet,
    DomainConstraint,
    ImplicationConstraint,
    LinearEquality,
    LinearInequality,
    PairInequality,
    Predicate,
    ProjectionAudit,
    SimplexConstraint,
    SumConstraint,
    _project_predicate,
    _row_changes,
)


def _python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return None
    return value


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("%s must be a positive integer" % name)
    value = int(value)
    if value < 1:
        raise ValueError("%s must be a positive integer" % name)
    return value


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / np.sum(exponential, axis=1, keepdims=True)


def _decode_composition(
    logits: np.ndarray, totals: np.ndarray, epsilon: float
) -> np.ndarray:
    """Invert smoothed log-ratios without clipping generated compositions."""

    shares = _softmax(logits)
    width = logits.shape[1]
    candidate = (
        shares * (totals[:, None] + float(width) * epsilon) - epsilon
    )
    tolerance = max(epsilon * 1e-6, 1e-14)
    exact = np.min(candidate, axis=1) >= -tolerance
    result = shares * totals[:, None]
    if bool(exact.any()):
        restored = candidate[exact]
        restored[np.abs(restored) <= tolerance] = 0.0
        # Floating subtraction above is theoretically sum-preserving.  Put
        # its last few ulps on the largest member for exact row totals.
        correction = totals[exact] - np.sum(restored, axis=1)
        largest = np.argmax(restored, axis=1)
        restored[np.arange(len(restored)), largest] += correction
        result[exact] = restored
    return result


def _expand_condition(value: Any, rows: int) -> List[Any]:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return [value.item()] * rows
    if isinstance(value, (list, np.ndarray, pd.Series, pd.Index)):
        values = list(value)
        if len(values) == 1:
            return values * rows
        if len(values) != rows:
            raise ValueError(
                "per-row condition values must have length %d" % rows
            )
        return values
    return [value] * rows


def _condition_frame(
    values: Mapping[Any, Sequence[Any]], columns: Sequence[Any], rows: int
) -> pd.DataFrame:
    return pd.DataFrame(
        {column: list(values[column]) for column in columns},
        columns=list(columns),
        index=pd.RangeIndex(rows),
    )


def _predicate_on_values(
    predicate: Predicate, values: Sequence[Any]
) -> np.ndarray:
    return predicate.evaluate(pd.DataFrame({predicate.column: list(values)}))


def _constraint_values_valid(
    constraint: Any,
    values: Mapping[Any, Sequence[Any]],
    rows: int,
) -> bool:
    probe = _condition_frame(values, constraint.columns, rows)
    return bool(np.all(constraint.evaluate(probe)))


class StructuralConstraintTransformer:
    """Map constrained columns to unconstrained/dependent coordinates.

    Pair inequalities replace the ``big`` coordinate by a nonnegative gap.
    Reconstruction folds a generated signed gap with ``abs`` rather than
    clipping, so invalid draws are not accumulated on the equality boundary.

    Nonnegative sums use one total and ``k - 1`` additive log-ratios.
    Reconstruction appends the reference logit and applies softmax, which
    guarantees nonnegative members with the exact requested total.
    """

    FORMAT_VERSION = 1
    STATE_FILENAME = "constraint_preprocessor.json"

    def __init__(
        self,
        constraints: Optional[
            Union[ConstraintSet, Iterable[ConstraintLike], Mapping[str, Any]]
        ] = None,
        *,
        log_ratio_epsilon: float = 1e-8,
    ) -> None:
        if isinstance(constraints, ConstraintSet):
            constraint_set = constraints
        elif isinstance(constraints, Mapping) and constraints.get(
            "format"
        ) == "ganify-constraint-set":
            constraint_set = ConstraintSet.from_dict(constraints)
        else:
            constraint_set = ConstraintSet(constraints)
        epsilon = float(log_ratio_epsilon)
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("log_ratio_epsilon must be finite and positive")
        self.constraint_set = constraint_set
        self.log_ratio_epsilon = epsilon
        self.fitted_ = False

    @property
    def constraints(self) -> ConstraintSet:
        return self.constraint_set

    def fit(
        self,
        frame: pd.DataFrame,
        *,
        schema: Optional[TableSchema] = None,
    ) -> "StructuralConstraintTransformer":
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(
                "StructuralConstraintTransformer.fit requires a DataFrame"
            )
        if frame.columns.has_duplicates:
            raise ValueError("dataframe contains duplicate column names")
        if len(frame) < 1:
            raise ValueError("dataframe must contain at least one row")
        if schema is not None:
            if not isinstance(schema, TableSchema):
                raise TypeError("schema must be a TableSchema")
            schema.validate(frame)
        self.constraint_set.validate(frame, raise_on_violation=True)

        pair_indices: List[int] = []
        composition_indices: List[int] = []
        implication_indices: List[int] = []
        occupied_composition: Dict[Any, int] = {}
        pair_writers: Dict[Any, int] = {}

        for index, constraint in enumerate(self.constraint_set):
            if isinstance(constraint, SumConstraint):
                if constraint.allow_missing or not constraint.nonnegative:
                    continue
                overlap = [
                    column
                    for column in constraint.columns
                    if column in occupied_composition
                ]
                if overlap:
                    other = occupied_composition[overlap[0]]
                    raise ValueError(
                        "structural sum constraints %r and %r overlap at %r; "
                        "use one composition or express the overlap as a "
                        "linear fallback"
                        % (
                            self.constraint_set[other].label,
                            constraint.label,
                            overlap[0],
                        )
                    )
                values = constraint.member_matrix(frame)
                if bool(np.any(~np.isfinite(values))) or bool(
                    np.any(values < 0.0)
                ):
                    raise ValueError(
                        "structural sum %r requires finite nonnegative members"
                        % constraint.label
                    )
                totals = constraint.target(frame)
                if bool(np.any(~np.isfinite(totals))) or bool(
                    np.any(totals < 0.0)
                ):
                    raise ValueError(
                        "structural sum %r requires finite nonnegative totals"
                        % constraint.label
                    )
                composition_indices.append(index)
                for column in constraint.columns:
                    occupied_composition[column] = index
            elif isinstance(constraint, PairInequality):
                if constraint.allow_missing:
                    continue
                if constraint.big in pair_writers:
                    other = pair_writers[constraint.big]
                    raise ValueError(
                        "pair constraints %r and %r both parameterize %r"
                        % (
                            self.constraint_set[other].label,
                            constraint.label,
                            constraint.big,
                        )
                    )
                pair_indices.append(index)
                pair_writers[constraint.big] = index
            elif isinstance(constraint, ImplicationConstraint):
                if (
                    not constraint.allow_missing
                    and constraint.consequent.operator
                    in {"==", "in", "<", "<=", ">", ">="}
                ):
                    implication_indices.append(index)

        collisions = [
            column
            for column in pair_writers
            if column in occupied_composition
        ]
        if collisions:
            raise ValueError(
                "columns cannot simultaneously be pair-gap and composition "
                "coordinates; overlap=%r" % collisions
            )

        pair_order = self._topological_pair_order(pair_indices)
        self.original_columns_ = tuple(frame.columns)
        self.original_dtypes_ = {
            column: str(frame[column].dtype) for column in frame.columns
        }
        self.numeric_scales_ = self._numeric_scales(frame)
        self.pair_indices_ = tuple(pair_indices)
        self.pair_order_ = tuple(pair_order)
        self.composition_indices_ = tuple(composition_indices)
        self.implication_indices_ = tuple(implication_indices)
        self.structural_constraint_indices_ = tuple(
            sorted(
                set(pair_indices)
                | set(composition_indices)
                | set(implication_indices)
            )
        )
        self.original_schema_ = schema
        self.reference_row_ = {
            column: _python_scalar(frame.iloc[0][column])
            for column in frame.columns
        }
        self.fitted_ = True
        masks = self.conditional_masks(frame)
        self.conditional_mask_rates_ = {
            column: float(masks[column].mean())
            for column in masks.columns
        }
        parameterized = self.transform(frame, validate=False)
        self.reference_parameterized_row_ = {
            column: _python_scalar(parameterized.iloc[0][column])
            for column in parameterized.columns
        }
        return self

    def fit_transform(
        self,
        frame: pd.DataFrame,
        *,
        schema: Optional[TableSchema] = None,
    ) -> pd.DataFrame:
        return self.fit(frame, schema=schema).transform(
            frame, validate=False
        )

    def _topological_pair_order(
        self, pair_indices: Sequence[int]
    ) -> List[int]:
        writers = {
            self.constraint_set[index].big: index
            for index in pair_indices
        }
        dependencies: Dict[int, set] = {
            index: set() for index in pair_indices
        }
        for index in pair_indices:
            constraint = self.constraint_set[index]
            dependency = writers.get(constraint.small)
            if dependency is not None:
                dependencies[index].add(dependency)
        order: List[int] = []
        pending = dict(dependencies)
        while pending:
            ready = sorted(
                index
                for index, values in pending.items()
                if not values
            )
            if not ready:
                labels = [
                    self.constraint_set[index].label for index in pending
                ]
                raise ValueError(
                    "pair inequalities contain a dependency cycle: %r"
                    % labels
                )
            for index in ready:
                order.append(index)
                pending.pop(index)
            for values in pending.values():
                values.difference_update(ready)
        return order

    @staticmethod
    def _numeric_scales(frame: pd.DataFrame) -> Dict[Any, float]:
        scales: Dict[Any, float] = {}
        for column in frame.columns:
            try:
                values = pd.to_numeric(
                    frame[column], errors="raise"
                ).to_numpy(dtype=np.float64)
            except (TypeError, ValueError):
                continue
            finite = values[np.isfinite(values)]
            if not len(finite):
                continue
            spread = float(np.std(finite))
            observed_range = float(np.max(finite) - np.min(finite))
            magnitude = float(np.mean(np.abs(finite)))
            scales[column] = max(spread, observed_range, magnitude, 1.0)
        return scales

    def _check_fitted(self) -> None:
        if not getattr(self, "fitted_", False):
            raise RuntimeError(
                "StructuralConstraintTransformer is not fitted"
            )

    def _check_frame(self, frame: pd.DataFrame) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("constraint preprocessing requires a DataFrame")
        if tuple(frame.columns) != self.original_columns_:
            missing = [
                column
                for column in self.original_columns_
                if column not in frame.columns
            ]
            extra = [
                column
                for column in frame.columns
                if column not in self.original_columns_
            ]
            raise ValueError(
                "constraint preprocessing columns changed; missing=%r, extra=%r"
                % (missing, extra)
            )

    def conditional_masks(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return antecedent masks used for structural implications."""

        self._check_fitted()
        self._check_frame(frame)
        return pd.DataFrame(
            {
                self.constraint_set[index].label: self.constraint_set[
                    index
                ].antecedent.evaluate(frame)
                for index in self.implication_indices_
            },
            index=frame.index,
        )

    def transform(
        self, frame: pd.DataFrame, *, validate: bool = True
    ) -> pd.DataFrame:
        """Convert original rows to gap and log-ratio coordinates."""

        self._check_fitted()
        self._check_frame(frame)
        if validate:
            self.constraint_set.validate(frame, raise_on_violation=True)
        source = frame.copy(deep=True)
        result = frame.copy(deep=True)

        for index in self.pair_indices_:
            constraint = self.constraint_set[index]
            small = pd.to_numeric(
                source[constraint.small], errors="raise"
            ).to_numpy(dtype=np.float64)
            big = pd.to_numeric(
                source[constraint.big], errors="raise"
            ).to_numpy(dtype=np.float64)
            gap = big - small - constraint.minimum_gap
            if bool(np.any(gap < -constraint.tolerance)):
                raise ValueError(
                    "cannot parameterize violated pair constraint %r"
                    % constraint.label
                )
            result[constraint.big] = np.maximum(gap, 0.0)

        for index in self.composition_indices_:
            constraint = self.constraint_set[index]
            members = constraint.member_matrix(source)
            totals = constraint.target(source)
            if constraint.total_column is None:
                result[constraint.members[0]] = totals
                coordinate_columns = constraint.members[1:]
            else:
                coordinate_columns = constraint.members[:-1]
                # This anchor is intentionally redundant.  It gives the table
                # transformer a finite channel while the decoder fixes the
                # reference logit to zero.
                result[constraint.members[-1]] = 0.0
            denominator = totals[:, None]
            fractions = np.divide(
                members,
                denominator,
                out=np.full_like(members, 1.0 / members.shape[1]),
                where=denominator > 0.0,
            )
            epsilon = self.log_ratio_epsilon
            ratios = np.log(fractions[:, :-1] + epsilon) - np.log(
                fractions[:, -1:] + epsilon
            )
            for position, column in enumerate(coordinate_columns):
                result[column] = ratios[:, position]
        return result.loc[:, list(self.original_columns_)]

    def inverse_transform(
        self,
        frame: pd.DataFrame,
        *,
        restore_dtypes: bool = True,
    ) -> pd.DataFrame:
        """Reconstruct original columns with structural validity by design."""

        self._check_fitted()
        self._check_frame(frame)
        result = frame.copy(deep=True)

        for index in self.composition_indices_:
            constraint = self.constraint_set[index]
            width = len(constraint.members)
            if constraint.total_column is None:
                totals = np.full(
                    len(result), float(constraint.total), dtype=np.float64
                )
                coordinate_columns = constraint.members[1:]
            else:
                totals = np.abs(
                    pd.to_numeric(
                        result[constraint.total_column], errors="raise"
                    ).to_numpy(dtype=np.float64)
                )
                result[constraint.total_column] = totals
                coordinate_columns = constraint.members[:-1]
            logits = np.zeros((len(result), width), dtype=np.float64)
            for position, column in enumerate(coordinate_columns):
                logits[:, position] = pd.to_numeric(
                    result[column], errors="raise"
                ).to_numpy(dtype=np.float64)
            members = _decode_composition(
                logits, totals, self.log_ratio_epsilon
            )
            for position, column in enumerate(constraint.members):
                result[column] = members[:, position]

        for index in self.pair_order_:
            constraint = self.constraint_set[index]
            small = pd.to_numeric(
                result[constraint.small], errors="raise"
            ).to_numpy(dtype=np.float64)
            signed_gap = pd.to_numeric(
                result[constraint.big], errors="raise"
            ).to_numpy(dtype=np.float64)
            # Folding, unlike max(gap, 0), does not create a point mass at the
            # equality boundary when a generator emits a negative coordinate.
            result[constraint.big] = (
                small + constraint.minimum_gap + np.abs(signed_gap)
            )

        for index in self.implication_indices_:
            constraint = self.constraint_set[index]
            active = constraint.antecedent.evaluate(result)
            _project_predicate(result, constraint.consequent, active)

        result = result.loc[:, list(self.original_columns_)]
        if restore_dtypes:
            result = self._restore_dtypes(result)
        return result

    def _restore_dtypes(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy(deep=False)
        composition_members = {
            column
            for index in self.composition_indices_
            for column in self.constraint_set[index].members
        }
        for column, dtype in self.original_dtypes_.items():
            try:
                if (
                    column not in composition_members
                    and (
                        dtype.startswith("int")
                        or dtype.startswith("uint")
                        or dtype.startswith("Int")
                        or dtype.startswith("UInt")
                    )
                ):
                    numeric = pd.to_numeric(
                        result[column], errors="raise"
                    ).to_numpy(dtype=np.float64)
                    result[column] = np.rint(numeric)
                result[column] = result[column].astype(dtype)
            except (TypeError, ValueError, OverflowError):
                # Projection may legitimately make an integer-valued affine
                # system fractional.  Preserve validity instead of silently
                # rounding it back out of the feasible region.
                pass
        return result

    def transform_schema(self, schema: TableSchema) -> TableSchema:
        """Return the schema consumed by the inner ``TableTransformer``."""

        self._check_fitted()
        if not isinstance(schema, TableSchema):
            raise TypeError("schema must be a TableSchema")
        changed_to_continuous = set()
        forced_nonnullable = {
            column
            for constraint in self.constraint_set
            if not constraint.allow_missing
            for column in constraint.columns
        }
        for index in self.pair_indices_:
            constraint = self.constraint_set[index]
            small = schema.column(constraint.small)
            big = schema.column(constraint.big)
            if not (
                small.kind == ColumnType.COUNT.value
                and big.kind == ColumnType.COUNT.value
                and constraint.minimum_gap == int(constraint.minimum_gap)
            ):
                changed_to_continuous.add(constraint.big)
        for index in self.composition_indices_:
            constraint = self.constraint_set[index]
            changed_to_continuous.update(constraint.members)

        columns = []
        for spec in schema:
            if spec.name not in changed_to_continuous:
                columns.append(
                    replace(spec, nullable=False)
                    if spec.name in forced_nonnullable and spec.nullable
                    else spec
                )
                continue
            columns.append(
                replace(
                    spec,
                    kind=ColumnType.CONTINUOUS.value,
                    nullable=False,
                    dtype="float64",
                    categories=(),
                    ordered=False,
                    constant_value=None,
                    timezone=None,
                )
            )
        return TableSchema(tuple(columns), version=schema.version)

    def transform_conditions(
        self,
        conditions: Mapping[Any, Any],
        rows: int,
        *,
        table_columns: Optional[Sequence[Any]] = None,
    ) -> Dict[Any, Any]:
        """Map safe original-space conditions to parameter coordinates.

        A dependent coordinate is safe only when enough original values were
        supplied to compute it exactly.  The resulting errors name the missing
        condition and explain how to make the request unambiguous.
        """

        self._check_fitted()
        rows = _positive_int("rows", rows)
        if not isinstance(conditions, Mapping):
            raise TypeError("conditions must be a mapping")
        columns = (
            self.original_columns_
            if table_columns is None
            else tuple(table_columns)
        )
        table_names = set(columns)
        original: Dict[Any, List[Any]] = {
            name: _expand_condition(value, rows)
            for name, value in conditions.items()
            if name in table_names
        }

        # First protect explicit values from fallback projection.  If only a
        # subset of an affine relation is fixed, Euclidean projection could
        # move that fixed value and violate the API's condition guarantee.
        for constraint in self.constraint_set:
            present = [
                column
                for column in constraint.columns
                if column in original
            ]
            if isinstance(constraint, (LinearInequality, LinearEquality)):
                if present and len(present) != len(constraint.columns):
                    missing = [
                        column
                        for column in constraint.columns
                        if column not in original
                    ]
                    raise ValueError(
                        "condition is unsafe for %r: fallback projection may "
                        "move conditioned columns; also condition %r or remove "
                        "conditions on %r"
                        % (constraint.label, missing, present)
                    )
                if present and not _constraint_values_valid(
                    constraint, original, rows
                ):
                    raise ValueError(
                        "explicit conditions violate linear constraint %r"
                        % constraint.label
                    )
            elif isinstance(constraint, BoundConstraint):
                if present and not _constraint_values_valid(
                    constraint, original, rows
                ):
                    raise ValueError(
                        "explicit condition for %r lies outside bounds %r"
                        % (constraint.column, constraint.label)
                    )
            elif isinstance(constraint, DomainConstraint):
                if present and not _constraint_values_valid(
                    constraint, original, rows
                ):
                    raise ValueError(
                        "explicit condition for %r is outside domain %r"
                        % (constraint.column, constraint.label)
                    )

        for constraint in self.constraint_set:
            if not isinstance(constraint, ImplicationConstraint):
                continue
            antecedent_known = constraint.antecedent.column in original
            consequent_known = constraint.consequent.column in original
            if not consequent_known:
                continue
            consequent_valid = _predicate_on_values(
                constraint.consequent,
                original[constraint.consequent.column],
            )
            if bool(np.all(consequent_valid)):
                continue
            if not antecedent_known:
                raise ValueError(
                    "condition on implication consequence %r is unsafe: %r "
                    "may activate and overwrite it; condition the antecedent "
                    "or choose a consequence-satisfying value"
                    % (
                        constraint.consequent.column,
                        constraint.antecedent.column,
                    )
                )
            antecedent_active = _predicate_on_values(
                constraint.antecedent,
                original[constraint.antecedent.column],
            )
            if bool(np.any(antecedent_active & ~consequent_valid)):
                raise ValueError(
                    "explicit conditions contradict implication %r"
                    % constraint.label
                )

        transformed: Dict[Any, Any] = {
            name: list(values) for name, values in original.items()
        }

        # Compositions are handled before pair gaps, matching transform().
        for index in self.composition_indices_:
            constraint = self.constraint_set[index]
            member_presence = [
                column in original for column in constraint.members
            ]
            if any(member_presence) and not all(member_presence):
                missing = [
                    column
                    for column in constraint.members
                    if column not in original
                ]
                raise ValueError(
                    "condition is unsafe for composition %r: individual "
                    "members use log-ratio coordinates; condition every member "
                    "(missing %r) or only the total"
                    % (constraint.label, missing)
                )
            if (
                constraint.total_column is not None
                and constraint.total_column in original
            ):
                totals = np.asarray(
                    original[constraint.total_column], dtype=np.float64
                )
                if bool(np.any(totals < 0.0)):
                    raise ValueError(
                        "composition total condition must be nonnegative"
                    )
            if not all(member_presence):
                continue
            members = np.column_stack(
                [
                    np.asarray(original[column], dtype=np.float64)
                    for column in constraint.members
                ]
            )
            if bool(np.any(members < 0.0)) or not bool(
                np.isfinite(members).all()
            ):
                raise ValueError(
                    "composition member conditions must be finite and "
                    "nonnegative"
                )
            sums = np.sum(members, axis=1)
            if constraint.total_column is None:
                totals = np.full(rows, float(constraint.total))
                if not bool(
                    np.allclose(
                        sums,
                        totals,
                        atol=constraint.tolerance,
                        rtol=0.0,
                    )
                ):
                    raise ValueError(
                        "composition conditions do not sum to %g for %r"
                        % (constraint.total, constraint.label)
                    )
                transformed[constraint.members[0]] = totals.tolist()
                coordinate_columns = constraint.members[1:]
            else:
                if constraint.total_column in original:
                    totals = np.asarray(
                        original[constraint.total_column], dtype=np.float64
                    )
                    if not bool(
                        np.allclose(
                            sums,
                            totals,
                            atol=constraint.tolerance,
                            rtol=0.0,
                        )
                    ):
                        raise ValueError(
                            "member and total conditions contradict %r"
                            % constraint.label
                        )
                else:
                    totals = sums
                    transformed[constraint.total_column] = totals.tolist()
                coordinate_columns = constraint.members[:-1]
                transformed[constraint.members[-1]] = [0.0] * rows
            fractions = np.divide(
                members,
                totals[:, None],
                out=np.full_like(members, 1.0 / members.shape[1]),
                where=totals[:, None] > 0.0,
            )
            epsilon = self.log_ratio_epsilon
            ratios = np.log(fractions[:, :-1] + epsilon) - np.log(
                fractions[:, -1:] + epsilon
            )
            for position, column in enumerate(coordinate_columns):
                transformed[column] = ratios[:, position].tolist()

        for index in self.pair_order_:
            constraint = self.constraint_set[index]
            if constraint.big not in original:
                continue
            if constraint.small not in original:
                raise ValueError(
                    "condition on dependent column %r is unsafe for %r; "
                    "also condition its base column %r so the nonnegative gap "
                    "can be computed"
                    % (constraint.big, constraint.label, constraint.small)
                )
            small = np.asarray(original[constraint.small], dtype=np.float64)
            big = np.asarray(original[constraint.big], dtype=np.float64)
            gap = big - small - constraint.minimum_gap
            if bool(np.any(gap < -constraint.tolerance)):
                raise ValueError(
                    "explicit conditions violate pair constraint %r"
                    % constraint.label
                )
            transformed[constraint.big] = np.maximum(gap, 0.0).tolist()

        # Preserve target/non-table conditions exactly.
        for name, value in conditions.items():
            if name not in table_names:
                transformed[name] = value
        return transformed

    def to_dict(self) -> Dict[str, Any]:
        self._check_fitted()
        return {
            "format": "ganify-structural-constraint-transformer",
            "version": self.FORMAT_VERSION,
            "constraints": self.constraint_set.to_dict(),
            "options": {
                "log_ratio_epsilon": self.log_ratio_epsilon,
            },
            "original_columns": [
                encode_json_value(column) for column in self.original_columns_
            ],
            "original_dtypes": [
                {
                    "column": encode_json_value(column),
                    "dtype": dtype,
                }
                for column, dtype in self.original_dtypes_.items()
            ],
            "numeric_scales": [
                {
                    "column": encode_json_value(column),
                    "scale": scale,
                }
                for column, scale in self.numeric_scales_.items()
            ],
            "pair_indices": list(self.pair_indices_),
            "pair_order": list(self.pair_order_),
            "composition_indices": list(self.composition_indices_),
            "implication_indices": list(self.implication_indices_),
            "structural_constraint_indices": list(
                self.structural_constraint_indices_
            ),
            "conditional_mask_rates": dict(
                self.conditional_mask_rates_
            ),
            "original_schema": (
                None
                if self.original_schema_ is None
                else self.original_schema_.to_dict()
            ),
            "reference_row": [
                {
                    "column": encode_json_value(column),
                    "value": encode_json_value(value),
                }
                for column, value in self.reference_row_.items()
            ],
            "reference_parameterized_row": [
                {
                    "column": encode_json_value(column),
                    "value": encode_json_value(value),
                }
                for column, value in self.reference_parameterized_row_.items()
            ],
        }

    state_dict = to_dict

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "StructuralConstraintTransformer":
        if not isinstance(payload, Mapping):
            raise TypeError(
                "structural constraint transformer state must be a mapping"
            )
        if (
            payload.get("format")
            != "ganify-structural-constraint-transformer"
        ):
            raise ValueError(
                "not a GANify structural constraint transformer state"
            )
        if int(payload.get("version", -1)) != cls.FORMAT_VERSION:
            raise ValueError(
                "unsupported structural constraint transformer format"
            )
        options = payload.get("options", {})
        instance = cls(
            ConstraintSet.from_dict(payload["constraints"]),
            log_ratio_epsilon=float(
                options.get("log_ratio_epsilon", 1e-8)
            ),
        )
        instance.original_columns_ = tuple(
            decode_json_value(column)
            for column in payload.get("original_columns", ())
        )
        instance.original_dtypes_ = {
            decode_json_value(item["column"]): str(item["dtype"])
            for item in payload.get("original_dtypes", ())
        }
        instance.numeric_scales_ = {
            decode_json_value(item["column"]): float(item["scale"])
            for item in payload.get("numeric_scales", ())
        }
        instance.pair_indices_ = tuple(
            int(value) for value in payload.get("pair_indices", ())
        )
        instance.pair_order_ = tuple(
            int(value) for value in payload.get("pair_order", ())
        )
        instance.composition_indices_ = tuple(
            int(value)
            for value in payload.get("composition_indices", ())
        )
        instance.implication_indices_ = tuple(
            int(value)
            for value in payload.get("implication_indices", ())
        )
        instance.structural_constraint_indices_ = tuple(
            int(value)
            for value in payload.get("structural_constraint_indices", ())
        )
        instance.conditional_mask_rates_ = {
            str(name): float(value)
            for name, value in payload.get(
                "conditional_mask_rates", {}
            ).items()
        }
        schema = payload.get("original_schema")
        instance.original_schema_ = (
            None if schema is None else TableSchema.from_dict(schema)
        )
        instance.reference_row_ = {
            decode_json_value(item["column"]): decode_json_value(item["value"])
            for item in payload.get("reference_row", ())
        }
        instance.reference_parameterized_row_ = {
            decode_json_value(item["column"]): decode_json_value(item["value"])
            for item in payload.get("reference_parameterized_row", ())
        }
        instance.fitted_ = True
        return instance

    from_state_dict = from_dict

    def to_json(self, *, indent: Optional[int] = None) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=None if indent else (",", ":"),
            indent=indent,
            allow_nan=False,
        )

    @classmethod
    def from_json(
        cls, value: Union[str, bytes]
    ) -> "StructuralConstraintTransformer":
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return cls.from_dict(json.loads(value))

    def save(self, path: Union[str, Path]) -> Path:
        destination = Path(path)
        if destination.suffix.lower() != ".json":
            destination.mkdir(parents=True, exist_ok=True)
            destination = destination / self.STATE_FILENAME
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            self.to_json(indent=2) + "\n", encoding="utf-8"
        )
        return destination

    @classmethod
    def load(
        cls, path: Union[str, Path]
    ) -> "StructuralConstraintTransformer":
        source = Path(path)
        if source.is_dir():
            source = source / cls.STATE_FILENAME
        return cls.from_json(source.read_text(encoding="utf-8"))


class ConstrainedTableTransformer:
    """Reversible composition around a fitted :class:`TableTransformer`.

    The inner transformer owns neural matrix metadata.  This wrapper presents
    the user-facing original table: forward calls parameterize constraints
    before numeric encoding and inverse calls decode numeric values before
    structural reconstruction.
    """

    FORMAT_VERSION = 1
    STATE_FILENAME = "constrained_table_transformer.json"

    def __init__(
        self,
        structural: StructuralConstraintTransformer,
        table: TableTransformer,
    ) -> None:
        if not isinstance(structural, StructuralConstraintTransformer):
            raise TypeError(
                "structural must be a StructuralConstraintTransformer"
            )
        if not getattr(structural, "fitted_", False):
            raise RuntimeError("structural constraint transformer is not fitted")
        if not isinstance(table, TableTransformer):
            raise TypeError("table must be a fitted TableTransformer")
        if not getattr(table, "fitted_", False):
            raise RuntimeError("inner TableTransformer is not fitted")
        if tuple(table.schema_.names) != structural.original_columns_:
            raise ValueError(
                "inner and structural preprocessors have different columns"
            )
        self.structural = structural
        self.table = table
        self.fitted_ = True

    @property
    def constraints(self) -> ConstraintSet:
        return self.structural.constraint_set

    @property
    def schema_(self) -> Optional[TableSchema]:
        return self.structural.original_schema_

    @property
    def parameter_schema_(self) -> TableSchema:
        return self.table.schema_

    def transform(self, frame: pd.DataFrame):
        parameterized = self.structural.transform(frame)
        return self.table.transform(parameterized)

    def fit_transform(self, frame: pd.DataFrame):
        # Both owned transforms are already fitted; this spelling mirrors the
        # sklearn/TableTransformer API without silently refitting either state.
        return self.transform(frame)

    def inverse_transform(
        self,
        values: Any,
        *,
        index: Optional[Iterable[Any]] = None,
    ) -> pd.DataFrame:
        parameterized = self.table.inverse_transform(values, index=index)
        return self.structural.inverse_transform(parameterized)

    def inverse_transform_with_parameterized(
        self,
        values: Any,
        *,
        index: Optional[Iterable[Any]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return ``(reconstructed, raw_parameterized)`` for auditing."""

        parameterized = self.table.inverse_transform(values, index=index)
        reconstructed = self.structural.inverse_transform(parameterized)
        return reconstructed, parameterized

    inverse = inverse_transform

    @property
    def head_metadata(self):
        return self.table.head_metadata

    @property
    def output_dim_(self) -> int:
        return self.table.output_dim_

    @property
    def column_layout_(self):
        return self.table.column_layout_

    @property
    def output_slices(self):
        return self.table.output_slices

    def get_feature_names_out(self) -> np.ndarray:
        return self.table.get_feature_names_out()

    def get_metadata(self) -> Mapping[str, Any]:
        metadata = dict(self.table.get_metadata())
        metadata["constraint_parameterization"] = {
            "structural_constraints": [
                self.structural.constraint_set[index].label
                for index in self.structural.structural_constraint_indices_
            ],
            "original_schema": (
                None
                if self.structural.original_schema_ is None
                else self.structural.original_schema_.to_dict()
            ),
        }
        return metadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": "ganify-constrained-table-transformer",
            "version": self.FORMAT_VERSION,
            "structural": self.structural.to_dict(),
            "table": self.table.to_dict(),
        }

    state_dict = to_dict

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ConstrainedTableTransformer":
        if not isinstance(payload, Mapping):
            raise TypeError(
                "constrained table transformer state must be a mapping"
            )
        if payload.get("format") != "ganify-constrained-table-transformer":
            raise ValueError(
                "not a GANify constrained table transformer state"
            )
        if int(payload.get("version", -1)) != cls.FORMAT_VERSION:
            raise ValueError(
                "unsupported constrained table transformer format"
            )
        return cls(
            StructuralConstraintTransformer.from_dict(
                payload["structural"]
            ),
            TableTransformer.from_dict(payload["table"]),
        )

    from_state_dict = from_dict

    def to_json(self, *, indent: Optional[int] = None) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=None if indent else (",", ":"),
            indent=indent,
            allow_nan=False,
        )

    @classmethod
    def from_json(
        cls, value: Union[str, bytes]
    ) -> "ConstrainedTableTransformer":
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return cls.from_dict(json.loads(value))

    def save(self, path: Union[str, Path]) -> Path:
        destination = Path(path)
        if destination.suffix.lower() != ".json":
            destination.mkdir(parents=True, exist_ok=True)
            destination = destination / self.STATE_FILENAME
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            self.to_json(indent=2) + "\n", encoding="utf-8"
        )
        return destination

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ConstrainedTableTransformer":
        source = Path(path)
        if source.is_dir():
            source = source / cls.STATE_FILENAME
        return cls.from_json(source.read_text(encoding="utf-8"))


def constraint_audit(
    constraints: ConstraintSet,
    before: pd.DataFrame,
    after: pd.DataFrame,
    *,
    raw: Optional[pd.DataFrame] = None,
    numeric_scales: Optional[Mapping[Any, float]] = None,
    projection: Optional[ProjectionAudit] = None,
    boundary_tolerance: float = 1e-8,
) -> Dict[str, Any]:
    """Build the stable per-sample audit recorded by the conditional engine."""

    if not isinstance(constraints, ConstraintSet):
        constraints = ConstraintSet(constraints)
    if len(before) != len(after):
        raise ValueError("constraint audit frames must have equal row counts")
    raw_frame = before if raw is None else raw
    if len(raw_frame) != len(before):
        raise ValueError("raw constraint audit frame has the wrong row count")
    raw_report = constraints.violation_report(raw_frame)
    pre_report = constraints.violation_report(before)
    post_report = constraints.violation_report(after)
    raw_joint = float(
        raw_report.loc[
            raw_report["constraint"] == "__all__", "violation_rate"
        ].iloc[0]
    )
    pre_joint = float(
        pre_report.loc[
            pre_report["constraint"] == "__all__", "violation_rate"
        ].iloc[0]
    )
    post_joint = float(
        post_report.loc[
            post_report["constraint"] == "__all__", "violation_rate"
        ].iloc[0]
    )
    raw_rates = {
        str(row.constraint): float(row.violation_rate)
        for row in raw_report.itertuples(index=False)
        if row.constraint != "__all__"
    }
    pre_rates = {
        str(row.constraint): float(row.violation_rate)
        for row in pre_report.itertuples(index=False)
        if row.constraint != "__all__"
    }
    post_rates = {
        str(row.constraint): float(row.violation_rate)
        for row in post_report.itertuples(index=False)
        if row.constraint != "__all__"
    }
    raw_margins = constraints.margin_summaries(
        raw_frame, boundary_tolerance=boundary_tolerance
    )
    pre_margins = constraints.margin_summaries(
        before, boundary_tolerance=boundary_tolerance
    )
    post_margins = constraints.margin_summaries(
        after, boundary_tolerance=boundary_tolerance
    )
    changed = _row_changes(before, after)
    scales = {} if numeric_scales is None else dict(numeric_scales)
    involved = tuple(
        dict.fromkeys(
            column
            for constraint in constraints
            for column in constraint.columns
            if column in before.columns
        )
    )
    squared = np.zeros(len(before), dtype=np.float64)
    dimensions = 0
    for column in involved:
        left = before[column]
        right = after[column]
        try:
            left_numeric = pd.to_numeric(
                left, errors="raise"
            ).to_numpy(dtype=np.float64)
            right_numeric = pd.to_numeric(
                right, errors="raise"
            ).to_numpy(dtype=np.float64)
            scale = max(float(scales.get(column, 1.0)), 1e-12)
            delta = np.nan_to_num(
                (right_numeric - left_numeric) / scale,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            squared += np.square(delta)
        except (TypeError, ValueError):
            equal = left.eq(right) | (left.isna() & right.isna())
            squared += (~equal).to_numpy(dtype=np.float64)
        dimensions += 1
    normalized = (
        0.0
        if not len(before) or not dimensions
        else float(np.mean(np.sqrt(squared / float(dimensions))))
    )
    boundary_values = [
        float(summary["boundary_mass"])
        for summary in post_margins.values()
    ]
    equality_values = [
        float(post_margins[constraint.label]["boundary_mass"])
        for constraint in constraints
        if constraint.is_equality
    ]
    boundary_mass = (
        0.0 if not boundary_values else float(np.mean(boundary_values))
    )
    equality_mass = (
        0.0 if not equality_values else float(np.mean(equality_values))
    )
    projection_dict = (
        {
            "method": "none",
            "iterations": 0,
            "converged": post_joint == 0.0,
            "pre_violation_rate": pre_joint,
            "post_violation_rate": post_joint,
            "changed_row_fraction": (
                0.0 if not len(before) else float(np.mean(changed))
            ),
            "steps": [],
        }
        if projection is None
        else projection.to_dict()
    )
    return {
        "raw_violation_rate": raw_joint,
        "pre_projection_violation_rate": pre_joint,
        "post_projection_violation_rate": post_joint,
        "post_violation_rate": post_joint,
        "post_rate": post_joint,
        "raw_constraint_violation_rates": raw_rates,
        "pre_constraint_violation_rates": pre_rates,
        "post_constraint_violation_rates": post_rates,
        "changed_row_fraction": (
            0.0 if not len(before) else float(np.mean(changed))
        ),
        "normalized_intervention_magnitude": normalized,
        "equality_mass": equality_mass,
        "boundary_mass": boundary_mass,
        "equality_boundary_mass": max(equality_mass, boundary_mass),
        "raw_constraint_margins": raw_margins,
        "pre_constraint_margins": pre_margins,
        "constraint_margins": post_margins,
        "constraint_margin_summaries": post_margins,
        "margin_distributions": post_margins,
        "equality_boundary_masses": {
            "equality": equality_mass,
            "boundary": boundary_mass,
        },
        "projection": projection_dict,
    }


def empty_constraint_audit() -> Dict[str, Any]:
    """Return the audit shape used when an engine has no constraints."""

    return {
        "raw_violation_rate": 0.0,
        "pre_projection_violation_rate": 0.0,
        "post_projection_violation_rate": 0.0,
        "post_violation_rate": 0.0,
        "post_rate": 0.0,
        "raw_constraint_violation_rates": {},
        "pre_constraint_violation_rates": {},
        "post_constraint_violation_rates": {},
        "changed_row_fraction": 0.0,
        "normalized_intervention_magnitude": 0.0,
        "equality_mass": 0.0,
        "boundary_mass": 0.0,
        "equality_boundary_mass": 0.0,
        "raw_constraint_margins": {},
        "pre_constraint_margins": {},
        "constraint_margins": {},
        "constraint_margin_summaries": {},
        "margin_distributions": {},
        "equality_boundary_masses": {
            "equality": 0.0,
            "boundary": 0.0,
        },
        "projection": {
            "method": "none",
            "iterations": 0,
            "converged": True,
            "pre_violation_rate": 0.0,
            "post_violation_rate": 0.0,
            "changed_row_fraction": 0.0,
            "steps": [],
        },
    }


# Preprocessor-oriented aliases kept intentionally explicit.
ConstraintTransformer = StructuralConstraintTransformer
StructuralPreprocessor = StructuralConstraintTransformer
ConstraintPreprocessor = ConstrainedTableTransformer


__all__ = [
    "ConstrainedTableTransformer",
    "ConstraintPreprocessor",
    "ConstraintTransformer",
    "StructuralConstraintTransformer",
    "StructuralPreprocessor",
    "constraint_audit",
    "empty_constraint_audit",
]
