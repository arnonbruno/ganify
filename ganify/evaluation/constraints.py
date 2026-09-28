"""Declarative and callable table-constraint validity metrics."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol, Sequence, Union

import numpy as np
import pandas as pd

from ._utils import as_frame


class Constraint(Protocol):
    """Protocol implemented by first-class evaluation constraints."""

    name: str

    def evaluate(self, data: pd.DataFrame) -> np.ndarray:
        """Return one boolean validity value per row."""


@dataclass(frozen=True)
class RangeConstraint:
    """Require a column to stay inside optional inclusive bounds."""

    column: object
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    allow_missing: bool = False
    name: str = ""

    def evaluate(self, data: pd.DataFrame) -> np.ndarray:
        """Evaluate the range constraint for every row."""

        if self.column not in data.columns:
            raise ValueError("constraint column %r is missing" % self.column)
        values = pd.to_numeric(data[self.column], errors="coerce")
        numeric = values.to_numpy(dtype=float)
        valid = np.ones(len(data), dtype=bool)
        if self.minimum is not None:
            valid &= numeric >= self.minimum
        if self.maximum is not None:
            valid &= numeric <= self.maximum
        missing = ~np.isfinite(numeric)
        valid[missing] = self.allow_missing
        return valid

    @property
    def label(self) -> str:
        """Return a stable human-readable name."""

        return self.name or "range:%s" % self.column


@dataclass(frozen=True)
class InequalityConstraint:
    """Compare two columns, or one column and a scalar, row by row."""

    left: object
    operator: str
    right: object
    tolerance: float = 0.0
    allow_missing: bool = False
    name: str = ""

    def evaluate(self, data: pd.DataFrame) -> np.ndarray:
        """Evaluate the inequality for every row."""

        if self.left not in data.columns:
            raise ValueError("constraint column %r is missing" % self.left)
        left = pd.to_numeric(data[self.left], errors="coerce").to_numpy(dtype=float)
        if self.right in data.columns:
            right = pd.to_numeric(data[self.right], errors="coerce").to_numpy(
                dtype=float
            )
        elif np.isscalar(self.right):
            right = np.full(len(data), float(self.right), dtype=float)
        else:
            raise ValueError("constraint right operand %r is missing" % self.right)
        valid = _compare(left, right, self.operator, self.tolerance)
        missing = ~np.isfinite(left) | ~np.isfinite(right)
        valid[missing] = self.allow_missing
        return valid

    @property
    def label(self) -> str:
        """Return a stable human-readable name."""

        return self.name or "%s%s%s" % (self.left, self.operator, self.right)


@dataclass(frozen=True)
class AllowedValuesConstraint:
    """Require a categorical column to belong to a fixed support."""

    column: object
    values: Sequence[Any]
    allow_missing: bool = False
    name: str = ""

    def evaluate(self, data: pd.DataFrame) -> np.ndarray:
        """Evaluate support membership for every row."""

        if self.column not in data.columns:
            raise ValueError("constraint column %r is missing" % self.column)
        series = data[self.column]
        valid = series.isin(list(self.values)).to_numpy(dtype=bool)
        if self.allow_missing:
            valid |= series.isna().to_numpy()
        return valid

    @property
    def label(self) -> str:
        """Return a stable human-readable name."""

        return self.name or "allowed:%s" % self.column


@dataclass(frozen=True)
class CallableConstraint:
    """Name a user-supplied vectorized constraint function."""

    name: str
    function: Callable[[pd.DataFrame], object]

    def evaluate(self, data: pd.DataFrame) -> np.ndarray:
        """Evaluate and validate the callable result."""

        return _boolean_vector(self.function(data), len(data), self.name)


ConstraintLike = Union[Constraint, Mapping[str, Any], Callable[[pd.DataFrame], object]]


def _compare(
    left: np.ndarray, right: np.ndarray, operation: str, tolerance: float
) -> np.ndarray:
    if tolerance < 0.0:
        raise ValueError("constraint tolerance must be non-negative")
    operations = {
        "<=": lambda a, b: a <= b + tolerance,
        "<": lambda a, b: a < b + tolerance,
        ">=": lambda a, b: a + tolerance >= b,
        ">": lambda a, b: a + tolerance > b,
        "==": lambda a, b: np.abs(a - b) <= tolerance,
        "!=": lambda a, b: np.abs(a - b) > tolerance,
    }
    if operation not in operations:
        raise ValueError("unsupported constraint operator %r" % operation)
    return np.asarray(operations[operation](left, right), dtype=bool)


def _boolean_vector(values: object, n_rows: int, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 0:
        array = np.full(n_rows, bool(array), dtype=bool)
    if array.shape != (n_rows,):
        raise ValueError(
            "constraint %r returned shape %r, expected (%d,)"
            % (name, array.shape, n_rows)
        )
    if array.dtype != bool:
        if bool(pd.isna(array).any()):
            raise ValueError("constraint %r returned missing validity values" % name)
        array = array.astype(bool)
    return array


def constraint_from_spec(spec: Mapping[str, Any]) -> Constraint:
    """Construct a first-class constraint from a JSON/YAML mapping."""

    kind = str(spec.get("type", "")).lower()
    if kind == "range":
        return RangeConstraint(
            column=spec["column"],
            minimum=spec.get("minimum", spec.get("min")),
            maximum=spec.get("maximum", spec.get("max")),
            allow_missing=bool(spec.get("allow_missing", False)),
            name=str(spec.get("name", "")),
        )
    if kind == "inequality":
        return InequalityConstraint(
            left=spec["left"],
            operator=str(spec.get("operator", "<=")),
            right=spec["right"],
            tolerance=float(spec.get("tolerance", 0.0)),
            allow_missing=bool(spec.get("allow_missing", False)),
            name=str(spec.get("name", "")),
        )
    if kind in {"allowed", "allowed_values", "domain"}:
        return AllowedValuesConstraint(
            column=spec["column"],
            values=tuple(spec["values"]),
            allow_missing=bool(spec.get("allow_missing", False)),
            name=str(spec.get("name", "")),
        )
    raise ValueError("unsupported constraint type %r" % kind)


def _resolve_constraint(item: ConstraintLike, index: int) -> Constraint:
    if isinstance(item, Mapping):
        return constraint_from_spec(item)
    if callable(item) and not hasattr(item, "evaluate"):
        return CallableConstraint(
            name=getattr(item, "__name__", "constraint_%d" % index),
            function=item,
        )
    if hasattr(item, "evaluate"):
        return item  # type: ignore[return-value]
    raise TypeError("constraint %d does not implement evaluate()" % index)


def constraint_validity(
    data: object, constraints: Iterable[ConstraintLike]
) -> pd.DataFrame:
    """Return per-constraint and all-constraints validity rates."""

    frame = as_frame(data)
    resolved = [
        _resolve_constraint(item, index) for index, item in enumerate(constraints)
    ]
    if not resolved:
        raise ValueError("at least one constraint is required")
    masks = []
    rows = []
    for index, constraint in enumerate(resolved):
        raw = constraint.evaluate(frame)
        name = getattr(
            constraint,
            "label",
            getattr(constraint, "name", "constraint_%d" % index),
        )
        valid = _boolean_vector(raw, len(frame), str(name))
        masks.append(valid)
        valid_count = int(valid.sum())
        rows.append(
            {
                "constraint": str(name),
                "valid_count": valid_count,
                "total_count": len(frame),
                "valid_rate": valid_count / float(len(frame)),
                "violation_rate": 1.0 - valid_count / float(len(frame)),
            }
        )
    all_valid = np.logical_and.reduce(masks)
    valid_count = int(all_valid.sum())
    rows.append(
        {
            "constraint": "__all__",
            "valid_count": valid_count,
            "total_count": len(frame),
            "valid_rate": valid_count / float(len(frame)),
            "violation_rate": 1.0 - valid_count / float(len(frame)),
        }
    )
    return pd.DataFrame(rows)


evaluate_constraints = constraint_validity


__all__ = [
    "AllowedValuesConstraint",
    "CallableConstraint",
    "Constraint",
    "ConstraintLike",
    "InequalityConstraint",
    "RangeConstraint",
    "constraint_from_spec",
    "constraint_validity",
    "evaluate_constraints",
]
