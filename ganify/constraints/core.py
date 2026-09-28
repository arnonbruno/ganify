"""Typed table constraints, reporting, and auditable Euclidean projection.

The classes in this module are deliberately independent from TensorFlow.  A
constraint owns three small pieces of behavior: row-wise validity, a signed
margin (positive is feasible), and a JSON-safe representation.  Projection is
implemented by :class:`ConstraintSet` and uses cyclic orthogonal projections
for affine half-spaces and hyperplanes.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ganify.schema import decode_json_value, encode_json_value


def _finite_float(name: str, value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("%s must be finite" % name)
    return result


def _nonnegative_float(name: str, value: Any) -> float:
    result = _finite_float(name, value)
    if result < 0.0:
        raise ValueError("%s must be non-negative" % name)
    return result


def _name(value: Any) -> Any:
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError("constraint column names must be hashable") from exc
    encode_json_value(value)
    return value


def _require_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("constraints require a pandas DataFrame")
    if frame.columns.has_duplicates:
        raise ValueError("constraint dataframe contains duplicate columns")
    return frame


def _require_columns(frame: pd.DataFrame, columns: Iterable[Any]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError("constraints reference missing columns: %r" % missing)


def _numeric(frame: pd.DataFrame, column: Any) -> np.ndarray:
    _require_columns(frame, (column,))
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(
        dtype=np.float64
    )
    original_missing = frame[column].isna().to_numpy(dtype=bool)
    invalid = ~np.isfinite(values) & ~original_missing
    if bool(invalid.any()):
        raise ValueError(
            "constraint column %r contains non-numeric or infinite values"
            % (column,)
        )
    return values


def _missing_mask(frame: pd.DataFrame, columns: Iterable[Any]) -> np.ndarray:
    columns = tuple(columns)
    if not columns:
        return np.zeros(len(frame), dtype=bool)
    return np.logical_or.reduce(
        [frame[column].isna().to_numpy(dtype=bool) for column in columns]
    )


def _valid_with_missing(
    margin: np.ndarray, missing: np.ndarray, allow_missing: bool
) -> np.ndarray:
    valid = np.asarray(margin >= 0.0, dtype=bool)
    valid[missing] = bool(allow_missing)
    return valid


def _label(default: str, configured: str) -> str:
    return configured if configured else default


class TableConstraint:
    """Runtime base class shared by all serializable table constraints."""

    type_name = "constraint"
    name = ""
    tolerance = 0.0
    allow_missing = False

    @property
    def columns(self) -> Tuple[Any, ...]:
        raise NotImplementedError

    @property
    def label(self) -> str:
        return _label(self.type_name, str(getattr(self, "name", "")))

    @property
    def is_equality(self) -> bool:
        return False

    def margin(self, frame: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def evaluate(self, frame: pd.DataFrame) -> np.ndarray:
        frame = _require_frame(frame)
        margins = np.asarray(self.margin(frame), dtype=np.float64)
        if margins.shape != (len(frame),):
            raise RuntimeError(
                "constraint %r emitted invalid margin shape %r"
                % (self.label, margins.shape)
            )
        missing = _missing_mask(frame, self.columns)
        return _valid_with_missing(
            margins, missing, bool(getattr(self, "allow_missing", False))
        )

    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    def to_json(self, *, indent: Optional[int] = None) -> str:
        """Serialize this one constraint without a wrapping ``ConstraintSet``."""

        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=None if indent else (",", ":"),
            indent=indent,
            allow_nan=False,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TableConstraint":
        """Restore one typed constraint, checking a concrete class request."""

        constraint = constraint_from_dict(payload)
        if cls is not TableConstraint and not isinstance(constraint, cls):
            raise ValueError(
                "constraint state contains %s, not %s"
                % (type(constraint).__name__, cls.__name__)
            )
        return constraint

    @classmethod
    def from_json(cls, value: Union[str, bytes]) -> "TableConstraint":
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True, init=False)
class BoundConstraint(TableConstraint):
    """Require a numeric column to lie inside inclusive optional bounds."""

    column: Any
    lower: Optional[float] = None
    upper: Optional[float] = None
    allow_missing: bool = False
    tolerance: float = 1e-9
    name: str = ""

    type_name = "bounds"

    def __init__(
        self,
        column: Any,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        *,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        allow_missing: bool = False,
        tolerance: float = 1e-9,
        name: str = "",
    ) -> None:
        lower_aliases = [
            value for value in (minimum, min_value) if value is not None
        ]
        upper_aliases = [
            value for value in (maximum, max_value) if value is not None
        ]
        if lower is not None and lower_aliases:
            raise ValueError("pass one lower-bound spelling")
        if upper is not None and upper_aliases:
            raise ValueError("pass one upper-bound spelling")
        if len(lower_aliases) > 1 or len(upper_aliases) > 1:
            raise ValueError("pass one bound alias")
        if lower_aliases:
            lower = lower_aliases[0]
        if upper_aliases:
            upper = upper_aliases[0]
        object.__setattr__(self, "column", _name(column))
        lower = (
            None if lower is None else _finite_float("lower", lower)
        )
        upper = (
            None if upper is None else _finite_float("upper", upper)
        )
        if lower is None and upper is None:
            raise ValueError("a bound constraint needs lower and/or upper")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("lower bound cannot exceed upper bound")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(
            self, "tolerance", _nonnegative_float("tolerance", tolerance)
        )
        object.__setattr__(self, "allow_missing", bool(allow_missing))
        object.__setattr__(self, "name", str(name))

    @property
    def minimum(self) -> Optional[float]:
        return self.lower

    @property
    def maximum(self) -> Optional[float]:
        return self.upper

    @property
    def columns(self) -> Tuple[Any, ...]:
        return (self.column,)

    @property
    def label(self) -> str:
        return _label("bounds:%s" % self.column, self.name)

    def margin(self, frame: pd.DataFrame) -> np.ndarray:
        values = _numeric(_require_frame(frame), self.column)
        pieces: List[np.ndarray] = []
        if self.lower is not None:
            pieces.append(values - self.lower + self.tolerance)
        if self.upper is not None:
            pieces.append(self.upper - values + self.tolerance)
        return pieces[0] if len(pieces) == 1 else np.minimum.reduce(pieces)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type_name,
            "column": encode_json_value(self.column),
            "lower": self.lower,
            "upper": self.upper,
            "allow_missing": self.allow_missing,
            "tolerance": self.tolerance,
            "name": self.name,
        }


@dataclass(frozen=True, init=False)
class DomainConstraint(TableConstraint):
    """Require a column to belong to a finite, serialized domain."""

    column: Any
    values: Tuple[Any, ...]
    allow_missing: bool = False
    name: str = ""

    type_name = "domain"
    tolerance = 0.0

    def __init__(
        self,
        column: Any,
        values: Optional[Sequence[Any]] = None,
        *,
        allowed_values: Optional[Sequence[Any]] = None,
        domain: Optional[Sequence[Any]] = None,
        allow_missing: bool = False,
        name: str = "",
    ) -> None:
        aliases = [
            value
            for value in (allowed_values, domain)
            if value is not None
        ]
        if values is not None and aliases:
            raise ValueError("pass values, allowed_values, or domain once")
        if len(aliases) > 1:
            raise ValueError("pass one domain alias")
        if values is None:
            values = aliases[0] if aliases else None
        if values is None:
            raise ValueError("a domain constraint needs values")
        object.__setattr__(self, "column", _name(column))
        values = tuple(values)
        if not values:
            raise ValueError("a domain constraint needs at least one value")
        for value in values:
            encode_json_value(value)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "allow_missing", bool(allow_missing))
        object.__setattr__(self, "name", str(name))

    @property
    def allowed_values(self) -> Tuple[Any, ...]:
        return self.values

    @property
    def columns(self) -> Tuple[Any, ...]:
        return (self.column,)

    @property
    def label(self) -> str:
        return _label("domain:%s" % self.column, self.name)

    def margin(self, frame: pd.DataFrame) -> np.ndarray:
        frame = _require_frame(frame)
        _require_columns(frame, self.columns)
        valid = frame[self.column].isin(list(self.values)).to_numpy(
            dtype=bool, copy=True
        )
        missing = frame[self.column].isna().to_numpy(dtype=bool)
        valid[missing] = self.allow_missing
        return np.where(valid, 1.0, -1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type_name,
            "column": encode_json_value(self.column),
            "values": [encode_json_value(value) for value in self.values],
            "allow_missing": self.allow_missing,
            "name": self.name,
        }


@dataclass(frozen=True, init=False)
class PairInequality(TableConstraint):
    """Require ``small + minimum_gap <= big`` for every observed row."""

    small: Any
    big: Any
    minimum_gap: float
    allow_missing: bool
    tolerance: float
    name: str

    type_name = "pair_inequality"

    def __init__(
        self,
        small: Any = None,
        big: Any = None,
        *,
        left: Any = None,
        right: Any = None,
        minimum_gap: float = 0.0,
        allow_missing: bool = False,
        tolerance: float = 1e-9,
        name: str = "",
    ) -> None:
        if left is not None:
            if small is not None:
                raise ValueError("pass small or left, not both")
            small = left
        if right is not None:
            if big is not None:
                raise ValueError("pass big or right, not both")
            big = right
        if small is None or big is None:
            raise ValueError("pair inequality requires small and big columns")
        small = _name(small)
        big = _name(big)
        if small == big:
            raise ValueError("pair inequality columns must be different")
        object.__setattr__(self, "small", small)
        object.__setattr__(self, "big", big)
        object.__setattr__(
            self,
            "minimum_gap",
            _nonnegative_float("minimum_gap", minimum_gap),
        )
        object.__setattr__(self, "allow_missing", bool(allow_missing))
        object.__setattr__(
            self, "tolerance", _nonnegative_float("tolerance", tolerance)
        )
        object.__setattr__(self, "name", str(name))

    @property
    def left(self) -> Any:
        return self.small

    @property
    def right(self) -> Any:
        return self.big

    @property
    def columns(self) -> Tuple[Any, ...]:
        return (self.small, self.big)

    @property
    def label(self) -> str:
        return _label("%s<=%s" % (self.small, self.big), self.name)

    def margin(self, frame: pd.DataFrame) -> np.ndarray:
        frame = _require_frame(frame)
        return (
            _numeric(frame, self.big)
            - _numeric(frame, self.small)
            - self.minimum_gap
            + self.tolerance
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type_name,
            "small": encode_json_value(self.small),
            "big": encode_json_value(self.big),
            "minimum_gap": self.minimum_gap,
            "allow_missing": self.allow_missing,
            "tolerance": self.tolerance,
            "name": self.name,
        }


def _normalise_coefficients(
    coefficients: Union[Mapping[Any, Any], Sequence[Tuple[Any, Any]]]
) -> Tuple[Tuple[Any, float], ...]:
    items = (
        list(coefficients.items())
        if isinstance(coefficients, Mapping)
        else list(coefficients)
    )
    if not items:
        raise ValueError("linear constraints need at least one coefficient")
    result: List[Tuple[Any, float]] = []
    seen = set()
    for column, coefficient in items:
        column = _name(column)
        if column in seen:
            raise ValueError(
                "linear constraint repeats column %r" % (column,)
            )
        seen.add(column)
        value = _finite_float("coefficient", coefficient)
        if value != 0.0:
            result.append((column, value))
    if not result:
        raise ValueError("linear constraint coefficients cannot all be zero")
    return tuple(result)


@dataclass(frozen=True, init=False)
class LinearInequality(TableConstraint):
    """An affine half-space ``a @ x <= rhs`` or ``a @ x >= rhs``."""

    _coefficients: Tuple[Tuple[Any, float], ...]
    rhs: float
    operator: str
    allow_missing: bool
    tolerance: float
    name: str

    type_name = "linear_inequality"

    def __init__(
        self,
        coefficients: Union[Mapping[Any, Any], Sequence[Tuple[Any, Any]]],
        rhs: Any = None,
        operator: Any = "<=",
        *,
        bound: Any = None,
        allow_missing: bool = False,
        tolerance: float = 1e-9,
        name: str = "",
    ) -> None:
        # Also accept the common positional form (coefficients, "<=", rhs).
        if isinstance(rhs, str) and rhs in {"<=", ">="}:
            rhs, operator = operator, rhs
        if bound is not None:
            if rhs is not None:
                raise ValueError("pass rhs or bound, not both")
            rhs = bound
        if rhs is None:
            raise ValueError("linear inequality requires rhs")
        operation = str(operator)
        if operation not in {"<=", ">="}:
            raise ValueError("linear inequality operator must be '<=' or '>='")
        object.__setattr__(
            self, "_coefficients", _normalise_coefficients(coefficients)
        )
        object.__setattr__(self, "rhs", _finite_float("rhs", rhs))
        object.__setattr__(self, "operator", operation)
        object.__setattr__(self, "allow_missing", bool(allow_missing))
        object.__setattr__(
            self, "tolerance", _nonnegative_float("tolerance", tolerance)
        )
        object.__setattr__(self, "name", str(name))

    @property
    def coefficients(self) -> Dict[Any, float]:
        return dict(self._coefficients)

    @property
    def columns(self) -> Tuple[Any, ...]:
        return tuple(column for column, _ in self._coefficients)

    @property
    def label(self) -> str:
        expression = "+".join(
            "%g*%s" % (coefficient, column)
            for column, coefficient in self._coefficients
        )
        return _label(
            "linear:%s%s%g" % (expression, self.operator, self.rhs),
            self.name,
        )

    def value(self, frame: pd.DataFrame) -> np.ndarray:
        frame = _require_frame(frame)
        result = np.zeros(len(frame), dtype=np.float64)
        for column, coefficient in self._coefficients:
            result += coefficient * _numeric(frame, column)
        return result

    def margin(self, frame: pd.DataFrame) -> np.ndarray:
        value = self.value(frame)
        if self.operator == "<=":
            return self.rhs - value + self.tolerance
        return value - self.rhs + self.tolerance

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type_name,
            "coefficients": [
                {
                    "column": encode_json_value(column),
                    "coefficient": coefficient,
                }
                for column, coefficient in self._coefficients
            ],
            "rhs": self.rhs,
            "operator": self.operator,
            "allow_missing": self.allow_missing,
            "tolerance": self.tolerance,
            "name": self.name,
        }


@dataclass(frozen=True, init=False)
class LinearEquality(TableConstraint):
    """An affine hyperplane ``a @ x == rhs``."""

    _coefficients: Tuple[Tuple[Any, float], ...]
    rhs: float
    allow_missing: bool
    tolerance: float
    name: str

    type_name = "linear_equality"

    def __init__(
        self,
        coefficients: Union[Mapping[Any, Any], Sequence[Tuple[Any, Any]]],
        rhs: Any = None,
        *,
        value: Any = None,
        allow_missing: bool = False,
        tolerance: float = 1e-8,
        name: str = "",
    ) -> None:
        if value is not None:
            if rhs is not None:
                raise ValueError("pass rhs or value, not both")
            rhs = value
        if rhs is None:
            raise ValueError("linear equality requires rhs")
        object.__setattr__(
            self, "_coefficients", _normalise_coefficients(coefficients)
        )
        object.__setattr__(self, "rhs", _finite_float("rhs", rhs))
        object.__setattr__(self, "allow_missing", bool(allow_missing))
        object.__setattr__(
            self, "tolerance", _nonnegative_float("tolerance", tolerance)
        )
        object.__setattr__(self, "name", str(name))

    @property
    def coefficients(self) -> Dict[Any, float]:
        return dict(self._coefficients)

    @property
    def columns(self) -> Tuple[Any, ...]:
        return tuple(column for column, _ in self._coefficients)

    @property
    def label(self) -> str:
        expression = "+".join(
            "%g*%s" % (coefficient, column)
            for column, coefficient in self._coefficients
        )
        return _label(
            "linear:%s==%g" % (expression, self.rhs), self.name
        )

    @property
    def is_equality(self) -> bool:
        return True

    def value(self, frame: pd.DataFrame) -> np.ndarray:
        frame = _require_frame(frame)
        result = np.zeros(len(frame), dtype=np.float64)
        for column, coefficient in self._coefficients:
            result += coefficient * _numeric(frame, column)
        return result

    def margin(self, frame: pd.DataFrame) -> np.ndarray:
        return self.tolerance - np.abs(self.value(frame) - self.rhs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type_name,
            "coefficients": [
                {
                    "column": encode_json_value(column),
                    "coefficient": coefficient,
                }
                for column, coefficient in self._coefficients
            ],
            "rhs": self.rhs,
            "allow_missing": self.allow_missing,
            "tolerance": self.tolerance,
            "name": self.name,
        }


@dataclass(frozen=True, init=False)
class SumConstraint(TableConstraint):
    """Require member columns to sum to a fixed or row-varying total."""

    members: Tuple[Any, ...]
    total: Optional[float] = None
    total_column: Any = None
    nonnegative: bool = True
    allow_missing: bool = False
    tolerance: float = 1e-8
    name: str = ""

    type_name = "sum"

    def __init__(
        self,
        members: Optional[Sequence[Any]] = None,
        total: Optional[float] = None,
        total_column: Any = None,
        *,
        columns: Optional[Sequence[Any]] = None,
        nonnegative: bool = True,
        allow_missing: bool = False,
        tolerance: float = 1e-8,
        name: str = "",
    ) -> None:
        if columns is not None:
            if members is not None:
                raise ValueError("pass members or columns, not both")
            members = columns
        if members is None:
            raise ValueError("sum constraint requires member columns")
        members = tuple(_name(column) for column in members)
        if len(members) < 2 or len(set(members)) != len(members):
            raise ValueError(
                "sum constraints need at least two distinct member columns"
            )
        object.__setattr__(self, "members", members)
        if (total is None) == (total_column is None):
            raise ValueError(
                "sum constraint requires exactly one of total or total_column"
            )
        if total is not None:
            object.__setattr__(
                self, "total", _finite_float("total", total)
            )
        else:
            object.__setattr__(self, "total", None)
        if total_column is not None:
            total_column = _name(total_column)
            if total_column in members:
                raise ValueError(
                    "sum total_column cannot also be a member column"
                )
            object.__setattr__(self, "total_column", total_column)
        else:
            object.__setattr__(self, "total_column", None)
        object.__setattr__(self, "nonnegative", bool(nonnegative))
        if nonnegative and total is not None and float(total) < 0.0:
            raise ValueError("a nonnegative sum cannot have a negative total")
        object.__setattr__(self, "allow_missing", bool(allow_missing))
        object.__setattr__(
            self, "tolerance", _nonnegative_float("tolerance", tolerance)
        )
        object.__setattr__(self, "name", str(name))

    @property
    def columns(self) -> Tuple[Any, ...]:
        return self.members + (
            () if self.total_column is None else (self.total_column,)
        )

    @property
    def label(self) -> str:
        target = self.total if self.total_column is None else self.total_column
        return _label("sum:%s=%s" % (",".join(map(str, self.members)), target), self.name)

    @property
    def is_equality(self) -> bool:
        return True

    def target(self, frame: pd.DataFrame) -> np.ndarray:
        if self.total_column is None:
            return np.full(len(frame), float(self.total), dtype=np.float64)
        return _numeric(frame, self.total_column)

    def member_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        return np.column_stack(
            [_numeric(frame, column) for column in self.members]
        )

    def margin(self, frame: pd.DataFrame) -> np.ndarray:
        frame = _require_frame(frame)
        values = self.member_matrix(frame)
        equality = self.tolerance - np.abs(
            np.sum(values, axis=1) - self.target(frame)
        )
        if not self.nonnegative:
            return equality
        lower = np.min(values, axis=1) + self.tolerance
        target = self.target(frame) + self.tolerance
        return np.minimum(np.minimum(equality, lower), target)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type_name,
            "members": [encode_json_value(column) for column in self.members],
            "total": self.total,
            "total_column": (
                None
                if self.total_column is None
                else encode_json_value(self.total_column)
            ),
            "nonnegative": self.nonnegative,
            "allow_missing": self.allow_missing,
            "tolerance": self.tolerance,
            "name": self.name,
        }


@dataclass(frozen=True, init=False)
class FixedSumConstraint(SumConstraint):
    """A named fixed-total specialization of :class:`SumConstraint`."""

    type_name = "fixed_sum"

    def __init__(
        self,
        columns: Optional[Sequence[Any]] = None,
        total: float = 1.0,
        *,
        members: Optional[Sequence[Any]] = None,
        nonnegative: bool = True,
        allow_missing: bool = False,
        tolerance: float = 1e-8,
        name: str = "",
    ) -> None:
        if members is not None:
            if columns is not None:
                raise ValueError("pass columns or members, not both")
            columns = members
        if columns is None:
            raise ValueError("fixed sum constraint requires columns")
        SumConstraint.__init__(
            self,
            members=tuple(columns),
            total=total,
            nonnegative=nonnegative,
            allow_missing=allow_missing,
            tolerance=tolerance,
            name=name,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = SumConstraint.to_dict(self)
        payload["type"] = self.type_name
        return payload


@dataclass(frozen=True, init=False)
class VariableSumConstraint(SumConstraint):
    """A named row-varying-total specialization of ``SumConstraint``."""

    type_name = "variable_sum"

    def __init__(
        self,
        columns: Optional[Sequence[Any]] = None,
        total_column: Any = None,
        *,
        members: Optional[Sequence[Any]] = None,
        nonnegative: bool = True,
        allow_missing: bool = False,
        tolerance: float = 1e-8,
        name: str = "",
    ) -> None:
        if members is not None:
            if columns is not None:
                raise ValueError("pass columns or members, not both")
            columns = members
        if columns is None:
            raise ValueError("variable sum constraint requires columns")
        if total_column is None:
            raise ValueError(
                "variable sum constraint requires total_column"
            )
        SumConstraint.__init__(
            self,
            members=tuple(columns),
            total_column=total_column,
            nonnegative=nonnegative,
            allow_missing=allow_missing,
            tolerance=tolerance,
            name=name,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = SumConstraint.to_dict(self)
        payload["type"] = self.type_name
        return payload


@dataclass(frozen=True, init=False)
class SimplexConstraint(SumConstraint):
    """A nonnegative fixed-sum composition (unit simplex by default)."""

    type_name = "simplex"

    def __init__(
        self,
        columns: Optional[Sequence[Any]] = None,
        total: float = 1.0,
        *,
        members: Optional[Sequence[Any]] = None,
        allow_missing: bool = False,
        tolerance: float = 1e-8,
        name: str = "",
    ) -> None:
        if members is not None:
            if columns is not None:
                raise ValueError("pass columns or members, not both")
            columns = members
        if columns is None:
            raise ValueError("simplex constraint requires columns")
        SumConstraint.__init__(
            self,
            members=tuple(columns),
            total=total,
            total_column=None,
            nonnegative=True,
            allow_missing=allow_missing,
            tolerance=tolerance,
            name=name,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = SumConstraint.to_dict(self)
        payload["type"] = self.type_name
        return payload


_PREDICATE_OPERATORS = {
    "==",
    "!=",
    "<",
    "<=",
    ">",
    ">=",
    "in",
    "not in",
}


@dataclass(frozen=True)
class Predicate:
    """A serializable scalar row predicate used by logical implications."""

    column: Any
    operator: str
    value: Any
    tolerance: float = 1e-9

    def __post_init__(self) -> None:
        object.__setattr__(self, "column", _name(self.column))
        operation = str(self.operator).strip().lower().replace("_", " ")
        if operation not in _PREDICATE_OPERATORS:
            raise ValueError("unsupported predicate operator %r" % operation)
        value = self.value
        if operation in {"in", "not in"}:
            if isinstance(value, (str, bytes)) or not isinstance(
                value, Sequence
            ):
                raise TypeError(
                    "predicate operator %r requires a value sequence"
                    % operation
                )
            value = tuple(value)
            if not value:
                raise ValueError("predicate value sequence cannot be empty")
        encode_json_value(value)
        object.__setattr__(self, "operator", operation)
        object.__setattr__(self, "value", value)
        object.__setattr__(
            self, "tolerance", _nonnegative_float("tolerance", self.tolerance)
        )

    def evaluate(self, frame: pd.DataFrame) -> np.ndarray:
        frame = _require_frame(frame)
        _require_columns(frame, (self.column,))
        series = frame[self.column]
        operation = self.operator
        if operation == "in":
            return series.isin(list(self.value)).to_numpy(dtype=bool)
        if operation == "not in":
            return (~series.isin(list(self.value))).to_numpy(dtype=bool)
        if operation in {"==", "!="}:
            if isinstance(self.value, (int, float, np.number)) and not isinstance(
                self.value, (bool, np.bool_)
            ):
                numeric = pd.to_numeric(series, errors="coerce").to_numpy(
                    dtype=np.float64
                )
                equal = np.abs(numeric - float(self.value)) <= self.tolerance
            else:
                equal = (series == self.value).fillna(False).to_numpy(
                    dtype=bool
                )
            return equal if operation == "==" else ~equal
        numeric = _numeric(frame, self.column)
        target = _finite_float("predicate value", self.value)
        if operation == "<":
            return numeric < target
        if operation == "<=":
            return numeric <= target + self.tolerance
        if operation == ">":
            return numeric > target
        return numeric + self.tolerance >= target

    def margin(self, frame: pd.DataFrame) -> np.ndarray:
        valid = self.evaluate(frame)
        if self.operator in {"<", "<=", ">", ">="}:
            numeric = _numeric(frame, self.column)
            target = float(self.value)
            if self.operator in {"<", "<="}:
                return target - numeric + self.tolerance
            return numeric - target + self.tolerance
        if self.operator == "==" and isinstance(
            self.value, (int, float, np.number)
        ) and not isinstance(self.value, (bool, np.bool_)):
            numeric = _numeric(frame, self.column)
            return self.tolerance - np.abs(numeric - float(self.value))
        return np.where(valid, 1.0, -1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column": encode_json_value(self.column),
            "operator": self.operator,
            "value": encode_json_value(self.value),
            "tolerance": self.tolerance,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Predicate":
        column = payload["column"]
        value = payload["value"]
        return cls(
            column=(
                decode_json_value(column)
                if isinstance(column, Mapping) and "type" in column
                else column
            ),
            operator=str(payload.get("operator", "==")),
            value=(
                decode_json_value(value)
                if isinstance(value, Mapping) and "type" in value
                else value
            ),
            tolerance=float(payload.get("tolerance", 1e-9)),
        )


def _coerce_predicate(
    value: Union[Predicate, Mapping[str, Any]]
) -> Predicate:
    if isinstance(value, Predicate):
        return value
    if isinstance(value, Mapping):
        return Predicate.from_dict(value)
    raise TypeError("implication predicates must be Predicate objects or mappings")


@dataclass(frozen=True, init=False)
class ImplicationConstraint(TableConstraint):
    """Require ``antecedent => consequent`` row by row."""

    antecedent: Predicate
    consequent: Predicate
    allow_missing: bool
    tolerance: float
    name: str

    type_name = "implication"

    def __init__(
        self,
        *args: Any,
        antecedent: Optional[Union[Predicate, Mapping[str, Any]]] = None,
        consequent: Optional[Union[Predicate, Mapping[str, Any]]] = None,
        if_column: Any = None,
        if_value: Any = None,
        if_operator: str = "==",
        then_column: Any = None,
        then_value: Any = None,
        then_operator: str = "==",
        allow_missing: bool = False,
        tolerance: float = 1e-9,
        name: str = "",
    ) -> None:
        if args:
            if len(args) == 2 and all(
                isinstance(item, (Predicate, Mapping)) for item in args
            ):
                if antecedent is not None or consequent is not None:
                    raise ValueError("implication predicates were supplied twice")
                antecedent, consequent = args
            elif len(args) == 4:
                if any(
                    value is not None
                    for value in (
                        antecedent,
                        consequent,
                        if_column,
                        then_column,
                    )
                ):
                    raise ValueError("implication operands were supplied twice")
                if_column, if_value, then_column, then_value = args
            else:
                raise TypeError(
                    "ImplicationConstraint accepts two predicates or "
                    "(if_column, if_value, then_column, then_value)"
                )
        if antecedent is None:
            if if_column is None:
                raise ValueError("implication requires an antecedent")
            antecedent = Predicate(
                if_column, if_operator, if_value, tolerance=tolerance
            )
        if consequent is None:
            if then_column is None:
                raise ValueError("implication requires a consequent")
            consequent = Predicate(
                then_column, then_operator, then_value, tolerance=tolerance
            )
        object.__setattr__(self, "antecedent", _coerce_predicate(antecedent))
        object.__setattr__(self, "consequent", _coerce_predicate(consequent))
        object.__setattr__(self, "allow_missing", bool(allow_missing))
        object.__setattr__(
            self, "tolerance", _nonnegative_float("tolerance", tolerance)
        )
        object.__setattr__(self, "name", str(name))

    @property
    def columns(self) -> Tuple[Any, ...]:
        values = (self.antecedent.column, self.consequent.column)
        return tuple(dict.fromkeys(values))

    @property
    def label(self) -> str:
        default = "if:%s%s%s=>%s%s%s" % (
            self.antecedent.column,
            self.antecedent.operator,
            self.antecedent.value,
            self.consequent.column,
            self.consequent.operator,
            self.consequent.value,
        )
        return _label(default, self.name)

    def margin(self, frame: pd.DataFrame) -> np.ndarray:
        frame = _require_frame(frame)
        active = self.antecedent.evaluate(frame)
        consequence_margin = self.consequent.margin(frame)
        # Inactive implications have no finite distance to their boundary.
        return np.where(active, consequence_margin, np.inf)

    def evaluate(self, frame: pd.DataFrame) -> np.ndarray:
        frame = _require_frame(frame)
        _require_columns(frame, self.columns)
        active = self.antecedent.evaluate(frame)
        valid = ~active | self.consequent.evaluate(frame)
        missing = _missing_mask(frame, self.columns)
        valid[missing] = self.allow_missing
        return valid

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type_name,
            "antecedent": self.antecedent.to_dict(),
            "consequent": self.consequent.to_dict(),
            "allow_missing": self.allow_missing,
            "tolerance": self.tolerance,
            "name": self.name,
        }


Constraint = Union[
    BoundConstraint,
    DomainConstraint,
    PairInequality,
    LinearInequality,
    LinearEquality,
    SumConstraint,
    FixedSumConstraint,
    VariableSumConstraint,
    SimplexConstraint,
    ImplicationConstraint,
]
ConstraintLike = Union[Constraint, Mapping[str, Any]]


def _decoded_name(value: Any) -> Any:
    if isinstance(value, Mapping) and "type" in value:
        return decode_json_value(value)
    return value


def _coefficients_from_payload(
    payload: Any,
) -> Union[Mapping[Any, Any], Sequence[Tuple[Any, Any]]]:
    if isinstance(payload, Mapping):
        return payload
    return [
        (
            _decoded_name(item.get("column")),
            item.get("coefficient", item.get("value")),
        )
        for item in payload
    ]


def constraint_from_dict(payload: Mapping[str, Any]) -> Constraint:
    """Deserialize a typed constraint from a JSON/YAML-style mapping."""

    if not isinstance(payload, Mapping):
        raise TypeError("constraint specification must be a mapping")
    kind = str(payload.get("type", payload.get("kind", ""))).strip().lower()
    kind = kind.replace("-", "_").replace(" ", "_")
    if kind in {"bounds", "bound", "range"}:
        return BoundConstraint(
            column=_decoded_name(payload["column"]),
            lower=payload.get("lower", payload.get("minimum", payload.get("min"))),
            upper=payload.get("upper", payload.get("maximum", payload.get("max"))),
            allow_missing=bool(payload.get("allow_missing", False)),
            tolerance=float(payload.get("tolerance", 1e-9)),
            name=str(payload.get("name", "")),
        )
    if kind in {"domain", "allowed", "allowed_values"}:
        return DomainConstraint(
            column=_decoded_name(payload["column"]),
            values=tuple(
                _decoded_name(value) for value in payload.get("values", ())
            ),
            allow_missing=bool(payload.get("allow_missing", False)),
            name=str(payload.get("name", "")),
        )
    if kind in {"pair", "pair_inequality", "inequality_pair"}:
        operator = str(payload.get("operator", "<="))
        small = payload.get("small", payload.get("left"))
        big = payload.get("big", payload.get("right"))
        if operator == ">=":
            small, big = big, small
        elif operator != "<=":
            raise ValueError(
                "pair inequality operator must be '<=' or '>='"
            )
        return PairInequality(
            small=_decoded_name(small),
            big=_decoded_name(big),
            minimum_gap=float(payload.get("minimum_gap", 0.0)),
            allow_missing=bool(payload.get("allow_missing", False)),
            tolerance=float(payload.get("tolerance", 1e-9)),
            name=str(payload.get("name", "")),
        )
    if kind in {"linear_inequality", "linear", "affine_inequality"}:
        return LinearInequality(
            _coefficients_from_payload(payload["coefficients"]),
            rhs=payload.get("rhs", payload.get("bound")),
            operator=payload.get("operator", "<="),
            allow_missing=bool(payload.get("allow_missing", False)),
            tolerance=float(payload.get("tolerance", 1e-9)),
            name=str(payload.get("name", "")),
        )
    if kind in {"linear_equality", "affine_equality"}:
        return LinearEquality(
            _coefficients_from_payload(payload["coefficients"]),
            rhs=payload.get("rhs", payload.get("value")),
            allow_missing=bool(payload.get("allow_missing", False)),
            tolerance=float(payload.get("tolerance", 1e-8)),
            name=str(payload.get("name", "")),
        )
    if kind in {"sum", "fixed_sum", "variable_sum"}:
        members = payload.get("members", payload.get("columns"))
        decoded_members = tuple(
            _decoded_name(value) for value in members
        )
        options = {
            "nonnegative": bool(payload.get("nonnegative", True)),
            "allow_missing": bool(payload.get("allow_missing", False)),
            "tolerance": float(payload.get("tolerance", 1e-8)),
            "name": str(payload.get("name", "")),
        }
        total_column = (
            None
            if payload.get("total_column") is None
            else _decoded_name(payload.get("total_column"))
        )
        if kind == "fixed_sum":
            return FixedSumConstraint(
                decoded_members,
                total=float(payload["total"]),
                **options,
            )
        if kind == "variable_sum":
            return VariableSumConstraint(
                decoded_members,
                total_column=total_column,
                **options,
            )
        return SumConstraint(
            members=decoded_members,
            total=payload.get("total"),
            total_column=total_column,
            **options,
        )
    if kind == "simplex":
        members = payload.get("members", payload.get("columns"))
        return SimplexConstraint(
            columns=tuple(_decoded_name(value) for value in members),
            total=float(payload.get("total", 1.0)),
            allow_missing=bool(payload.get("allow_missing", False)),
            tolerance=float(payload.get("tolerance", 1e-8)),
            name=str(payload.get("name", "")),
        )
    if kind in {"implication", "logical_implication", "implies"}:
        common = {
            "allow_missing": bool(payload.get("allow_missing", False)),
            "tolerance": float(payload.get("tolerance", 1e-9)),
            "name": str(payload.get("name", "")),
        }
        if "antecedent" in payload or "consequent" in payload:
            if "antecedent" not in payload or "consequent" not in payload:
                raise ValueError(
                    "implication needs both antecedent and consequent"
                )
            return ImplicationConstraint(
                antecedent=Predicate.from_dict(payload["antecedent"]),
                consequent=Predicate.from_dict(payload["consequent"]),
                **common,
            )
        return ImplicationConstraint(
            if_column=_decoded_name(payload.get("if_column")),
            if_value=_decoded_name(payload.get("if_value")),
            if_operator=str(payload.get("if_operator", "==")),
            then_column=_decoded_name(payload.get("then_column")),
            then_value=_decoded_name(payload.get("then_value")),
            then_operator=str(payload.get("then_operator", "==")),
            **common,
        )
    raise ValueError("unsupported constraint type %r" % kind)


def coerce_constraint(value: ConstraintLike) -> Constraint:
    if isinstance(value, TableConstraint):
        return value  # type: ignore[return-value]
    if isinstance(value, Mapping):
        return constraint_from_dict(value)
    raise TypeError(
        "constraints must be typed constraint objects or specification mappings"
    )


def _margin_summary(
    values: np.ndarray, *, boundary_tolerance: float
) -> Dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    summary: Dict[str, Any] = {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "inactive_count": int(array.size - finite.size),
    }
    if not finite.size:
        summary.update(
            {
                "minimum": None,
                "min": None,
                "q05": None,
                "p05": None,
                "q25": None,
                "p25": None,
                "median": None,
                "p50": None,
                "q75": None,
                "p75": None,
                "q95": None,
                "p95": None,
                "maximum": None,
                "max": None,
                "mean": None,
                "std": None,
                "boundary_mass": 0.0,
            }
        )
        return summary
    quantiles = np.quantile(finite, [0.05, 0.25, 0.5, 0.75, 0.95])
    summary.update(
        {
            "minimum": float(np.min(finite)),
            "min": float(np.min(finite)),
            "q05": float(quantiles[0]),
            "p05": float(quantiles[0]),
            "q25": float(quantiles[1]),
            "p25": float(quantiles[1]),
            "median": float(quantiles[2]),
            "p50": float(quantiles[2]),
            "q75": float(quantiles[3]),
            "p75": float(quantiles[3]),
            "q95": float(quantiles[4]),
            "p95": float(quantiles[4]),
            "maximum": float(np.max(finite)),
            "max": float(np.max(finite)),
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
            "boundary_mass": float(
                np.mean(np.abs(finite) <= boundary_tolerance)
            ),
        }
    )
    return summary


@dataclass(frozen=True)
class ConstraintValidation:
    """Combined validity and margin reports returned by ``validate``."""

    valid: bool
    violations: pd.DataFrame
    margins: pd.DataFrame

    def __bool__(self) -> bool:
        return self.valid

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "violations": self.violations.to_dict(orient="records"),
            "margins": self.margins.to_dict(orient="records"),
        }


@dataclass(frozen=True)
class ProjectionAudit:
    """Auditable trace of cyclic Euclidean projection."""

    iterations: int
    converged: bool
    pre_violation_rate: float
    post_violation_rate: float
    changed_row_fraction: float
    steps: Tuple[Mapping[str, Any], ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": "cyclic_euclidean_projection",
            "iterations": self.iterations,
            "converged": self.converged,
            "pre_violation_rate": self.pre_violation_rate,
            "post_violation_rate": self.post_violation_rate,
            "changed_row_fraction": self.changed_row_fraction,
            "steps": [dict(step) for step in self.steps],
        }


def _simplex_projection(values: np.ndarray, total: float) -> np.ndarray:
    """Orthogonally project one vector onto ``x >= 0, sum(x) = total``."""

    if total < 0.0:
        raise ValueError("cannot project onto a simplex with negative total")
    if total == 0.0:
        return np.zeros_like(values)
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered) - total
    indices = np.arange(1, len(values) + 1, dtype=np.float64)
    supported = ordered - cumulative / indices > 0.0
    rho = int(np.flatnonzero(supported)[-1])
    threshold = cumulative[rho] / float(rho + 1)
    return np.maximum(values - threshold, 0.0)


def _assign_numeric(frame: pd.DataFrame, column: Any, values: np.ndarray) -> None:
    frame[column] = pd.Series(values, index=frame.index, name=column)


def _project_predicate(
    frame: pd.DataFrame, predicate: Predicate, active: np.ndarray
) -> np.ndarray:
    before = frame[predicate.column].copy()
    operation = predicate.operator
    selected = np.flatnonzero(active & ~predicate.evaluate(frame))
    if not len(selected):
        return np.zeros(len(frame), dtype=np.float64)
    if operation in {"==", "in"}:
        replacement = (
            predicate.value
            if operation == "=="
            else tuple(predicate.value)[0]
        )
        values = frame[predicate.column].astype(object).to_numpy(copy=True)
        values[selected] = replacement
        frame[predicate.column] = pd.Series(values, index=frame.index)
    elif operation in {"<=", "<", ">=", ">"}:
        values = _numeric(frame, predicate.column)
        target = float(predicate.value)
        if operation == "<":
            target = float(np.nextafter(target, -np.inf))
        elif operation == ">":
            target = float(np.nextafter(target, np.inf))
        values[selected] = target
        _assign_numeric(frame, predicate.column, values)
    elif operation in {"!=", "not in"}:
        raise ValueError(
            "cannot deterministically project implication consequence %r; "
            "use ==, in, or a numeric bound" % operation
        )
    after = frame[predicate.column]
    changed = ~(before.eq(after) | (before.isna() & after.isna()))
    return changed.to_numpy(dtype=np.float64)


def _project_constraint(
    frame: pd.DataFrame, constraint: Constraint
) -> np.ndarray:
    rows = len(frame)
    adjustment = np.zeros(rows, dtype=np.float64)
    missing = _missing_mask(frame, constraint.columns)
    eligible = ~missing
    if isinstance(constraint, BoundConstraint):
        values = _numeric(frame, constraint.column)
        projected = values.copy()
        if constraint.lower is not None:
            projected[eligible] = np.maximum(
                projected[eligible], constraint.lower
            )
        if constraint.upper is not None:
            projected[eligible] = np.minimum(
                projected[eligible], constraint.upper
            )
        adjustment[eligible] = np.abs(
            projected[eligible] - values[eligible]
        )
        _assign_numeric(frame, constraint.column, projected)
        return adjustment
    if isinstance(constraint, DomainConstraint):
        invalid = ~constraint.evaluate(frame) & eligible
        if bool(invalid.any()):
            before = frame[constraint.column].copy()
            values = frame[constraint.column].astype(object).to_numpy(copy=True)
            values[invalid] = constraint.values[0]
            frame[constraint.column] = pd.Series(values, index=frame.index)
            after = frame[constraint.column]
            adjustment = (
                ~(before.eq(after) | (before.isna() & after.isna()))
            ).to_numpy(dtype=np.float64)
        return adjustment
    if isinstance(constraint, PairInequality):
        small = _numeric(frame, constraint.small)
        big = _numeric(frame, constraint.big)
        violation = small + constraint.minimum_gap - big
        active = eligible & (violation > constraint.tolerance)
        correction = np.where(active, violation / 2.0, 0.0)
        _assign_numeric(frame, constraint.small, small - correction)
        _assign_numeric(frame, constraint.big, big + correction)
        adjustment = np.sqrt(2.0) * np.abs(correction)
        return adjustment
    if isinstance(constraint, LinearInequality):
        value = constraint.value(frame)
        if constraint.operator == "<=":
            residual = value - constraint.rhs
            normal = constraint._coefficients
        else:
            residual = constraint.rhs - value
            normal = tuple(
                (column, -coefficient)
                for column, coefficient in constraint._coefficients
            )
        active = eligible & (residual > constraint.tolerance)
        norm_squared = sum(coefficient ** 2 for _, coefficient in normal)
        scale = np.where(active, residual / norm_squared, 0.0)
        for column, coefficient in normal:
            values = _numeric(frame, column)
            _assign_numeric(frame, column, values - scale * coefficient)
        adjustment = np.where(
            active, residual / math.sqrt(norm_squared), 0.0
        )
        return np.abs(adjustment)
    if isinstance(constraint, LinearEquality):
        residual = constraint.value(frame) - constraint.rhs
        active = eligible & (np.abs(residual) > constraint.tolerance)
        norm_squared = sum(
            coefficient ** 2
            for _, coefficient in constraint._coefficients
        )
        scale = np.where(active, residual / norm_squared, 0.0)
        for column, coefficient in constraint._coefficients:
            values = _numeric(frame, column)
            _assign_numeric(frame, column, values - scale * coefficient)
        return np.where(
            active, np.abs(residual) / math.sqrt(norm_squared), 0.0
        )
    if isinstance(constraint, SumConstraint):
        values = constraint.member_matrix(frame)
        target = constraint.target(frame)
        projected = values.copy()
        active = eligible & ~constraint.evaluate(frame)
        for row in np.flatnonzero(active):
            if constraint.nonnegative:
                projected[row] = _simplex_projection(
                    values[row], float(target[row])
                )
            else:
                delta = (
                    float(target[row]) - float(np.sum(values[row]))
                ) / float(values.shape[1])
                projected[row] = values[row] + delta
        adjustment = np.linalg.norm(projected - values, axis=1)
        for index, column in enumerate(constraint.members):
            _assign_numeric(frame, column, projected[:, index])
        return adjustment
    if isinstance(constraint, ImplicationConstraint):
        active = constraint.antecedent.evaluate(frame) & eligible
        return _project_predicate(frame, constraint.consequent, active)
    raise TypeError("projection does not support %s" % type(constraint).__name__)


def _row_changes(before: pd.DataFrame, after: pd.DataFrame) -> np.ndarray:
    if len(before) != len(after) or tuple(before.columns) != tuple(after.columns):
        raise ValueError("projection changed dataframe shape")
    changed = np.zeros(len(before), dtype=bool)
    for column in before.columns:
        left = before[column]
        right = after[column]
        equal = left.eq(right) | (left.isna() & right.isna())
        if pd.api.types.is_numeric_dtype(left.dtype) and pd.api.types.is_numeric_dtype(
            right.dtype
        ):
            left_values = pd.to_numeric(left, errors="coerce").to_numpy(
                dtype=np.float64
            )
            right_values = pd.to_numeric(right, errors="coerce").to_numpy(
                dtype=np.float64
            )
            finite = np.isfinite(left_values) & np.isfinite(right_values)
            numeric_equal = np.zeros(len(before), dtype=bool)
            numeric_equal[finite] = np.isclose(
                left_values[finite],
                right_values[finite],
                rtol=1e-12,
                atol=1e-12,
            )
            numeric_equal[~finite] = equal.to_numpy(dtype=bool)[~finite]
            changed |= ~numeric_equal
        else:
            changed |= ~equal.to_numpy(dtype=bool)
    return changed


class ConstraintSet:
    """An immutable ordered collection of typed table constraints."""

    FORMAT_VERSION = 1
    STATE_FILENAME = "constraints.json"

    def __init__(self, constraints: Optional[Iterable[ConstraintLike]] = None):
        if constraints is None:
            values: Tuple[Constraint, ...] = ()
        elif isinstance(constraints, ConstraintSet):
            values = constraints.constraints
        elif isinstance(constraints, Mapping) and (
            constraints.get("format") == "ganify-constraint-set"
            or (
                "constraints" in constraints
                and "type" not in constraints
                and "kind" not in constraints
            )
        ):
            if constraints.get("format") == "ganify-constraint-set" and int(
                constraints.get("version", -1)
            ) != self.FORMAT_VERSION:
                raise ValueError("unsupported constraint set format")
            values = tuple(
                coerce_constraint(value)
                for value in constraints.get("constraints", ())
            )
        elif isinstance(constraints, (TableConstraint, Mapping)):
            values = (coerce_constraint(constraints),)
        else:
            values = tuple(coerce_constraint(value) for value in constraints)
        labels = [constraint.label for constraint in values]
        duplicates = sorted(
            {label for label in labels if labels.count(label) > 1}
        )
        if duplicates:
            raise ValueError(
                "constraint labels must be unique; duplicates=%r" % duplicates
            )
        self.constraints = values

    def __iter__(self) -> Iterator[Constraint]:
        return iter(self.constraints)

    def __len__(self) -> int:
        return len(self.constraints)

    def __getitem__(self, index: int) -> Constraint:
        return self.constraints[index]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": "ganify-constraint-set",
            "version": self.FORMAT_VERSION,
            "constraints": [
                constraint.to_dict() for constraint in self.constraints
            ],
        }

    state_dict = to_dict

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConstraintSet":
        if not isinstance(payload, Mapping):
            raise TypeError("constraint set state must be a mapping")
        if payload.get("format") == "ganify-constraint-set":
            if int(payload.get("version", -1)) != cls.FORMAT_VERSION:
                raise ValueError("unsupported constraint set format")
            values = payload.get("constraints", ())
        elif "constraints" in payload and len(payload) == 1:
            values = payload["constraints"]
        else:
            values = (payload,)
        return cls(constraint_from_dict(value) for value in values)

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
    def from_json(cls, value: Union[str, bytes]) -> "ConstraintSet":
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
    def load(cls, path: Union[str, Path]) -> "ConstraintSet":
        source = Path(path)
        if source.is_dir():
            source = source / cls.STATE_FILENAME
        return cls.from_json(source.read_text(encoding="utf-8"))

    def _check_columns(self, frame: pd.DataFrame) -> None:
        frame = _require_frame(frame)
        referenced: List[Any] = []
        for constraint in self.constraints:
            referenced.extend(constraint.columns)
        _require_columns(frame, tuple(dict.fromkeys(referenced)))

    def masks(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = _require_frame(frame)
        self._check_columns(frame)
        return pd.DataFrame(
            {
                constraint.label: constraint.evaluate(frame)
                for constraint in self.constraints
            },
            index=frame.index,
        )

    def joint_mask(self, frame: pd.DataFrame) -> np.ndarray:
        frame = _require_frame(frame)
        if not self.constraints:
            return np.ones(len(frame), dtype=bool)
        return np.logical_and.reduce(
            [constraint.evaluate(frame) for constraint in self.constraints]
        )

    def violation_rate(self, frame: pd.DataFrame) -> float:
        if len(frame) == 0:
            return 0.0
        return float(1.0 - np.mean(self.joint_mask(frame)))

    def violation_report(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = _require_frame(frame)
        self._check_columns(frame)
        rows: List[Dict[str, Any]] = []
        masks: List[np.ndarray] = []
        for constraint in self.constraints:
            valid = constraint.evaluate(frame)
            masks.append(valid)
            valid_count = int(np.sum(valid))
            rows.append(
                {
                    "constraint": constraint.label,
                    "type": constraint.type_name,
                    "valid_count": valid_count,
                    "violation_count": int(len(frame) - valid_count),
                    "valid_rate": (
                        1.0 if not len(frame) else valid_count / float(len(frame))
                    ),
                    "violation_rate": (
                        0.0
                        if not len(frame)
                        else 1.0 - valid_count / float(len(frame))
                    ),
                }
            )
        joint = (
            np.ones(len(frame), dtype=bool)
            if not masks
            else np.logical_and.reduce(masks)
        )
        valid_count = int(np.sum(joint))
        rows.append(
            {
                "constraint": "__all__",
                "type": "joint",
                "valid_count": valid_count,
                "violation_count": int(len(frame) - valid_count),
                "valid_rate": (
                    1.0 if not len(frame) else valid_count / float(len(frame))
                ),
                "violation_rate": (
                    0.0
                    if not len(frame)
                    else 1.0 - valid_count / float(len(frame))
                ),
            }
        )
        return pd.DataFrame(rows)

    def margin_summaries(
        self,
        frame: pd.DataFrame,
        *,
        boundary_tolerance: float = 1e-8,
    ) -> Dict[str, Dict[str, Any]]:
        frame = _require_frame(frame)
        self._check_columns(frame)
        tolerance = _nonnegative_float(
            "boundary_tolerance", boundary_tolerance
        )
        return {
            constraint.label: {
                "type": constraint.type_name,
                **_margin_summary(
                    constraint.margin(frame),
                    boundary_tolerance=max(
                        tolerance,
                        float(getattr(constraint, "tolerance", 0.0)),
                    ),
                ),
            }
            for constraint in self.constraints
        }

    def margin_report(
        self,
        frame: pd.DataFrame,
        *,
        boundary_tolerance: float = 1e-8,
    ) -> pd.DataFrame:
        summaries = self.margin_summaries(
            frame, boundary_tolerance=boundary_tolerance
        )
        return pd.DataFrame(
            [
                {"constraint": label, **summary}
                for label, summary in summaries.items()
            ]
        )

    def validate(
        self,
        frame: pd.DataFrame,
        *,
        raise_on_violation: bool = False,
    ) -> ConstraintValidation:
        violations = self.violation_report(frame)
        margins = self.margin_report(frame)
        valid = bool(
            violations.loc[
                violations["constraint"] == "__all__", "violation_count"
            ].iloc[0]
            == 0
        )
        if raise_on_violation and not valid:
            failing = violations.loc[
                (violations["constraint"] != "__all__")
                & (violations["violation_count"] > 0),
                ["constraint", "violation_count"],
            ]
            details = ", ".join(
                "%s (%d rows)" % (row.constraint, row.violation_count)
                for row in failing.itertuples(index=False)
            )
            raise ValueError(
                "training data violates configured constraints: %s" % details
            )
        return ConstraintValidation(valid, violations, margins)

    def report(self, frame: pd.DataFrame) -> ConstraintValidation:
        """Alias returning both violation and margin reports."""

        return self.validate(frame)

    def assert_satisfied(self, frame: pd.DataFrame) -> None:
        self.validate(frame, raise_on_violation=True)

    def project(
        self,
        frame: pd.DataFrame,
        *,
        max_iterations: int = 100,
        tolerance: float = 1e-8,
        return_audit: bool = False,
    ):
        """Project rows with cyclic orthogonal projections.

        Linear inequalities use the exact Euclidean projection onto an affine
        half-space; linear equalities use the exact projection onto a
        hyperplane.  Repeating those projections is the standard POCS fallback
        for intersections of convex sets.  The trace identifies every applied
        constraint and the largest adjustment it made.
        """

        frame = _require_frame(frame)
        self._check_columns(frame)
        if isinstance(max_iterations, bool) or int(max_iterations) < 1:
            raise ValueError("max_iterations must be a positive integer")
        max_iterations = int(max_iterations)
        tolerance = _nonnegative_float("tolerance", tolerance)
        original = frame.copy(deep=True)
        result = frame.copy(deep=True)
        pre_rate = self.violation_rate(result)
        totals = [
            {
                "constraint": constraint.label,
                "type": constraint.type_name,
                "applications": 0,
                "changed_rows": 0,
                "maximum_adjustment": 0.0,
            }
            for constraint in self.constraints
        ]
        converged = pre_rate == 0.0
        iterations = 0
        for iteration in range(1, max_iterations + 1):
            if converged:
                break
            iterations = iteration
            changed_this_iteration = False
            for index, constraint in enumerate(self.constraints):
                adjustment = _project_constraint(result, constraint)
                active = adjustment > tolerance
                if bool(active.any()):
                    changed_this_iteration = True
                    totals[index]["applications"] += 1
                    totals[index]["changed_rows"] += int(np.sum(active))
                    totals[index]["maximum_adjustment"] = max(
                        float(totals[index]["maximum_adjustment"]),
                        float(np.max(adjustment)),
                    )
            converged = self.violation_rate(result) == 0.0
            if not changed_this_iteration:
                break
        post_rate = self.violation_rate(result)
        changed = _row_changes(original, result)
        audit = ProjectionAudit(
            iterations=iterations,
            converged=post_rate == 0.0,
            pre_violation_rate=pre_rate,
            post_violation_rate=post_rate,
            changed_row_fraction=(
                0.0 if not len(result) else float(np.mean(changed))
            ),
            steps=tuple(dict(step) for step in totals),
        )
        return (result, audit) if return_audit else result


def iterative_euclidean_projection(
    frame: pd.DataFrame,
    constraints: Union[ConstraintSet, Iterable[ConstraintLike]],
    *,
    max_iterations: int = 100,
    tolerance: float = 1e-8,
    return_audit: bool = False,
):
    """Public functional wrapper around ``ConstraintSet.project``."""

    constraint_set = (
        constraints
        if isinstance(constraints, ConstraintSet)
        else ConstraintSet(constraints)
    )
    return constraint_set.project(
        frame,
        max_iterations=max_iterations,
        tolerance=tolerance,
        return_audit=return_audit,
    )


project_linear_constraints = iterative_euclidean_projection
project_constraints = iterative_euclidean_projection


# Clear public aliases for users coming from schema/evaluation terminology.
BoundsConstraint = BoundConstraint
RangeConstraint = BoundConstraint
AllowedValuesConstraint = DomainConstraint
PairInequalityConstraint = PairInequality
LinearInequalityConstraint = LinearInequality
LinearEqualityConstraint = LinearEquality
LogicalImplication = ImplicationConstraint
LogicalImplicationConstraint = ImplicationConstraint
Bounds = BoundConstraint
Domain = DomainConstraint
Pair = PairInequality
FixedSum = FixedSumConstraint
VariableSum = VariableSumConstraint
Simplex = SimplexConstraint
Implication = ImplicationConstraint


__all__ = [
    "AllowedValuesConstraint",
    "BoundConstraint",
    "Bounds",
    "BoundsConstraint",
    "Constraint",
    "ConstraintLike",
    "ConstraintSet",
    "ConstraintValidation",
    "DomainConstraint",
    "Domain",
    "FixedSum",
    "FixedSumConstraint",
    "Implication",
    "ImplicationConstraint",
    "LinearEquality",
    "LinearEqualityConstraint",
    "LinearInequality",
    "LinearInequalityConstraint",
    "LogicalImplication",
    "LogicalImplicationConstraint",
    "Pair",
    "PairInequality",
    "PairInequalityConstraint",
    "Predicate",
    "ProjectionAudit",
    "RangeConstraint",
    "SimplexConstraint",
    "Simplex",
    "SumConstraint",
    "TableConstraint",
    "VariableSumConstraint",
    "VariableSum",
    "coerce_constraint",
    "constraint_from_dict",
    "iterative_euclidean_projection",
    "project_constraints",
    "project_linear_constraints",
]
