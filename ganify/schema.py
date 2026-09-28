"""Typed, immutable schemas for tabular data.

The schema module deliberately has no dependency on the GAN implementation.
It provides deterministic dataframe inference, explicit user overrides, and a
JSON-safe representation that can be persisted with a fitted preprocessor.
"""

from __future__ import annotations

import base64
import datetime as _datetime
import json
import math
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


class ColumnType(str, Enum):
    """Semantic column types understood by GANify preprocessing."""

    CONTINUOUS = "continuous"
    COUNT = "count"
    BINARY = "binary"
    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"
    DATETIME = "datetime"
    CONSTANT = "constant"


SUPPORTED_COLUMN_TYPES: Tuple[str, ...] = tuple(item.value for item in ColumnType)

_KIND_ALIASES = {
    "bool": ColumnType.BINARY.value,
    "boolean": ColumnType.BINARY.value,
    "category": ColumnType.CATEGORICAL.value,
    "const": ColumnType.CONSTANT.value,
    "date": ColumnType.DATETIME.value,
    "float": ColumnType.CONTINUOUS.value,
    "integer": ColumnType.COUNT.value,
    "numeric": ColumnType.CONTINUOUS.value,
    "number": ColumnType.CONTINUOUS.value,
    "timestamp": ColumnType.DATETIME.value,
}


def _normalise_kind(kind: Union[str, ColumnType]) -> str:
    if isinstance(kind, ColumnType):
        return kind.value
    if not isinstance(kind, str):
        raise TypeError("column kind must be a string or ColumnType")
    value = _KIND_ALIASES.get(kind.strip().lower(), kind.strip().lower())
    if value not in SUPPORTED_COLUMN_TYPES:
        raise ValueError(
            "unsupported column kind %r; expected one of %s"
            % (kind, ", ".join(SUPPORTED_COLUMN_TYPES))
        )
    return value


def _python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _is_missing_scalar(value: Any) -> bool:
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _value_equal(left: Any, right: Any) -> bool:
    if _is_missing_scalar(left) and _is_missing_scalar(right):
        return True
    try:
        result = left == right
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def encode_json_value(value: Any) -> Any:
    """Encode a scalar or tuple as JSON primitives without losing its type."""

    value = _python_scalar(value)
    if value is None:
        return {"type": "none"}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        # Decimal text avoids precision loss in JSON consumers using doubles.
        return {"type": "int", "value": str(value)}
    if isinstance(value, float):
        if math.isnan(value):
            text = "nan"
        elif math.isinf(value):
            text = "inf" if value > 0 else "-inf"
        else:
            text = value.hex()
        return {"type": "float", "value": text}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, bytes):
        encoded = base64.b64encode(value).decode("ascii")
        return {"type": "bytes", "value": encoded}
    if isinstance(value, pd.Timestamp):
        return {"type": "timestamp", "value": value.isoformat()}
    if isinstance(value, _datetime.datetime):
        return {"type": "datetime", "value": value.isoformat()}
    if isinstance(value, _datetime.date):
        return {"type": "date", "value": value.isoformat()}
    if isinstance(value, _datetime.time):
        return {"type": "time", "value": value.isoformat()}
    if isinstance(value, pd.Timedelta):
        return {"type": "timedelta", "value": str(value.value)}
    if isinstance(value, _datetime.timedelta):
        return {
            "type": "timedelta",
            "value": str(int(value.total_seconds() * 1_000_000_000)),
        }
    if isinstance(value, tuple):
        return {"type": "tuple", "value": [encode_json_value(item) for item in value]}
    raise TypeError(
        "value %r of type %s is not JSON-serializable by GANify"
        % (value, type(value).__name__)
    )


def decode_json_value(payload: Any) -> Any:
    """Decode a value produced by :func:`encode_json_value`."""

    if not isinstance(payload, Mapping) or "type" not in payload:
        raise ValueError("invalid encoded JSON value")
    kind = payload["type"]
    value = payload.get("value")
    if kind == "none":
        return None
    if kind == "bool":
        return bool(value)
    if kind == "int":
        return int(value)
    if kind == "float":
        if value == "nan":
            return float("nan")
        if value == "inf":
            return float("inf")
        if value == "-inf":
            return float("-inf")
        return float.fromhex(value)
    if kind == "str":
        return str(value)
    if kind == "bytes":
        return base64.b64decode(str(value).encode("ascii"))
    if kind == "timestamp":
        return pd.Timestamp(value)
    if kind == "datetime":
        return _datetime.datetime.fromisoformat(str(value))
    if kind == "date":
        return _datetime.date.fromisoformat(str(value))
    if kind == "time":
        return _datetime.time.fromisoformat(str(value))
    if kind == "timedelta":
        return pd.Timedelta(int(value), unit="ns")
    if kind == "tuple":
        return tuple(decode_json_value(item) for item in value)
    raise ValueError("unknown encoded JSON value type %r" % kind)


def _stable_key(value: Any) -> str:
    try:
        payload = encode_json_value(value)
    except TypeError:
        payload = {
            "type": type(value).__name__,
            "value": repr(value),
        }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def stable_unique(values: Iterable[Any]) -> Tuple[Any, ...]:
    """Return non-missing unique values in a deterministic order."""

    unique: List[Any] = []
    for raw in values:
        value = _python_scalar(raw)
        if _is_missing_scalar(value):
            continue
        if not any(_value_equal(value, present) for present in unique):
            unique.append(value)
    try:
        return tuple(sorted(unique))
    except TypeError:
        return tuple(sorted(unique, key=_stable_key))


@dataclass(frozen=True)
class ColumnSpec:
    """Immutable semantic description of one dataframe column.

    Parameters
    ----------
    name:
        Dataframe column label. JSON-safe scalar and tuple labels are supported.
    kind:
        One of :class:`ColumnType`.
    nullable:
        Whether missing values are valid. Nullable columns receive an explicit
        mask channel in :class:`ganify.preprocessing.TableTransformer`.
    dtype:
        Original pandas dtype string, used when reconstructing a dataframe.
    categories:
        Ordered vocabulary for binary, categorical, or ordinal columns.
    ordered:
        Whether a categorical vocabulary has an intrinsic order.
    constant_value:
        Exact value restored for a constant column.
    timezone:
        IANA/fixed-offset timezone name for timezone-aware datetimes.
    """

    name: Any
    kind: Union[str, ColumnType]
    nullable: bool = False
    dtype: str = "object"
    categories: Tuple[Any, ...] = field(default_factory=tuple)
    ordered: bool = False
    constant_value: Any = None
    timezone: Optional[str] = None

    def __post_init__(self) -> None:
        canonical_kind = _normalise_kind(self.kind)
        object.__setattr__(self, "kind", canonical_kind)
        object.__setattr__(self, "nullable", bool(self.nullable))
        object.__setattr__(self, "dtype", str(self.dtype))
        object.__setattr__(self, "categories", tuple(self.categories))
        object.__setattr__(self, "ordered", bool(self.ordered))
        if self.timezone is not None:
            object.__setattr__(self, "timezone", str(self.timezone))

        try:
            hash(self.name)
        except TypeError as exc:
            raise TypeError("column name must be hashable") from exc
        if _is_missing_scalar(self.name):
            raise ValueError("column name cannot be missing")
        # Fail at construction rather than much later during model saving.
        encode_json_value(self.name)
        for category in self.categories:
            encode_json_value(category)
        if canonical_kind == ColumnType.CONSTANT.value:
            encode_json_value(self.constant_value)
        if len(self.categories) != len(stable_unique(self.categories)):
            raise ValueError("column %r has duplicate or missing categories" % (self.name,))
        if canonical_kind == ColumnType.BINARY.value and len(self.categories) > 2:
            raise ValueError("binary column %r cannot have more than two values" % self.name)
        if canonical_kind not in (
            ColumnType.BINARY.value,
            ColumnType.CATEGORICAL.value,
            ColumnType.ORDINAL.value,
        ) and self.categories:
            raise ValueError(
                "categories are only valid for binary, categorical, and ordinal columns"
            )
        if canonical_kind == ColumnType.ORDINAL.value and self.categories:
            object.__setattr__(self, "ordered", True)
        if canonical_kind != ColumnType.DATETIME.value and self.timezone is not None:
            raise ValueError("timezone is only valid for datetime columns")

    @property
    def semantic_type(self) -> str:
        """Alias for ``kind`` used by schema-oriented APIs."""

        return str(self.kind)

    @property
    def type(self) -> str:
        """Short alias for ``kind``."""

        return str(self.kind)

    @property
    def vocabulary(self) -> Tuple[Any, ...]:
        """Alias for the immutable category vocabulary."""

        return self.categories

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe representation."""

        return {
            "name": encode_json_value(self.name),
            "kind": self.kind,
            "nullable": self.nullable,
            "dtype": self.dtype,
            "categories": [encode_json_value(value) for value in self.categories],
            "ordered": self.ordered,
            "constant_value": encode_json_value(self.constant_value),
            "timezone": self.timezone,
        }

    def to_json(self, *, indent: Optional[int] = None) -> str:
        """Serialize this column specification to JSON text."""

        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=None if indent else (",", ":"),
            indent=indent,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ColumnSpec":
        """Restore a column specification from :meth:`to_dict`."""

        if not isinstance(payload, Mapping):
            raise TypeError("column specification must be a mapping")
        return cls(
            name=decode_json_value(payload["name"]),
            kind=payload["kind"],
            nullable=bool(payload.get("nullable", False)),
            dtype=str(payload.get("dtype", "object")),
            categories=tuple(
                decode_json_value(value) for value in payload.get("categories", [])
            ),
            ordered=bool(payload.get("ordered", False)),
            constant_value=decode_json_value(
                payload.get("constant_value", {"type": "none"})
            ),
            timezone=payload.get("timezone"),
        )

    @classmethod
    def from_json(cls, value: Union[str, bytes]) -> "ColumnSpec":
        """Restore a column specification from JSON text."""

        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True)
class TableSchema:
    """Immutable ordered collection of :class:`ColumnSpec` objects."""

    columns: Tuple[ColumnSpec, ...]
    version: int = 1

    def __post_init__(self) -> None:
        columns = tuple(
            item if isinstance(item, ColumnSpec) else ColumnSpec.from_dict(item)
            for item in self.columns
        )
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "version", int(self.version))
        if self.version != 1:
            raise ValueError("unsupported table schema version %r" % self.version)
        names = [column.name for column in columns]
        if len(names) != len(set(names)):
            raise ValueError("schema contains duplicate column names")

    def __len__(self) -> int:
        return len(self.columns)

    def __iter__(self) -> Iterator[ColumnSpec]:
        return iter(self.columns)

    def __getitem__(self, key: Union[int, Any]) -> ColumnSpec:
        if isinstance(key, int):
            return self.columns[key]
        for column in self.columns:
            if column.name == key:
                return column
        raise KeyError(key)

    def get(self, name: Any, default: Optional[ColumnSpec] = None) -> Optional[ColumnSpec]:
        """Return a named specification, or ``default`` when absent."""

        for column in self.columns:
            if column.name == name:
                return column
        return default

    def column(self, name: Any) -> ColumnSpec:
        """Return a named specification and raise ``KeyError`` when absent."""

        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(name)

    @property
    def names(self) -> Tuple[Any, ...]:
        """Column labels in dataframe order."""

        return tuple(column.name for column in self.columns)

    @property
    def column_names(self) -> Tuple[Any, ...]:
        """Alias for :attr:`names`."""

        return self.names

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe representation."""

        return {
            "format": "ganify-table-schema",
            "version": self.version,
            "columns": [column.to_dict() for column in self.columns],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TableSchema":
        """Restore a schema from :meth:`to_dict`."""

        if not isinstance(payload, Mapping):
            raise TypeError("table schema must be a mapping")
        if payload.get("format", "ganify-table-schema") != "ganify-table-schema":
            raise ValueError("not a GANify table schema")
        return cls(
            columns=tuple(ColumnSpec.from_dict(item) for item in payload["columns"]),
            version=int(payload.get("version", 1)),
        )

    def to_json(self, *, indent: Optional[int] = None) -> str:
        """Serialize the schema to JSON."""

        return json.dumps(
            self.to_dict(), sort_keys=True, separators=None if indent else (",", ":"), indent=indent
        )

    @classmethod
    def from_json(cls, value: Union[str, bytes]) -> "TableSchema":
        """Deserialize a schema from JSON text."""

        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return cls.from_dict(json.loads(value))

    def save(self, path: Union[str, Path]) -> Path:
        """Write the schema as UTF-8 JSON and return the destination."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.to_json(indent=2) + "\n", encoding="utf-8")
        return destination

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TableSchema":
        """Load a schema written by :meth:`save`."""

        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    @classmethod
    def infer(
        cls,
        frame: pd.DataFrame,
        overrides: Optional[Mapping[Any, Any]] = None,
        *,
        categorical_max_unique: int = 32,
        categorical_ratio: float = 0.20,
        integer_tolerance: float = 1e-9,
    ) -> "TableSchema":
        """Infer a deterministic schema from a dataframe."""

        return infer_schema(
            frame,
            overrides=overrides,
            categorical_max_unique=categorical_max_unique,
            categorical_ratio=categorical_ratio,
            integer_tolerance=integer_tolerance,
        )

    from_dataframe = infer

    def validate(
        self,
        frame: pd.DataFrame,
        *,
        strict_columns: bool = True,
        allow_unknown: bool = False,
        integer_tolerance: float = 1e-9,
    ) -> "TableSchema":
        """Validate dataframe shape, domains, missingness, and semantic types.

        The schema itself is returned to make validation convenient in fluent
        setup code. No dataframe values are mutated.
        """

        _require_frame(frame)
        missing_columns = [name for name in self.names if name not in frame.columns]
        extra_columns = [name for name in frame.columns if name not in self.names]
        if missing_columns or (strict_columns and extra_columns):
            raise ValueError(
                "dataframe columns do not match schema; missing=%r, extra=%r"
                % (missing_columns, extra_columns)
            )
        for spec in self.columns:
            _validate_series(
                frame[spec.name],
                spec,
                allow_unknown=allow_unknown,
                integer_tolerance=integer_tolerance,
            )
        return self


def _require_frame(frame: pd.DataFrame) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("schema inference and validation require a pandas DataFrame")
    if frame.columns.has_duplicates:
        raise ValueError("dataframe contains duplicate column names")
    if len(frame.columns) == 0:
        raise ValueError("dataframe must contain at least one column")
    if len(frame) == 0:
        raise ValueError("dataframe must contain at least one row")


def _datetime_timezone(series: pd.Series) -> Optional[str]:
    dtype = series.dtype
    timezone = getattr(dtype, "tz", None)
    return None if timezone is None else str(timezone)


def _categorical_values(series: pd.Series) -> Tuple[Any, ...]:
    if isinstance(series.dtype, pd.CategoricalDtype):
        return tuple(_python_scalar(value) for value in series.cat.categories)
    return stable_unique(series.tolist())


def _numeric_observed(series: pd.Series, name: Any) -> np.ndarray:
    try:
        values = pd.to_numeric(series.dropna(), errors="raise").to_numpy()
    except (TypeError, ValueError) as exc:
        raise ValueError("column %r must contain numeric values" % (name,)) from exc
    if np.iscomplexobj(values):
        raise ValueError("column %r cannot contain complex values" % (name,))
    values = np.asarray(values, dtype=np.float64)
    if not bool(np.isfinite(values).all()):
        raise ValueError("column %r contains infinite values" % (name,))
    return values


def _is_integer_nature(values: np.ndarray, tolerance: float) -> bool:
    if len(values) == 0:
        return False
    return bool(np.all(np.abs(values - np.rint(values)) <= tolerance))


def _infer_column(
    series: pd.Series,
    *,
    categorical_max_unique: int,
    categorical_ratio: float,
    integer_tolerance: float,
) -> ColumnSpec:
    name = series.name
    nullable = bool(series.isna().any())
    dtype = str(series.dtype)
    observed = series[~series.isna()]
    unique = stable_unique(observed.tolist())
    n_unique = len(unique)

    # Validate numeric finiteness before the constant shortcut: an all-infinity
    # numeric column is not a usable exact constant for model preprocessing.
    if pd.api.types.is_numeric_dtype(series.dtype):
        _numeric_observed(series, name)

    if n_unique <= 1:
        value = unique[0] if unique else None
        return ColumnSpec(
            name=name,
            kind=ColumnType.CONSTANT,
            nullable=nullable,
            dtype=dtype,
            constant_value=value,
        )

    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return ColumnSpec(
            name=name,
            kind=ColumnType.DATETIME,
            nullable=nullable,
            dtype=dtype,
            timezone=_datetime_timezone(series),
        )
    if pd.api.types.is_timedelta64_dtype(series.dtype):
        raise ValueError(
            "column %r has timedelta dtype; use an explicit continuous override "
            "after choosing a unit" % (name,)
        )
    if pd.api.types.is_bool_dtype(series.dtype):
        return ColumnSpec(
            name=name,
            kind=ColumnType.BINARY,
            nullable=nullable,
            dtype=dtype,
            categories=(False, True),
        )
    if isinstance(series.dtype, pd.CategoricalDtype):
        ordered = bool(series.dtype.ordered)
        return ColumnSpec(
            name=name,
            kind=ColumnType.ORDINAL if ordered else ColumnType.CATEGORICAL,
            nullable=nullable,
            dtype=dtype,
            categories=_categorical_values(series),
            ordered=ordered,
        )
    if not pd.api.types.is_numeric_dtype(series.dtype):
        kind = ColumnType.BINARY if n_unique == 2 else ColumnType.CATEGORICAL
        return ColumnSpec(
            name=name,
            kind=kind,
            nullable=nullable,
            dtype=dtype,
            categories=unique,
        )

    values = _numeric_observed(series, name)
    integer_nature = _is_integer_nature(values, integer_tolerance)
    ratio = float(n_unique) / float(max(len(values), 1))
    if integer_nature and n_unique == 2:
        return ColumnSpec(
            name=name,
            kind=ColumnType.BINARY,
            nullable=nullable,
            dtype=dtype,
            categories=unique,
        )
    if integer_nature:
        low_cardinality = (
            n_unique <= categorical_max_unique and ratio <= categorical_ratio
        )
        if low_cardinality:
            return ColumnSpec(
                name=name,
                kind=ColumnType.ORDINAL,
                nullable=nullable,
                dtype=dtype,
                categories=unique,
                ordered=True,
            )
        if bool(np.all(values >= 0.0)):
            return ColumnSpec(
                name=name,
                kind=ColumnType.COUNT,
                nullable=nullable,
                dtype=dtype,
            )
    return ColumnSpec(
        name=name,
        kind=ColumnType.CONTINUOUS,
        nullable=nullable,
        dtype=dtype,
    )


def _override_spec(base: ColumnSpec, series: pd.Series, override: Any) -> ColumnSpec:
    if isinstance(override, ColumnSpec):
        if override.name != base.name:
            raise ValueError(
                "override for column %r has mismatched name %r"
                % (base.name, override.name)
            )
        return override
    if isinstance(override, (str, ColumnType)):
        changes: Dict[str, Any] = {"kind": _normalise_kind(override)}
    elif isinstance(override, Mapping):
        allowed = {
            "kind",
            "type",
            "semantic_type",
            "nullable",
            "dtype",
            "categories",
            "vocabulary",
            "ordered",
            "constant_value",
            "value",
            "timezone",
        }
        unknown = set(override) - allowed
        if unknown:
            raise ValueError(
                "unknown override options for column %r: %r"
                % (base.name, sorted(unknown))
            )
        changes = dict(override)
        aliases = [
            key for key in ("kind", "type", "semantic_type") if key in changes
        ]
        if len(aliases) > 1:
            raise ValueError("override must specify only one of kind, type, semantic_type")
        if aliases:
            changes["kind"] = _normalise_kind(changes.pop(aliases[0]))
        if "vocabulary" in changes:
            if "categories" in changes:
                raise ValueError("override cannot specify categories and vocabulary")
            changes["categories"] = changes.pop("vocabulary")
        if "value" in changes:
            if "constant_value" in changes:
                raise ValueError("override cannot specify value and constant_value")
            changes["constant_value"] = changes.pop("value")
    else:
        raise TypeError(
            "override for column %r must be a kind, mapping, or ColumnSpec"
            % (base.name,)
        )

    target_kind = _normalise_kind(changes.get("kind", base.kind))
    if "categories" not in changes:
        if target_kind in (
            ColumnType.BINARY.value,
            ColumnType.CATEGORICAL.value,
            ColumnType.ORDINAL.value,
        ):
            changes["categories"] = _categorical_values(series)
        else:
            changes["categories"] = ()
    if target_kind == ColumnType.ORDINAL.value and "ordered" not in changes:
        changes["ordered"] = True
    elif target_kind != ColumnType.ORDINAL.value and "ordered" not in changes:
        changes["ordered"] = False
    if target_kind == ColumnType.CONSTANT.value and "constant_value" not in changes:
        values = stable_unique(series.dropna().tolist())
        changes["constant_value"] = values[0] if values else None
    if target_kind == ColumnType.DATETIME.value and "timezone" not in changes:
        changes["timezone"] = _datetime_timezone(series)
    elif target_kind != ColumnType.DATETIME.value and "timezone" not in changes:
        changes["timezone"] = None
    return replace(base, **changes)


def _contains(values: Sequence[Any], candidate: Any) -> bool:
    return any(_value_equal(candidate, value) for value in values)


def _validate_series(
    series: pd.Series,
    spec: ColumnSpec,
    *,
    allow_unknown: bool,
    integer_tolerance: float,
) -> None:
    missing = series.isna()
    if bool(missing.any()) and not spec.nullable:
        raise ValueError(
            "column %r contains missing values but is not nullable" % (spec.name,)
        )
    observed = series[~missing]
    kind = spec.kind
    if pd.api.types.is_numeric_dtype(series.dtype):
        _numeric_observed(observed, spec.name)
    if kind in (ColumnType.CONTINUOUS.value, ColumnType.COUNT.value):
        values = _numeric_observed(observed, spec.name)
        if kind == ColumnType.COUNT.value and len(values):
            if not _is_integer_nature(values, integer_tolerance):
                raise ValueError("count column %r contains non-integer values" % spec.name)
            if bool(np.any(values < 0.0)):
                raise ValueError("count column %r contains negative values" % spec.name)
        return
    if kind == ColumnType.DATETIME.value:
        try:
            parsed = pd.to_datetime(observed, errors="raise")
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "datetime column %r contains invalid timestamps" % (spec.name,)
            ) from exc
        timezone = getattr(parsed.dtype, "tz", None)
        actual_timezone = None if timezone is None else str(timezone)
        if spec.timezone is None and actual_timezone is not None:
            raise ValueError(
                "datetime column %r is timezone-aware but schema is naive" % spec.name
            )
        if spec.timezone is not None and actual_timezone is None and len(parsed):
            raise ValueError(
                "datetime column %r is naive but schema expects timezone %r"
                % (spec.name, spec.timezone)
            )
        return
    if kind == ColumnType.CONSTANT.value:
        invalid = [
            value
            for value in observed.tolist()
            if not _value_equal(_python_scalar(value), spec.constant_value)
        ]
        if invalid:
            raise ValueError(
                "constant column %r contains a value different from %r"
                % (spec.name, spec.constant_value)
            )
        return
    if kind in (
        ColumnType.BINARY.value,
        ColumnType.CATEGORICAL.value,
        ColumnType.ORDINAL.value,
    ):
        if (
            kind == ColumnType.BINARY.value
            and not allow_unknown
            and len(stable_unique(observed.tolist())) > 2
        ):
            raise ValueError("binary column %r has more than two values" % spec.name)
        if spec.categories and not allow_unknown:
            unknown = stable_unique(
                value
                for value in observed.tolist()
                if not _contains(spec.categories, _python_scalar(value))
            )
            if unknown:
                raise ValueError(
                    "column %r contains values outside its vocabulary: %r"
                    % (spec.name, list(unknown))
                )
        return
    raise AssertionError("unhandled column kind %r" % kind)


def infer_schema(
    frame: pd.DataFrame,
    overrides: Optional[Mapping[Any, Any]] = None,
    *,
    categorical_max_unique: int = 32,
    categorical_ratio: float = 0.20,
    integer_tolerance: float = 1e-9,
) -> TableSchema:
    """Infer a deterministic :class:`TableSchema`.

    Numeric inference combines pandas dtype, observed integer nature, unique
    cardinality, and the unique-to-row ratio. Thus a wide-range integer count
    is not classified as categorical solely because it has fewer than an
    arbitrary number of values. Ambiguous columns can always be assigned an
    explicit override.
    """

    _require_frame(frame)
    if isinstance(categorical_max_unique, bool) or int(categorical_max_unique) < 2:
        raise ValueError("categorical_max_unique must be an integer of at least 2")
    categorical_max_unique = int(categorical_max_unique)
    categorical_ratio = float(categorical_ratio)
    if not 0.0 <= categorical_ratio <= 1.0:
        raise ValueError("categorical_ratio must be between 0 and 1")
    integer_tolerance = float(integer_tolerance)
    if integer_tolerance < 0.0 or not math.isfinite(integer_tolerance):
        raise ValueError("integer_tolerance must be a finite non-negative value")

    overrides = {} if overrides is None else dict(overrides)
    unknown_overrides = [name for name in overrides if name not in frame.columns]
    if unknown_overrides:
        raise ValueError("overrides reference unknown columns: %r" % unknown_overrides)

    columns: List[ColumnSpec] = []
    for name in frame.columns:
        series = frame[name]
        base = _infer_column(
            series,
            categorical_max_unique=categorical_max_unique,
            categorical_ratio=categorical_ratio,
            integer_tolerance=integer_tolerance,
        )
        spec = _override_spec(base, series, overrides[name]) if name in overrides else base
        columns.append(spec)
    schema = TableSchema(tuple(columns))
    schema.validate(frame)
    return schema


infer_table_schema = infer_schema


__all__ = [
    "ColumnSpec",
    "ColumnType",
    "SUPPORTED_COLUMN_TYPES",
    "TableSchema",
    "decode_json_value",
    "encode_json_value",
    "infer_schema",
    "infer_table_schema",
]
