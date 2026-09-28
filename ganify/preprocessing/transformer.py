"""Reversible mixed-type dataframe preprocessing."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ganify.schema import (
    ColumnSpec,
    ColumnType,
    TableSchema,
    decode_json_value,
    encode_json_value,
    infer_schema,
    stable_unique,
)

from ._numeric import (
    DiscreteCopula1D,
    EmpiricalCopula1D,
    ModeAware1D,
    numeric_transform_from_dict,
)


def _value_equal(left: Any, right: Any) -> bool:
    try:
        if bool(pd.isna(left)) and bool(pd.isna(right)):
            return True
    except (TypeError, ValueError):
        pass
    try:
        result = left == right
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _lookup(vocabulary: Sequence[Any], value: Any) -> Optional[int]:
    for index, candidate in enumerate(vocabulary):
        if _value_equal(candidate, value):
            return index
    return None


def _normalise_numeric_transform(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("continuous_transform must be a string")
    canonical = value.strip().lower().replace("-", "_")
    aliases = {
        "empirical": "copula",
        "empirical_cdf": "copula",
        "quantile": "copula",
        "mode": "mode_aware",
        "mode_specific": "mode_aware",
        "modes": "mode_aware",
    }
    canonical = aliases.get(canonical, canonical)
    if canonical not in ("copula", "mode_aware"):
        raise ValueError("continuous_transform must be 'copula' or 'mode-aware'")
    return canonical


def _normalise_unknown_policy(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("handle_unknown must be a string")
    canonical = value.strip().lower().replace("-", "_")
    aliases = {
        "raise": "error",
        "use_unknown": "use_encoded_value",
        "unknown": "use_encoded_value",
    }
    canonical = aliases.get(canonical, canonical)
    if canonical not in ("error", "ignore", "use_encoded_value"):
        raise ValueError(
            "handle_unknown must be 'error', 'ignore', or 'use_encoded_value'"
        )
    return canonical


class TransformedTable(np.ndarray):
    """Numeric ndarray carrying the source index for a direct inverse call."""

    source_index: Optional[pd.Index]

    def __new__(cls, values: np.ndarray, index: Optional[pd.Index] = None):
        instance = np.asarray(values, dtype=np.float64).view(cls)
        instance.source_index = None if index is None else index.copy()
        return instance

    def __array_finalize__(self, source: Optional[np.ndarray]) -> None:
        self.source_index = getattr(source, "source_index", None)


@dataclass(frozen=True)
class HeadSpec:
    """One neural output head in the transformed matrix."""

    column: Any
    name: str
    kind: str
    start: int
    stop: int
    activation: str

    @property
    def width(self) -> int:
        """Number of matrix channels in this head."""

        return self.stop - self.start

    @property
    def slice(self) -> slice:
        """Python slice selecting the head."""

        return slice(self.start, self.stop)

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-safe head metadata."""

        return {
            "column": encode_json_value(self.column),
            "name": self.name,
            "kind": self.kind,
            "start": self.start,
            "stop": self.stop,
            "width": self.width,
            "activation": self.activation,
        }


@dataclass(frozen=True)
class ColumnSlice:
    """Matrix layout for one original dataframe column."""

    column: Any
    kind: str
    start: int
    stop: int
    value_start: int
    value_stop: int
    mask_index: Optional[int]

    @property
    def width(self) -> int:
        """Total channels, including a nullable mask when present."""

        return self.stop - self.start

    @property
    def slice(self) -> slice:
        """Python slice selecting all channels for this column."""

        return slice(self.start, self.stop)

    @property
    def value_slice(self) -> slice:
        """Python slice selecting non-mask channels."""

        return slice(self.value_start, self.value_stop)

    @property
    def mask_slice(self) -> Optional[slice]:
        """One-channel missing-mask slice, if the column is nullable."""

        if self.mask_index is None:
            return None
        return slice(self.mask_index, self.mask_index + 1)

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-safe column layout metadata."""

        return {
            "column": encode_json_value(self.column),
            "kind": self.kind,
            "start": self.start,
            "stop": self.stop,
            "value_start": self.value_start,
            "value_stop": self.value_stop,
            "mask_index": self.mask_index,
        }


class TableTransformer:
    """Fit reversible, neural-friendly transforms to a typed dataframe.

    Parameters
    ----------
    schema:
        Optional explicit :class:`~ganify.schema.TableSchema`. When omitted, a
        deterministic schema is inferred at fit time.
    continuous_transform:
        ``"copula"`` (default) or ``"mode-aware"``. Both are reversible;
        mode-aware encoding emits a residual plus a softmax component head.
    handle_unknown:
        Behavior for categories not observed during fitting: ``"error"``,
        ``"ignore"``, or ``"use_encoded_value"``.
    unknown_value:
        Value reconstructed for explicitly encoded unknown categories.
    schema_overrides:
        Per-column overrides forwarded to :func:`ganify.schema.infer_schema`.

    Notes
    -----
    Every nullable column receives a separate ``missing`` sigmoid head. Value
    channels are filled with deterministic zeros where the mask is active, so
    the returned matrix always contains finite numbers.
    """

    FORMAT_VERSION = 1
    STATE_FILENAME = "preprocessor.json"

    def __init__(
        self,
        schema: Optional[Union[TableSchema, Mapping[str, Any]]] = None,
        *,
        continuous_transform: str = "copula",
        numeric_transform: Optional[str] = None,
        mode_aware: Optional[bool] = None,
        handle_unknown: str = "error",
        unknown_value: Any = "__unknown__",
        max_modes: int = 5,
        schema_overrides: Optional[Mapping[Any, Any]] = None,
        categorical_max_unique: int = 32,
        categorical_ratio: float = 0.20,
        integer_tolerance: float = 1e-9,
        random_state: Optional[int] = None,
    ) -> None:
        if numeric_transform is not None:
            if continuous_transform != "copula":
                raise ValueError(
                    "specify only one of continuous_transform and numeric_transform"
                )
            continuous_transform = numeric_transform
        if mode_aware is not None:
            requested = "mode_aware" if bool(mode_aware) else "copula"
            if continuous_transform != "copula" and (
                _normalise_numeric_transform(continuous_transform) != requested
            ):
                raise ValueError(
                    "mode_aware conflicts with continuous_transform"
                )
            continuous_transform = requested
        if isinstance(schema, Mapping):
            schema = TableSchema.from_dict(schema)
        if schema is not None and not isinstance(schema, TableSchema):
            raise TypeError("schema must be a TableSchema, mapping, or None")
        if isinstance(max_modes, bool) or int(max_modes) < 1:
            raise ValueError("max_modes must be a positive integer")
        if isinstance(categorical_max_unique, bool) or int(categorical_max_unique) < 2:
            raise ValueError("categorical_max_unique must be at least 2")
        if not 0.0 <= float(categorical_ratio) <= 1.0:
            raise ValueError("categorical_ratio must be between 0 and 1")
        if float(integer_tolerance) < 0.0 or not math.isfinite(
            float(integer_tolerance)
        ):
            raise ValueError("integer_tolerance must be finite and non-negative")
        if random_state is not None and (
            isinstance(random_state, bool)
            or not isinstance(random_state, (int, np.integer))
        ):
            raise ValueError("random_state must be an integer or None")

        self.schema = schema
        self.continuous_transform = _normalise_numeric_transform(
            continuous_transform
        )
        self.handle_unknown = _normalise_unknown_policy(handle_unknown)
        encode_json_value(unknown_value)
        self.unknown_value = unknown_value
        self.max_modes = int(max_modes)
        self.schema_overrides = (
            {} if schema_overrides is None else dict(schema_overrides)
        )
        self.categorical_max_unique = int(categorical_max_unique)
        self.categorical_ratio = float(categorical_ratio)
        self.integer_tolerance = float(integer_tolerance)
        self.random_state = None if random_state is None else int(random_state)
        self.fitted_ = False

    def fit(
        self,
        frame: pd.DataFrame,
        y: Any = None,
        *,
        schema: Optional[TableSchema] = None,
        overrides: Optional[Mapping[Any, Any]] = None,
    ) -> "TableTransformer":
        """Fit all per-column transforms and matrix layout metadata."""

        if y is not None:
            raise ValueError("TableTransformer does not consume a target y")
        self._require_frame(frame)
        if schema is not None and self.schema is not None:
            raise ValueError("schema was supplied both at construction and fit time")
        selected_schema = schema if schema is not None else self.schema
        if selected_schema is not None and not isinstance(selected_schema, TableSchema):
            raise TypeError("schema must be a TableSchema")
        if overrides is not None and selected_schema is not None:
            raise ValueError("overrides cannot be combined with an explicit schema")
        if overrides is not None and self.schema_overrides:
            raise ValueError(
                "overrides were supplied both at construction and fit time"
            )
        selected_overrides = (
            dict(overrides) if overrides is not None else self.schema_overrides
        )
        if selected_schema is None:
            selected_schema = infer_schema(
                frame,
                overrides=selected_overrides,
                categorical_max_unique=self.categorical_max_unique,
                categorical_ratio=self.categorical_ratio,
                integer_tolerance=self.integer_tolerance,
            )
        resolved_schema = self._resolve_schema(selected_schema, frame)
        resolved_schema.validate(
            frame,
            integer_tolerance=self.integer_tolerance,
        )

        states: List[Dict[str, Any]] = []
        for spec in resolved_schema:
            states.append(self._fit_column(frame[spec.name], spec))

        self.schema_ = resolved_schema
        self.fitted_schema_ = resolved_schema
        self.column_states_ = states
        self.n_features_in_ = len(resolved_schema)
        self.feature_names_in_ = np.asarray(resolved_schema.names, dtype=object)
        self._build_layout()
        self.fitted_ = True
        return self

    def fit_transform(
        self,
        frame: pd.DataFrame,
        y: Any = None,
        *,
        schema: Optional[TableSchema] = None,
        overrides: Optional[Mapping[Any, Any]] = None,
    ) -> TransformedTable:
        """Fit the transformer and return the encoded matrix."""

        return self.fit(frame, y=y, schema=schema, overrides=overrides).transform(
            frame
        )

    def transform(self, frame: pd.DataFrame) -> TransformedTable:
        """Encode a dataframe into a finite float64 matrix."""

        self._check_fitted()
        self._require_frame(frame)
        self.schema_.validate(
            frame,
            allow_unknown=self.handle_unknown != "error",
            integer_tolerance=self.integer_tolerance,
        )
        output = np.zeros((len(frame), self.output_dim_), dtype=np.float64)
        for spec, state, layout in zip(
            self.schema_, self.column_states_, self.column_layout_
        ):
            series = frame[spec.name]
            missing = series.isna().to_numpy(dtype=bool)
            values = output[:, layout.value_slice]
            self._transform_column(series, missing, spec, state, values)
            if layout.mask_index is not None:
                output[:, layout.mask_index] = missing.astype(np.float64)
        if not bool(np.isfinite(output).all()):
            raise RuntimeError("preprocessing produced a non-finite matrix")
        return TransformedTable(output, frame.index)

    def transform_with_metadata(
        self, frame: pd.DataFrame
    ) -> Tuple[TransformedTable, Mapping[str, Any]]:
        """Encode a dataframe and return matrix metadata alongside it."""

        return self.transform(frame), self.get_metadata()

    def inverse_transform(
        self,
        values: Any,
        *,
        index: Optional[Iterable[Any]] = None,
    ) -> pd.DataFrame:
        """Reconstruct a dataframe from a transformed numeric matrix."""

        self._check_fitted()
        source_index = getattr(values, "source_index", None)
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError("encoded table must be a two-dimensional matrix")
        if array.shape[1] != self.output_dim_:
            raise ValueError(
                "encoded table has %d columns; expected %d"
                % (array.shape[1], self.output_dim_)
            )
        if not bool(np.isfinite(array).all()):
            raise ValueError("encoded table contains NaN or infinite values")
        if index is None:
            result_index = (
                source_index
                if source_index is not None and len(source_index) == len(array)
                else pd.RangeIndex(len(array))
            )
        else:
            result_index = pd.Index(index)
            if len(result_index) != len(array):
                raise ValueError("inverse index length does not match encoded rows")

        columns: Dict[Any, pd.Series] = {}
        for spec, state, layout in zip(
            self.schema_, self.column_states_, self.column_layout_
        ):
            missing = (
                np.zeros(len(array), dtype=bool)
                if layout.mask_index is None
                else array[:, layout.mask_index] >= 0.5
            )
            decoded = self._inverse_column(
                array[:, layout.value_slice], missing, spec, state
            )
            columns[spec.name] = self._restore_series(
                decoded, missing, spec, result_index
            )
        return pd.DataFrame(columns, index=result_index).loc[:, list(self.schema_.names)]

    inverse = inverse_transform

    def get_feature_names_out(self) -> np.ndarray:
        """Return deterministic transformed channel names."""

        self._check_fitted()
        return np.asarray(self.feature_names_out_, dtype=object)

    def get_metadata(self) -> Mapping[str, Any]:
        """Return a defensive JSON-safe copy of output/head metadata."""

        self._check_fitted()
        return json.loads(json.dumps(self.metadata_))

    @property
    def output_slices(self) -> Mapping[Any, slice]:
        """Map original column names to transformed matrix slices."""

        self._check_fitted()
        return dict(self.output_slices_)

    @property
    def head_metadata(self) -> Tuple[HeadSpec, ...]:
        """Immutable neural-head metadata."""

        self._check_fitted()
        return self.head_specs_

    @property
    def metadata(self) -> Mapping[str, Any]:
        """JSON-safe transformed-layout metadata."""

        return self.get_metadata()

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-safe fitted state preserving inverse behavior."""

        self._check_fitted()
        return {
            "format": "ganify-table-transformer",
            "version": self.FORMAT_VERSION,
            "schema": self.schema_.to_dict(),
            "options": {
                "continuous_transform": self.continuous_transform,
                "handle_unknown": self.handle_unknown,
                "unknown_value": encode_json_value(self.unknown_value),
                "max_modes": self.max_modes,
                "categorical_max_unique": self.categorical_max_unique,
                "categorical_ratio": self.categorical_ratio,
                "integer_tolerance": self.integer_tolerance,
                "random_state": self.random_state,
            },
            "columns": [
                self._column_state_to_dict(state)
                for state in self.column_states_
            ],
        }

    state_dict = to_dict

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TableTransformer":
        """Restore a fitted transformer from :meth:`to_dict`."""

        if not isinstance(payload, Mapping):
            raise TypeError("transformer state must be a mapping")
        if payload.get("format") != "ganify-table-transformer":
            raise ValueError("not a GANify table transformer state")
        if int(payload.get("version", -1)) != cls.FORMAT_VERSION:
            raise ValueError(
                "unsupported table transformer format %r" % payload.get("version")
            )
        schema = TableSchema.from_dict(payload["schema"])
        options = payload.get("options", {})
        instance = cls(
            schema=schema,
            continuous_transform=options.get("continuous_transform", "copula"),
            handle_unknown=options.get("handle_unknown", "error"),
            unknown_value=decode_json_value(
                options.get(
                    "unknown_value",
                    encode_json_value("__unknown__"),
                )
            ),
            max_modes=int(options.get("max_modes", 5)),
            categorical_max_unique=int(
                options.get("categorical_max_unique", 32)
            ),
            categorical_ratio=float(options.get("categorical_ratio", 0.20)),
            integer_tolerance=float(options.get("integer_tolerance", 1e-9)),
            random_state=options.get("random_state"),
        )
        states = [
            instance._column_state_from_dict(item)
            for item in payload.get("columns", [])
        ]
        if len(states) != len(schema):
            raise ValueError("transformer state does not match its schema")
        instance.schema_ = schema
        instance.fitted_schema_ = schema
        instance.column_states_ = states
        instance.n_features_in_ = len(schema)
        instance.feature_names_in_ = np.asarray(schema.names, dtype=object)
        instance._build_layout()
        instance.fitted_ = True
        return instance

    from_state_dict = from_dict

    def to_json(self, *, indent: Optional[int] = None) -> str:
        """Serialize fitted state to JSON text."""

        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=None if indent else (",", ":"),
            indent=indent,
        )

    @classmethod
    def from_json(cls, value: Union[str, bytes]) -> "TableTransformer":
        """Restore fitted state from JSON text."""

        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return cls.from_dict(json.loads(value))

    def save(self, path: Union[str, Path]) -> Path:
        """Persist state to a JSON file or a preprocessor directory."""

        destination = Path(path)
        if destination.suffix.lower() == ".json":
            destination.parent.mkdir(parents=True, exist_ok=True)
        else:
            destination.mkdir(parents=True, exist_ok=True)
            destination = destination / self.STATE_FILENAME
        destination.write_text(self.to_json(indent=2) + "\n", encoding="utf-8")
        return destination

    save_state = save

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TableTransformer":
        """Load state written by :meth:`save`."""

        source = Path(path)
        if source.is_dir():
            source = source / cls.STATE_FILENAME
        return cls.from_json(source.read_text(encoding="utf-8"))

    load_state = load

    def _resolve_schema(
        self, schema: TableSchema, frame: pd.DataFrame
    ) -> TableSchema:
        missing = [name for name in schema.names if name not in frame.columns]
        extra = [name for name in frame.columns if name not in schema.names]
        if missing or extra:
            raise ValueError(
                "dataframe columns do not match schema; missing=%r, extra=%r"
                % (missing, extra)
            )
        columns: List[ColumnSpec] = []
        for spec in schema:
            series = frame[spec.name]
            changes: Dict[str, Any] = {}
            if spec.kind in (
                ColumnType.BINARY.value,
                ColumnType.CATEGORICAL.value,
                ColumnType.ORDINAL.value,
            ) and not spec.categories:
                if isinstance(series.dtype, pd.CategoricalDtype):
                    vocabulary = tuple(series.cat.categories.tolist())
                else:
                    vocabulary = stable_unique(series.dropna().tolist())
                changes["categories"] = vocabulary
            if (
                spec.kind == ColumnType.CONSTANT.value
                and spec.constant_value is None
            ):
                observed = stable_unique(series.dropna().tolist())
                if observed:
                    changes["constant_value"] = observed[0]
            columns.append(replace(spec, **changes) if changes else spec)
        return TableSchema(tuple(columns), version=schema.version)

    def _new_numeric_transform(self):
        if self.continuous_transform == "copula":
            return EmpiricalCopula1D()
        return ModeAware1D(max_modes=self.max_modes)

    def _fit_column(
        self, series: pd.Series, spec: ColumnSpec
    ) -> Dict[str, Any]:
        missing = series.isna().to_numpy(dtype=bool)
        kind = spec.kind
        state: Dict[str, Any] = {"kind": kind}
        if kind == ColumnType.CONTINUOUS.value:
            numeric = pd.to_numeric(series[~missing], errors="raise").to_numpy(
                dtype=np.float64
            )
            all_missing = len(numeric) == 0
            fitting = np.asarray([0.0]) if all_missing else numeric
            transform = self._new_numeric_transform().fit(fitting)
            state.update(
                {
                    "transform": transform,
                    "all_missing": all_missing,
                    "minimum": float(np.min(fitting)),
                    "maximum": float(np.max(fitting)),
                }
            )
            return state
        if kind == ColumnType.COUNT.value:
            integers = self._count_values(series[~missing])
            all_missing = len(integers) == 0
            fitting = [0] if all_missing else integers
            origin = min(fitting)
            if self.continuous_transform == "copula":
                transform = DiscreteCopula1D().fit(fitting)
            else:
                deltas = np.asarray(
                    [value - origin for value in fitting], dtype=np.float64
                )
                transform = ModeAware1D(max_modes=self.max_modes).fit(deltas)
            state.update(
                {
                    "transform": transform,
                    "all_missing": all_missing,
                    "minimum": min(fitting),
                    "maximum": max(fitting),
                    "count_origin": origin,
                }
            )
            return state
        if kind in (
            ColumnType.BINARY.value,
            ColumnType.CATEGORICAL.value,
            ColumnType.ORDINAL.value,
        ):
            state["vocabulary"] = tuple(spec.categories)
            return state
        if kind == ColumnType.DATETIME.value:
            nanoseconds = self._datetime_nanoseconds(series, spec)
            observed_ns = nanoseconds[~missing]
            all_missing = len(observed_ns) == 0
            origin_ns = 0 if all_missing else int(np.min(observed_ns))
            seconds = (
                np.asarray([0.0])
                if all_missing
                else (observed_ns.astype(object) - origin_ns).astype(np.float64)
                / 1_000_000_000.0
            )
            transform = EmpiricalCopula1D().fit(seconds)
            if all_missing:
                year_min = 1970
                year_max = 1970
            else:
                calendar = self._datetime_index(observed_ns, spec)
                years = np.asarray(calendar.year, dtype=np.int64)
                year_min = int(years.min())
                year_max = int(years.max())
            state.update(
                {
                    "transform": transform,
                    "all_missing": all_missing,
                    "origin_ns": origin_ns,
                    "year_min": year_min,
                    "year_max": year_max,
                }
            )
            return state
        if kind == ColumnType.CONSTANT.value:
            state["constant_value"] = spec.constant_value
            return state
        raise AssertionError("unhandled column kind %r" % kind)

    def _build_layout(self) -> None:
        offset = 0
        columns: List[ColumnSlice] = []
        heads: List[HeadSpec] = []
        names: List[str] = []
        for spec, state in zip(self.schema_, self.column_states_):
            start = offset
            value_start = offset
            label = str(spec.name)
            if spec.kind in (
                ColumnType.CONTINUOUS.value,
                ColumnType.COUNT.value,
            ):
                transform = state["transform"]
                if isinstance(transform, (EmpiricalCopula1D, DiscreteCopula1D)):
                    heads.append(
                        HeadSpec(
                            spec.name,
                            "value",
                            spec.kind,
                            offset,
                            offset + 1,
                            "tanh",
                        )
                    )
                    names.append("%s__value" % label)
                    offset += 1
                else:
                    heads.append(
                        HeadSpec(
                            spec.name,
                            "residual",
                            spec.kind,
                            offset,
                            offset + 1,
                            "tanh",
                        )
                    )
                    names.append("%s__residual" % label)
                    offset += 1
                    components = transform.output_width - 1
                    heads.append(
                        HeadSpec(
                            spec.name,
                            "component",
                            spec.kind,
                            offset,
                            offset + components,
                            "softmax",
                        )
                    )
                    names.extend(
                        "%s__component_%d" % (label, index)
                        for index in range(components)
                    )
                    offset += components
            elif spec.kind == ColumnType.BINARY.value:
                heads.append(
                    HeadSpec(
                        spec.name,
                        "binary",
                        spec.kind,
                        offset,
                        offset + 1,
                        "sigmoid",
                    )
                )
                names.append("%s__binary" % label)
                offset += 1
                if self.handle_unknown == "use_encoded_value":
                    heads.append(
                        HeadSpec(
                            spec.name,
                            "unknown",
                            spec.kind,
                            offset,
                            offset + 1,
                            "sigmoid",
                        )
                    )
                    names.append("%s__unknown" % label)
                    offset += 1
            elif spec.kind in (
                ColumnType.CATEGORICAL.value,
                ColumnType.ORDINAL.value,
            ):
                width = len(state["vocabulary"])
                if self.handle_unknown == "use_encoded_value":
                    width += 1
                width = max(width, 1)
                heads.append(
                    HeadSpec(
                        spec.name,
                        "vocabulary",
                        spec.kind,
                        offset,
                        offset + width,
                        "softmax",
                    )
                )
                names.extend(
                    "%s__category_%d" % (label, index) for index in range(width)
                )
                offset += width
            elif spec.kind == ColumnType.DATETIME.value:
                heads.append(
                    HeadSpec(
                        spec.name,
                        "timestamp",
                        spec.kind,
                        offset,
                        offset + 1,
                        "tanh",
                    )
                )
                names.append("%s__timestamp" % label)
                offset += 1
                heads.append(
                    HeadSpec(
                        spec.name,
                        "calendar",
                        spec.kind,
                        offset,
                        offset + 9,
                        "tanh",
                    )
                )
                names.extend(
                    [
                        "%s__year" % label,
                        "%s__month_sin" % label,
                        "%s__month_cos" % label,
                        "%s__day_sin" % label,
                        "%s__day_cos" % label,
                        "%s__weekday_sin" % label,
                        "%s__weekday_cos" % label,
                        "%s__time_sin" % label,
                        "%s__time_cos" % label,
                    ]
                )
                offset += 9
            elif spec.kind == ColumnType.CONSTANT.value:
                heads.append(
                    HeadSpec(
                        spec.name,
                        "constant",
                        spec.kind,
                        offset,
                        offset + 1,
                        "linear",
                    )
                )
                names.append("%s__constant" % label)
                offset += 1
            else:
                raise AssertionError("unhandled column kind %r" % spec.kind)

            value_stop = offset
            mask_index: Optional[int] = None
            if spec.nullable:
                mask_index = offset
                heads.append(
                    HeadSpec(
                        spec.name,
                        "missing",
                        "missing",
                        offset,
                        offset + 1,
                        "sigmoid",
                    )
                )
                names.append("%s__missing" % label)
                offset += 1
            columns.append(
                ColumnSlice(
                    column=spec.name,
                    kind=spec.kind,
                    start=start,
                    stop=offset,
                    value_start=value_start,
                    value_stop=value_stop,
                    mask_index=mask_index,
                )
            )

        self.column_layout_ = tuple(columns)
        self.head_specs_ = tuple(heads)
        self.output_info_ = self.head_specs_
        self.output_dim_ = offset
        self.n_features_out_ = offset
        self.feature_names_out_ = tuple(names)
        self.output_slices_ = {
            layout.column: layout.slice for layout in self.column_layout_
        }
        self.slices_ = dict(self.output_slices_)
        self.column_slices_ = dict(self.output_slices_)
        self.value_slices_ = {
            layout.column: layout.value_slice for layout in self.column_layout_
        }
        self.mask_slices_ = {
            layout.column: layout.mask_slice
            for layout in self.column_layout_
            if layout.mask_slice is not None
        }
        self.metadata_ = {
            "output_dim": self.output_dim_,
            "feature_names": list(self.feature_names_out_),
            "columns": [layout.to_dict() for layout in self.column_layout_],
            "heads": [head.to_dict() for head in self.head_specs_],
        }

    def _transform_column(
        self,
        series: pd.Series,
        missing: np.ndarray,
        spec: ColumnSpec,
        state: Mapping[str, Any],
        destination: np.ndarray,
    ) -> None:
        observed = ~missing
        kind = spec.kind
        if kind == ColumnType.CONTINUOUS.value:
            if bool(observed.any()):
                numeric = pd.to_numeric(
                    series[observed], errors="raise"
                ).to_numpy(dtype=np.float64)
                destination[observed, :] = state["transform"].transform(numeric)
            return
        if kind == ColumnType.COUNT.value:
            if bool(observed.any()):
                integers = self._count_values(series[observed])
                if isinstance(state["transform"], DiscreteCopula1D):
                    transformed = state["transform"].transform(integers)
                else:
                    deltas = np.asarray(
                        [value - state["count_origin"] for value in integers],
                        dtype=np.float64,
                    )
                    transformed = state["transform"].transform(deltas)
                destination[observed, :] = transformed
            return
        if kind == ColumnType.BINARY.value:
            vocabulary = state["vocabulary"]
            for row in np.flatnonzero(observed):
                index = _lookup(vocabulary, series.iloc[row])
                if index is None:
                    if self.handle_unknown == "error":
                        raise ValueError(
                            "unknown category %r in column %r"
                            % (series.iloc[row], spec.name)
                        )
                    if self.handle_unknown == "use_encoded_value":
                        destination[row, 1] = 1.0
                    else:
                        destination[row, 0] = 0.5
                else:
                    destination[row, 0] = float(index)
            return
        if kind in (ColumnType.CATEGORICAL.value, ColumnType.ORDINAL.value):
            vocabulary = state["vocabulary"]
            unknown_index = (
                len(vocabulary)
                if self.handle_unknown == "use_encoded_value"
                else None
            )
            for row in np.flatnonzero(observed):
                index = _lookup(vocabulary, series.iloc[row])
                if index is None:
                    if self.handle_unknown == "error":
                        raise ValueError(
                            "unknown category %r in column %r"
                            % (series.iloc[row], spec.name)
                        )
                    if unknown_index is not None:
                        destination[row, unknown_index] = 1.0
                else:
                    destination[row, index] = 1.0
            return
        if kind == ColumnType.DATETIME.value:
            if bool(observed.any()):
                nanoseconds = self._datetime_nanoseconds(series, spec)
                observed_ns = nanoseconds[observed]
                seconds = (
                    (observed_ns.astype(object) - state["origin_ns"])
                    .astype(np.float64)
                    / 1_000_000_000.0
                )
                destination[observed, 0:1] = state["transform"].transform(
                    seconds
                )
                destination[observed, 1:] = self._calendar_channels(
                    observed_ns, spec, state
                )
            return
        if kind == ColumnType.CONSTANT.value:
            # A zero channel lets all-constant tables still have a trainable
            # matrix shape; inverse ignores this channel exactly.
            return
        raise AssertionError("unhandled column kind %r" % kind)

    def _inverse_column(
        self,
        values: np.ndarray,
        missing: np.ndarray,
        spec: ColumnSpec,
        state: Mapping[str, Any],
    ) -> Any:
        kind = spec.kind
        if kind == ColumnType.CONTINUOUS.value:
            return state["transform"].inverse_transform(values)
        if kind == ColumnType.COUNT.value:
            restored = state["transform"].inverse_transform(values)
            if isinstance(state["transform"], DiscreteCopula1D):
                return restored
            result = []
            for value in restored:
                integer = int(round(float(value))) + state["count_origin"]
                integer = min(
                    state["maximum"], max(state["minimum"], integer)
                )
                result.append(integer)
            return result
        if kind == ColumnType.BINARY.value:
            vocabulary = state["vocabulary"]
            restored: List[Any] = []
            for row in range(len(values)):
                if missing[row]:
                    restored.append(None)
                    continue
                unknown = (
                    self.handle_unknown == "use_encoded_value"
                    and values.shape[1] > 1
                    and values[row, 1] >= 0.5
                )
                ignored = (
                    self.handle_unknown == "ignore"
                    and abs(values[row, 0] - 0.5) <= 1e-12
                )
                if unknown or ignored or not vocabulary:
                    restored.append(self.unknown_value)
                else:
                    index = 1 if values[row, 0] >= 0.5 else 0
                    index = min(index, len(vocabulary) - 1)
                    restored.append(vocabulary[index])
            return restored
        if kind in (ColumnType.CATEGORICAL.value, ColumnType.ORDINAL.value):
            vocabulary = state["vocabulary"]
            restored = []
            for row in range(len(values)):
                if missing[row]:
                    restored.append(None)
                    continue
                if not vocabulary or float(np.max(values[row])) <= 0.0:
                    restored.append(self.unknown_value)
                    continue
                index = int(np.argmax(values[row]))
                if index >= len(vocabulary):
                    restored.append(self.unknown_value)
                else:
                    restored.append(vocabulary[index])
            return restored
        if kind == ColumnType.DATETIME.value:
            seconds = state["transform"].inverse_transform(values[:, 0:1])
            delta_ns = np.rint(seconds * 1_000_000_000.0)
            lower = float(np.iinfo(np.int64).min + 1 - state["origin_ns"])
            upper = float(np.iinfo(np.int64).max - state["origin_ns"])
            delta_ns = np.clip(delta_ns, lower, upper).astype(np.int64)
            return delta_ns + np.int64(state["origin_ns"])
        if kind == ColumnType.CONSTANT.value:
            return [state["constant_value"]] * len(values)
        raise AssertionError("unhandled column kind %r" % kind)

    def _restore_series(
        self,
        decoded: Any,
        missing: np.ndarray,
        spec: ColumnSpec,
        index: pd.Index,
    ) -> pd.Series:
        kind = spec.kind
        if kind == ColumnType.DATETIME.value:
            nanoseconds = np.asarray(decoded, dtype=np.int64).copy()
            nanoseconds[missing] = np.iinfo(np.int64).min
            if spec.timezone is None:
                result = pd.Series(
                    pd.to_datetime(nanoseconds, unit="ns", errors="coerce"),
                    index=index,
                    name=spec.name,
                )
            else:
                timestamps = pd.to_datetime(
                    nanoseconds, unit="ns", errors="coerce", utc=True
                ).tz_convert(spec.timezone)
                result = pd.Series(timestamps, index=index, name=spec.name)
            return result

        values = list(decoded)
        for row in np.flatnonzero(missing):
            values[int(row)] = None
        if kind in (ColumnType.CATEGORICAL.value, ColumnType.ORDINAL.value):
            known = all(
                value is None or _lookup(spec.categories, value) is not None
                for value in values
            )
            if spec.dtype == "category":
                if known:
                    categorical = pd.Categorical(
                        values,
                        categories=list(spec.categories),
                        ordered=spec.ordered,
                    )
                    return pd.Series(categorical, index=index, name=spec.name)
        series = pd.Series(values, index=index, name=spec.name)
        return self._cast_series(series, spec)

    @staticmethod
    def _cast_series(series: pd.Series, spec: ColumnSpec) -> pd.Series:
        try:
            if spec.dtype == "object":
                return series.astype(object)
            if spec.dtype.startswith("string"):
                return series.astype(spec.dtype)
            if spec.dtype == "category":
                return series
            return series.astype(spec.dtype)
        except (TypeError, ValueError, OverflowError):
            # Unknown sentinels cannot be inserted into a closed categorical
            # dtype. Returning object preserves the configured behavior.
            return series

    @staticmethod
    def _datetime_nanoseconds(
        series: pd.Series, spec: ColumnSpec
    ) -> np.ndarray:
        parsed = pd.to_datetime(series, errors="raise")
        index = pd.DatetimeIndex(parsed)
        timezone = index.tz
        if spec.timezone is None and timezone is not None:
            raise ValueError(
                "datetime column %r is timezone-aware but schema is naive"
                % spec.name
            )
        if spec.timezone is not None:
            if timezone is None and bool(series.notna().any()):
                raise ValueError(
                    "datetime column %r is naive but schema expects timezone %r"
                    % (spec.name, spec.timezone)
                )
            if timezone is not None:
                index = index.tz_convert(spec.timezone)
        if hasattr(index, "as_unit"):
            index = index.as_unit("ns")
        return np.asarray(index.asi8, dtype=np.int64)

    @staticmethod
    def _datetime_index(
        nanoseconds: np.ndarray, spec: ColumnSpec
    ) -> pd.DatetimeIndex:
        if spec.timezone is None:
            return pd.DatetimeIndex(pd.to_datetime(nanoseconds, unit="ns"))
        return pd.DatetimeIndex(
            pd.to_datetime(nanoseconds, unit="ns", utc=True)
        ).tz_convert(spec.timezone)

    def _calendar_channels(
        self,
        nanoseconds: np.ndarray,
        spec: ColumnSpec,
        state: Mapping[str, Any],
    ) -> np.ndarray:
        calendar = self._datetime_index(nanoseconds, spec)
        year = np.asarray(calendar.year, dtype=np.float64)
        year_span = float(state["year_max"] - state["year_min"])
        year_channel = (
            np.zeros(len(calendar), dtype=np.float64)
            if year_span == 0.0
            else ((year - state["year_min"]) / year_span) * 2.0 - 1.0
        )
        month_angle = (
            2.0 * math.pi * (np.asarray(calendar.month) - 1.0) / 12.0
        )
        day_angle = (
            2.0 * math.pi * (np.asarray(calendar.dayofyear) - 1.0) / 366.0
        )
        weekday_angle = (
            2.0 * math.pi * np.asarray(calendar.dayofweek) / 7.0
        )
        seconds = (
            np.asarray(calendar.hour, dtype=np.float64) * 3600.0
            + np.asarray(calendar.minute, dtype=np.float64) * 60.0
            + np.asarray(calendar.second, dtype=np.float64)
            + np.asarray(calendar.microsecond, dtype=np.float64) / 1_000_000.0
            + np.asarray(calendar.nanosecond, dtype=np.float64)
            / 1_000_000_000.0
        )
        time_angle = 2.0 * math.pi * seconds / 86_400.0
        return np.column_stack(
            [
                year_channel,
                np.sin(month_angle),
                np.cos(month_angle),
                np.sin(day_angle),
                np.cos(day_angle),
                np.sin(weekday_angle),
                np.cos(weekday_angle),
                np.sin(time_angle),
                np.cos(time_angle),
            ]
        )

    def _count_values(self, series: pd.Series) -> List[int]:
        values: List[int] = []
        for raw in series.tolist():
            try:
                integer = int(round(raw))
                close = abs(raw - integer) <= self.integer_tolerance
            except (TypeError, ValueError, OverflowError):
                close = False
                integer = -1
            if not close or integer < 0:
                raise ValueError("count values must be non-negative integers")
            values.append(integer)
        return values

    def _column_state_to_dict(
        self, state: Mapping[str, Any]
    ) -> Dict[str, Any]:
        kind = state["kind"]
        payload: Dict[str, Any] = {"kind": kind}
        if kind == ColumnType.CONTINUOUS.value:
            payload.update(
                {
                    "transform": state["transform"].to_dict(),
                    "all_missing": bool(state["all_missing"]),
                    "minimum": float(state["minimum"]),
                    "maximum": float(state["maximum"]),
                }
            )
        elif kind == ColumnType.COUNT.value:
            payload.update(
                {
                    "transform": state["transform"].to_dict(),
                    "all_missing": bool(state["all_missing"]),
                    "minimum": str(state["minimum"]),
                    "maximum": str(state["maximum"]),
                    "count_origin": str(state["count_origin"]),
                }
            )
        elif kind in (
            ColumnType.BINARY.value,
            ColumnType.CATEGORICAL.value,
            ColumnType.ORDINAL.value,
        ):
            payload["vocabulary"] = [
                encode_json_value(value) for value in state["vocabulary"]
            ]
        elif kind == ColumnType.DATETIME.value:
            payload.update(
                {
                    "transform": state["transform"].to_dict(),
                    "all_missing": bool(state["all_missing"]),
                    "origin_ns": str(state["origin_ns"]),
                    "year_min": int(state["year_min"]),
                    "year_max": int(state["year_max"]),
                }
            )
        elif kind == ColumnType.CONSTANT.value:
            payload["constant_value"] = encode_json_value(
                state["constant_value"]
            )
        else:
            raise AssertionError("unhandled column kind %r" % kind)
        return payload

    @staticmethod
    def _column_state_from_dict(payload: Mapping[str, Any]) -> Dict[str, Any]:
        kind = payload["kind"]
        state: Dict[str, Any] = {"kind": kind}
        if kind == ColumnType.CONTINUOUS.value:
            state.update(
                {
                    "transform": numeric_transform_from_dict(
                        payload["transform"]
                    ),
                    "all_missing": bool(payload.get("all_missing", False)),
                    "minimum": float(payload["minimum"]),
                    "maximum": float(payload["maximum"]),
                }
            )
        elif kind == ColumnType.COUNT.value:
            state.update(
                {
                    "transform": numeric_transform_from_dict(
                        payload["transform"]
                    ),
                    "all_missing": bool(payload.get("all_missing", False)),
                    "minimum": int(payload["minimum"]),
                    "maximum": int(payload["maximum"]),
                    "count_origin": int(payload["count_origin"]),
                }
            )
        elif kind in (
            ColumnType.BINARY.value,
            ColumnType.CATEGORICAL.value,
            ColumnType.ORDINAL.value,
        ):
            state["vocabulary"] = tuple(
                decode_json_value(value)
                for value in payload.get("vocabulary", [])
            )
        elif kind == ColumnType.DATETIME.value:
            state.update(
                {
                    "transform": numeric_transform_from_dict(
                        payload["transform"]
                    ),
                    "all_missing": bool(payload.get("all_missing", False)),
                    "origin_ns": int(payload["origin_ns"]),
                    "year_min": int(payload["year_min"]),
                    "year_max": int(payload["year_max"]),
                }
            )
        elif kind == ColumnType.CONSTANT.value:
            state["constant_value"] = decode_json_value(
                payload["constant_value"]
            )
        else:
            raise ValueError("unknown column state kind %r" % kind)
        return state

    @staticmethod
    def _require_frame(frame: pd.DataFrame) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("TableTransformer requires a pandas DataFrame")
        if frame.columns.has_duplicates:
            raise ValueError("dataframe contains duplicate column names")
        if len(frame) == 0:
            raise ValueError("dataframe must contain at least one row")
        if len(frame.columns) == 0:
            raise ValueError("dataframe must contain at least one column")

    def _check_fitted(self) -> None:
        if not getattr(self, "fitted_", False):
            raise RuntimeError("TableTransformer is not fitted")


__all__ = [
    "ColumnSlice",
    "HeadSpec",
    "TableTransformer",
    "TransformedTable",
]
