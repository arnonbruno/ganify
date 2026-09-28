"""CTGAN-style condition discovery and deterministic row sampling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ganify.preprocessing import HeadSpec, TableTransformer
from ganify.schema import (
    ColumnType,
    decode_json_value,
    encode_json_value,
    stable_unique,
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


def _is_missing(value: Any) -> bool:
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("%s must be a positive integer" % name)
    value = int(value)
    if value < 1:
        raise ValueError("%s must be a positive integer" % name)
    return value


def _json_safe(value: Any) -> Any:
    """Recursively convert NumPy RNG state to strict JSON primitives."""

    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@dataclass(frozen=True)
class ConditionGroup:
    """One mutually exclusive group in the concatenated condition vector."""

    name: Any
    kind: str
    source: str
    head_name: str
    condition_start: int
    condition_stop: int
    output_start: int
    output_stop: int
    activation: str
    labels: Tuple[Any, ...]
    counts: Tuple[int, ...]
    row_indices: Tuple[Tuple[int, ...], ...]
    bin_edges: Tuple[float, ...] = ()
    bin_ids: Tuple[int, ...] = ()
    presence_index: Optional[int] = None

    @property
    def width(self) -> int:
        return self.condition_stop - self.condition_start

    @property
    def condition_slice(self) -> slice:
        return slice(self.condition_start, self.condition_stop)

    @property
    def output_slice(self) -> slice:
        return slice(self.output_start, self.output_stop)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": encode_json_value(self.name),
            "kind": self.kind,
            "source": self.source,
            "head_name": self.head_name,
            "condition_start": self.condition_start,
            "condition_stop": self.condition_stop,
            "output_start": self.output_start,
            "output_stop": self.output_stop,
            "activation": self.activation,
            "labels": [encode_json_value(value) for value in self.labels],
            "counts": list(self.counts),
            "row_indices": [list(rows) for rows in self.row_indices],
            "bin_edges": list(self.bin_edges),
            "bin_ids": list(self.bin_ids),
            "presence_index": self.presence_index,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConditionGroup":
        return cls(
            name=decode_json_value(payload["name"]),
            kind=str(payload["kind"]),
            source=str(payload["source"]),
            head_name=str(payload["head_name"]),
            condition_start=int(payload["condition_start"]),
            condition_stop=int(payload["condition_stop"]),
            output_start=int(payload["output_start"]),
            output_stop=int(payload["output_stop"]),
            activation=str(payload["activation"]),
            labels=tuple(
                decode_json_value(value) for value in payload.get("labels", [])
            ),
            counts=tuple(int(value) for value in payload.get("counts", [])),
            row_indices=tuple(
                tuple(int(row) for row in rows)
                for rows in payload.get("row_indices", [])
            ),
            bin_edges=tuple(float(value) for value in payload.get("bin_edges", [])),
            bin_ids=tuple(int(value) for value in payload.get("bin_ids", [])),
            presence_index=(
                None
                if payload.get("presence_index") is None
                else int(payload["presence_index"])
            ),
        )


@dataclass
class ConditionBatch:
    """A sampled condition and the rows/targets that satisfy it.

    Iteration intentionally yields the four public training values, making
    ``condition, indices, target, mask = sampler.sample_training(...)`` work
    while retaining group/value diagnostics for the engine.
    """

    condition: np.ndarray
    real_indices: np.ndarray
    output_target: np.ndarray
    output_mask: np.ndarray
    group_indices: np.ndarray
    value_indices: np.ndarray

    @property
    def condition_vector(self) -> np.ndarray:
        return self.condition

    @property
    def indices(self) -> np.ndarray:
        return self.real_indices

    @property
    def target(self) -> np.ndarray:
        return self.output_target

    @property
    def mask(self) -> np.ndarray:
        return self.output_mask

    def __iter__(self) -> Iterator[np.ndarray]:
        yield self.condition
        yield self.real_indices
        yield self.output_target
        yield self.output_mask

    def __len__(self) -> int:
        return 4


class ConditionSampler:
    """Discover table conditions and perform CTGAN-style training sampling.

    Discrete values are sampled with ``log(count + 1)`` training weights after
    choosing a condition group uniformly. Generation uses empirical value
    frequencies. Nullable masks and a multiclass target are regular condition
    groups, including the valid one-state case.
    """

    FORMAT_VERSION = 1
    STATE_FILENAME = "condition_sampler.json"

    def __init__(
        self,
        *,
        continuous_bins: Union[int, bool] = 0,
        target_name: Any = None,
        random_state: Optional[int] = None,
    ) -> None:
        if isinstance(continuous_bins, bool):
            continuous_bins = 10 if continuous_bins else 0
        if (
            not isinstance(continuous_bins, (int, np.integer))
            or int(continuous_bins) < 0
            or int(continuous_bins) == 1
        ):
            raise ValueError("continuous_bins must be 0 or an integer >= 2")
        if random_state is not None and (
            isinstance(random_state, bool)
            or not isinstance(random_state, (int, np.integer))
        ):
            raise ValueError("random_state must be an integer or None")
        if target_name is not None:
            encode_json_value(target_name)

        self.continuous_bins = int(continuous_bins)
        self.target_name = target_name
        self.random_state = None if random_state is None else int(random_state)
        self._rng = np.random.default_rng(self.random_state)
        self.fitted_ = False

    @property
    def condition_dim(self) -> int:
        self._check_fitted()
        return self.condition_dim_

    @property
    def output_dim(self) -> int:
        self._check_fitted()
        return self.output_dim_

    @property
    def groups(self) -> Tuple[ConditionGroup, ...]:
        self._check_fitted()
        return self.groups_

    def fit(
        self,
        encoded: Any,
        transformer: TableTransformer,
        target: Any = None,
        *,
        y: Any = None,
        target_name: Any = None,
    ) -> "ConditionSampler":
        if y is not None:
            if target is not None:
                raise ValueError("pass the target once, as target or y")
            target = y
        if not isinstance(transformer, TableTransformer):
            raise TypeError("transformer must be a fitted TableTransformer")
        if not getattr(transformer, "fitted_", False):
            raise RuntimeError("transformer is not fitted")

        source_index = getattr(encoded, "source_index", None)
        target_index = getattr(target, "index", None)
        if (
            source_index is not None
            and isinstance(target_index, pd.Index)
            and not source_index.equals(target_index)
        ):
            raise ValueError("encoded table and target must have the same index")
        matrix = np.asarray(encoded, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("encoded table must be two-dimensional")
        if matrix.shape[0] < 1:
            raise ValueError("encoded table must contain at least one row")
        if matrix.shape[1] != transformer.output_dim_:
            raise ValueError(
                "encoded table has %d columns; transformer emits %d"
                % (matrix.shape[1], transformer.output_dim_)
            )
        if not bool(np.isfinite(matrix).all()):
            raise ValueError("encoded table contains NaN or infinite values")

        resolved_target_name = (
            self.target_name if target_name is None else target_name
        )
        (
            target_matrix,
            target_values,
            resolved_target_name,
            target_dtype,
        ) = self._prepare_target(
            target,
            len(matrix),
            resolved_target_name,
            transformer,
        )

        self.transformer_ = transformer
        self.data_dim_ = int(matrix.shape[1])
        self.target_values_ = target_values
        self.target_name_ = resolved_target_name
        self.target_dtype_ = target_dtype
        self.target_dim_ = (
            0 if target_matrix is None else int(target_matrix.shape[1])
        )
        self.output_dim_ = self.data_dim_ + self.target_dim_
        self.output_matrix_ = (
            matrix.copy()
            if target_matrix is None
            else np.concatenate([matrix, target_matrix], axis=1)
        )
        self.known_table_names_ = tuple(transformer.schema_.names)

        groups: List[ConditionGroup] = []
        offset = 0
        layouts = {
            layout.column: layout for layout in transformer.column_layout_
        }
        states = {
            spec.name: state
            for spec, state in zip(
                transformer.schema_, transformer.column_states_
            )
        }

        for head in transformer.head_metadata:
            layout = layouts[head.column]
            state = states[head.column]
            presence = layout.mask_index
            if head.name == "binary" and head.kind == ColumnType.BINARY.value:
                observed = np.ones(len(matrix), dtype=bool)
                if presence is not None:
                    observed = matrix[:, presence] < 0.5
                codes = (matrix[:, head.start] >= 0.5).astype(np.int64)
                vocabulary = tuple(state.get("vocabulary", ()))
                supported = [
                    code
                    for code in (0, 1)
                    if bool(np.any(observed & (codes == code)))
                ]
                labels = tuple(
                    vocabulary[code] if code < len(vocabulary) else code
                    for code in supported
                )
                rows = tuple(
                    tuple(np.flatnonzero(observed & (codes == code)).tolist())
                    for code in supported
                )
                group = self._new_group(
                    name=head.column,
                    kind=ColumnType.BINARY.value,
                    source="table",
                    head_name=head.name,
                    condition_start=offset,
                    output_start=head.start,
                    output_stop=head.stop,
                    activation="sigmoid",
                    labels=labels,
                    rows=rows,
                    presence_index=presence,
                )
                if group is not None:
                    groups.append(group)
                    offset = group.condition_stop
            elif (
                head.name == "vocabulary"
                and head.kind
                in (ColumnType.CATEGORICAL.value, ColumnType.ORDINAL.value)
            ):
                values = matrix[:, head.slice]
                observed = np.sum(values, axis=1) > 0.0
                if presence is not None:
                    observed &= matrix[:, presence] < 0.5
                codes = np.argmax(values, axis=1)
                vocabulary = tuple(state.get("vocabulary", ()))
                if transformer.handle_unknown == "use_encoded_value":
                    vocabulary = vocabulary + (transformer.unknown_value,)
                supported = [
                    code
                    for code in range(head.width)
                    if bool(np.any(observed & (codes == code)))
                ]
                labels = tuple(
                    vocabulary[code] if code < len(vocabulary) else code
                    for code in supported
                )
                rows = tuple(
                    tuple(np.flatnonzero(observed & (codes == code)).tolist())
                    for code in supported
                )
                group = self._new_group(
                    name=head.column,
                    kind=head.kind,
                    source="table",
                    head_name=head.name,
                    condition_start=offset,
                    output_start=head.start,
                    output_stop=head.stop,
                    activation="softmax",
                    labels=labels,
                    rows=rows,
                    presence_index=presence,
                )
                if group is not None:
                    groups.append(group)
                    offset = group.condition_stop
            elif head.name == "missing" and head.kind == "missing":
                codes = (matrix[:, head.start] >= 0.5).astype(np.int64)
                supported = [
                    code for code in (0, 1) if bool(np.any(codes == code))
                ]
                labels = tuple(bool(code) for code in supported)
                rows = tuple(
                    tuple(np.flatnonzero(codes == code).tolist())
                    for code in supported
                )
                group = self._new_group(
                    name=head.column,
                    kind="missing",
                    source="table",
                    head_name=head.name,
                    condition_start=offset,
                    output_start=head.start,
                    output_stop=head.stop,
                    activation="sigmoid",
                    labels=labels,
                    rows=rows,
                    presence_index=None,
                )
                if group is not None:
                    groups.append(group)
                    offset = group.condition_stop

        if self.continuous_bins:
            decoded = transformer.inverse_transform(matrix)
            for spec, layout in zip(
                transformer.schema_, transformer.column_layout_
            ):
                if spec.kind != ColumnType.CONTINUOUS.value:
                    continue
                observed = ~decoded[spec.name].isna().to_numpy()
                numeric = pd.to_numeric(
                    decoded.loc[observed, spec.name], errors="raise"
                ).to_numpy(dtype=np.float64)
                if len(numeric) == 0:
                    continue
                quantiles = np.linspace(
                    0.0, 1.0, self.continuous_bins + 1
                )[1:-1]
                if len(quantiles):
                    try:
                        cuts = np.quantile(numeric, quantiles, method="linear")
                    except TypeError:  # NumPy < 1.22
                        cuts = np.quantile(
                            numeric, quantiles, interpolation="linear"
                        )
                    edges = np.unique(np.asarray(cuts, dtype=np.float64))
                else:
                    edges = np.empty(0, dtype=np.float64)
                original_rows = np.flatnonzero(observed)
                raw_codes = np.searchsorted(edges, numeric, side="right")
                supported = sorted(np.unique(raw_codes).astype(int).tolist())
                rows = tuple(
                    tuple(original_rows[raw_codes == code].tolist())
                    for code in supported
                )
                labels: List[Any] = []
                for code in supported:
                    lower = None if code == 0 else float(edges[code - 1])
                    upper = (
                        None if code == len(edges) else float(edges[code])
                    )
                    labels.append((lower, upper))
                group = self._new_group(
                    name=spec.name,
                    kind="continuous_bin",
                    source="table",
                    head_name="quantile_bin",
                    condition_start=offset,
                    output_start=layout.value_start,
                    output_stop=layout.value_stop,
                    activation="continuous_bin",
                    labels=tuple(labels),
                    rows=rows,
                    bin_edges=tuple(float(value) for value in edges),
                    bin_ids=tuple(supported),
                    presence_index=layout.mask_index,
                )
                if group is not None:
                    groups.append(group)
                    offset = group.condition_stop

        if target_matrix is not None:
            codes = np.argmax(target_matrix, axis=1)
            rows = tuple(
                tuple(np.flatnonzero(codes == code).tolist())
                for code in range(target_matrix.shape[1])
            )
            group = self._new_group(
                name=resolved_target_name,
                kind="target",
                source="target",
                head_name="target",
                condition_start=offset,
                output_start=self.data_dim_,
                output_stop=self.output_dim_,
                activation="softmax",
                labels=target_values,
                rows=rows,
                presence_index=None,
            )
            if group is not None:
                groups.append(group)
                offset = group.condition_stop

        self.groups_ = tuple(groups)
        self.condition_dim_ = int(offset)
        self._rng = np.random.default_rng(self.random_state)
        self.fitted_ = True
        return self

    fit_transform = fit

    @staticmethod
    def _new_group(
        *,
        name: Any,
        kind: str,
        source: str,
        head_name: str,
        condition_start: int,
        output_start: int,
        output_stop: int,
        activation: str,
        labels: Tuple[Any, ...],
        rows: Tuple[Tuple[int, ...], ...],
        bin_edges: Tuple[float, ...] = (),
        bin_ids: Tuple[int, ...] = (),
        presence_index: Optional[int] = None,
    ) -> Optional[ConditionGroup]:
        if not labels:
            return None
        counts = tuple(len(value) for value in rows)
        if len(labels) != len(rows) or not any(counts):
            return None
        return ConditionGroup(
            name=name,
            kind=kind,
            source=source,
            head_name=head_name,
            condition_start=condition_start,
            condition_stop=condition_start + len(labels),
            output_start=output_start,
            output_stop=output_stop,
            activation=activation,
            labels=tuple(labels),
            counts=counts,
            row_indices=rows,
            bin_edges=bin_edges,
            bin_ids=bin_ids,
            presence_index=presence_index,
        )

    def _prepare_target(
        self,
        target: Any,
        rows: int,
        target_name: Any,
        transformer: TableTransformer,
    ) -> Tuple[Optional[np.ndarray], Tuple[Any, ...], Any, Optional[str]]:
        if target is None:
            if target_name is not None:
                raise ValueError("target_name was supplied without a target")
            return None, (), None, None

        target_index = getattr(target, "index", None)
        if target_index is not None and len(target_index) != rows:
            raise ValueError("target length does not match encoded table")
        inferred_name = getattr(target, "name", None)
        name = target_name
        if name is None:
            name = inferred_name if inferred_name is not None else "target"
            while name in transformer.schema_.names:
                name = "_%s_" % name
        if name in transformer.schema_.names:
            raise ValueError(
                "target_name %r conflicts with a table column" % (name,)
            )
        encode_json_value(name)

        dtype: Optional[str] = None
        if isinstance(target, pd.DataFrame):
            if target.shape[1] == 1:
                dtype = str(target.dtypes.iloc[0])
                array = target.iloc[:, 0].to_numpy()
            else:
                numeric = target.to_numpy(dtype=np.float64)
                self._validate_indicator_target(numeric, rows)
                classes = tuple(target.columns.tolist())
                encode = np.argmax(numeric, axis=1)
                one_hot = np.zeros((rows, len(classes)), dtype=np.float64)
                one_hot[np.arange(rows), encode] = 1.0
                return one_hot, classes, name, "object"
        else:
            dtype = str(getattr(target, "dtype", "object"))
            array = np.asarray(target)

        if array.ndim == 2 and array.shape[1] > 1:
            numeric = np.asarray(array, dtype=np.float64)
            self._validate_indicator_target(numeric, rows)
            classes = tuple(range(numeric.shape[1]))
            encode = np.argmax(numeric, axis=1)
            one_hot = np.zeros((rows, len(classes)), dtype=np.float64)
            one_hot[np.arange(rows), encode] = 1.0
            return one_hot, classes, name, dtype
        if array.ndim == 2 and array.shape[1] == 1:
            array = array[:, 0]
        if array.ndim != 1 or len(array) != rows:
            raise ValueError(
                "target must be one-dimensional labels or a two-dimensional "
                "one-hot matrix"
            )
        values = [value.item() if isinstance(value, np.generic) else value for value in array]
        if any(_is_missing(value) for value in values):
            raise ValueError("target cannot contain missing values")
        classes = stable_unique(values)
        if not classes:
            raise ValueError("target must contain at least one class")
        one_hot = np.zeros((rows, len(classes)), dtype=np.float64)
        for row, value in enumerate(values):
            index = next(
                (
                    position
                    for position, candidate in enumerate(classes)
                    if _value_equal(value, candidate)
                ),
                None,
            )
            if index is None:
                raise RuntimeError("target encoding failed")
            one_hot[row, index] = 1.0
        return one_hot, tuple(classes), name, dtype

    @staticmethod
    def _validate_indicator_target(array: np.ndarray, rows: int) -> None:
        if array.ndim != 2 or array.shape[0] != rows or array.shape[1] < 1:
            raise ValueError("target one-hot matrix has an invalid shape")
        if not bool(np.isfinite(array).all()):
            raise ValueError("target one-hot matrix contains non-finite values")
        if bool(np.any(array < 0.0)) or bool(np.any(np.sum(array, axis=1) <= 0.0)):
            raise ValueError("target one-hot matrix must have a positive class per row")

    @property
    def target_head_spec(self) -> Optional[HeadSpec]:
        self._check_fitted()
        if not self.target_dim_:
            return None
        return HeadSpec(
            column=self.target_name_,
            name="target",
            kind="target",
            start=self.data_dim_,
            stop=self.output_dim_,
            activation="softmax",
        )

    @property
    def output_head_specs(self) -> Tuple[HeadSpec, ...]:
        self._check_fitted()
        if not hasattr(self, "transformer_") or self.transformer_ is None:
            raise RuntimeError(
                "restored sampler does not carry transformer head metadata"
            )
        target_head = self.target_head_spec
        return tuple(self.transformer_.head_metadata) + (
            () if target_head is None else (target_head,)
        )

    def sample_training(self, batch_size: int) -> ConditionBatch:
        self._check_fitted()
        batch_size = _positive_int("batch_size", batch_size)
        condition, target, mask = self._empty_batch(batch_size)
        real = np.empty(batch_size, dtype=np.int64)
        group_indices = np.full(batch_size, -1, dtype=np.int64)
        value_indices = np.full(batch_size, -1, dtype=np.int64)

        if not self.groups_:
            real[:] = self._rng.integers(0, len(self.output_matrix_), size=batch_size)
            return ConditionBatch(
                condition, real, target, mask, group_indices, value_indices
            )

        group_indices[:] = self._rng.integers(
            0, len(self.groups_), size=batch_size
        )
        for row, group_index in enumerate(group_indices.tolist()):
            group = self.groups_[group_index]
            weights = np.log1p(np.asarray(group.counts, dtype=np.float64))
            weights[np.asarray(group.counts) <= 0] = 0.0
            probabilities = weights / np.sum(weights)
            value_index = int(self._rng.choice(group.width, p=probabilities))
            members = group.row_indices[value_index]
            source_row = int(self._rng.choice(members))
            real[row] = source_row
            value_indices[row] = value_index
            self._apply_assignment(
                condition,
                target,
                mask,
                row,
                group,
                value_index,
                source_row,
            )
        return ConditionBatch(
            condition, real, target, mask, group_indices, value_indices
        )

    sample_train = sample_training
    sample_conditional_batch = sample_training

    def sample_generation(
        self,
        rows: int,
        conditions: Optional[Mapping[Any, Any]] = None,
        *,
        return_info: bool = False,
    ) -> Union[np.ndarray, ConditionBatch]:
        self._check_fitted()
        rows = _positive_int("rows", rows)
        if conditions is not None and not isinstance(conditions, Mapping):
            raise TypeError("conditions must be a mapping of names to values")
        condition, target, mask = self._empty_batch(rows)
        real = np.full(rows, -1, dtype=np.int64)
        group_indices = np.full(rows, -1, dtype=np.int64)
        value_indices = np.full(rows, -1, dtype=np.int64)

        if conditions is None:
            if self.groups_:
                group_indices[:] = self._rng.integers(
                    0, len(self.groups_), size=rows
                )
                for row, group_index in enumerate(group_indices.tolist()):
                    group = self.groups_[group_index]
                    weights = np.asarray(group.counts, dtype=np.float64)
                    probabilities = weights / np.sum(weights)
                    value_index = int(
                        self._rng.choice(group.width, p=probabilities)
                    )
                    members = group.row_indices[value_index]
                    source_row = int(self._rng.choice(members))
                    real[row] = source_row
                    value_indices[row] = value_index
                    self._apply_assignment(
                        condition,
                        target,
                        mask,
                        row,
                        group,
                        value_index,
                        source_row,
                    )
        else:
            self._encode_explicit_conditions(
                conditions,
                condition,
                target,
                mask,
                real,
                group_indices,
                value_indices,
            )

        batch = ConditionBatch(
            condition, real, target, mask, group_indices, value_indices
        )
        return batch if return_info else batch.condition

    sample_empirical = sample_generation

    def encode_conditions(
        self,
        conditions: Mapping[Any, Any],
        rows: int = 1,
        *,
        return_info: bool = False,
    ) -> Union[np.ndarray, ConditionBatch]:
        return self.sample_generation(
            rows, conditions=conditions, return_info=return_info
        )

    def _empty_batch(
        self, rows: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.zeros((rows, self.condition_dim_), dtype=np.float32),
            np.zeros((rows, self.output_dim_), dtype=np.float32),
            np.zeros((rows, self.output_dim_), dtype=np.float32),
        )

    def _apply_assignment(
        self,
        condition: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray,
        row: int,
        group: ConditionGroup,
        value_index: int,
        source_row: int,
    ) -> None:
        condition[row, group.condition_start + value_index] = 1.0
        target[row, group.output_slice] = self.output_matrix_[
            source_row, group.output_slice
        ]
        mask[row, group.output_slice] = 1.0
        if group.presence_index is not None and group.kind != "missing":
            target[row, group.presence_index] = 0.0
            mask[row, group.presence_index] = 1.0

    def _encode_explicit_conditions(
        self,
        conditions: Mapping[Any, Any],
        condition: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray,
        real: np.ndarray,
        group_indices: np.ndarray,
        value_indices: np.ndarray,
    ) -> None:
        rows = len(condition)
        known_names = self.known_table_names_ + (
            (() if self.target_name_ is None else (self.target_name_,))
        )
        unknown = [
            name
            for name in conditions
            if not any(_value_equal(name, known) for known in known_names)
        ]
        if unknown:
            raise ValueError("conditions reference unknown names: %r" % unknown)

        for name, raw_values in conditions.items():
            values = self._expand_values(raw_values, rows)
            primary = [
                (index, group)
                for index, group in enumerate(self.groups_)
                if _value_equal(group.name, name) and group.kind != "missing"
            ]
            missing_groups = [
                (index, group)
                for index, group in enumerate(self.groups_)
                if _value_equal(group.name, name) and group.kind == "missing"
            ]
            for row, value in enumerate(values):
                assignments: List[Tuple[int, ConditionGroup, int]] = []
                if _is_missing(value):
                    if not missing_groups:
                        raise ValueError(
                            "column %r has no missing-value condition" % (name,)
                        )
                    group_index, group = missing_groups[0]
                    assignments.append(
                        (group_index, group, self._label_index(group, True))
                    )
                else:
                    if missing_groups:
                        group_index, group = missing_groups[0]
                        assignments.append(
                            (group_index, group, self._label_index(group, False))
                        )
                    if primary:
                        group_index, group = primary[0]
                        if group.kind == "continuous_bin":
                            raw_bin = int(
                                np.searchsorted(
                                    group.bin_edges, float(value), side="right"
                                )
                            )
                            try:
                                selected = group.bin_ids.index(raw_bin)
                            except ValueError as exc:
                                raise ValueError(
                                    "value %r falls in an unsupported bin for %r"
                                    % (value, name)
                                ) from exc
                        else:
                            selected = self._label_index(group, value)
                        assignments.append((group_index, group, selected))
                    elif (
                        self.target_name_ is not None
                        and _value_equal(name, self.target_name_)
                    ):
                        raise ValueError("target has no condition group")

                for group_index, group, selected in assignments:
                    members = group.row_indices[selected]
                    if not members:
                        raise ValueError(
                            "condition %r=%r has no training support"
                            % (name, value)
                        )
                    source_row = int(members[0])
                    self._apply_assignment(
                        condition,
                        target,
                        mask,
                        row,
                        group,
                        selected,
                        source_row,
                    )
                    if group_indices[row] < 0:
                        group_indices[row] = group_index
                        value_indices[row] = selected
                        real[row] = source_row

    @staticmethod
    def _expand_values(value: Any, rows: int) -> List[Any]:
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

    @staticmethod
    def _label_index(group: ConditionGroup, value: Any) -> int:
        for index, candidate in enumerate(group.labels):
            if _value_equal(value, candidate):
                return index
        raise ValueError(
            "unsupported condition value %r for %r; expected one of %r"
            % (value, group.name, list(group.labels))
        )

    def decode_target(self, generated: Any) -> Optional[pd.Series]:
        self._check_fitted()
        if not self.target_dim_:
            return None
        array = np.asarray(generated)
        if array.ndim != 2 or array.shape[1] != self.output_dim_:
            raise ValueError("generated matrix has the wrong output width")
        codes = np.argmax(array[:, self.data_dim_ : self.output_dim_], axis=1)
        values = [self.target_values_[int(code)] for code in codes]
        result = pd.Series(values, name=self.target_name_)
        if self.target_dtype_ is not None:
            try:
                result = result.astype(self.target_dtype_)
            except (TypeError, ValueError):
                pass
        return result

    def to_dict(self) -> Dict[str, Any]:
        self._check_fitted()
        return {
            "format": "ganify-condition-sampler",
            "version": self.FORMAT_VERSION,
            "options": {
                "continuous_bins": self.continuous_bins,
                "target_name": (
                    None
                    if self.target_name is None
                    else encode_json_value(self.target_name)
                ),
                "random_state": self.random_state,
            },
            "data_dim": self.data_dim_,
            "output_dim": self.output_dim_,
            "condition_dim": self.condition_dim_,
            "known_table_names": [
                encode_json_value(value) for value in self.known_table_names_
            ],
            "target": {
                "name": (
                    None
                    if self.target_name_ is None
                    else encode_json_value(self.target_name_)
                ),
                "values": [
                    encode_json_value(value) for value in self.target_values_
                ],
                "dtype": self.target_dtype_,
                "dim": self.target_dim_,
            },
            "groups": [group.to_dict() for group in self.groups_],
            "output_matrix": self.output_matrix_.tolist(),
            "rng_state": _json_safe(self._rng.bit_generator.state),
        }

    state_dict = to_dict

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConditionSampler":
        if not isinstance(payload, Mapping):
            raise TypeError("condition sampler state must be a mapping")
        if payload.get("format") != "ganify-condition-sampler":
            raise ValueError("not a GANify condition sampler state")
        if int(payload.get("version", -1)) != cls.FORMAT_VERSION:
            raise ValueError("unsupported condition sampler format")
        options = payload.get("options", {})
        encoded_name = options.get("target_name")
        instance = cls(
            continuous_bins=int(options.get("continuous_bins", 0)),
            target_name=(
                None if encoded_name is None else decode_json_value(encoded_name)
            ),
            random_state=options.get("random_state"),
        )
        instance.data_dim_ = int(payload["data_dim"])
        instance.output_dim_ = int(payload["output_dim"])
        instance.condition_dim_ = int(payload["condition_dim"])
        instance.known_table_names_ = tuple(
            decode_json_value(value)
            for value in payload.get("known_table_names", [])
        )
        target = payload.get("target", {})
        target_name = target.get("name")
        instance.target_name_ = (
            None if target_name is None else decode_json_value(target_name)
        )
        instance.target_values_ = tuple(
            decode_json_value(value) for value in target.get("values", [])
        )
        instance.target_dtype_ = target.get("dtype")
        instance.target_dim_ = int(target.get("dim", 0))
        instance.groups_ = tuple(
            ConditionGroup.from_dict(group)
            for group in payload.get("groups", [])
        )
        instance.output_matrix_ = np.asarray(
            payload.get("output_matrix", []), dtype=np.float64
        )
        if instance.output_matrix_.ndim != 2 or (
            instance.output_matrix_.shape[1] != instance.output_dim_
        ):
            raise ValueError("invalid condition sampler output matrix")
        if not bool(np.isfinite(instance.output_matrix_).all()):
            raise ValueError("condition sampler output matrix is non-finite")
        instance.transformer_ = None
        rng_state = payload.get("rng_state")
        if rng_state is not None:
            instance._rng.bit_generator.state = dict(rng_state)
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
    def from_json(cls, value: Union[str, bytes]) -> "ConditionSampler":
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
        destination.write_text(self.to_json(indent=2) + "\n", encoding="utf-8")
        return destination

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ConditionSampler":
        source = Path(path)
        if source.is_dir():
            source = source / cls.STATE_FILENAME
        return cls.from_json(source.read_text(encoding="utf-8"))

    def _check_fitted(self) -> None:
        if not getattr(self, "fitted_", False):
            raise RuntimeError("ConditionSampler is not fitted")


__all__ = ["ConditionBatch", "ConditionGroup", "ConditionSampler"]
