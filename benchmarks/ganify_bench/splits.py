"""Deterministic, auditable train/validation/test split handling."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _canonical(value: Any) -> str:
    """Encode row IDs without conflating values such as ``1`` and ``"1"``."""

    if isinstance(value, np.generic):
        value = value.item()
    return json.dumps(
        {"type": type(value).__name__, "value": value},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _score(value: Any, seed: int, group: str) -> str:
    payload = "%d\0%s\0%s" % (seed, group, _canonical(value))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _partition_hash(
    train: Sequence[Any], validation: Sequence[Any], test: Sequence[Any]
) -> str:
    values = {
        "train": sorted(_canonical(item) for item in train),
        "validation": sorted(_canonical(item) for item in validation),
        "test": sorted(_canonical(item) for item in test),
    }
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SplitIndices:
    """Stable row IDs and content hash for one data split."""

    train: Tuple[Any, ...]
    validation: Tuple[Any, ...]
    test: Tuple[Any, ...]
    seed: int
    split_hash: str

    def as_dict(self) -> Dict[str, Any]:
        """Return JSON-friendly split metadata."""

        return {
            "train": list(self.train),
            "validation": list(self.validation),
            "test": list(self.test),
            "seed": self.seed,
            "split_hash": self.split_hash,
        }

    def assert_no_leakage(self) -> None:
        """Raise if any canonical row ID appears in multiple partitions."""

        assert_no_leakage(self)


def assert_no_leakage(split: SplitIndices) -> None:
    """Validate that train, validation, and test IDs are pairwise disjoint."""

    partitions = {
        "train": {_canonical(item) for item in split.train},
        "validation": {_canonical(item) for item in split.validation},
        "test": {_canonical(item) for item in split.test},
    }
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        overlap = partitions[left].intersection(partitions[right])
        if overlap:
            raise ValueError(
                "split leakage between %s and %s for %d row IDs"
                % (left, right, len(overlap))
            )
    observed_hash = _partition_hash(split.train, split.validation, split.test)
    if observed_hash != split.split_hash:
        raise ValueError(
            "split hash mismatch: expected %s, observed %s"
            % (split.split_hash, observed_hash)
        )


def deterministic_split(
    row_ids: Iterable[Any],
    *,
    test_size: float = 0.2,
    validation_size: float = 0.1,
    seed: int = 0,
    stratify: Optional[Iterable[Any]] = None,
) -> SplitIndices:
    """Create an order-independent hash split, optionally stratified.

    Split assignment depends only on row ID, seed, and stratum. Reordering the
    source table cannot move rows between partitions. Duplicate IDs are rejected
    because they make leakage detection ambiguous.
    """

    ids = list(row_ids)
    if len(ids) < 3:
        raise ValueError("at least three row IDs are required")
    if not (0.0 <= test_size < 1.0):
        raise ValueError("test_size must be in [0, 1)")
    if not (0.0 <= validation_size < 1.0):
        raise ValueError("validation_size must be in [0, 1)")
    if test_size + validation_size >= 1.0:
        raise ValueError("test_size + validation_size must be below 1")
    tokens = [_canonical(item) for item in ids]
    if len(tokens) != len(set(tokens)):
        raise ValueError("row IDs must be unique")

    if stratify is None:
        strata = ["__all__"] * len(ids)
    else:
        raw_strata = list(stratify)
        if len(raw_strata) != len(ids):
            raise ValueError("stratify must be row-aligned with row_ids")
        strata = [_canonical(item) for item in raw_strata]

    groups: Dict[str, list] = {}
    for row_id, group in zip(ids, strata):
        groups.setdefault(group, []).append(row_id)

    train = []
    validation = []
    test = []
    for group in sorted(groups):
        ordered = sorted(groups[group], key=lambda item: _score(item, seed, group))
        n_group = len(ordered)
        n_test = int(round(n_group * test_size))
        n_validation = int(round(n_group * validation_size))
        if n_test + n_validation >= n_group and n_group > 0:
            overflow = n_test + n_validation - n_group + 1
            reduction = min(overflow, n_validation)
            n_validation -= reduction
            n_test -= overflow - reduction
        test.extend(ordered[:n_test])
        validation.extend(ordered[n_test : n_test + n_validation])
        train.extend(ordered[n_test + n_validation :])

    # Stable order makes manifests byte-reproducible while assignment remains
    # independent of source order.
    train_tuple = tuple(sorted(train, key=_canonical))
    validation_tuple = tuple(sorted(validation, key=_canonical))
    test_tuple = tuple(sorted(test, key=_canonical))
    split = SplitIndices(
        train=train_tuple,
        validation=validation_tuple,
        test=test_tuple,
        seed=int(seed),
        split_hash=_partition_hash(train_tuple, validation_tuple, test_tuple),
    )
    split.assert_no_leakage()
    if len(split.train) == 0:
        raise ValueError("split produced an empty training partition")
    return split


def split_frame(
    frame: pd.DataFrame, split: SplitIndices, *, id_column: object
) -> Dict[str, pd.DataFrame]:
    """Apply ID-based partitions to a dataframe and recheck coverage."""

    if id_column not in frame.columns:
        raise ValueError("id column %r is missing" % id_column)
    if bool(frame[id_column].duplicated().any()):
        raise ValueError("id column %r must be unique" % id_column)
    indexed = frame.set_index(id_column, drop=False)
    expected = {_canonical(value) for value in indexed.index}
    split_ids = {
        _canonical(value)
        for values in (split.train, split.validation, split.test)
        for value in values
    }
    if split_ids != expected:
        raise ValueError(
            "split IDs do not exactly cover the dataframe: missing=%d, extra=%d"
            % (len(expected - split_ids), len(split_ids - expected))
        )
    return {
        "train": indexed.loc[list(split.train)].reset_index(drop=True),
        "validation": indexed.loc[list(split.validation)].reset_index(drop=True),
        "test": indexed.loc[list(split.test)].reset_index(drop=True),
    }


__all__ = [
    "SplitIndices",
    "assert_no_leakage",
    "deterministic_split",
    "split_frame",
]
