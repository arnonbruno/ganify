"""Internal validation and preprocessing helpers for table evaluation."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def as_frame(data: object, *, name: str = "data") -> pd.DataFrame:
    """Return *data* as a defensive-copy dataframe.

    Arrays are assigned stable ``column_<n>`` names. Evaluation intentionally
    rejects non-two-dimensional inputs because silently flattening a table can
    invalidate dependence metrics.
    """

    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    else:
        array = np.asarray(data)
        if array.ndim != 2:
            raise ValueError("%s must be a two-dimensional table" % name)
        frame = pd.DataFrame(
            array, columns=["column_%d" % index for index in range(array.shape[1])]
        )
    if frame.columns.has_duplicates:
        raise ValueError("%s contains duplicate column names" % name)
    if len(frame) == 0:
        raise ValueError("%s must contain at least one row" % name)
    return frame


def aligned_frames(
    reference: object, candidate: object
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Validate and align two tables without silently dropping columns."""

    left = as_frame(reference, name="reference")
    right = as_frame(candidate, name="candidate")
    missing = [column for column in left.columns if column not in right.columns]
    extra = [column for column in right.columns if column not in left.columns]
    if missing or extra:
        raise ValueError(
            "candidate columns do not match reference; missing=%r, extra=%r"
            % (missing, extra)
        )
    return left, right.loc[:, left.columns].copy()


def infer_categorical_columns(
    frame: pd.DataFrame, categorical_columns: Optional[Iterable[object]] = None
) -> List[object]:
    """Resolve categorical columns, validating any explicit column names."""

    if categorical_columns is None:
        return [
            column
            for column in frame.columns
            if (
                pd.api.types.is_object_dtype(frame[column].dtype)
                or isinstance(frame[column].dtype, pd.CategoricalDtype)
                or pd.api.types.is_bool_dtype(frame[column].dtype)
                or pd.api.types.is_string_dtype(frame[column].dtype)
            )
        ]
    columns = list(categorical_columns)
    unknown = [column for column in columns if column not in frame.columns]
    if unknown:
        raise ValueError("unknown categorical columns: %r" % unknown)
    return columns


def numeric_columns(
    frame: pd.DataFrame, categorical_columns: Sequence[object]
) -> List[object]:
    """Return columns treated as numeric and reject non-convertible values."""

    categorical = set(categorical_columns)
    columns = [column for column in frame.columns if column not in categorical]
    for column in columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        introduced = converted.isna() & ~frame[column].isna()
        if bool(introduced.any()):
            raise ValueError(
                "column %r is not numeric; declare it categorical" % column
            )
    return columns


def make_preprocessor(
    frame: pd.DataFrame,
    categorical_columns: Optional[Iterable[object]] = None,
) -> Tuple[ColumnTransformer, List[object], List[object]]:
    """Build a deterministic mixed-table encoder fitted by the caller."""

    categorical = infer_categorical_columns(frame, categorical_columns)
    numeric = numeric_columns(frame, categorical)
    transformers = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(
                                strategy="constant", fill_value="__missing__"
                            ),
                        ),
                        (
                            "one_hot",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                categorical,
            )
        )
    if not transformers:
        raise ValueError("table must contain at least one evaluable column")
    return ColumnTransformer(transformers, remainder="drop"), numeric, categorical


def dense_float(values: object) -> np.ndarray:
    """Convert a dense or scipy-sparse matrix to a finite float array."""

    if hasattr(values, "toarray"):
        values = values.toarray()
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("encoded table must be two-dimensional")
    if not bool(np.isfinite(array).all()):
        raise ValueError("encoded table contains non-finite values")
    return array


def finite_numeric(values: object, *, name: str) -> np.ndarray:
    """Return finite, non-missing numeric observations from one column."""

    series = pd.to_numeric(pd.Series(values), errors="coerce")
    array = series.to_numpy(dtype=float)
    if bool(np.isinf(array).any()):
        raise ValueError("%s contains infinite values" % name)
    return array[np.isfinite(array)]
