"""Anti-gaming controls for validating synthetic-table metrics."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ._utils import as_frame


def bootstrap_control(
    real: object,
    *,
    n_rows: Optional[int] = None,
    random_state: int = 0,
) -> pd.DataFrame:
    """Sample real rows with replacement.

    This positive fidelity control deliberately memorizes: every output row is
    an exact training match. A novelty metric that does not expose that fact is
    unsuitable for release gating.
    """

    frame = as_frame(real, name="real")
    count = len(frame) if n_rows is None else int(n_rows)
    if count <= 0:
        raise ValueError("n_rows must be positive")
    rng = np.random.default_rng(random_state)
    indices = rng.integers(0, len(frame), size=count)
    return frame.iloc[indices].reset_index(drop=True)


def independently_permuted_columns(
    real: object,
    *,
    random_state: int = 0,
) -> pd.DataFrame:
    """Preserve each empirical marginal while destroying row dependence.

    Every column receives a separate deterministic permutation. The operation
    keeps values, dtypes, missingness, and sample size exactly unchanged.
    """

    frame = as_frame(real, name="real")
    rng = np.random.default_rng(random_state)
    output = frame.copy()
    for column in output.columns:
        permutation = rng.permutation(len(output))
        output[column] = frame[column].iloc[permutation].to_numpy()
    return output.reset_index(drop=True)


def permuted_columns_control(
    real: object,
    *,
    random_state: int = 0,
) -> pd.DataFrame:
    """Alias for :func:`independently_permuted_columns`."""

    return independently_permuted_columns(real, random_state=random_state)


__all__ = [
    "bootstrap_control",
    "independently_permuted_columns",
    "permuted_columns_control",
]
