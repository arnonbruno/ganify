"""Pinned, truth-known synthetic tables for offline release checks.

The controlled panel is deliberately generated rather than downloaded.  It
contains several failure modes that a tabular synthesizer can otherwise hide:
multi-modal continuous data, nonlinear and XOR relationships, asymmetric tail
dependence, a long-tailed categorical distribution, three missingness
mechanisms, and exact structural constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from .run import hash_dataframe, hash_json


CONTROLLED_PANEL_DATASET_ID = "controlled_panel"
CONTROLLED_PANEL_VERSION = "ganify-controlled-v1"
CONTROLLED_PANEL_SEED = 20240517
CONTROLLED_PANEL_ROWS = 10_000
CONTROLLED_PANEL_COLUMNS = 100


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(
        dtype=np.float64
    )


@dataclass(frozen=True)
class KnownConstraint:
    """Small serializable constraint used by the generated fixture.

    It implements the ``evaluate(frame)`` protocol consumed by
    :mod:`ganify.evaluation` without importing TensorFlow-backed package APIs.
    """

    name: str
    kind: str
    columns: Tuple[str, ...]
    options: Mapping[str, Any]

    def evaluate(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [column for column in self.columns if column not in frame]
        if missing:
            raise ValueError(
                "controlled constraint %r is missing columns %r"
                % (self.name, missing)
            )
        tolerance = float(self.options.get("tolerance", 1e-9))
        if self.kind == "range":
            values = _numeric(frame, self.columns[0])
            valid = np.isfinite(values)
            lower = self.options.get("minimum")
            upper = self.options.get("maximum")
            if lower is not None:
                valid &= values + tolerance >= float(lower)
            if upper is not None:
                valid &= values <= float(upper) + tolerance
            return valid
        if self.kind == "inequality":
            left = _numeric(frame, self.columns[0])
            right = _numeric(frame, self.columns[1])
            return (
                np.isfinite(left)
                & np.isfinite(right)
                & (left <= right + tolerance)
            )
        if self.kind == "equality":
            right = np.zeros(len(frame), dtype=np.float64)
            coefficients = tuple(
                float(value)
                for value in self.options.get(
                    "coefficients", (1.0,) * len(self.columns)
                )
            )
            if len(coefficients) != len(self.columns):
                raise RuntimeError("controlled equality coefficients are invalid")
            for column, coefficient in zip(self.columns, coefficients):
                right += coefficient * _numeric(frame, column)
            target = float(self.options.get("target", 0.0))
            return np.isfinite(right) & (np.abs(right - target) <= tolerance)
        if self.kind == "allowed":
            return frame[self.columns[0]].isin(
                list(self.options["values"])
            ).to_numpy(dtype=bool)
        if self.kind == "implication":
            antecedent = (
                frame[self.columns[0]] == self.options["if_value"]
            ).fillna(False)
            consequent = (
                frame[self.columns[1]] == self.options["then_value"]
            ).fillna(False)
            return ((~antecedent) | consequent).to_numpy(dtype=bool)
        raise RuntimeError("unknown controlled constraint kind %r" % self.kind)

    @property
    def label(self) -> str:
        return self.name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "columns": list(self.columns),
            "options": dict(self.options),
        }


def _constraints() -> Tuple[KnownConstraint, ...]:
    return (
        KnownConstraint(
            "duration_nonnegative",
            "range",
            ("duration",),
            {"minimum": 0.0, "tolerance": 1e-12},
        ),
        KnownConstraint(
            "start_not_after_end",
            "inequality",
            ("start", "end"),
            {"tolerance": 1e-12},
        ),
        KnownConstraint(
            "end_equals_start_plus_duration",
            "equality",
            ("end", "start", "duration"),
            {
                "coefficients": (1.0, -1.0, -1.0),
                "target": 0.0,
                "tolerance": 1e-10,
            },
        ),
        KnownConstraint(
            "shares_sum_to_one",
            "equality",
            ("share_a", "share_b", "share_c"),
            {
                "coefficients": (1.0, 1.0, 1.0),
                "target": 1.0,
                "tolerance": 1e-10,
            },
        ),
        KnownConstraint(
            "child_count_not_above_total",
            "inequality",
            ("child_count", "total_count"),
            {"tolerance": 0.0},
        ),
        KnownConstraint(
            "vip_requires_premium",
            "implication",
            ("segment", "premium"),
            {"if_value": "vip", "then_value": True},
        ),
    )


@dataclass(frozen=True)
class ControlledPanel:
    """Generated data together with its pinned data-generating truth."""

    frame: pd.DataFrame
    truth_frame: pd.DataFrame
    truth: Mapping[str, Any]
    constraints: Tuple[KnownConstraint, ...]
    schema_overrides: Mapping[str, str]

    @property
    def data(self) -> pd.DataFrame:
        """Alias used by dataset-oriented callers."""

        return self.frame

    @property
    def complete_data(self) -> pd.DataFrame:
        """Unmasked values and propensities, excluded from model inputs."""

        return self.truth_frame

    @property
    def dataset_id(self) -> str:
        return CONTROLLED_PANEL_DATASET_ID

    @property
    def data_hash(self) -> str:
        return hash_dataframe(self.frame)

    @property
    def truth_hash(self) -> str:
        return hash_json(self.truth)

    def manifest(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "version": CONTROLLED_PANEL_VERSION,
            "rows": len(self.frame),
            "columns": len(self.frame.columns),
            "data_sha256": self.data_hash,
            "truth_data_sha256": hash_dataframe(self.truth_frame),
            "truth_sha256": self.truth_hash,
            "constraints": [
                constraint.to_dict() for constraint in self.constraints
            ],
        }


def generate_controlled_panel(
    rows: int = CONTROLLED_PANEL_ROWS,
    *,
    seed: int = CONTROLLED_PANEL_SEED,
    n_rows: Optional[int] = None,
    columns: int = CONTROLLED_PANEL_COLUMNS,
    zipf_categories: int = 64,
) -> ControlledPanel:
    """Generate the versioned controlled panel deterministically.

    ``n_rows`` is accepted as an explicit alias for callers that use dataset
    terminology.  The same arguments produce byte-identical dataframe hashes
    across repeated runs on the same NumPy/pandas serialization contract.
    """

    if n_rows is not None:
        if rows != CONTROLLED_PANEL_ROWS and int(rows) != int(n_rows):
            raise ValueError("pass rows or n_rows, not conflicting values")
        rows = int(n_rows)
    if isinstance(rows, bool) or int(rows) < 32:
        raise ValueError("controlled panel rows must be an integer >= 32")
    if isinstance(seed, bool):
        raise ValueError("controlled panel seed must be an integer")
    if isinstance(zipf_categories, bool) or int(zipf_categories) < 8:
        raise ValueError("zipf_categories must be an integer >= 8")
    if isinstance(columns, bool) or int(columns) < 32:
        raise ValueError("controlled panel columns must be an integer >= 32")
    rows = int(rows)
    seed = int(seed)
    columns = int(columns)
    zipf_categories = int(zipf_categories)
    rng = np.random.default_rng(seed)

    # Multi-modal and nonlinear relationships.
    mixture_component = rng.choice(3, size=rows, p=(0.55, 0.30, 0.15))
    mixture_location = np.asarray((-2.5, 0.75, 4.5))[mixture_component]
    mixture_scale = np.asarray((0.45, 1.15, 0.30))[mixture_component]
    mixture_x = mixture_location + mixture_scale * rng.normal(size=rows)
    mixture_y = (
        -0.35 * mixture_location
        + np.asarray((0.25, 0.80, 0.45))[mixture_component]
        * rng.normal(size=rows)
        + 0.65 * mixture_x
    )
    nonlinear_x = rng.uniform(-3.0, 3.0, size=rows)
    nonlinear_y = (
        np.sin(1.7 * nonlinear_x)
        + 0.18 * np.square(nonlinear_x)
        + rng.normal(scale=0.12, size=rows)
    )
    xor_left = rng.integers(0, 2, size=rows, dtype=np.int8)
    xor_right = rng.integers(0, 2, size=rows, dtype=np.int8)
    target = np.logical_xor(xor_left, xor_right).astype(np.int8)

    # Marshall-Olkin sampling for Clayton lower-tail dependence.  A survival
    # transform provides a paired upper-tail regime without SciPy.
    clayton_theta = 2.0
    common_lower = rng.gamma(
        shape=1.0 / clayton_theta, scale=1.0, size=rows
    )
    exponential_lower = rng.exponential(size=(rows, 2))
    lower_tail = np.power(
        1.0 + exponential_lower / common_lower[:, None],
        -1.0 / clayton_theta,
    )
    common_upper = rng.gamma(
        shape=1.0 / clayton_theta, scale=1.0, size=rows
    )
    exponential_upper = rng.exponential(size=(rows, 2))
    upper_tail = 1.0 - np.power(
        1.0 + exponential_upper / common_upper[:, None],
        -1.0 / clayton_theta,
    )

    zipf_rank = np.minimum(
        rng.zipf(a=1.35, size=rows), zipf_categories
    ).astype(np.int64)
    zipf_category = np.asarray(
        ["zipf_%03d" % value for value in zipf_rank], dtype=object
    )

    # Exact structural relationships.
    start = rng.uniform(0.0, 100.0, size=rows)
    duration = rng.gamma(shape=2.0, scale=2.5, size=rows)
    end = start + duration
    shares = rng.dirichlet((0.7, 1.3, 2.1), size=rows)
    total_count = rng.poisson(lam=12.0, size=rows).astype(np.int64)
    child_count = np.asarray(
        [rng.binomial(int(total), 0.35) for total in total_count],
        dtype=np.int64,
    )
    segment = rng.choice(
        np.asarray(["standard", "plus", "vip"], dtype=object),
        size=rows,
        p=(0.72, 0.23, 0.05),
    )
    premium = rng.random(rows) < 0.28
    premium[segment == "vip"] = True

    # The observed drivers make MAR identifiable; MNAR depends on the masked
    # value itself.  Mask columns retain truth for mechanism-specific metrics.
    missing_driver = (
        0.55 * nonlinear_x
        + 0.45 * xor_left.astype(np.float64)
        - 0.25 * mixture_component
    )
    mcar_complete = rng.normal(size=rows)
    mar_complete = 0.75 * mixture_x + rng.normal(scale=0.7, size=rows)
    mnar_complete = rng.standard_t(df=4.0, size=rows)
    mcar_probability = 0.12
    mar_probability = _sigmoid(-2.0 + 0.70 * missing_driver)
    mnar_probability = _sigmoid(-2.1 + 0.90 * mnar_complete)
    mcar_missing = rng.random(rows) < mcar_probability
    mar_missing = rng.random(rows) < mar_probability
    mnar_missing = rng.random(rows) < mnar_probability
    mcar_value = mcar_complete.copy()
    mar_value = mar_complete.copy()
    mnar_value = mnar_complete.copy()
    mcar_value[mcar_missing] = np.nan
    mar_value[mar_missing] = np.nan
    mnar_value[mnar_missing] = np.nan

    frame_values: Dict[str, Any] = {
            "row_id": [
                "controlled-v1-%08d" % index for index in range(rows)
            ],
            "mixture_component": np.asarray(
                ["component_%d" % value for value in mixture_component],
                dtype=object,
            ),
            "mixture_x": mixture_x,
            "mixture_y": mixture_y,
            "nonlinear_x": nonlinear_x,
            "nonlinear_y": nonlinear_y,
            "xor_left": xor_left,
            "xor_right": xor_right,
            "lower_tail_u": lower_tail[:, 0],
            "lower_tail_v": lower_tail[:, 1],
            "upper_tail_u": upper_tail[:, 0],
            "upper_tail_v": upper_tail[:, 1],
            "zipf_rank": zipf_rank,
            "zipf_category": zipf_category,
            "missing_driver": missing_driver,
            "mcar_value": mcar_value,
            "mar_value": mar_value,
            "mnar_value": mnar_value,
            "mcar_missing": mcar_missing,
            "mar_missing": mar_missing,
            "mnar_missing": mnar_missing,
            "start": start,
            "duration": duration,
            "end": end,
            "share_a": shares[:, 0],
            "share_b": shares[:, 1],
            "share_c": shares[:, 2],
            "total_count": total_count,
            "child_count": child_count,
            "segment": segment,
            "premium": premium,
            "target": target,
    }
    nuisance_count = columns - len(frame_values)
    if nuisance_count:
        factors = rng.normal(size=(rows, 5))
        weights = rng.normal(
            scale=0.45, size=(nuisance_count, factors.shape[1])
        )
        noise_scale = np.linspace(0.15, 1.0, nuisance_count)
        for index in range(nuisance_count):
            frame_values["nuisance_%04d" % index] = (
                factors @ weights[index]
                + rng.normal(scale=noise_scale[index], size=rows)
            )
    frame = pd.DataFrame(frame_values)
    frame.attrs.update(
        {
            "dataset_id": CONTROLLED_PANEL_DATASET_ID,
            "version": CONTROLLED_PANEL_VERSION,
            "seed": seed,
        }
    )
    truth_frame = pd.DataFrame(
        {
            "row_id": frame["row_id"].copy(),
            "mcar_complete": mcar_complete,
            "mar_complete": mar_complete,
            "mnar_complete": mnar_complete,
            "mcar_probability": np.full(rows, mcar_probability),
            "mar_probability": mar_probability,
            "mnar_probability": mnar_probability,
            "mixture_component_code": mixture_component,
            "xor_target": target,
        }
    )
    truth_frame.attrs.update(frame.attrs)
    constraints = _constraints()
    if not all(bool(np.all(item.evaluate(frame))) for item in constraints):
        raise RuntimeError("controlled panel generator violated pinned constraints")

    truth: Dict[str, Any] = {
        "dataset_id": CONTROLLED_PANEL_DATASET_ID,
        "version": CONTROLLED_PANEL_VERSION,
        "seed": seed,
        "rows": rows,
        "columns": columns,
        "mixture": {
            "weights": [0.55, 0.30, 0.15],
            "locations": [-2.5, 0.75, 4.5],
            "scales": [0.45, 1.15, 0.30],
        },
        "nonlinear": {
            "equation": "sin(1.7*x) + 0.18*x^2 + Normal(0, 0.12)",
        },
        "xor": {
            "equation": "target = xor_left XOR xor_right",
            "label_noise": 0.0,
        },
        "tail_dependence": {
            "family": "Clayton and survival-Clayton",
            "theta": clayton_theta,
            "lower_tail_coefficient": float(
                np.power(2.0, -1.0 / clayton_theta)
            ),
            "upper_tail_coefficient": float(
                np.power(2.0, -1.0 / clayton_theta)
            ),
        },
        "zipf": {
            "exponent": 1.35,
            "categories": zipf_categories,
            "clipped_final_bucket": True,
        },
        "missingness": {
            "mcar": {
                "probability": mcar_probability,
                "observed_rate": float(np.mean(mcar_missing)),
            },
            "mar": {
                "logit": "-2.0 + 0.70*missing_driver",
                "observed_rate": float(np.mean(mar_missing)),
            },
            "mnar": {
                "logit": "-2.1 + 0.90*complete_value",
                "observed_rate": float(np.mean(mnar_missing)),
            },
        },
        "constraints": [item.to_dict() for item in constraints],
        "target": "target",
        "id_column": "row_id",
        "nuisance_features": {
            "count": nuisance_count,
            "latent_factors": 5,
            "noise_scale_range": [0.15, 1.0],
        },
    }
    schema_overrides = {
        "mixture_component": "categorical",
        "xor_left": "binary",
        "xor_right": "binary",
        "zipf_rank": "count",
        "zipf_category": "categorical",
        "mcar_missing": "binary",
        "mar_missing": "binary",
        "mnar_missing": "binary",
        "total_count": "count",
        "child_count": "count",
        "segment": "categorical",
        "premium": "binary",
        "target": "binary",
    }
    return ControlledPanel(
        frame=frame,
        truth_frame=truth_frame,
        truth=truth,
        constraints=constraints,
        schema_overrides=schema_overrides,
    )


build_controlled_panel = generate_controlled_panel
controlled_panel = generate_controlled_panel


__all__ = [
    "CONTROLLED_PANEL_COLUMNS",
    "CONTROLLED_PANEL_DATASET_ID",
    "CONTROLLED_PANEL_ROWS",
    "CONTROLLED_PANEL_SEED",
    "CONTROLLED_PANEL_VERSION",
    "ControlledPanel",
    "KnownConstraint",
    "build_controlled_panel",
    "controlled_panel",
    "generate_controlled_panel",
]
