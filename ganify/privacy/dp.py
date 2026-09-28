"""Conservative differential-privacy configuration and accounting primitives."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


def _default_orders() -> Tuple[float, ...]:
    return tuple(float(value) for value in range(2, 65)) + (
        96.0,
        128.0,
        256.0,
    )


@dataclass(frozen=True)
class DPConfig:
    """Configuration for optional DP critic training.

    Accounting deliberately uses a full-batch Gaussian upper bound with
    replace-one sensitivity. It does not claim privacy amplification from
    minibatch sampling, making the resulting epsilon conservative.
    ``noise_seed`` exists for mechanism tests; configuring it makes the engine
    refuse an end-to-end formal claim.
    """

    enabled: bool = True
    l2_norm_clip: float = 1.0
    noise_multiplier: float = 1.0
    microbatch_size: int = 1
    delta: float = 1e-5
    accountant_orders: Tuple[float, ...] = field(default_factory=_default_orders)
    preprocessing_public: bool = False
    conditional_frequencies_public: bool = False
    dataset_size_public: bool = True
    require_end_to_end: bool = False
    noise_seed: Optional[int] = None

    def __post_init__(self) -> None:
        clip = float(self.l2_norm_clip)
        noise = float(self.noise_multiplier)
        delta = float(self.delta)
        if not math.isfinite(clip) or clip <= 0.0:
            raise ValueError("l2_norm_clip must be finite and positive")
        if not math.isfinite(noise) or noise <= 0.0:
            raise ValueError("noise_multiplier must be finite and positive")
        if (
            isinstance(self.microbatch_size, bool)
            or int(self.microbatch_size) < 1
        ):
            raise ValueError("microbatch_size must be a positive integer")
        if not math.isfinite(delta) or not 0.0 < delta < 1.0:
            raise ValueError("delta must be strictly between zero and one")
        orders = tuple(float(value) for value in self.accountant_orders)
        if not orders or any(
            not math.isfinite(value) or value <= 1.0 for value in orders
        ):
            raise ValueError("accountant_orders must all be finite and > 1")
        if self.noise_seed is not None and (
            isinstance(self.noise_seed, bool)
            or not isinstance(self.noise_seed, (int, np.integer))
        ):
            raise ValueError("noise_seed must be an integer or None")
        object.__setattr__(self, "l2_norm_clip", clip)
        object.__setattr__(self, "noise_multiplier", noise)
        object.__setattr__(self, "microbatch_size", int(self.microbatch_size))
        object.__setattr__(self, "delta", delta)
        object.__setattr__(self, "accountant_orders", orders)
        object.__setattr__(
            self,
            "noise_seed",
            None if self.noise_seed is None else int(self.noise_seed),
        )

    def to_dict(self) -> Dict[str, Any]:
        output = asdict(self)
        output["accountant_orders"] = list(self.accountant_orders)
        return output

    as_dict = to_dict

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DPConfig":
        values = dict(payload)
        if "accountant_orders" in values:
            values["accountant_orders"] = tuple(values["accountant_orders"])
        return cls(**values)


class PrivacyAccountant:
    """RDP accountant for a conservative Gaussian-mechanism composition.

    Each recorded update is treated as a full-data Gaussian mechanism under
    replace-one adjacency. Clipped contributions have norm at most ``C`` and
    replacing one contribution changes their sum by at most ``2C``. Noise has
    standard deviation ``sigma*C``; therefore one update has
    ``RDP(alpha) = 2*alpha/sigma**2``. This intentionally ignores any sampling
    amplification and remains valid for non-uniform or with-replacement
    sampling, conditional on fixed/public preprocessing.
    """

    FORMAT_VERSION = 1

    def __init__(
        self,
        noise_multiplier: Any,
        *,
        orders: Optional[Iterable[float]] = None,
    ) -> None:
        if isinstance(noise_multiplier, DPConfig):
            if orders is None:
                orders = noise_multiplier.accountant_orders
            noise_multiplier = noise_multiplier.noise_multiplier
        noise = float(noise_multiplier)
        if not math.isfinite(noise) or noise <= 0.0:
            raise ValueError("noise_multiplier must be finite and positive")
        resolved = _default_orders() if orders is None else tuple(
            float(value) for value in orders
        )
        if not resolved or any(
            not math.isfinite(value) or value <= 1.0 for value in resolved
        ):
            raise ValueError("orders must all be finite and > 1")
        self.noise_multiplier = noise
        self.orders = tuple(resolved)
        self.rdp = np.zeros(len(self.orders), dtype=np.float64)
        self.steps = 0
        self.sample_rates = []
        self.valid = True
        self.invalid_reason: Optional[str] = None

    def record_step(
        self, *, sample_rate: Optional[float] = None, count: int = 1
    ) -> None:
        if isinstance(count, bool) or int(count) < 1:
            raise ValueError("count must be a positive integer")
        if sample_rate is not None:
            sample_rate = float(sample_rate)
            if (
                not math.isfinite(sample_rate)
                or sample_rate <= 0.0
                or sample_rate > 1.0
            ):
                raise ValueError("sample_rate must be in (0, 1]")
            self.sample_rates.extend([sample_rate] * int(count))
        per_step = (
            2.0
            * np.asarray(self.orders, dtype=np.float64)
            / (self.noise_multiplier ** 2)
        )
        self.rdp += int(count) * per_step
        self.steps += int(count)

    step = record_step

    def invalidate(self, reason: str) -> None:
        if not str(reason).strip():
            raise ValueError("invalid accountant reason cannot be empty")
        self.valid = False
        self.invalid_reason = str(reason)

    def epsilon(self, delta: float) -> float:
        """Return epsilon only when the mechanism accounting is valid."""

        if not self.valid:
            raise RuntimeError(
                "privacy accountant is invalid: %s" % self.invalid_reason
            )
        delta = float(delta)
        if not math.isfinite(delta) or not 0.0 < delta < 1.0:
            raise ValueError("delta must be strictly between zero and one")
        if self.steps == 0:
            return 0.0
        orders = np.asarray(self.orders, dtype=np.float64)
        epsilon = self.rdp + math.log(1.0 / delta) / (orders - 1.0)
        return float(np.min(epsilon))

    get_epsilon = epsilon

    def optimal_order(self, delta: float) -> float:
        if not self.valid:
            raise RuntimeError(
                "privacy accountant is invalid: %s" % self.invalid_reason
            )
        if self.steps == 0:
            return float(self.orders[-1])
        delta = float(delta)
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must be strictly between zero and one")
        orders = np.asarray(self.orders, dtype=np.float64)
        epsilon = self.rdp + math.log(1.0 / delta) / (orders - 1.0)
        return float(orders[int(np.argmin(epsilon))])

    def report(self, delta: float) -> Dict[str, Any]:
        epsilon: Optional[float]
        order: Optional[float]
        if self.valid:
            epsilon = self.epsilon(delta)
            order = self.optimal_order(delta)
        else:
            epsilon = None
            order = None
        return {
            "accountant_valid": bool(self.valid),
            "invalid_reason": self.invalid_reason,
            "accountant": "conservative_full_batch_gaussian_rdp",
            "adjacency": "replace_one",
            "sampling_amplification_used": False,
            "noise_multiplier": self.noise_multiplier,
            "steps": self.steps,
            "delta": float(delta),
            "epsilon": epsilon,
            "optimal_order": order,
            "orders": list(self.orders),
            "rdp": self.rdp.tolist(),
            "sample_rates": list(self.sample_rates),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "format": "ganify-privacy-accountant",
            "version": self.FORMAT_VERSION,
            "noise_multiplier": self.noise_multiplier,
            "orders": list(self.orders),
            "rdp": self.rdp.tolist(),
            "steps": self.steps,
            "sample_rates": list(self.sample_rates),
            "valid": self.valid,
            "invalid_reason": self.invalid_reason,
        }

    to_dict = state_dict

    @classmethod
    def from_state_dict(
        cls, payload: Mapping[str, Any]
    ) -> "PrivacyAccountant":
        if payload.get("format") != "ganify-privacy-accountant":
            raise ValueError("not a GANify privacy accountant state")
        if int(payload.get("version", -1)) != cls.FORMAT_VERSION:
            raise ValueError("unsupported privacy accountant state")
        instance = cls(
            float(payload["noise_multiplier"]),
            orders=payload["orders"],
        )
        rdp = np.asarray(payload["rdp"], dtype=np.float64)
        if rdp.shape != (len(instance.orders),) or not np.isfinite(rdp).all():
            raise ValueError("invalid persisted RDP vector")
        instance.rdp = rdp
        instance.steps = int(payload.get("steps", 0))
        instance.sample_rates = [
            float(value) for value in payload.get("sample_rates", [])
        ]
        instance.valid = bool(payload.get("valid", True))
        instance.invalid_reason = payload.get("invalid_reason")
        return instance

    from_dict = from_state_dict


def clip_and_noise_gradients(
    per_example_gradients: Sequence[np.ndarray],
    *,
    l2_norm_clip: float,
    noise_multiplier: float,
    random_state: Optional[int] = None,
) -> Tuple[Tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
    """Clip per-example gradient vectors globally and add Gaussian noise.

    Arrays must share their first (privacy-unit) dimension. Returned gradients
    are noisy averages, followed by pre-clipping norms and clipping factors.
    This NumPy primitive is used for deterministic mechanism tests; the engine
    applies the same operation to TensorFlow Jacobians.
    """

    if not per_example_gradients:
        raise ValueError("per_example_gradients cannot be empty")
    clip = float(l2_norm_clip)
    noise = float(noise_multiplier)
    if not math.isfinite(clip) or clip <= 0.0:
        raise ValueError("l2_norm_clip must be finite and positive")
    if not math.isfinite(noise) or noise < 0.0:
        raise ValueError("noise_multiplier must be finite and non-negative")
    arrays = tuple(np.asarray(value, dtype=np.float64) for value in per_example_gradients)
    units = arrays[0].shape[0] if arrays[0].ndim else 0
    if units < 1 or any(value.ndim < 1 or value.shape[0] != units for value in arrays):
        raise ValueError("gradient arrays must share a non-empty first dimension")
    squared = np.zeros(units, dtype=np.float64)
    for value in arrays:
        squared += np.sum(
            np.square(value).reshape(units, -1), axis=1
        )
    norms = np.sqrt(squared)
    factors = np.minimum(1.0, clip / np.maximum(norms, 1e-30))
    rng = np.random.default_rng(random_state)
    output = []
    for value in arrays:
        reshape = (units,) + (1,) * (value.ndim - 1)
        summed = np.sum(value * factors.reshape(reshape), axis=0)
        if noise:
            summed = summed + rng.normal(
                0.0, noise * clip, size=summed.shape
            )
        output.append(summed / float(units))
    return tuple(output), norms, factors


clip_gradients = clip_and_noise_gradients


__all__ = [
    "DPConfig",
    "PrivacyAccountant",
    "clip_and_noise_gradients",
    "clip_gradients",
]
