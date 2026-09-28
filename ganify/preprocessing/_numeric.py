"""Deterministic one-dimensional numeric transforms used by table preprocessing."""

from __future__ import annotations

import bisect
import math
from collections import Counter
from typing import Any, Dict, Mapping

import numpy as np


def _finite_vector(values: Any, *, name: str = "values") -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("%s must be one-dimensional" % name)
    if len(array) == 0:
        raise ValueError("%s must contain at least one value" % name)
    if not bool(np.isfinite(array).all()):
        raise ValueError("%s contains NaN or infinite values" % name)
    return array


class EmpiricalCopula1D:
    """Reversible empirical-CDF transform for one finite numeric feature.

    Distinct observations are mapped to the midpoint of their empirical mass.
    Linear interpolation between these knots supports unseen values while
    mapping every fitted observation back exactly (up to floating precision).
    """

    def __init__(self) -> None:
        self.values_: np.ndarray
        self.coordinates_: np.ndarray
        self.fitted_ = False

    @property
    def output_width(self) -> int:
        """Number of numeric channels emitted by :meth:`transform`."""

        return 1

    def fit(self, values: Any) -> "EmpiricalCopula1D":
        """Fit empirical quantile knots."""

        array = _finite_vector(values)
        unique, counts = np.unique(array, return_counts=True)
        cumulative = np.cumsum(counts, dtype=np.float64)
        midpoint = (cumulative - 0.5 * counts.astype(np.float64)) / float(len(array))
        self.values_ = np.asarray(unique, dtype=np.float64)
        self.coordinates_ = np.asarray(midpoint * 2.0 - 1.0, dtype=np.float64)
        if len(unique) == 1:
            self.coordinates_[0] = 0.0
        self.fitted_ = True
        return self

    def transform(self, values: Any) -> np.ndarray:
        """Map values into a bounded empirical rank channel."""

        self._check_fitted()
        array = _finite_vector(values)
        if len(self.values_) == 1:
            return np.zeros((len(array), 1), dtype=np.float64)
        transformed = np.interp(
            array,
            self.values_,
            self.coordinates_,
            left=-1.0,
            right=1.0,
        )
        return transformed.reshape(-1, 1)

    def inverse_transform(self, values: Any) -> np.ndarray:
        """Map an empirical rank channel back to the fitted numeric domain."""

        self._check_fitted()
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 2 and array.shape[1] == 1:
            array = array[:, 0]
        if array.ndim != 1:
            raise ValueError("encoded copula values must have one channel")
        if not bool(np.isfinite(array).all()):
            raise ValueError("encoded copula values contain NaN or infinite values")
        if len(self.values_) == 1:
            return np.full(len(array), self.values_[0], dtype=np.float64)
        return np.interp(
            np.clip(array, -1.0, 1.0),
            self.coordinates_,
            self.values_,
            left=self.values_[0],
            right=self.values_[-1],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-safe fitted state."""

        self._check_fitted()
        return {
            "type": "empirical_copula",
            "values": self.values_.tolist(),
            "coordinates": self.coordinates_.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EmpiricalCopula1D":
        """Restore fitted state from :meth:`to_dict`."""

        if payload.get("type") != "empirical_copula":
            raise ValueError("not an empirical copula state")
        instance = cls()
        instance.values_ = _finite_vector(payload["values"], name="copula values")
        instance.coordinates_ = _finite_vector(
            payload["coordinates"], name="copula coordinates"
        )
        if len(instance.values_) != len(instance.coordinates_):
            raise ValueError("copula values and coordinates have different lengths")
        if bool(np.any(np.diff(instance.values_) <= 0.0)):
            raise ValueError("copula values must be strictly increasing")
        if len(instance.coordinates_) > 1 and bool(
            np.any(np.diff(instance.coordinates_) <= 0.0)
        ):
            raise ValueError("copula coordinates must be strictly increasing")
        instance.fitted_ = True
        return instance

    def _check_fitted(self) -> None:
        if not getattr(self, "fitted_", False):
            raise RuntimeError("numeric transform is not fitted")


class DiscreteCopula1D:
    """Empirical step transform preserving arbitrary-size integer counts."""

    def __init__(self) -> None:
        self.fitted_ = False

    @property
    def output_width(self) -> int:
        """Number of numeric channels emitted by :meth:`transform`."""

        return 1

    def fit(self, values: Any) -> "DiscreteCopula1D":
        """Fit exact integer support and empirical midpoint coordinates."""

        integers = self._integers(values)
        if not integers:
            raise ValueError("values must contain at least one count")
        frequencies = Counter(integers)
        support = sorted(frequencies)
        cumulative = 0
        coordinates = []
        for value in support:
            count = frequencies[value]
            coordinates.append(
                ((cumulative + 0.5 * count) / float(len(integers))) * 2.0 - 1.0
            )
            cumulative += count
        if len(support) == 1:
            coordinates[0] = 0.0
        self.values_ = tuple(support)
        self.coordinates_ = np.asarray(coordinates, dtype=np.float64)
        self.lookup_ = {
            value: self.coordinates_[index]
            for index, value in enumerate(self.values_)
        }
        self.fitted_ = True
        return self

    def transform(self, values: Any) -> np.ndarray:
        """Map integer counts to empirical rank coordinates."""

        self._check_fitted()
        integers = self._integers(values)
        output = np.empty(len(integers), dtype=np.float64)
        for row, value in enumerate(integers):
            exact = self.lookup_.get(value)
            if exact is not None:
                output[row] = exact
                continue
            position = bisect.bisect_left(self.values_, value)
            if position == 0:
                output[row] = -1.0
            elif position == len(self.values_):
                output[row] = 1.0
            else:
                lower = self.values_[position - 1]
                upper = self.values_[position]
                fraction = float(value - lower) / float(upper - lower)
                output[row] = (
                    self.coordinates_[position - 1] * (1.0 - fraction)
                    + self.coordinates_[position] * fraction
                )
        return output.reshape(-1, 1)

    def inverse_transform(self, values: Any) -> np.ndarray:
        """Decode to the nearest observed integer count exactly."""

        self._check_fitted()
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 2 and array.shape[1] == 1:
            array = array[:, 0]
        if array.ndim != 1:
            raise ValueError("encoded discrete values must have one channel")
        if not bool(np.isfinite(array).all()):
            raise ValueError("encoded discrete values contain NaN or infinite values")
        result = np.empty(len(array), dtype=object)
        for row, value in enumerate(np.clip(array, -1.0, 1.0)):
            position = int(np.searchsorted(self.coordinates_, value, side="left"))
            if position <= 0:
                index = 0
            elif position >= len(self.coordinates_):
                index = len(self.coordinates_) - 1
            else:
                before = self.coordinates_[position - 1]
                after = self.coordinates_[position]
                index = position - 1 if value - before <= after - value else position
            result[row] = self.values_[index]
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-safe fitted state with decimal integer support."""

        self._check_fitted()
        return {
            "type": "discrete_copula",
            "values": [str(value) for value in self.values_],
            "coordinates": self.coordinates_.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DiscreteCopula1D":
        """Restore fitted state from :meth:`to_dict`."""

        if payload.get("type") != "discrete_copula":
            raise ValueError("not a discrete copula state")
        instance = cls()
        instance.values_ = tuple(int(value) for value in payload["values"])
        instance.coordinates_ = _finite_vector(
            payload["coordinates"], name="discrete coordinates"
        )
        if not instance.values_ or len(instance.values_) != len(instance.coordinates_):
            raise ValueError("invalid discrete copula state")
        if any(
            right <= left
            for left, right in zip(instance.values_[:-1], instance.values_[1:])
        ):
            raise ValueError("discrete values must be strictly increasing")
        if len(instance.coordinates_) > 1 and bool(
            np.any(np.diff(instance.coordinates_) <= 0.0)
        ):
            raise ValueError("discrete coordinates must be strictly increasing")
        instance.lookup_ = {
            value: instance.coordinates_[index]
            for index, value in enumerate(instance.values_)
        }
        instance.fitted_ = True
        return instance

    @staticmethod
    def _integers(values: Any):
        array = np.asarray(values, dtype=object)
        if array.ndim != 1:
            raise ValueError("count values must be one-dimensional")
        result = []
        for value in array.tolist():
            integer = int(value)
            if value != integer:
                raise ValueError("count values must be integers")
            if integer < 0:
                raise ValueError("count values must be non-negative")
            result.append(integer)
        return result

    def _check_fitted(self) -> None:
        if not getattr(self, "fitted_", False):
            raise RuntimeError("numeric transform is not fitted")


class ModeAware1D:
    """Deterministic mode-specific normalization without scikit-learn.

    A one-dimensional k-means partition supplies a softmax component channel,
    while an arctangent residual records the exact position within the chosen
    component. The arctangent keeps every output finite and bounded.
    """

    def __init__(self, max_modes: int = 5, max_iter: int = 100) -> None:
        if isinstance(max_modes, bool) or int(max_modes) < 1:
            raise ValueError("max_modes must be a positive integer")
        if isinstance(max_iter, bool) or int(max_iter) < 1:
            raise ValueError("max_iter must be a positive integer")
        self.max_modes = int(max_modes)
        self.max_iter = int(max_iter)
        self.fitted_ = False

    @property
    def output_width(self) -> int:
        """Residual channel plus one component channel per fitted mode."""

        self._check_fitted()
        return 1 + len(self.centers_)

    def fit(self, values: Any) -> "ModeAware1D":
        """Fit a deterministic one-dimensional mode partition."""

        # Sorting makes fitted centers bitwise stable under dataframe row
        # permutations, not merely stable for a repeated call in one order.
        array = np.sort(_finite_vector(values), kind="mergesort")
        unique_count = len(np.unique(array))
        # Avoid tiny, nearly empty modes. A component needs roughly eight rows,
        # while genuinely small datasets still retain at least one component.
        supported = max(1, len(array) // 8)
        components = min(self.max_modes, unique_count, supported)
        components = max(1, int(components))

        if components == 1:
            centers = np.asarray([float(np.median(array))], dtype=np.float64)
            labels = np.zeros(len(array), dtype=np.int64)
        else:
            probabilities = (np.arange(components, dtype=np.float64) + 0.5) / components
            try:
                centers = np.asarray(
                    np.quantile(array, probabilities, method="linear"),
                    dtype=np.float64,
                )
            except TypeError:  # NumPy < 1.22
                centers = np.asarray(
                    np.quantile(array, probabilities, interpolation="linear"),
                    dtype=np.float64,
                )
            labels = np.zeros(len(array), dtype=np.int64)
            for _ in range(self.max_iter):
                distances = np.abs(array[:, None] - centers[None, :])
                new_labels = np.argmin(distances, axis=1)
                new_centers = centers.copy()
                for component in range(components):
                    selected = array[new_labels == component]
                    if len(selected):
                        new_centers[component] = float(np.mean(selected))
                order = np.argsort(new_centers, kind="mergesort")
                new_centers = new_centers[order]
                remap = np.empty_like(order)
                remap[order] = np.arange(components)
                new_labels = remap[new_labels]
                if np.array_equal(new_labels, labels) and np.allclose(
                    new_centers, centers, rtol=0.0, atol=1e-14
                ):
                    centers = new_centers
                    labels = new_labels
                    break
                centers = new_centers
                labels = new_labels

        global_scale = float(np.std(array))
        fallback = max(global_scale * 0.05, np.finfo(np.float64).eps)
        scales = np.empty(len(centers), dtype=np.float64)
        weights = np.empty(len(centers), dtype=np.float64)
        for component in range(len(centers)):
            selected = array[labels == component]
            scale = float(np.std(selected)) if len(selected) > 1 else fallback
            scales[component] = max(scale, fallback)
            weights[component] = float(len(selected)) / float(len(array))

        self.centers_ = centers
        self.scales_ = scales
        self.weights_ = weights
        self.fitted_ = True
        return self

    def _labels(self, array: np.ndarray) -> np.ndarray:
        distance = np.abs(array[:, None] - self.centers_[None, :])
        # Normalize by scale so broad modes do not absorb narrow neighbours.
        distance = distance / self.scales_[None, :]
        return np.argmin(distance, axis=1)

    def transform(self, values: Any) -> np.ndarray:
        """Return bounded residuals and one-hot mode assignments."""

        self._check_fitted()
        array = _finite_vector(values)
        labels = self._labels(array)
        standardized = (
            array - self.centers_[labels]
        ) / self.scales_[labels]
        residual = (2.0 / math.pi) * np.arctan(standardized)
        output = np.zeros((len(array), self.output_width), dtype=np.float64)
        output[:, 0] = residual
        output[np.arange(len(array)), labels + 1] = 1.0
        return output

    def inverse_transform(self, values: Any) -> np.ndarray:
        """Reconstruct values from residual and component channels."""

        self._check_fitted()
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != self.output_width:
            raise ValueError(
                "mode-aware values must have %d channels" % self.output_width
            )
        if not bool(np.isfinite(array).all()):
            raise ValueError("mode-aware values contain NaN or infinite values")
        labels = np.argmax(array[:, 1:], axis=1)
        residual = np.clip(array[:, 0], -1.0 + 1e-12, 1.0 - 1e-12)
        standardized = np.tan(residual * (math.pi / 2.0))
        return self.centers_[labels] + standardized * self.scales_[labels]

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-safe fitted state."""

        self._check_fitted()
        return {
            "type": "mode_aware",
            "max_modes": self.max_modes,
            "max_iter": self.max_iter,
            "centers": self.centers_.tolist(),
            "scales": self.scales_.tolist(),
            "weights": self.weights_.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModeAware1D":
        """Restore fitted state from :meth:`to_dict`."""

        if payload.get("type") != "mode_aware":
            raise ValueError("not a mode-aware transform state")
        instance = cls(
            max_modes=int(payload.get("max_modes", 5)),
            max_iter=int(payload.get("max_iter", 100)),
        )
        instance.centers_ = _finite_vector(payload["centers"], name="mode centers")
        instance.scales_ = _finite_vector(payload["scales"], name="mode scales")
        instance.weights_ = _finite_vector(payload["weights"], name="mode weights")
        if not (
            len(instance.centers_)
            == len(instance.scales_)
            == len(instance.weights_)
        ):
            raise ValueError("mode-aware state arrays have different lengths")
        if bool(np.any(instance.scales_ <= 0.0)):
            raise ValueError("mode scales must be positive")
        if bool(np.any(np.diff(instance.centers_) < 0.0)):
            raise ValueError("mode centers must be sorted")
        instance.fitted_ = True
        return instance

    def _check_fitted(self) -> None:
        if not getattr(self, "fitted_", False):
            raise RuntimeError("numeric transform is not fitted")


def numeric_transform_from_dict(payload: Mapping[str, Any]):
    """Restore either supported numeric transform from JSON-safe state."""

    kind = payload.get("type")
    if kind == "empirical_copula":
        return EmpiricalCopula1D.from_dict(payload)
    if kind == "discrete_copula":
        return DiscreteCopula1D.from_dict(payload)
    if kind == "mode_aware":
        return ModeAware1D.from_dict(payload)
    raise ValueError("unknown numeric transform type %r" % kind)


EmpiricalCopulaScaler = EmpiricalCopula1D
ModeAwareTransformer = ModeAware1D


__all__ = [
    "DiscreteCopula1D",
    "EmpiricalCopula1D",
    "EmpiricalCopulaScaler",
    "ModeAware1D",
    "ModeAwareTransformer",
    "numeric_transform_from_dict",
]
