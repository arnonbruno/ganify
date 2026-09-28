"""Adapters separating release orchestration from synthesizer APIs.

The three statistical controls are dependency-free and always available.
External frontier adapters are explicit protocol placeholders: attempting to
fit one raises an actionable installation error, which the release runner
records as a failed run instead of silently reducing benchmark coverage.
"""

from __future__ import annotations

import copy
import json
import math
from statistics import NormalDist
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

import numpy as np
import pandas as pd


@runtime_checkable
class ModelAdapter(Protocol):
    """Minimal deterministic interface required by the benchmark runner."""

    name: str

    def fit(
        self,
        train: pd.DataFrame,
        *,
        seed: int,
        target: Optional[str] = None,
    ) -> "ModelAdapter":
        """Fit using only the supplied training partition."""

    def sample(self, n_rows: int, *, seed: int) -> pd.DataFrame:
        """Draw a deterministic synthetic table with the training schema."""

    def get_config(self) -> Mapping[str, Any]:
        """Return fully resolved, JSON-serializable model configuration."""


class AdapterUnavailableError(ImportError):
    """Raised when a preregistered external protocol is not installed."""


def _require_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("model adapters require a pandas DataFrame")
    if len(frame) < 1 or len(frame.columns) < 1:
        raise ValueError("model adapter training data must not be empty")
    if frame.columns.has_duplicates:
        raise ValueError("model adapter training columns must be unique")
    return frame


def _positive_rows(n_rows: int) -> int:
    if isinstance(n_rows, bool) or not isinstance(n_rows, (int, np.integer)):
        raise ValueError("n_rows must be a positive integer")
    if int(n_rows) < 1:
        raise ValueError("n_rows must be a positive integer")
    return int(n_rows)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_safe(value.to_dict())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        "adapter configuration value %r (%s) is not JSON-serializable"
        % (value, type(value).__name__)
    )


class BootstrapAdapter:
    """Positive memorization control: sample complete rows with replacement."""

    name = "bootstrap"

    def __init__(self) -> None:
        self._train: Optional[pd.DataFrame] = None
        self._fit_seed: Optional[int] = None

    def fit(
        self,
        train: pd.DataFrame,
        *,
        seed: int,
        target: Optional[str] = None,
    ) -> "BootstrapAdapter":
        del target
        self._train = _require_frame(train).reset_index(drop=True).copy(deep=True)
        self._fit_seed = int(seed)
        return self

    def sample(self, n_rows: int, *, seed: int) -> pd.DataFrame:
        if self._train is None:
            raise RuntimeError("BootstrapAdapter is not fitted")
        count = _positive_rows(n_rows)
        rng = np.random.default_rng(int(seed))
        indices = rng.integers(0, len(self._train), size=count)
        return self._train.iloc[indices].reset_index(drop=True).copy(deep=True)

    def get_config(self) -> Mapping[str, Any]:
        return {"adapter": self.name, "sampling": "rows_with_replacement"}

    def clone(self) -> "BootstrapAdapter":
        return type(self)()


class IndependentMarginalsAdapter:
    """Negative dependence control with exact empirical marginals."""

    name = "independent_marginals"

    def __init__(self) -> None:
        self._train: Optional[pd.DataFrame] = None
        self._fit_seed: Optional[int] = None

    def fit(
        self,
        train: pd.DataFrame,
        *,
        seed: int,
        target: Optional[str] = None,
    ) -> "IndependentMarginalsAdapter":
        del target
        self._train = _require_frame(train).reset_index(drop=True).copy(deep=True)
        self._fit_seed = int(seed)
        return self

    def sample(self, n_rows: int, *, seed: int) -> pd.DataFrame:
        if self._train is None:
            raise RuntimeError("IndependentMarginalsAdapter is not fitted")
        count = _positive_rows(n_rows)
        rng = np.random.default_rng(int(seed))
        columns = []
        for column in self._train.columns:
            indices = rng.integers(0, len(self._train), size=count)
            sampled = self._train[column].iloc[indices].reset_index(drop=True)
            sampled.name = column
            columns.append(sampled)
        return pd.concat(columns, axis=1).loc[:, self._train.columns]

    def get_config(self) -> Mapping[str, Any]:
        return {
            "adapter": self.name,
            "sampling": "independent_empirical_columns",
        }

    def clone(self) -> "IndependentMarginalsAdapter":
        return type(self)()


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _sort_key(value: Any) -> Tuple[Any, ...]:
    value = _scalar(value)
    try:
        missing = bool(pd.isna(value))
    except (TypeError, ValueError):
        missing = False
    if missing:
        return (3, "", "")
    if isinstance(value, (bool, np.bool_)):
        return (0, int(value), "")
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if math.isfinite(numeric):
            return (0, numeric, "")
    if isinstance(value, pd.Timestamp):
        return (1, int(value.value), str(value.tz))
    return (2, type(value).__name__, repr(value))


def _normal_scores(values: Sequence[Any]) -> Tuple[np.ndarray, np.ndarray]:
    keys = [_sort_key(value) for value in values]
    order = np.asarray(
        sorted(range(len(keys)), key=lambda index: keys[index]),
        dtype=np.int64,
    )
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and keys[int(order[stop])] == keys[int(order[start])]:
            stop += 1
        # Positions are zero-based; the midpoint maps strictly inside (0, 1).
        midpoint = 0.5 * (start + stop - 1)
        ranks[order[start:stop]] = midpoint
        start = stop
    probabilities = (ranks + 0.5) / float(len(values))
    normal = NormalDist()
    scores = np.fromiter(
        (normal.inv_cdf(float(value)) for value in probabilities),
        dtype=np.float64,
        count=len(probabilities),
    )
    return scores, order


class GaussianCopulaAdapter:
    """Empirical-marginal Gaussian copula control.

    The adapter uses mid-rank normal scores and a regularized correlation
    matrix.  Its inverse map selects rows from each original Series, preserving
    nullable, categorical, and extension dtypes without optional SciPy.
    """

    name = "gaussian_copula"

    def __init__(self, *, eigenvalue_floor: float = 1e-6) -> None:
        value = float(eigenvalue_floor)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("eigenvalue_floor must be finite and positive")
        self.eigenvalue_floor = value
        self._train: Optional[pd.DataFrame] = None
        self._orders: Tuple[np.ndarray, ...] = ()
        self._correlation: Optional[np.ndarray] = None

    def fit(
        self,
        train: pd.DataFrame,
        *,
        seed: int,
        target: Optional[str] = None,
    ) -> "GaussianCopulaAdapter":
        del target
        frame = _require_frame(train).reset_index(drop=True).copy(deep=True)
        if len(frame) < 2:
            raise ValueError("GaussianCopulaAdapter needs at least two rows")
        score_columns = []
        orders = []
        for column in frame.columns:
            scores, order = _normal_scores(frame[column].tolist())
            score_columns.append(scores)
            orders.append(order)
        matrix = np.column_stack(score_columns)
        if matrix.shape[1] == 1:
            correlation = np.ones((1, 1), dtype=np.float64)
        else:
            correlation = np.corrcoef(matrix, rowvar=False)
            correlation = np.nan_to_num(
                correlation, nan=0.0, posinf=0.0, neginf=0.0
            )
            correlation = 0.5 * (correlation + correlation.T)
            np.fill_diagonal(correlation, 1.0)
            eigenvalues, eigenvectors = np.linalg.eigh(correlation)
            eigenvalues = np.maximum(eigenvalues, self.eigenvalue_floor)
            correlation = (
                eigenvectors
                @ np.diag(eigenvalues)
                @ eigenvectors.T
            )
            scale = np.sqrt(np.maximum(np.diag(correlation), 1e-15))
            correlation = correlation / np.outer(scale, scale)
            correlation = 0.5 * (correlation + correlation.T)
            np.fill_diagonal(correlation, 1.0)
        self._train = frame
        self._orders = tuple(orders)
        self._correlation = correlation
        self._fit_seed = int(seed)
        return self

    def sample(self, n_rows: int, *, seed: int) -> pd.DataFrame:
        if self._train is None or self._correlation is None:
            raise RuntimeError("GaussianCopulaAdapter is not fitted")
        count = _positive_rows(n_rows)
        rng = np.random.default_rng(int(seed))
        latent = rng.multivariate_normal(
            np.zeros(len(self._train.columns), dtype=np.float64),
            self._correlation,
            size=count,
            check_valid="raise",
        )
        if latent.ndim == 1:
            latent = latent.reshape(-1, 1)
        normal = NormalDist()
        columns = []
        for index, column in enumerate(self._train.columns):
            probabilities = np.fromiter(
                (normal.cdf(float(value)) for value in latent[:, index]),
                dtype=np.float64,
                count=count,
            )
            quantiles = np.minimum(
                (probabilities * len(self._train)).astype(np.int64),
                len(self._train) - 1,
            )
            source_indices = self._orders[index][quantiles]
            sampled = (
                self._train[column]
                .iloc[source_indices]
                .reset_index(drop=True)
            )
            sampled.name = column
            columns.append(sampled)
        return pd.concat(columns, axis=1).loc[:, self._train.columns]

    def get_config(self) -> Mapping[str, Any]:
        return {
            "adapter": self.name,
            "method": "empirical_midrank_gaussian_copula",
            "eigenvalue_floor": self.eigenvalue_floor,
        }

    def clone(self) -> "GaussianCopulaAdapter":
        return type(self)(eigenvalue_floor=self.eigenvalue_floor)


class GanifyConditionalAdapter:
    """Adapter for the current :class:`ConditionalGANEngine`."""

    name = "ganify_conditional"

    def __init__(
        self,
        engine_config: Optional[Mapping[str, Any]] = None,
        **engine_options: Any,
    ) -> None:
        nested = engine_options.pop("engine", None)
        if nested is not None:
            if engine_config is not None:
                raise ValueError(
                    "pass engine_config or engine, not both"
                )
            engine_config = nested
        if engine_config is not None and not isinstance(
            engine_config, Mapping
        ):
            raise TypeError("engine_config must be a mapping")
        configured = dict(engine_config or {})
        overlap = set(configured).intersection(engine_options)
        if overlap:
            raise ValueError(
                "duplicate Ganify engine options: %r" % sorted(overlap)
            )
        configured.update(engine_options)
        if "random_state" in configured:
            raise ValueError(
                "GanifyConditionalAdapter receives random_state from the "
                "release model seed"
            )
        self.engine_config = configured
        # Fail during registration rather than after a long benchmark.
        json.dumps(_json_safe(self.engine_config), sort_keys=True)
        self._engine: Any = None
        self._columns: Tuple[Any, ...] = ()
        self._target: Optional[str] = None

    def fit(
        self,
        train: pd.DataFrame,
        *,
        seed: int,
        target: Optional[str] = None,
    ) -> "GanifyConditionalAdapter":
        frame = _require_frame(train)
        if target is not None and target not in frame.columns:
            raise ValueError("target column %r is missing" % target)
        from ganify.conditional import ConditionalGANEngine

        self._columns = tuple(frame.columns)
        self._target = target
        features = (
            frame
            if target is None
            else frame.drop(columns=[target])
        )
        labels = None if target is None else frame[target]
        options = dict(self.engine_config)
        options["random_state"] = int(seed)
        self._engine = ConditionalGANEngine(**options)
        self._engine.fit(
            features,
            target=labels,
            target_name=target,
            verbose=0,
        )
        return self

    def sample(self, n_rows: int, *, seed: int) -> pd.DataFrame:
        if self._engine is None:
            raise RuntimeError("GanifyConditionalAdapter is not fitted")
        count = _positive_rows(n_rows)
        self._engine.set_sampling_seed(int(seed))
        sampled = self._engine.sample(
            count, return_target=self._target is not None
        )
        if self._target is None:
            frame = sampled
        else:
            features, target = sampled
            frame = features.copy()
            frame[self._target] = np.asarray(target)
        return frame.loc[:, list(self._columns)].reset_index(drop=True)

    def get_config(self) -> Mapping[str, Any]:
        return {
            "adapter": self.name,
            "engine": _json_safe(self.engine_config),
        }

    @property
    def privacy_report(self) -> Optional[Mapping[str, Any]]:
        if self._engine is None:
            return None
        return self._engine.privacy_report

    def clone(self) -> "GanifyConditionalAdapter":
        return type(self)(**copy.deepcopy(self.engine_config))


class ExternalProtocolAdapter:
    """Named placeholder for an external preregistered implementation."""

    name = "external"
    project = "external synthesizer"
    install = "Install the implementation and register a concrete adapter."

    def __init__(self, **config: Any) -> None:
        self.config = dict(config)
        json.dumps(_json_safe(self.config), sort_keys=True)

    def _error(self) -> AdapterUnavailableError:
        return AdapterUnavailableError(
            "%s adapter is a release-protocol placeholder and cannot be "
            "silently skipped. %s Then replace/register this placeholder "
            "with a ModelAdapter-compatible implementation named %r."
            % (self.project, self.install, self.name)
        )

    def fit(
        self,
        train: pd.DataFrame,
        *,
        seed: int,
        target: Optional[str] = None,
    ) -> "ExternalProtocolAdapter":
        del train, seed, target
        raise self._error()

    def sample(self, n_rows: int, *, seed: int) -> pd.DataFrame:
        del n_rows, seed
        raise self._error()

    def get_config(self) -> Mapping[str, Any]:
        return {
            "adapter": self.name,
            "status": "protocol_placeholder",
            "project": self.project,
            "installation": self.install,
            "config": _json_safe(self.config),
        }

    def clone(self) -> "ExternalProtocolAdapter":
        return type(self)(**copy.deepcopy(self.config))


class CTGANAdapter(ExternalProtocolAdapter):
    name = "ctgan"
    project = "CTGAN"
    install = (
        "Install the pinned SDV CTGAN package (for example "
        "`python -m pip install ctgan`) and record its version."
    )


class CTABGANPlusAdapter(ExternalProtocolAdapter):
    name = "ctab_gan_plus"
    project = "CTAB-GAN+"
    install = (
        "Install a pinned CTAB-GAN+ checkout from "
        "https://github.com/Team-TUD/CTAB-GAN-Plus and its requirements."
    )


class TAEGANAdapter(ExternalProtocolAdapter):
    name = "taegan"
    project = "TAEGAN"
    install = (
        "Install the authors' pinned TAEGAN implementation and expose its "
        "fit/sample API through ModelAdapter."
    )


class ARFAdapter(ExternalProtocolAdapter):
    name = "arf"
    project = "Adversarial Random Forests (ARF)"
    install = (
        "Install a pinned ARF implementation (including its R/Python runtime "
        "when required) and register a deterministic bridge."
    )


class TabDDPMAdapter(ExternalProtocolAdapter):
    name = "tabddpm"
    project = "TabDDPM"
    install = (
        "Install a pinned checkout of "
        "https://github.com/yandex-research/tab-ddpm and its model assets."
    )


class TabSynAdapter(ExternalProtocolAdapter):
    name = "tabsyn"
    project = "TabSyn"
    install = (
        "Install a pinned checkout of "
        "https://github.com/amazon-science/tabsyn and its model assets."
    )


class TabDiffAdapter(ExternalProtocolAdapter):
    name = "tabdiff"
    project = "TabDiff"
    install = (
        "Install the pinned TabDiff reference implementation and its model "
        "assets, then register a deterministic ModelAdapter bridge."
    )


AdapterFactory = Callable[[], ModelAdapter]
AdapterLike = Union[ModelAdapter, AdapterFactory]


_ADAPTER_TYPES: Dict[str, type] = {
    "bootstrap": BootstrapAdapter,
    "independent_marginals": IndependentMarginalsAdapter,
    "gaussian_copula": GaussianCopulaAdapter,
    "ganify_conditional": GanifyConditionalAdapter,
    "ctgan": CTGANAdapter,
    "ctab_gan_plus": CTABGANPlusAdapter,
    "taegan": TAEGANAdapter,
    "arf": ARFAdapter,
    "tabddpm": TabDDPMAdapter,
    "tabsyn": TabSynAdapter,
    "tabdiff": TabDiffAdapter,
}
_ADAPTER_ALIASES = {
    "ganify": "ganify_conditional",
    "conditional_ganify": "ganify_conditional",
    "independent": "independent_marginals",
    "independent_marginal": "independent_marginals",
    "copula": "gaussian_copula",
    "gaussian-copula": "gaussian_copula",
    "ctab-gan+": "ctab_gan_plus",
    "ctabgan+": "ctab_gan_plus",
    "ctabganplus": "ctab_gan_plus",
    "ctab_gan+": "ctab_gan_plus",
    "tab-ddpm": "tabddpm",
    "tab_ddpm": "tabddpm",
    "tab-syn": "tabsyn",
    "tab_syn": "tabsyn",
    "tab-diff": "tabdiff",
    "tab_diff": "tabdiff",
}


def normalize_adapter_name(name: str) -> str:
    value = str(name).strip().lower().replace(" ", "_")
    return _ADAPTER_ALIASES.get(value, value)


def create_adapter(
    name: str, config: Optional[Mapping[str, Any]] = None
) -> ModelAdapter:
    """Construct one built-in control, Ganify adapter, or placeholder."""

    normalized = normalize_adapter_name(name)
    if normalized not in _ADAPTER_TYPES:
        raise KeyError(
            "unknown adapter %r; available adapters: %s"
            % (name, ", ".join(sorted(_ADAPTER_TYPES)))
        )
    options = dict(config or {})
    return validate_adapter(_ADAPTER_TYPES[normalized](**options))


def clone_adapter(adapter: AdapterLike) -> ModelAdapter:
    """Return an unfitted adapter for one release run."""

    if isinstance(adapter, type):
        return validate_adapter(adapter())
    if not isinstance(adapter, ModelAdapter) and callable(adapter):
        return validate_adapter(adapter())
    validated = validate_adapter(adapter)
    clone = getattr(validated, "clone", None)
    if callable(clone):
        return validate_adapter(clone())
    try:
        return validate_adapter(copy.deepcopy(validated))
    except Exception as error:
        raise TypeError(
            "adapter %r cannot be cloned; pass a zero-argument factory"
            % validated.name
        ) from error


def validate_adapter(adapter: object) -> ModelAdapter:
    """Validate an adapter early, before an expensive benchmark run."""

    if not isinstance(adapter, ModelAdapter):
        missing = [
            name
            for name in ("name", "fit", "sample", "get_config")
            if not hasattr(adapter, name)
        ]
        raise TypeError("model adapter is missing: %s" % ", ".join(missing))
    name = getattr(adapter, "name")
    if not isinstance(name, str) or not name:
        raise ValueError("model adapter name must be a non-empty string")
    config = adapter.get_config()
    if not isinstance(config, Mapping):
        raise TypeError("model adapter get_config() must return a mapping")
    try:
        json.dumps(config, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "model adapter get_config() must be JSON-serializable"
        ) from error
    return adapter


MANDATORY_CONTROL_NAMES = (
    "bootstrap",
    "independent_marginals",
    "gaussian_copula",
)
FRONTIER_ADAPTER_NAMES = (
    "ctgan",
    "ctab_gan_plus",
    "taegan",
    "arf",
    "tabddpm",
    "tabsyn",
    "tabdiff",
)


# Explicit control aliases make release configs and user code equally clear.
BootstrapControl = BootstrapAdapter
IndependentMarginalsControl = IndependentMarginalsAdapter
GaussianCopulaControl = GaussianCopulaAdapter
CTABGANAdapter = CTABGANPlusAdapter


__all__ = [
    "ARFAdapter",
    "AdapterFactory",
    "AdapterLike",
    "AdapterUnavailableError",
    "BootstrapAdapter",
    "BootstrapControl",
    "CTABGANPlusAdapter",
    "CTABGANAdapter",
    "CTGANAdapter",
    "ExternalProtocolAdapter",
    "FRONTIER_ADAPTER_NAMES",
    "GaussianCopulaAdapter",
    "GaussianCopulaControl",
    "GanifyConditionalAdapter",
    "IndependentMarginalsAdapter",
    "IndependentMarginalsControl",
    "MANDATORY_CONTROL_NAMES",
    "ModelAdapter",
    "TAEGANAdapter",
    "TabDDPMAdapter",
    "TabDiffAdapter",
    "TabSynAdapter",
    "clone_adapter",
    "create_adapter",
    "normalize_adapter_name",
    "validate_adapter",
]
