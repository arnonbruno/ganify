"""Self-contained conditional mixed-type WGAN-GP training engine."""

from __future__ import annotations

import json
import math
import secrets
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf

from ganify.constraints import (
    ConstrainedTableTransformer,
    ConstraintSet,
    StructuralConstraintTransformer,
    constraint_audit,
    empty_constraint_audit,
)
from ganify.preprocessing import TableTransformer
from ganify.privacy.dp import DPConfig, PrivacyAccountant
from ganify.schema import (
    ColumnSpec,
    TableSchema,
    decode_json_value,
    encode_json_value,
)

from .networks import (
    ConditionalCritic,
    ConditionalDenoisingEncoder,
    ConditionalGenerator,
    _critic_architecture,
)
from .sampler import ConditionSampler


def _positive_int(name: str, value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) < 1
    ):
        raise ValueError("%s must be a positive integer" % name)
    return int(value)


def _nonnegative_int(name: str, value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
    ):
        raise ValueError("%s must be a non-negative integer" % name)
    return int(value)


def _nonnegative_float(name: str, value: Any) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("%s must be finite and non-negative" % name)
    return value


def _positive_float(name: str, value: Any) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("%s must be finite and positive" % name)
    return value


def _probability(name: str, value: Any, *, upper_open: bool = False) -> float:
    value = float(value)
    valid = 0.0 <= value < 1.0 if upper_open else 0.0 <= value <= 1.0
    if not math.isfinite(value) or not valid:
        interval = "[0, 1)" if upper_open else "[0, 1]"
        raise ValueError("%s must be finite and in %s" % (name, interval))
    return value


def _dims(name: str, values: Sequence[int]) -> Tuple[int, ...]:
    result = tuple(_positive_int(name, value) for value in values)
    if not result:
        raise ValueError("%s must contain at least one width" % name)
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _is_missing(value: Any) -> bool:
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _encode_override_value(value: Any) -> Dict[str, Any]:
    if isinstance(value, ColumnSpec):
        return {"kind": "column_spec", "value": value.to_dict()}
    if isinstance(value, Mapping):
        return {
            "kind": "mapping",
            "items": [
                {
                    "key": encode_json_value(key),
                    "value": _encode_override_value(item),
                }
                for key, item in value.items()
            ],
        }
    if isinstance(value, (list, tuple)):
        return {
            "kind": "tuple" if isinstance(value, tuple) else "list",
            "items": [_encode_override_value(item) for item in value],
        }
    return {"kind": "scalar", "value": encode_json_value(value)}


def _decode_override_value(payload: Mapping[str, Any]) -> Any:
    kind = payload.get("kind")
    if kind == "column_spec":
        return ColumnSpec.from_dict(payload["value"])
    if kind == "mapping":
        return {
            decode_json_value(item["key"]): _decode_override_value(
                item["value"]
            )
            for item in payload.get("items", [])
        }
    if kind in {"list", "tuple"}:
        values = [
            _decode_override_value(item)
            for item in payload.get("items", [])
        ]
        return tuple(values) if kind == "tuple" else values
    if kind == "scalar":
        return decode_json_value(payload["value"])
    raise ValueError("invalid persisted schema override")


def _encode_schema_overrides(
    values: Mapping[Any, Any]
) -> Dict[str, Any]:
    return {
        "format": "ganify-schema-overrides",
        "items": [
            {
                "name": encode_json_value(name),
                "value": _encode_override_value(value),
            }
            for name, value in values.items()
        ],
    }


def _decode_schema_overrides(payload: Mapping[str, Any]) -> Dict[Any, Any]:
    if payload.get("format") != "ganify-schema-overrides":
        return dict(payload)
    return {
        decode_json_value(item["name"]): _decode_override_value(item["value"])
        for item in payload.get("items", [])
    }


class ConditionalGANEngine:
    """Conditional mixed-type synthesizer using WGAN-GP and typed heads.

    The engine is deliberately independent from :class:`ganify.Ganify`.
    Tables are encoded by :class:`~ganify.preprocessing.TableTransformer`;
    categorical columns, nullable masks, optional quantile bins, and an
    optional multiclass target become equally eligible CTGAN conditions.
    """

    FORMAT_VERSION = 1

    def __init__(
        self,
        *,
        random_state: Optional[int] = 42,
        noise_dim: int = 32,
        generator_dims: Sequence[int] = (128, 128),
        critic_dims: Sequence[int] = (128, 128),
        batch_size: int = 64,
        epochs: int = 10,
        n_critic: int = 5,
        gradient_penalty: float = 10.0,
        conditional_weight: float = 1.0,
        generator_learning_rate: float = 1e-4,
        critic_learning_rate: float = 2e-4,
        beta_1: float = 0.0,
        beta_2: float = 0.9,
        ema_decay: float = 0.999,
        continuous_transform: str = "copula",
        continuous_bins: Union[int, bool] = 0,
        handle_unknown: str = "error",
        max_modes: int = 5,
        schema: Optional[Union[TableSchema, Mapping[str, Any]]] = None,
        schema_overrides: Optional[Mapping[Any, Any]] = None,
        constraints: Any = None,
        spectral_normalization: bool = False,
        compile: bool = False,
        critic_architecture: str = "residual",
        pac: int = 1,
        attention_heads: int = 4,
        attention_dim: Optional[int] = None,
        fourier_features: int = 32,
        fourier_scale: float = 1.0,
        match_critic_parameters: bool = False,
        critic_parameter_budget: Optional[int] = None,
        warmup_epochs: int = 0,
        warmup_mask_probability: float = 0.15,
        warmup_learning_rate: Optional[float] = None,
        interaction_weight: float = 0.0,
        interaction_temperature: float = 0.1,
        feature_matching_weight: float = 0.0,
        swa: bool = False,
        swa_start_epoch: int = 1,
        swa_frequency: int = 1,
        swag: bool = False,
        swag_start_epoch: int = 1,
        swag_frequency: int = 1,
        swag_max_rank: int = 20,
        swag_scale: float = 1.0,
        dp_config: Optional[Union[DPConfig, Mapping[str, Any]]] = None,
    ) -> None:
        if random_state is not None and (
            isinstance(random_state, bool)
            or not isinstance(random_state, (int, np.integer))
        ):
            raise ValueError("random_state must be an integer or None")
        self.random_state = (
            None if random_state is None else int(random_state)
        )
        self.noise_dim = _positive_int("noise_dim", noise_dim)
        self.generator_dims = _dims("generator_dims", generator_dims)
        self.critic_dims = _dims("critic_dims", critic_dims)
        self.batch_size = _positive_int("batch_size", batch_size)
        self.epochs = _positive_int("epochs", epochs)
        self.n_critic = _positive_int("n_critic", n_critic)
        self.pac = _positive_int("pac", pac)
        if self.batch_size % self.pac:
            raise ValueError(
                "batch_size must be divisible by pac; batch_size counts raw "
                "rows and the effective critic batch is batch_size // pac"
            )
        self.gradient_penalty = _nonnegative_float(
            "gradient_penalty", gradient_penalty
        )
        self.conditional_weight = _nonnegative_float(
            "conditional_weight", conditional_weight
        )
        self.generator_learning_rate = _positive_float(
            "generator_learning_rate", generator_learning_rate
        )
        self.critic_learning_rate = _positive_float(
            "critic_learning_rate", critic_learning_rate
        )
        self.beta_1 = _probability("beta_1", beta_1, upper_open=True)
        self.beta_2 = _probability("beta_2", beta_2, upper_open=True)
        self.ema_decay = _probability(
            "ema_decay", ema_decay, upper_open=True
        )
        if isinstance(continuous_bins, bool):
            continuous_bins = 10 if continuous_bins else 0
        if (
            not isinstance(continuous_bins, (int, np.integer))
            or int(continuous_bins) < 0
            or int(continuous_bins) == 1
        ):
            raise ValueError("continuous_bins must be 0 or an integer >= 2")
        self.continuous_bins = int(continuous_bins)
        self.continuous_transform = str(continuous_transform)
        self.handle_unknown = str(handle_unknown)
        self.max_modes = _positive_int("max_modes", max_modes)
        if isinstance(schema, Mapping):
            schema = TableSchema.from_dict(schema)
        if schema is not None and not isinstance(schema, TableSchema):
            raise TypeError("schema must be a TableSchema, mapping, or None")
        self.schema = schema
        if (
            isinstance(schema_overrides, Mapping)
            and schema_overrides.get("format")
            == "ganify-schema-overrides"
        ):
            schema_overrides = _decode_schema_overrides(schema_overrides)
        self.schema_overrides = (
            {} if schema_overrides is None else dict(schema_overrides)
        )
        if self.schema is not None and self.schema_overrides:
            raise ValueError("schema and schema_overrides cannot both be set")
        if isinstance(constraints, ConstraintSet):
            self.constraints = constraints
        elif (
            isinstance(constraints, Mapping)
            and constraints.get("format") == "ganify-constraint-set"
        ):
            self.constraints = ConstraintSet.from_dict(constraints)
        else:
            self.constraints = ConstraintSet(constraints)
        self.spectral_normalization = bool(spectral_normalization)
        self.compile = bool(compile)
        self.critic_architecture = _critic_architecture(
            critic_architecture
        )
        self.attention_heads = _positive_int(
            "attention_heads", attention_heads
        )
        self.attention_dim = (
            None
            if attention_dim is None
            else _positive_int("attention_dim", attention_dim)
        )
        self.fourier_features = _positive_int(
            "fourier_features", fourier_features
        )
        self.fourier_scale = _positive_float(
            "fourier_scale", fourier_scale
        )
        self.match_critic_parameters = bool(match_critic_parameters)
        self.critic_parameter_budget = (
            None
            if critic_parameter_budget is None
            else _positive_int(
                "critic_parameter_budget", critic_parameter_budget
            )
        )
        if (
            self.match_critic_parameters
            and self.critic_parameter_budget is not None
        ):
            raise ValueError(
                "match_critic_parameters and critic_parameter_budget are "
                "mutually exclusive"
            )
        self.warmup_epochs = _nonnegative_int(
            "warmup_epochs", warmup_epochs
        )
        self.warmup_mask_probability = _probability(
            "warmup_mask_probability", warmup_mask_probability
        )
        self.warmup_learning_rate = (
            self.generator_learning_rate
            if warmup_learning_rate is None
            else _positive_float(
                "warmup_learning_rate", warmup_learning_rate
            )
        )
        self.interaction_weight = _nonnegative_float(
            "interaction_weight", interaction_weight
        )
        self.interaction_temperature = _positive_float(
            "interaction_temperature", interaction_temperature
        )
        self.feature_matching_weight = _nonnegative_float(
            "feature_matching_weight", feature_matching_weight
        )
        self.swa = bool(swa)
        self.swa_start_epoch = _positive_int(
            "swa_start_epoch", swa_start_epoch
        )
        self.swa_frequency = _positive_int(
            "swa_frequency", swa_frequency
        )
        self.swag = bool(swag)
        self.swag_start_epoch = _positive_int(
            "swag_start_epoch", swag_start_epoch
        )
        self.swag_frequency = _positive_int(
            "swag_frequency", swag_frequency
        )
        self.swag_max_rank = _positive_int(
            "swag_max_rank", swag_max_rank
        )
        self.swag_scale = _nonnegative_float("swag_scale", swag_scale)
        if isinstance(dp_config, Mapping):
            dp_config = DPConfig.from_dict(dp_config)
        if dp_config is not None and not isinstance(dp_config, DPConfig):
            raise TypeError("dp_config must be a DPConfig, mapping, or None")
        self.dp_config = dp_config
        self.dp_enabled = bool(
            dp_config is not None and dp_config.enabled
        )
        self._validate_dp_configuration()

        self._rng = np.random.default_rng(self.random_state)
        self.fitted_ = False
        self.history_: Dict[str, List[float]] = {
            "critic": [],
            "generator": [],
            "gradient_penalty": [],
            "conditional": [],
            "interaction": [],
            "feature_matching": [],
            "warmup": [],
        }
        self.epochs_ran_ = 0
        self.warmup_epochs_ran_ = 0
        self.constraint_transformer_ = None
        self.constraint_preprocessor_ = None
        self.preprocessor_ = None
        self.last_constraint_audit_ = empty_constraint_audit()
        self.privacy_accountant_: Optional[PrivacyAccountant] = None
        self.privacy_report_: Optional[Dict[str, Any]] = None

    def _validate_dp_configuration(self) -> None:
        if not self.dp_enabled:
            return
        incompatible = []
        if self.gradient_penalty:
            incompatible.append("gradient_penalty must be 0")
        if self.pac != 1:
            incompatible.append("pac must be 1")
        if self.feature_matching_weight:
            incompatible.append("feature_matching_weight must be 0")
        if self.interaction_weight:
            incompatible.append("interaction_weight must be 0")
        if self.warmup_epochs:
            incompatible.append("warmup_epochs must be 0")
        if self.spectral_normalization:
            incompatible.append("spectral_normalization must be disabled")
        if self.compile:
            incompatible.append("compile must be False")
        if incompatible:
            raise ValueError(
                "DP critic training is incompatible with: %s"
                % "; ".join(incompatible)
            )

    def fit(
        self,
        frame: pd.DataFrame,
        target: Any = None,
        *,
        y: Any = None,
        target_name: Any = None,
        schema: Optional[TableSchema] = None,
        schema_overrides: Optional[Mapping[Any, Any]] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        n_critic: Optional[int] = None,
        verbose: int = 0,
        compile: Optional[bool] = None,
        preprocessor: Optional[TableTransformer] = None,
    ) -> "ConditionalGANEngine":
        """Fit a conditional WGAN-GP to a pandas mixed-type table."""

        if y is not None:
            if target is not None:
                raise ValueError("pass the target once, as target or y")
            target = y
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("ConditionalGANEngine.fit requires a DataFrame")
        if frame.columns.has_duplicates:
            raise ValueError("dataframe contains duplicate column names")
        if len(frame) < 1 or len(frame.columns) < 1:
            raise ValueError("dataframe must contain rows and columns")
        target_index = getattr(target, "index", None)
        if (
            target_index is not None
            and isinstance(target_index, pd.Index)
            and not frame.index.equals(target_index)
        ):
            raise ValueError("frame and target must have the same index")

        fit_epochs = self.epochs if epochs is None else _positive_int(
            "epochs", epochs
        )
        fit_batch = self.batch_size if batch_size is None else _positive_int(
            "batch_size", batch_size
        )
        if fit_batch % self.pac:
            raise ValueError(
                "batch_size must be divisible by pac; batch_size counts raw "
                "rows and the effective critic batch is batch_size // pac"
            )
        fit_n_critic = self.n_critic if n_critic is None else _positive_int(
            "n_critic", n_critic
        )
        fit_compile = self.compile if compile is None else bool(compile)
        if self.dp_enabled and fit_compile:
            raise ValueError("compile=True is not supported for DP training")
        if preprocessor is not None:
            if not self.dp_enabled:
                raise ValueError(
                    "a fixed preprocessor hook is available only in DP mode"
                )
            if not isinstance(preprocessor, TableTransformer):
                raise TypeError("preprocessor must be a TableTransformer")
            if not getattr(preprocessor, "fitted_", False):
                raise RuntimeError("the fixed preprocessor is not fitted")
            if len(self.constraints):
                raise ValueError(
                    "a fixed public preprocessor cannot be combined with "
                    "engine-fitted structural constraints"
                )
            if schema is not None or schema_overrides is not None:
                raise ValueError(
                    "schema options cannot be combined with a fixed preprocessor"
                )
        if self.dp_enabled:
            if target is not None:
                raise ValueError(
                    "DP mode currently refuses private targets; generator "
                    "training must remain postprocessing of the DP critic"
                )
            if fit_batch > len(frame):
                raise ValueError(
                    "DP batch_size cannot exceed the number of training rows; "
                    "critic batches are sampled without replacement"
                )
            if fit_batch % self.dp_config.microbatch_size:
                raise ValueError(
                    "DP batch_size must be divisible by microbatch_size"
                )
        fit_schema = self.schema if schema is None else schema
        fit_overrides = (
            self.schema_overrides
            if schema_overrides is None
            else dict(schema_overrides)
        )
        if fit_schema is not None and fit_overrides:
            raise ValueError("schema and schema_overrides cannot both be set")
        if preprocessor is not None and (
            fit_schema is not None or fit_overrides
        ):
            raise ValueError(
                "configured schema options cannot be combined with a fixed "
                "preprocessor"
            )

        self._rng = np.random.default_rng(self.random_state)
        training_frame = frame
        transformer_schema = fit_schema
        transformer_overrides = fit_overrides
        self.constraint_transformer_ = None
        if len(self.constraints):
            # Resolve the user-facing schema before changing coordinates.  The
            # inner TableTransformer then receives a derived schema in which
            # gap/log-ratio coordinates are numeric continuous features.
            schema_probe = TableTransformer(
                schema=fit_schema,
                schema_overrides=fit_overrides,
                continuous_transform=self.continuous_transform,
                handle_unknown=self.handle_unknown,
                max_modes=self.max_modes,
                random_state=self.random_state,
            ).fit(frame)
            self.constraint_transformer_ = (
                StructuralConstraintTransformer(self.constraints).fit(
                    frame, schema=schema_probe.schema_
                )
            )
            training_frame = self.constraint_transformer_.transform(
                frame, validate=False
            )
            transformer_schema = (
                self.constraint_transformer_.transform_schema(
                    schema_probe.schema_
                )
            )
            transformer_overrides = {}
        self._dp_fixed_preprocessor_ = preprocessor is not None
        if preprocessor is None:
            self.transformer_ = TableTransformer(
                schema=transformer_schema,
                schema_overrides=transformer_overrides,
                continuous_transform=self.continuous_transform,
                handle_unknown=self.handle_unknown,
                max_modes=self.max_modes,
                random_state=self.random_state,
            )
            encoded = np.asarray(
                self.transformer_.fit_transform(training_frame),
                dtype=np.float64,
            )
        else:
            self.transformer_ = preprocessor
            encoded = np.asarray(
                self.transformer_.transform(training_frame),
                dtype=np.float64,
            )
        self.constraint_preprocessor_ = (
            None
            if self.constraint_transformer_ is None
            else ConstrainedTableTransformer(
                self.constraint_transformer_, self.transformer_
            )
        )
        self.preprocessor_ = (
            self.transformer_
            if self.constraint_preprocessor_ is None
            else self.constraint_preprocessor_
        )
        sampler_seed = (
            None if self.random_state is None else self.random_state + 104729
        )
        self.sampler_ = ConditionSampler(
            continuous_bins=self.continuous_bins,
            target_name=target_name,
            random_state=sampler_seed,
        ).fit(encoded, self.transformer_, target)
        if self.dp_enabled:
            # Private category frequencies and row-membership lists must not
            # drive critic or generator updates. DP mode is deliberately
            # unconditional; explicit sample-time table values remain ordinary
            # caller-provided postprocessing.
            self.sampler_.groups_ = ()
            self.sampler_.condition_dim_ = 0
        self._sync_metadata()
        self.head_specs_ = tuple(self.transformer_.head_metadata)
        target_head = self.sampler_.target_head_spec
        if target_head is not None:
            self.head_specs_ += (target_head,)

        # Non-DP sampling is with replacement. DP critic batches are instead
        # selected uniformly without replacement so one person contributes to
        # at most one clipped microbatch in an update.
        self._fit_batch_size_ = fit_batch
        self._effective_critic_batch_size_ = fit_batch // self.pac
        self._fit_n_critic_ = fit_n_critic
        self._fit_compile_ = fit_compile
        self._training_rows_ = len(frame)
        if self.dp_enabled:
            self._parameter_condition_defaults_ = (
                self._decoded_zero_defaults(self.transformer_)
            )
            self._condition_defaults_ = dict(
                self._parameter_condition_defaults_
            )
            self._initialize_dp_state()
        else:
            self._condition_defaults_ = self._condition_defaults(frame)
            self._parameter_condition_defaults_ = self._condition_defaults(
                training_frame
            )
        self._build_models(compile_steps=fit_compile)
        self.history_ = {
            "critic": [],
            "generator": [],
            "gradient_penalty": [],
            "conditional": [],
            "interaction": [],
            "feature_matching": [],
            "warmup": [],
        }
        self.epochs_ran_ = 0
        self.warmup_epochs_ran_ = 0

        steps = max(1, int(math.ceil(len(frame) / float(fit_batch))))
        if self.warmup_epochs:
            self._run_warmup(steps, verbose=verbose)
            self.ema_generator_.set_weights(self.generator_.get_weights())
        for epoch in range(1, fit_epochs + 1):
            critic_values: List[float] = []
            penalty_values: List[float] = []
            generator_values: List[float] = []
            conditional_values: List[float] = []
            interaction_values: List[float] = []
            feature_matching_values: List[float] = []
            for _ in range(steps):
                for _ in range(fit_n_critic):
                    if self.dp_enabled:
                        indices = self._rng.choice(
                            len(encoded),
                            size=self._fit_batch_size_,
                            replace=False,
                        )
                        real = encoded[indices].astype(
                            np.float32, copy=False
                        )
                        condition = np.zeros(
                            (
                                self._fit_batch_size_,
                                self.sampler_.condition_dim_,
                            ),
                            dtype=np.float32,
                        )
                    else:
                        batch = self.sampler_.sample_training(
                            self._fit_batch_size_
                        )
                        real = self.sampler_.output_matrix_[
                            batch.real_indices
                        ].astype(np.float32, copy=False)
                        condition = batch.condition
                    noise = self._noise(self._fit_batch_size_)
                    epsilon = self._rng.random(
                        (self._fit_batch_size_, 1)
                    ).astype(np.float32)
                    critic_loss, penalty = self._critic_step_fn(
                        tf.convert_to_tensor(real),
                        tf.convert_to_tensor(condition),
                        tf.convert_to_tensor(noise),
                        tf.convert_to_tensor(epsilon),
                    )
                    if self.dp_enabled:
                        self.privacy_accountant_.record_step(
                            sample_rate=min(
                                1.0,
                                self._fit_batch_size_
                                / float(self._training_rows_),
                            )
                        )
                    critic_values.append(float(critic_loss.numpy()))
                    penalty_values.append(float(penalty.numpy()))

                if self.dp_enabled:
                    condition = np.zeros(
                        (
                            self._fit_batch_size_,
                            self.sampler_.condition_dim_,
                        ),
                        dtype=np.float32,
                    )
                    target_values = np.zeros(
                        (
                            self._fit_batch_size_,
                            self.sampler_.output_dim_,
                        ),
                        dtype=np.float32,
                    )
                    target_mask = np.zeros_like(target_values)
                    group_indices = np.full(
                        self._fit_batch_size_, -1, dtype=np.int64
                    )
                    # Direct real-data generator losses are forbidden in DP
                    # mode. This placeholder is never consumed because
                    # interaction and feature matching were rejected.
                    real = np.zeros_like(target_values)
                else:
                    batch = self.sampler_.sample_training(
                        self._fit_batch_size_
                    )
                    real = self.sampler_.output_matrix_[
                        batch.real_indices
                    ].astype(np.float32, copy=False)
                    condition = batch.condition
                    target_values = batch.output_target
                    target_mask = batch.output_mask
                    group_indices = batch.group_indices
                (
                    generator_loss,
                    conditional_loss,
                    interaction_loss,
                    feature_matching_loss,
                ) = self._generator_step_fn(
                    tf.convert_to_tensor(self._noise(self._fit_batch_size_)),
                    tf.convert_to_tensor(condition),
                    tf.convert_to_tensor(target_values),
                    tf.convert_to_tensor(target_mask),
                    tf.convert_to_tensor(group_indices),
                    tf.convert_to_tensor(real),
                )
                generator_values.append(float(generator_loss.numpy()))
                conditional_values.append(float(conditional_loss.numpy()))
                interaction_values.append(float(interaction_loss.numpy()))
                feature_matching_values.append(
                    float(feature_matching_loss.numpy())
                )

            values = (
                critic_values
                + penalty_values
                + generator_values
                + conditional_values
                + interaction_values
                + feature_matching_values
            )
            if not all(math.isfinite(value) for value in values):
                raise RuntimeError(
                    "conditional GAN training produced a non-finite loss"
                )
            self.history_["critic"].append(float(np.mean(critic_values)))
            self.history_["gradient_penalty"].append(
                float(np.mean(penalty_values))
            )
            self.history_["generator"].append(
                float(np.mean(generator_values))
            )
            self.history_["conditional"].append(
                float(np.mean(conditional_values))
            )
            self.history_["interaction"].append(
                float(np.mean(interaction_values))
            )
            self.history_["feature_matching"].append(
                float(np.mean(feature_matching_values))
            )
            self.epochs_ran_ = epoch
            self._collect_generator_snapshot(epoch)
            if verbose:
                print(
                    "Epoch %d/%d critic=%.4f generator=%.4f "
                    "conditional=%.4f interaction=%.4f "
                    "feature_matching=%.4f gp=%.4f"
                    % (
                        epoch,
                        fit_epochs,
                        self.history_["critic"][-1],
                        self.history_["generator"][-1],
                        self.history_["conditional"][-1],
                        self.history_["interaction"][-1],
                        self.history_["feature_matching"][-1],
                        self.history_["gradient_penalty"][-1],
                    )
                )
        if self.dp_enabled:
            self._finalize_privacy_report()
            # ConditionSampler normally retains the complete encoded training
            # matrix. DP mode never needs it after fitting, so remove it before
            # either in-memory inspection or persistence.
            self.sampler_.output_matrix_ = np.zeros(
                (1, self.sampler_.output_dim_), dtype=np.float64
            )
        self.fitted_ = True
        return self

    def _build_models(self, *, compile_steps: bool) -> None:
        self.generator_ = ConditionalGenerator(
            self.head_specs_,
            noise_dim=self.noise_dim,
            condition_dim=self.sampler_.condition_dim_,
            hidden_dims=self.generator_dims,
            random_state=self.random_state,
        )
        critic_seed = (
            None if self.random_state is None else self.random_state + 7919
        )
        parameter_budget = self.critic_parameter_budget
        if self.match_critic_parameters:
            reference = ConditionalCritic(
                self.sampler_.output_dim_,
                condition_dim=self.sampler_.condition_dim_,
                hidden_dims=self.critic_dims,
                spectral_normalization=self.spectral_normalization,
                random_state=critic_seed,
                architecture="residual",
                pac=1,
                name="conditional_critic_parameter_reference",
            )
            reference(
                [
                    tf.zeros(
                        (1, self.sampler_.output_dim_), dtype=tf.float32
                    ),
                    tf.zeros(
                        (1, self.sampler_.condition_dim_), dtype=tf.float32
                    ),
                ],
                training=False,
            )
            parameter_budget = int(
                sum(
                    int(np.prod(variable.shape))
                    for variable in reference.trainable_variables
                )
            )
        self.critic_ = ConditionalCritic(
            self.sampler_.output_dim_,
            condition_dim=self.sampler_.condition_dim_,
            hidden_dims=self.critic_dims,
            spectral_normalization=self.spectral_normalization,
            random_state=critic_seed,
            architecture=self.critic_architecture,
            pac=self.pac,
            attention_heads=self.attention_heads,
            attention_dim=self.attention_dim,
            fourier_features=self.fourier_features,
            fourier_scale=self.fourier_scale,
            parameter_budget=parameter_budget,
        )
        self.ema_generator_ = ConditionalGenerator(
            self.head_specs_,
            noise_dim=self.noise_dim,
            condition_dim=self.sampler_.condition_dim_,
            hidden_dims=self.generator_dims,
            random_state=self.random_state,
            name="conditional_generator_ema",
        )
        dummy_condition = tf.zeros(
            (self.pac, self.sampler_.condition_dim_), dtype=tf.float32
        )
        dummy_noise = tf.zeros((1, self.noise_dim), dtype=tf.float32)
        dummy_features = tf.zeros(
            (self.pac, self.sampler_.output_dim_), dtype=tf.float32
        )
        generator_condition = dummy_condition[:1]
        self.generator_(
            [dummy_noise, generator_condition], training=False
        )
        self.critic_([dummy_features, dummy_condition], training=False)
        self.ema_generator_(
            [dummy_noise, generator_condition], training=False
        )
        self.ema_generator_.set_weights(self.generator_.get_weights())
        self.critic_parameter_count_ = int(
            sum(
                int(np.prod(variable.shape))
                for variable in self.critic_.trainable_variables
            )
        )
        self.critic_parameter_target_ = (
            self.critic_parameter_count_
            if parameter_budget is None
            else int(parameter_budget)
        )
        self.critic_parameter_gap_ = (
            self.critic_parameter_count_ - self.critic_parameter_target_
        )

        self.warmup_encoder_ = None
        if self.warmup_epochs:
            encoder_seed = (
                None
                if self.random_state is None
                else self.random_state + 16127
            )
            self.warmup_encoder_ = ConditionalDenoisingEncoder(
                self.sampler_.output_dim_,
                self.noise_dim,
                condition_dim=self.sampler_.condition_dim_,
                hidden_dims=tuple(reversed(self.generator_dims)),
                random_state=encoder_seed,
            )
            self.warmup_encoder_(
                [
                    tf.zeros(
                        (1, self.sampler_.output_dim_), dtype=tf.float32
                    ),
                    tf.zeros(
                        (1, self.sampler_.output_dim_), dtype=tf.float32
                    ),
                    generator_condition,
                ],
                training=False,
            )

        self._reset_weight_collection()

        self.generator_optimizer_ = tf.keras.optimizers.Adam(
            learning_rate=self.generator_learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
        )
        self.critic_optimizer_ = tf.keras.optimizers.Adam(
            learning_rate=self.critic_learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
        )
        self.warmup_optimizer_ = (
            tf.keras.optimizers.Adam(
                learning_rate=self.warmup_learning_rate,
                beta_1=self.beta_1,
                beta_2=self.beta_2,
            )
            if self.warmup_encoder_ is not None
            else None
        )
        if self.dp_enabled:
            if compile_steps:
                raise ValueError("compiled DP training steps are unsupported")
            self._ensure_dp_noise_generator()
            self._critic_step_fn = self._dp_critic_train_step
            self._generator_step_fn = self._generator_train_step
            self._warmup_step_fn = self._warmup_train_step
        elif compile_steps:
            self._critic_step_fn = tf.function(self._critic_train_step)
            self._generator_step_fn = tf.function(
                self._generator_train_step
            )
            self._warmup_step_fn = tf.function(self._warmup_train_step)
        else:
            self._critic_step_fn = self._critic_train_step
            self._generator_step_fn = self._generator_train_step
            self._warmup_step_fn = self._warmup_train_step

    @staticmethod
    def _apply_gradients(
        optimizer: tf.keras.optimizers.Optimizer,
        gradients: Sequence[Optional[tf.Tensor]],
        variables: Sequence[tf.Variable],
        owner: str,
    ) -> None:
        pairs = [
            (gradient, variable)
            for gradient, variable in zip(gradients, variables)
            if gradient is not None
        ]
        if not pairs:
            raise RuntimeError("no gradients were computed for %s" % owner)
        for gradient, _ in pairs:
            tf.debugging.assert_all_finite(
                gradient, "%s produced a non-finite gradient" % owner
            )
        optimizer.apply_gradients(pairs)

    def _critic_train_step(
        self,
        real: tf.Tensor,
        condition: tf.Tensor,
        noise: tf.Tensor,
        epsilon: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        variables = self.critic_.trainable_variables
        with tf.GradientTape() as tape:
            fake = tf.stop_gradient(
                self.generator_([noise, condition], training=True)
            )
            real_score = self.critic_([real, condition], training=True)
            fake_score = self.critic_([fake, condition], training=True)
            interpolated = epsilon * real + (1.0 - epsilon) * fake
            with tf.GradientTape() as penalty_tape:
                penalty_tape.watch(interpolated)
                mixed_score = self.critic_(
                    [interpolated, condition], training=True
                )
            gradients = penalty_tape.gradient(
                mixed_score, interpolated
            )
            packed_gradients = tf.reshape(
                gradients,
                [
                    tf.shape(gradients)[0] // self.pac,
                    self.pac * self.sampler_.output_dim_,
                ],
            )
            slopes = tf.sqrt(
                tf.reduce_sum(
                    tf.square(packed_gradients), axis=1
                )
                + 1e-12
            )
            penalty = tf.reduce_mean(tf.square(slopes - 1.0))
            loss = (
                tf.reduce_mean(fake_score)
                - tf.reduce_mean(real_score)
                + self.gradient_penalty * penalty
            )
        gradients = tape.gradient(loss, variables)
        self._apply_gradients(
            self.critic_optimizer_, gradients, variables, "critic"
        )
        return loss, penalty

    def _dp_critic_train_step(
        self,
        real: tf.Tensor,
        condition: tf.Tensor,
        noise: tf.Tensor,
        epsilon: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Apply clipped microbatch critic gradients plus Gaussian noise.

        ``epsilon`` is accepted for train-step signature parity but is unused:
        WGAN-GP is intentionally forbidden because a batch-coupled gradient
        penalty is not compatible with this per-unit mechanism.
        """

        del epsilon
        if not self.dp_enabled or self.dp_config is None:
            raise RuntimeError("DP critic step called without DPConfig")
        variables = self.critic_.trainable_variables
        microbatch_size = self.dp_config.microbatch_size
        tf.debugging.assert_equal(
            tf.math.floormod(tf.shape(real)[0], microbatch_size),
            0,
            message="DP batch must be divisible by microbatch_size",
        )
        with tf.GradientTape(persistent=True) as tape:
            fake = tf.stop_gradient(
                self.generator_([noise, condition], training=True)
            )
            real_score = tf.reshape(
                self.critic_([real, condition], training=True), [-1]
            )
            fake_score = tf.reshape(
                self.critic_([fake, condition], training=True), [-1]
            )
            per_example_loss = fake_score - real_score
            microbatch_loss = tf.reduce_mean(
                tf.reshape(
                    per_example_loss, [-1, microbatch_size]
                ),
                axis=1,
            )
        jacobians = [
            tape.jacobian(
                microbatch_loss,
                variable,
                experimental_use_pfor=False,
            )
            for variable in variables
        ]
        del tape
        if any(gradient is None for gradient in jacobians):
            raise RuntimeError(
                "no per-microbatch gradients were computed for DP critic"
            )
        squared_norm = tf.zeros(
            tf.shape(microbatch_loss), dtype=microbatch_loss.dtype
        )
        for gradient in jacobians:
            axes = tf.range(1, tf.rank(gradient))
            squared_norm += tf.reduce_sum(tf.square(gradient), axis=axes)
        norm = tf.sqrt(squared_norm + tf.cast(1e-30, squared_norm.dtype))
        clip = tf.cast(
            self.dp_config.l2_norm_clip, microbatch_loss.dtype
        )
        factors = tf.minimum(tf.ones_like(norm), clip / norm)
        private_units = tf.cast(tf.shape(microbatch_loss)[0], tf.float32)
        noisy_gradients = []
        for gradient, variable in zip(jacobians, variables):
            rank = gradient.shape.rank
            if rank is None:
                raise RuntimeError("DP gradient rank must be statically known")
            scale = tf.reshape(factors, (-1,) + (1,) * (rank - 1))
            clipped_sum = tf.reduce_sum(gradient * scale, axis=0)
            noise_value = self._dp_noise_generator_.normal(
                shape=tf.shape(clipped_sum),
                dtype=clipped_sum.dtype,
            )
            noise_value *= tf.cast(
                self.dp_config.noise_multiplier
                * self.dp_config.l2_norm_clip,
                clipped_sum.dtype,
            )
            noisy_gradients.append(
                (clipped_sum + noise_value)
                / tf.cast(private_units, clipped_sum.dtype)
            )
        self._apply_gradients(
            self.critic_optimizer_,
            noisy_gradients,
            variables,
            "DP critic",
        )
        zero = tf.zeros((), dtype=microbatch_loss.dtype)
        # Raw critic losses are direct functions of private examples and are
        # therefore not exposed through history or verbose logging.
        return zero, zero

    def _generator_train_step(
        self,
        noise: tf.Tensor,
        condition: tf.Tensor,
        target: tf.Tensor,
        mask: tf.Tensor,
        group_indices: tf.Tensor,
        real: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        variables = self.generator_.trainable_variables
        with tf.GradientTape() as tape:
            fake = self.generator_([noise, condition], training=True)
            score, fake_features = self.critic_(
                [fake, condition],
                training=False,
                return_features=True,
            )
            conditional = self._conditional_loss(
                fake, target, mask, group_indices
            )
            interaction = (
                self._interaction_loss(real, fake)
                if self.interaction_weight
                else tf.zeros((), dtype=fake.dtype)
            )
            if self.feature_matching_weight:
                real_features = tf.stop_gradient(
                    self.critic_.extract_features(
                        [real, condition], training=False
                    )
                )
                feature_matching = tf.reduce_mean(
                    tf.square(
                        tf.reduce_mean(fake_features, axis=0)
                        - tf.reduce_mean(real_features, axis=0)
                    )
                )
            else:
                feature_matching = tf.zeros((), dtype=fake.dtype)
            loss = (
                -tf.reduce_mean(score)
                + self.conditional_weight * conditional
                + self.interaction_weight * interaction
                + self.feature_matching_weight * feature_matching
            )
        gradients = tape.gradient(loss, variables)
        self._apply_gradients(
            self.generator_optimizer_, gradients, variables, "generator"
        )
        for averaged, live in zip(
            self.ema_generator_.weights, self.generator_.weights
        ):
            averaged.assign(
                self.ema_decay * averaged
                + (1.0 - self.ema_decay) * live
            )
        return loss, conditional, interaction, feature_matching

    def _conditional_loss(
        self,
        generated: tf.Tensor,
        target: tf.Tensor,
        mask: tf.Tensor,
        group_indices: tf.Tensor,
    ) -> tf.Tensor:
        if not self.sampler_.groups_:
            return tf.zeros((), dtype=generated.dtype)
        total = tf.zeros((), dtype=generated.dtype)
        active_groups = tf.zeros((), dtype=generated.dtype)
        epsilon = tf.cast(1e-7, generated.dtype)
        for index, group in enumerate(self.sampler_.groups_):
            active = tf.cast(
                tf.equal(group_indices, index), generated.dtype
            )
            count = tf.reduce_sum(active)
            predicted = generated[:, group.output_slice]
            expected = target[:, group.output_slice]
            if group.activation == "softmax":
                per_row = -tf.reduce_sum(
                    expected
                    * tf.math.log(tf.clip_by_value(predicted, epsilon, 1.0)),
                    axis=1,
                )
            elif group.activation == "sigmoid":
                clipped = tf.clip_by_value(
                    predicted, epsilon, 1.0 - epsilon
                )
                per_row = -tf.reduce_mean(
                    expected * tf.math.log(clipped)
                    + (1.0 - expected) * tf.math.log(1.0 - clipped),
                    axis=1,
                )
            else:
                selected_mask = mask[:, group.output_slice]
                denominator = tf.maximum(
                    tf.reduce_sum(selected_mask, axis=1), 1.0
                )
                per_row = (
                    tf.reduce_sum(
                        tf.square(predicted - expected) * selected_mask,
                        axis=1,
                    )
                    / denominator
                )
            enabled = tf.cast(count > 0.0, generated.dtype)
            total += (
                tf.reduce_sum(per_row * active)
                / tf.maximum(count, 1.0)
            ) * enabled
            active_groups += enabled
        return total / tf.maximum(active_groups, 1.0)

    def _interaction_loss(
        self, real: tf.Tensor, generated: tf.Tensor
    ) -> tf.Tensor:
        """Match off-diagonal Pearson and differentiable rank covariance."""

        real = tf.cast(real, generated.dtype)
        width = tf.shape(generated)[1]
        off_diagonal = tf.ones(
            [width, width], dtype=generated.dtype
        ) - tf.eye(width, dtype=generated.dtype)
        denominator = tf.maximum(
            tf.reduce_sum(off_diagonal),
            tf.cast(1.0, generated.dtype),
        )

        def standardized_covariance(values: tf.Tensor) -> tf.Tensor:
            centered = values - tf.reduce_mean(values, axis=0, keepdims=True)
            scale = tf.sqrt(
                tf.reduce_mean(tf.square(centered), axis=0, keepdims=True)
                + tf.cast(1e-6, values.dtype)
            )
            normalized = centered / scale
            rows = tf.cast(tf.shape(values)[0], values.dtype)
            return tf.matmul(
                normalized, normalized, transpose_a=True
            ) / tf.maximum(rows - 1.0, 1.0)

        def soft_rank_covariance(values: tf.Tensor) -> tf.Tensor:
            differences = (
                tf.expand_dims(values, axis=1)
                - tf.expand_dims(values, axis=0)
            ) / tf.cast(self.interaction_temperature, values.dtype)
            ranks = tf.reduce_mean(tf.math.sigmoid(differences), axis=1)
            return standardized_covariance(ranks)

        real_covariance = tf.stop_gradient(
            standardized_covariance(real)
        )
        fake_covariance = standardized_covariance(generated)
        real_rank_covariance = tf.stop_gradient(
            soft_rank_covariance(real)
        )
        fake_rank_covariance = soft_rank_covariance(generated)
        covariance_loss = tf.reduce_sum(
            tf.square(fake_covariance - real_covariance) * off_diagonal
        ) / denominator
        rank_loss = tf.reduce_sum(
            tf.square(
                fake_rank_covariance - real_rank_covariance
            )
            * off_diagonal
        ) / denominator
        return 0.5 * (covariance_loss + rank_loss)

    def _typed_reconstruction_loss(
        self, expected: tf.Tensor, reconstructed: tf.Tensor
    ) -> tf.Tensor:
        losses: List[tf.Tensor] = []
        epsilon = tf.cast(1e-7, reconstructed.dtype)
        for head in self.head_specs_:
            target = expected[:, head.slice]
            predicted = reconstructed[:, head.slice]
            if head.activation == "softmax":
                loss = -tf.reduce_mean(
                    tf.reduce_sum(
                        target
                        * tf.math.log(
                            tf.clip_by_value(predicted, epsilon, 1.0)
                        ),
                        axis=1,
                    )
                )
            elif head.activation == "sigmoid":
                clipped = tf.clip_by_value(
                    predicted, epsilon, 1.0 - epsilon
                )
                loss = -tf.reduce_mean(
                    target * tf.math.log(clipped)
                    + (1.0 - target) * tf.math.log(1.0 - clipped)
                )
            else:
                loss = tf.reduce_mean(tf.square(predicted - target))
            losses.append(loss)
        return tf.add_n(losses) / tf.cast(
            len(losses), reconstructed.dtype
        )

    def _warmup_train_step(
        self,
        real: tf.Tensor,
        condition: tf.Tensor,
        corruption_mask: tf.Tensor,
    ) -> tf.Tensor:
        if self.warmup_encoder_ is None or self.warmup_optimizer_ is None:
            raise RuntimeError("masked-denoising warmup is disabled")
        variables = (
            self.warmup_encoder_.trainable_variables
            + self.generator_.trainable_variables
        )
        with tf.GradientTape() as tape:
            corrupted = real * (1.0 - corruption_mask)
            latent = self.warmup_encoder_(
                [corrupted, corruption_mask, condition],
                training=True,
            )
            reconstructed = self.generator_(
                [latent, condition], training=True
            )
            loss = self._typed_reconstruction_loss(real, reconstructed)
        gradients = tape.gradient(loss, variables)
        self._apply_gradients(
            self.warmup_optimizer_,
            gradients,
            variables,
            "masked-denoising warmup",
        )
        return loss

    def _run_warmup(self, steps: int, *, verbose: int) -> None:
        for epoch in range(1, self.warmup_epochs + 1):
            losses: List[float] = []
            for _ in range(steps):
                batch = self.sampler_.sample_training(
                    self._fit_batch_size_
                )
                real = self.sampler_.output_matrix_[
                    batch.real_indices
                ].astype(np.float32, copy=False)
                corruption_mask = (
                    self._rng.random(real.shape)
                    < self.warmup_mask_probability
                ).astype(np.float32)
                loss = self._warmup_step_fn(
                    tf.convert_to_tensor(real),
                    tf.convert_to_tensor(batch.condition),
                    tf.convert_to_tensor(corruption_mask),
                )
                losses.append(float(loss.numpy()))
            if not all(math.isfinite(value) for value in losses):
                raise RuntimeError(
                    "masked-denoising warmup produced a non-finite loss"
                )
            self.history_["warmup"].append(float(np.mean(losses)))
            self.warmup_epochs_ran_ = epoch
            if verbose:
                print(
                    "Warmup %d/%d reconstruction=%.4f"
                    % (
                        epoch,
                        self.warmup_epochs,
                        self.history_["warmup"][-1],
                    )
                )

    def _reset_weight_collection(self) -> None:
        self.swa_collections_ = 0
        self.swag_collections_ = 0
        self._swa_weights_: Optional[List[np.ndarray]] = None
        self._swag_mean_: Optional[np.ndarray] = None
        self._swag_square_mean_: Optional[np.ndarray] = None
        self._swag_snapshots_: List[np.ndarray] = []
        weights = self.generator_.get_weights()
        self._generator_weight_shapes_ = [
            tuple(int(value) for value in weight.shape) for weight in weights
        ]
        self._generator_weight_sizes_ = [
            int(weight.size) for weight in weights
        ]

    @staticmethod
    def _flatten_weights(weights: Sequence[np.ndarray]) -> np.ndarray:
        return np.concatenate(
            [np.asarray(weight, dtype=np.float64).reshape(-1) for weight in weights]
        )

    def _unflatten_generator_weights(
        self, vector: np.ndarray
    ) -> List[np.ndarray]:
        values: List[np.ndarray] = []
        offset = 0
        current = self.generator_.get_weights()
        for shape, size, reference in zip(
            self._generator_weight_shapes_,
            self._generator_weight_sizes_,
            current,
        ):
            values.append(
                np.asarray(vector[offset : offset + size])
                .reshape(shape)
                .astype(reference.dtype, copy=False)
            )
            offset += size
        if offset != len(vector):
            raise ValueError("generator weight vector has the wrong size")
        return values

    def _collect_generator_snapshot(self, epoch: int) -> None:
        weights = self.generator_.get_weights()
        if (
            self.swa
            and epoch >= self.swa_start_epoch
            and (epoch - self.swa_start_epoch) % self.swa_frequency == 0
        ):
            self.swa_collections_ += 1
            if self._swa_weights_ is None:
                self._swa_weights_ = [
                    np.asarray(value, dtype=np.float64).copy()
                    for value in weights
                ]
            else:
                rate = 1.0 / float(self.swa_collections_)
                for average, value in zip(self._swa_weights_, weights):
                    average += rate * (
                        np.asarray(value, dtype=np.float64) - average
                    )

        if (
            self.swag
            and epoch >= self.swag_start_epoch
            and (epoch - self.swag_start_epoch) % self.swag_frequency == 0
        ):
            vector = self._flatten_weights(weights)
            self.swag_collections_ += 1
            if self._swag_mean_ is None:
                self._swag_mean_ = vector.copy()
                self._swag_square_mean_ = np.square(vector)
            else:
                rate = 1.0 / float(self.swag_collections_)
                self._swag_mean_ += rate * (vector - self._swag_mean_)
                if self._swag_square_mean_ is None:
                    raise RuntimeError("SWAG second moment is unavailable")
                self._swag_square_mean_ += rate * (
                    np.square(vector) - self._swag_square_mean_
                )
            self._swag_snapshots_.append(vector.copy())
            if len(self._swag_snapshots_) > self.swag_max_rank:
                self._swag_snapshots_.pop(0)

    def sample_swag_weights(
        self,
        random_state: Optional[int] = None,
        *,
        scale: Optional[float] = None,
    ) -> List[np.ndarray]:
        """Sample one deterministic low-rank SWAG generator state."""

        if (
            not self.swag_collections_
            or self._swag_mean_ is None
            or self._swag_square_mean_ is None
        ):
            raise RuntimeError("no SWAG generator snapshots were collected")
        sample_scale = (
            self.swag_scale
            if scale is None
            else _nonnegative_float("scale", scale)
        )
        seed = (
            int(self._rng.integers(0, np.iinfo(np.int32).max))
            if random_state is None
            else int(random_state)
        )
        rng = np.random.default_rng(seed)
        diagonal_variance = np.maximum(
            self._swag_square_mean_ - np.square(self._swag_mean_),
            0.0,
        )
        diagonal = np.sqrt(diagonal_variance) * rng.normal(
            size=len(self._swag_mean_)
        )
        low_rank = np.zeros_like(self._swag_mean_)
        if len(self._swag_snapshots_) > 1:
            deviations = np.stack(self._swag_snapshots_) - self._swag_mean_
            coefficients = rng.normal(size=len(deviations))
            low_rank = (
                coefficients @ deviations
            ) / math.sqrt(float(len(deviations) - 1))
        sampled = self._swag_mean_ + (
            sample_scale / math.sqrt(2.0)
        ) * (diagonal + low_rank)
        return self._unflatten_generator_weights(sampled)

    def swa_weights(self) -> List[np.ndarray]:
        """Return a copy of the collected stochastic weight average."""

        if not self.swa_collections_ or self._swa_weights_ is None:
            raise RuntimeError("no SWA generator snapshots were collected")
        current = self.generator_.get_weights()
        return [
            average.astype(reference.dtype, copy=True)
            for average, reference in zip(self._swa_weights_, current)
        ]

    def _noise(self, rows: int) -> np.ndarray:
        return self._rng.normal(
            0.0, 1.0, size=(int(rows), self.noise_dim)
        ).astype(np.float32)

    def set_sampling_seed(self, random_state: int) -> None:
        """Reset generation RNGs without changing fitted model weights."""

        if isinstance(random_state, bool) or not isinstance(
            random_state, (int, np.integer)
        ):
            raise ValueError("random_state must be an integer")
        self._check_fitted()
        seed = int(random_state)
        self._rng = np.random.default_rng(seed)
        self.sampler_._rng = np.random.default_rng(seed + 104729)

    @staticmethod
    def _decoded_zero_defaults(
        transformer: TableTransformer,
    ) -> Dict[Any, Any]:
        encoded = np.zeros((1, transformer.output_dim_), dtype=np.float64)
        decoded = transformer.inverse_transform(encoded)
        return {
            name: (
                value.item() if isinstance(value, np.generic) else value
            )
            for name, value in decoded.iloc[0].items()
        }

    def _ensure_dp_noise_generator(self) -> None:
        if not self.dp_enabled or self.dp_config is None:
            return
        if hasattr(self, "_dp_noise_generator_"):
            return
        seed = self.dp_config.noise_seed
        if seed is None:
            # Draw from the operating system rather than the reproducible model
            # RNG. The seed and generator state are intentionally not persisted.
            seed = secrets.randbelow(2**31 - 1)
        generator_type = getattr(
            tf.random, "Generator", tf.random.experimental.Generator
        )
        self._dp_noise_generator_ = generator_type.from_seed(
            int(seed) % (2**31 - 1)
        )

    def _initialize_dp_state(self) -> None:
        if not self.dp_enabled or self.dp_config is None:
            return
        self.privacy_accountant_ = PrivacyAccountant(
            self.dp_config.noise_multiplier,
            orders=self.dp_config.accountant_orders,
        )
        if hasattr(self, "_dp_noise_generator_"):
            del self._dp_noise_generator_
        self._ensure_dp_noise_generator()
        self.privacy_report_ = None

    def _finalize_privacy_report(self) -> None:
        if (
            not self.dp_enabled
            or self.dp_config is None
            or self.privacy_accountant_ is None
        ):
            return
        preprocessing_accounted = bool(
            getattr(self, "_dp_fixed_preprocessor_", False)
            and self.dp_config.preprocessing_public
            and self.constraint_transformer_ is None
        )
        # DP mode removes all condition groups before model construction.
        # Neither critic nor generator samples private conditional frequencies.
        conditional_accounted = True
        dataset_size_public = bool(self.dp_config.dataset_size_public)
        noise_seed_private = self.dp_config.noise_seed is None
        delta_recommended = bool(
            self.dp_config.delta < 1.0 / float(self._training_rows_)
        )
        accountant = self.privacy_accountant_.report(
            self.dp_config.delta
        )
        finite_epsilon = bool(
            accountant["epsilon"] is not None
            and math.isfinite(float(accountant["epsilon"]))
        )
        formal = bool(
            accountant["accountant_valid"]
            and accountant["steps"] > 0
            and finite_epsilon
            and preprocessing_accounted
            and conditional_accounted
            and dataset_size_public
            and noise_seed_private
            and delta_recommended
        )
        scope = "end_to_end" if formal else "training_only_dp"
        reasons = []
        if not preprocessing_accounted:
            reasons.append(
                "preprocessing was fitted on private data or was not declared "
                "public/fixed"
            )
        if not finite_epsilon:
            reasons.append("the composed epsilon is not finite")
        if not dataset_size_public:
            reasons.append("the persisted dataset size is not declared public")
        if not noise_seed_private:
            reasons.append(
                "a deterministic DP noise seed was configured; this is for "
                "mechanism testing, not a formal privacy release"
            )
        if not delta_recommended:
            reasons.append("delta is not smaller than 1 / training_rows")
        self.privacy_report_ = {
            "enabled": True,
            "mechanism_applied": True,
            "scope": scope,
            "formal_dp": formal,
            "epsilon_claim": (
                accountant["epsilon"] if formal else None
            ),
            "accountant": accountant,
            "privacy_boundary": {
                "preprocessing_accounted": preprocessing_accounted,
                "preprocessing_source": (
                    "caller_declared_public_fixed"
                    if preprocessing_accounted
                    else "private_or_unverified"
                ),
                "conditional_frequencies_accounted": conditional_accounted,
                "conditional_frequencies_source": "disabled",
                "dataset_size_public": dataset_size_public,
                "noise_randomness_accounted": noise_seed_private,
                "noise_seed_persisted": False,
                "delta_smaller_than_inverse_dataset": delta_recommended,
                "generator_accessed_private_rows": False,
                "generator_is_postprocessing_of_private_critic": True,
            },
            "training": {
                "rows": self._training_rows_,
                "batch_size": self._fit_batch_size_,
                "microbatch_size": self.dp_config.microbatch_size,
                "clip_norm": self.dp_config.l2_norm_clip,
                "critic_updates": self.privacy_accountant_.steps,
                "sampling": "uniform_without_replacement",
                "conditioning": "disabled",
            },
            "formal_claim_refused_reasons": reasons,
            "limitations": [
                "The accountant is deliberately conservative and does not use "
                "privacy amplification from minibatch sampling.",
                "When scope is training_only_dp, epsilon is diagnostic for the "
                "clipped/noised critic updates conditional on fixed encoded "
                "inputs; it is not an end-to-end privacy claim.",
                "Private fitted preprocessors and ordinary non-DP model "
                "artifacts must not be released as if they were DP outputs.",
                "An end-to-end claim is conditional on the caller's declaration "
                "that the supplied fixed preprocessor and its schema/domain "
                "were obtained without private training data.",
            ],
        }
        if self.dp_config.require_end_to_end and not formal:
            raise RuntimeError(
                "require_end_to_end=True but a formal DP claim is unavailable: "
                + "; ".join(reasons)
            )

    @staticmethod
    def _condition_defaults(frame: pd.DataFrame) -> Dict[Any, Any]:
        defaults: Dict[Any, Any] = {}
        for name in frame.columns:
            observed = frame[name].dropna()
            value = None if len(observed) == 0 else observed.iloc[0]
            defaults[name] = (
                value.item() if isinstance(value, np.generic) else value
            )
        return defaults

    def sample(
        self,
        rows: int,
        conditions: Optional[Mapping[Any, Any]] = None,
        *,
        return_target: bool = False,
        use_ema: bool = True,
        weight_source: Optional[str] = None,
        weight_seed: Optional[int] = None,
        use_swa: bool = False,
        use_swag: bool = False,
        return_audit: bool = False,
    ):
        """Generate decoded rows from pure noise and an optional weight state.

        ``weight_source`` may be ``"live"``, ``"ema"``, ``"swa"``, or
        ``"swag"``. The legacy ``use_ema`` flag remains authoritative when no
        source is given. ``weight_seed`` makes SWAG draws reproducible.
        ``return_audit`` appends the deterministic constraint audit to the
        returned table (and target, when requested).
        """

        self._check_fitted()
        rows = _positive_int("rows", rows)
        requested_sources = int(bool(use_swa)) + int(bool(use_swag))
        if requested_sources > 1:
            raise ValueError("use_swa and use_swag are mutually exclusive")
        if weight_source is not None and requested_sources:
            raise ValueError(
                "pass weight_source or use_swa/use_swag, not both"
            )
        if use_swa:
            weight_source = "swa"
        elif use_swag:
            weight_source = "swag"
        source = (
            ("ema" if use_ema else "live")
            if weight_source is None
            else str(weight_source).strip().lower()
        )
        if source not in {"live", "ema", "swa", "swag"}:
            raise ValueError(
                "weight_source must be 'live', 'ema', 'swa', or 'swag'"
            )
        parameter_conditions = conditions
        if conditions and self.constraint_transformer_ is not None:
            parameter_conditions = (
                self.constraint_transformer_.transform_conditions(
                    conditions,
                    rows,
                    table_columns=self.constraint_transformer_.original_columns_,
                )
            )
            self._validate_parameter_conditions(
                parameter_conditions, rows
            )
        if self.dp_enabled:
            if parameter_conditions:
                unknown = [
                    name
                    for name in parameter_conditions
                    if not any(
                        self._value_equal(name, column)
                        for column in self.transformer_.schema_.names
                    )
                ]
                if unknown:
                    raise ValueError(
                        "conditions reference unknown names: %r" % unknown
                    )
            condition_batch = self.sampler_.sample_generation(
                rows, conditions=None, return_info=True
            )
        else:
            condition_batch = self.sampler_.sample_generation(
                rows, conditions=parameter_conditions, return_info=True
            )
        if source == "ema":
            generator = self.ema_generator_
        elif source == "live":
            generator = self.generator_
        else:
            weights = (
                self.swa_weights()
                if source == "swa"
                else self.sample_swag_weights(weight_seed)
            )
            generator = ConditionalGenerator(
                self.head_specs_,
                noise_dim=self.noise_dim,
                condition_dim=self.sampler_.condition_dim_,
                hidden_dims=self.generator_dims,
                random_state=self.random_state,
                name="conditional_generator_%s_sample" % source,
            )
            generator(
                [
                    tf.zeros((1, self.noise_dim), dtype=tf.float32),
                    tf.zeros(
                        (1, self.sampler_.condition_dim_), dtype=tf.float32
                    ),
                ],
                training=False,
            )
            generator.set_weights(weights)
        generated = generator(
            [
                tf.convert_to_tensor(self._noise(rows)),
                tf.convert_to_tensor(condition_batch.condition),
            ],
            training=False,
        ).numpy()

        selected = condition_batch.output_mask > 0.0
        generated[selected] = condition_batch.output_target[selected]
        if parameter_conditions:
            self._apply_explicit_table_conditions(
                generated, parameter_conditions, rows
            )
        if self.constraint_transformer_ is not None:
            (
                table,
                raw_parameterized,
            ) = self.constraint_preprocessor_.inverse_transform_with_parameterized(
                generated[:, : self.sampler_.data_dim_]
            )
            pre_projection = table.copy(deep=True)
            table, projection = self.constraints.project(
                table, return_audit=True
            )
            self.last_constraint_audit_ = constraint_audit(
                self.constraints,
                pre_projection,
                table,
                numeric_scales=(
                    self.constraint_transformer_.numeric_scales_
                ),
                projection=projection,
                raw=raw_parameterized,
            )
        else:
            table = self.transformer_.inverse_transform(
                generated[:, : self.sampler_.data_dim_]
            )
            self.last_constraint_audit_ = empty_constraint_audit()
        if not return_target:
            return (
                (
                    table,
                    json.loads(json.dumps(self.last_constraint_audit_)),
                )
                if return_audit
                else table
            )
        target = self.sampler_.decode_target(generated)
        if target is not None:
            target.index = table.index
        if return_audit:
            return (
                table,
                target,
                json.loads(json.dumps(self.last_constraint_audit_)),
            )
        return table, target

    generate = sample

    def sample_conditioned(
        self,
        rows: int,
        conditions: Mapping[Any, Any],
        *,
        return_target: bool = False,
        use_ema: bool = True,
        weight_source: Optional[str] = None,
        weight_seed: Optional[int] = None,
        return_audit: bool = False,
    ):
        return self.sample(
            rows,
            conditions=conditions,
            return_target=return_target,
            use_ema=use_ema,
            weight_source=weight_source,
            weight_seed=weight_seed,
            return_audit=return_audit,
        )

    def _apply_explicit_table_conditions(
        self,
        generated: np.ndarray,
        conditions: Mapping[Any, Any],
        rows: int,
    ) -> None:
        table_conditions = {
            name: value
            for name, value in conditions.items()
            if any(
                self._value_equal(name, column)
                for column in self.transformer_.schema_.names
            )
        }
        if not table_conditions:
            return
        defaults = getattr(
            self,
            "_parameter_condition_defaults_",
            self._condition_defaults_,
        )
        values: Dict[Any, List[Any]] = {
            name: [defaults[name]] * rows
            for name in self.transformer_.schema_.names
        }
        for requested_name, raw in table_conditions.items():
            actual_name = next(
                name
                for name in self.transformer_.schema_.names
                if self._value_equal(requested_name, name)
            )
            values[actual_name] = self._expand_values(raw, rows)
        probe = pd.DataFrame(values, columns=list(self.transformer_.schema_.names))
        encoded = np.asarray(
            self.transformer_.transform(probe), dtype=np.float64
        )
        layouts = {
            layout.column: layout
            for layout in self.transformer_.column_layout_
        }
        for requested_name in table_conditions:
            actual_name = next(
                name
                for name in self.transformer_.schema_.names
                if self._value_equal(requested_name, name)
            )
            layout = layouts[actual_name]
            generated[:, layout.slice] = encoded[:, layout.slice]

    def _validate_parameter_conditions(
        self,
        conditions: Mapping[Any, Any],
        rows: int,
    ) -> None:
        """Reject coordinates the fitted inner transform cannot preserve."""

        table_conditions = {
            name: value
            for name, value in conditions.items()
            if any(
                self._value_equal(name, column)
                for column in self.transformer_.schema_.names
            )
        }
        if not table_conditions:
            return
        defaults = self._parameter_condition_defaults_
        values: Dict[Any, List[Any]] = {
            name: [defaults[name]] * rows
            for name in self.transformer_.schema_.names
        }
        actual_names: Dict[Any, Any] = {}
        for requested_name, raw in table_conditions.items():
            actual_name = next(
                name
                for name in self.transformer_.schema_.names
                if self._value_equal(requested_name, name)
            )
            actual_names[requested_name] = actual_name
            values[actual_name] = self._expand_values(raw, rows)
        probe = pd.DataFrame(
            values, columns=list(self.transformer_.schema_.names)
        )
        encoded = self.transformer_.transform(probe)
        restored = self.transformer_.inverse_transform(encoded)
        unsafe: List[Any] = []
        for requested_name, actual_name in actual_names.items():
            expected = probe[actual_name]
            actual = restored[actual_name]
            equal = expected.eq(actual) | (expected.isna() & actual.isna())
            try:
                left = pd.to_numeric(
                    expected, errors="raise"
                ).to_numpy(dtype=np.float64)
                right = pd.to_numeric(
                    actual, errors="raise"
                ).to_numpy(dtype=np.float64)
                finite = np.isfinite(left) & np.isfinite(right)
                numeric_equal = equal.to_numpy(dtype=bool, copy=True)
                numeric_equal[finite] = np.isclose(
                    left[finite],
                    right[finite],
                    rtol=1e-7,
                    atol=1e-8,
                )
                preserved = bool(np.all(numeric_equal))
            except (TypeError, ValueError):
                preserved = bool(equal.all())
            if not preserved:
                unsafe.append(requested_name)
        if unsafe:
            raise ValueError(
                "explicit conditions %r cannot be transformed safely by the "
                "fitted constraint coordinates (they are outside learned "
                "parameter support); use supported values or refit with "
                "representative constrained rows" % unsafe
            )

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

    def _save_weight_collection(self, path: Path) -> None:
        total = int(sum(self._generator_weight_sizes_))
        swa_flat = (
            np.empty(0, dtype=np.float64)
            if self._swa_weights_ is None
            else self._flatten_weights(self._swa_weights_)
        )
        swag_mean = (
            np.empty(0, dtype=np.float64)
            if self._swag_mean_ is None
            else self._swag_mean_
        )
        swag_square_mean = (
            np.empty(0, dtype=np.float64)
            if self._swag_square_mean_ is None
            else self._swag_square_mean_
        )
        swag_snapshots = (
            np.empty((0, total), dtype=np.float64)
            if not self._swag_snapshots_
            else np.stack(self._swag_snapshots_)
        )
        np.savez_compressed(
            path,
            swa_collections=np.asarray(
                [self.swa_collections_], dtype=np.int64
            ),
            swag_collections=np.asarray(
                [self.swag_collections_], dtype=np.int64
            ),
            swa_flat=swa_flat,
            swag_mean=swag_mean,
            swag_square_mean=swag_square_mean,
            swag_snapshots=swag_snapshots,
        )

    def _load_weight_collection(self, path: Path) -> None:
        if not path.is_file():
            return
        total = int(sum(self._generator_weight_sizes_))
        with np.load(path, allow_pickle=False) as state:
            self.swa_collections_ = int(state["swa_collections"][0])
            self.swag_collections_ = int(state["swag_collections"][0])
            swa_flat = np.asarray(state["swa_flat"], dtype=np.float64)
            swag_mean = np.asarray(state["swag_mean"], dtype=np.float64)
            swag_square_mean = np.asarray(
                state["swag_square_mean"], dtype=np.float64
            )
            snapshots = np.asarray(
                state["swag_snapshots"], dtype=np.float64
            )
        if swa_flat.size:
            if swa_flat.size != total:
                raise ValueError("persisted SWA state has the wrong size")
            self._swa_weights_ = self._unflatten_generator_weights(swa_flat)
            self._swa_weights_ = [
                np.asarray(value, dtype=np.float64)
                for value in self._swa_weights_
            ]
        if swag_mean.size:
            if (
                swag_mean.size != total
                or swag_square_mean.size != total
            ):
                raise ValueError("persisted SWAG state has the wrong size")
            self._swag_mean_ = swag_mean
            self._swag_square_mean_ = swag_square_mean
        if snapshots.size:
            if snapshots.ndim != 2 or snapshots.shape[1] != total:
                raise ValueError(
                    "persisted SWAG low-rank state has the wrong shape"
                )
            self._swag_snapshots_ = [
                row.copy() for row in snapshots
            ]

    def save(self, directory: Union[str, Path]) -> str:
        """Persist networks, preprocessing, sampler, config, and history."""

        self._check_fitted()
        destination = Path(directory)
        if destination.exists() and not destination.is_dir():
            raise ValueError("save path must be a directory")
        destination.mkdir(parents=True, exist_ok=True)
        self.generator_.save_weights(
            str(destination / "generator.weights.h5")
        )
        self.critic_.save_weights(
            str(destination / "critic.weights.h5")
        )
        self.ema_generator_.save_weights(
            str(destination / "ema_generator.weights.h5")
        )
        if self.warmup_encoder_ is not None:
            self.warmup_encoder_.save_weights(
                str(destination / "warmup_encoder.weights.h5")
            )
        self._save_weight_collection(
            destination / "generator_averages.npz"
        )
        self.transformer_.save(destination / "transformer.json")
        self.sampler_.save(destination / "sampler.json")
        if self.constraint_transformer_ is not None:
            self.constraint_transformer_.save(
                destination / "constraint_preprocessor.json"
            )
        metadata = {
            "format": "ganify-conditional-engine",
            "version": self.FORMAT_VERSION,
            "config": self._config_dict(),
            "history": self.history_,
            "epochs_ran": self.epochs_ran_,
            "warmup_epochs_ran": self.warmup_epochs_ran_,
            "fit_batch_size": self._fit_batch_size_,
            "effective_critic_batch_size": (
                self._effective_critic_batch_size_
            ),
            "fit_n_critic": self._fit_n_critic_,
            "fit_compile": self._fit_compile_,
            "training_rows": self._training_rows_,
            "condition_defaults": [
                {
                    "name": encode_json_value(name),
                    "value": encode_json_value(
                        None if _is_missing(value) else value
                    ),
                }
                for name, value in self._condition_defaults_.items()
            ],
            "parameter_condition_defaults": [
                {
                    "name": encode_json_value(name),
                    "value": encode_json_value(
                        None if _is_missing(value) else value
                    ),
                }
                for name, value in self._parameter_condition_defaults_.items()
            ],
            "critic_parameter_count": self.critic_parameter_count_,
            "critic_parameter_target": self.critic_parameter_target_,
            "critic_parameter_gap": self.critic_parameter_gap_,
            "swa_collections": self.swa_collections_,
            "swag_collections": self.swag_collections_,
            "rng_state": _json_safe(self._rng.bit_generator.state),
            "last_constraint_audit": _json_safe(
                self.last_constraint_audit_
            ),
            "dp_fixed_preprocessor": bool(
                getattr(self, "_dp_fixed_preprocessor_", False)
            ),
            "privacy_accountant": (
                None
                if self.privacy_accountant_ is None
                else self.privacy_accountant_.state_dict()
            ),
            "privacy_report": _json_safe(self.privacy_report_),
        }
        (destination / "metadata.json").write_text(
            json.dumps(
                metadata, indent=2, sort_keys=True, allow_nan=False
            )
            + "\n",
            encoding="utf-8",
        )
        return str(destination)

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "ConditionalGANEngine":
        """Restore an engine written by :meth:`save`."""

        source = Path(directory)
        metadata_path = source / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                "no conditional engine metadata at %s" % metadata_path
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("format") != "ganify-conditional-engine":
            raise ValueError("not a GANify conditional engine")
        if int(metadata.get("version", -1)) != cls.FORMAT_VERSION:
            raise ValueError("unsupported conditional engine format")
        instance = cls(**metadata["config"])
        instance.transformer_ = TableTransformer.load(
            source / "transformer.json"
        )
        constraint_state = source / "constraint_preprocessor.json"
        if len(instance.constraints):
            if not constraint_state.is_file():
                raise ValueError(
                    "conditional engine config contains constraints but the "
                    "reversible constraint preprocessor state is missing"
                )
            instance.constraint_transformer_ = (
                StructuralConstraintTransformer.load(constraint_state)
            )
            if (
                instance.constraint_transformer_.constraint_set.to_dict()
                != instance.constraints.to_dict()
            ):
                raise ValueError(
                    "persisted constraint config and preprocessor disagree"
                )
        else:
            instance.constraint_transformer_ = None
        instance.constraint_preprocessor_ = (
            None
            if instance.constraint_transformer_ is None
            else ConstrainedTableTransformer(
                instance.constraint_transformer_,
                instance.transformer_,
            )
        )
        instance.preprocessor_ = (
            instance.transformer_
            if instance.constraint_preprocessor_ is None
            else instance.constraint_preprocessor_
        )
        instance.sampler_ = ConditionSampler.load(source / "sampler.json")
        instance.sampler_.transformer_ = instance.transformer_
        instance._sync_metadata()
        instance.head_specs_ = tuple(instance.transformer_.head_metadata)
        target_head = instance.sampler_.target_head_spec
        if target_head is not None:
            instance.head_specs_ += (target_head,)
        instance._fit_batch_size_ = int(metadata["fit_batch_size"])
        instance._effective_critic_batch_size_ = int(
            metadata.get(
                "effective_critic_batch_size",
                instance._fit_batch_size_ // instance.pac,
            )
        )
        instance._fit_n_critic_ = int(metadata["fit_n_critic"])
        instance._fit_compile_ = bool(metadata.get("fit_compile", False))
        instance._training_rows_ = int(metadata["training_rows"])
        instance._condition_defaults_ = {
            decode_json_value(item["name"]): decode_json_value(item["value"])
            for item in metadata.get("condition_defaults", [])
        }
        instance._parameter_condition_defaults_ = {
            decode_json_value(item["name"]): decode_json_value(item["value"])
            for item in metadata.get(
                "parameter_condition_defaults",
                metadata.get("condition_defaults", []),
            )
        }
        instance._dp_fixed_preprocessor_ = bool(
            metadata.get("dp_fixed_preprocessor", False)
        )
        instance._build_models(compile_steps=instance._fit_compile_)
        instance.generator_.load_weights(
            str(source / "generator.weights.h5")
        )
        instance.critic_.load_weights(
            str(source / "critic.weights.h5")
        )
        instance.ema_generator_.load_weights(
            str(source / "ema_generator.weights.h5")
        )
        warmup_weights = source / "warmup_encoder.weights.h5"
        if warmup_weights.is_file():
            if instance.warmup_encoder_ is None:
                raise ValueError(
                    "persisted warmup encoder has no configured model"
                )
            instance.warmup_encoder_.load_weights(str(warmup_weights))
        instance._load_weight_collection(
            source / "generator_averages.npz"
        )
        instance.history_ = {
            name: [float(value) for value in values]
            for name, values in metadata.get("history", {}).items()
        }
        for name in (
            "critic",
            "generator",
            "gradient_penalty",
            "conditional",
            "interaction",
            "feature_matching",
            "warmup",
        ):
            instance.history_.setdefault(name, [])
        instance.epochs_ran_ = int(metadata.get("epochs_ran", 0))
        instance.warmup_epochs_ran_ = int(
            metadata.get("warmup_epochs_ran", 0)
        )
        rng_state = metadata.get("rng_state")
        if rng_state is not None:
            instance._rng.bit_generator.state = dict(rng_state)
        instance.last_constraint_audit_ = dict(
            metadata.get(
                "last_constraint_audit", empty_constraint_audit()
            )
        )
        accountant_state = metadata.get("privacy_accountant")
        if accountant_state is not None:
            instance.privacy_accountant_ = (
                PrivacyAccountant.from_state_dict(accountant_state)
            )
        instance.privacy_report_ = metadata.get("privacy_report")
        instance.fitted_ = True
        return instance

    def _config_dict(self) -> Dict[str, Any]:
        return {
            "random_state": self.random_state,
            "noise_dim": self.noise_dim,
            "generator_dims": list(self.generator_dims),
            "critic_dims": list(self.critic_dims),
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "n_critic": self.n_critic,
            "gradient_penalty": self.gradient_penalty,
            "conditional_weight": self.conditional_weight,
            "generator_learning_rate": self.generator_learning_rate,
            "critic_learning_rate": self.critic_learning_rate,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "ema_decay": self.ema_decay,
            "continuous_transform": self.continuous_transform,
            "continuous_bins": self.continuous_bins,
            "handle_unknown": self.handle_unknown,
            "max_modes": self.max_modes,
            "schema": (
                None if self.schema is None else self.schema.to_dict()
            ),
            "schema_overrides": _encode_schema_overrides(
                self.schema_overrides
            ),
            "constraints": (
                None
                if not len(self.constraints)
                else self.constraints.to_dict()
            ),
            "spectral_normalization": self.spectral_normalization,
            "compile": self.compile,
            "critic_architecture": self.critic_architecture,
            "pac": self.pac,
            "attention_heads": self.attention_heads,
            "attention_dim": self.attention_dim,
            "fourier_features": self.fourier_features,
            "fourier_scale": self.fourier_scale,
            "match_critic_parameters": self.match_critic_parameters,
            "critic_parameter_budget": self.critic_parameter_budget,
            "warmup_epochs": self.warmup_epochs,
            "warmup_mask_probability": self.warmup_mask_probability,
            "warmup_learning_rate": self.warmup_learning_rate,
            "interaction_weight": self.interaction_weight,
            "interaction_temperature": self.interaction_temperature,
            "feature_matching_weight": self.feature_matching_weight,
            "swa": self.swa,
            "swa_start_epoch": self.swa_start_epoch,
            "swa_frequency": self.swa_frequency,
            "swag": self.swag,
            "swag_start_epoch": self.swag_start_epoch,
            "swag_frequency": self.swag_frequency,
            "swag_max_rank": self.swag_max_rank,
            "swag_scale": self.swag_scale,
            "dp_config": (
                None
                if self.dp_config is None
                else self.dp_config.to_dict()
            ),
        }

    @property
    def transformer(self) -> TableTransformer:
        self._check_fitted()
        return self.transformer_

    @property
    def preprocessor(self):
        """User-facing reversible preprocessor (constraint-aware when set)."""

        self._check_fitted()
        return self.preprocessor_

    @property
    def constraint_preprocessor(self):
        """Return the structural wrapper, or ``None`` when unconstrained."""

        self._check_fitted()
        return self.constraint_preprocessor_

    @property
    def sampler(self) -> ConditionSampler:
        self._check_fitted()
        return self.sampler_

    @property
    def generator(self) -> ConditionalGenerator:
        self._check_fitted()
        return self.generator_

    @property
    def critic(self) -> ConditionalCritic:
        self._check_fitted()
        return self.critic_

    @property
    def history(self) -> Mapping[str, List[float]]:
        return self.history_

    @property
    def privacy_report(self) -> Optional[Mapping[str, Any]]:
        """Return the explicit DP boundary report, or ``None`` when disabled."""

        return self.privacy_report_

    @property
    def effective_critic_batch_size(self) -> int:
        """Number of packed examples seen by each critic update."""

        if hasattr(self, "_effective_critic_batch_size_"):
            return self._effective_critic_batch_size_
        return self.batch_size // self.pac

    def _sync_metadata(self) -> None:
        self.schema_ = (
            self.transformer_.schema_
            if self.constraint_transformer_ is None
            else self.constraint_transformer_.original_schema_
        )
        self.target_name_ = self.sampler_.target_name_
        self.target_classes_ = self.sampler_.target_values_
        self.condition_dim_ = self.sampler_.condition_dim_
        self.output_dim_ = self.sampler_.output_dim_

    def _check_fitted(self) -> None:
        if not getattr(self, "fitted_", False):
            raise RuntimeError("ConditionalGANEngine is not fitted")


__all__ = ["ConditionalGANEngine", "DPConfig", "PrivacyAccountant"]
