"""Neural networks used by the standalone conditional GAN engine."""

from __future__ import annotations

import math
from numbers import Integral
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import tensorflow as tf

from ganify.preprocessing import HeadSpec
from ganify.schema import decode_json_value


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("%s must be a positive integer" % name)
    value = int(value)
    if value < 1:
        raise ValueError("%s must be a positive integer" % name)
    return value


def _hidden_dims(values: Sequence[int]) -> Tuple[int, ...]:
    result = tuple(_positive_int("hidden dimension", value) for value in values)
    if not result:
        raise ValueError("hidden_dims must contain at least one width")
    return result


def _nonnegative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("%s must be a non-negative integer" % name)
    value = int(value)
    if value < 0:
        raise ValueError("%s must be a non-negative integer" % name)
    return value


def _critic_architecture(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("architecture must be a string")
    canonical = value.strip().lower().replace("-", "_")
    aliases = {
        "mlp": "residual",
        "residual_mlp": "residual",
        "self_attention": "attention",
        "feature_attention": "attention",
        "feature_token": "attention",
        "feature_token_attention": "attention",
        "fourier_feature": "fourier",
        "fourier_features": "fourier",
        "fct": "fourier",
        "fct_style": "fourier",
    }
    canonical = aliases.get(canonical, canonical)
    if canonical not in {"residual", "attention", "fourier"}:
        raise ValueError(
            "architecture must be 'residual', 'attention', or 'fourier'"
        )
    return canonical


def _residual_parameter_count(
    input_width: int, hidden_dims: Sequence[int]
) -> int:
    dims = tuple(int(value) for value in hidden_dims)
    count = (input_width + 1) * dims[0]
    current = dims[0]
    for width in dims:
        count += (current + 1) * width
        count += (width + 1) * width
        if current != width:
            count += (current + 1) * width
        current = width
    return count + current + 1


def _attention_parameter_count(
    input_width: int, token_dim: int, heads: int, layers: int
) -> int:
    key_dim = max(1, int(math.ceil(token_dim / float(heads))))
    projected = heads * key_dim
    tokenizer = 2 * input_width * token_dim
    attention = (
        3 * (token_dim * projected + projected)
        + projected * token_dim
        + token_dim
    )
    feed_forward = (
        token_dim * (2 * token_dim)
        + 2 * token_dim
        + (2 * token_dim) * token_dim
        + token_dim
    )
    normalizations = 4 * token_dim
    return (
        tokenizer
        + layers * (attention + feed_forward + normalizations)
        + token_dim
        + 1
    )


def _fourier_parameter_count(
    input_width: int,
    hidden_dims: Sequence[int],
    fourier_features: int,
) -> int:
    mapped_width = input_width + 2 * fourier_features
    return _residual_parameter_count(mapped_width, hidden_dims)


def _matched_width(
    architecture: str,
    *,
    input_width: int,
    target: int,
    layers: int,
    attention_heads: int,
    fourier_features: int,
) -> int:
    """Return the closest uniform width for a trainable parameter budget."""

    target = _positive_int("parameter_budget", target)
    # The upper bound is deliberately generous for small tabular models while
    # remaining finite for malformed multi-billion-parameter budgets.
    upper = min(4096, max(16, int(math.sqrt(target) * 4.0) + 8))
    if architecture == "attention":
        candidates = range(attention_heads, upper + 1, attention_heads)
    else:
        candidates = range(1, upper + 1)

    best_width = 1
    best_gap: Optional[int] = None
    for width in candidates:
        dims = (width,) * layers
        if architecture == "attention":
            count = _attention_parameter_count(
                input_width, width, attention_heads, layers
            )
        elif architecture == "fourier":
            count = _fourier_parameter_count(
                input_width, dims, fourier_features
            )
        else:
            count = _residual_parameter_count(input_width, dims)
        gap = abs(count - target)
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_width = width
    return best_width


def _coerce_head(value: Any) -> HeadSpec:
    if isinstance(value, HeadSpec):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("head_specs must contain HeadSpec objects or mappings")
    column = value.get("column")
    if isinstance(column, Mapping) and "type" in column:
        column = decode_json_value(column)
    return HeadSpec(
        column=column,
        name=str(value["name"]),
        kind=str(value["kind"]),
        start=int(value["start"]),
        stop=int(value["stop"]),
        activation=str(value["activation"]),
    )


def _head_specs(values: Sequence[Any]) -> Tuple[HeadSpec, ...]:
    heads = tuple(_coerce_head(value) for value in values)
    if not heads:
        raise ValueError("head_specs must contain at least one output head")
    offset = 0
    for head in heads:
        if head.start != offset or head.stop <= head.start:
            raise ValueError(
                "head_specs must be contiguous, ordered, and have positive width"
            )
        if head.activation not in {"tanh", "sigmoid", "softmax", "linear"}:
            raise ValueError(
                "unsupported head activation %r" % head.activation
            )
        offset = head.stop
    return heads


def _initializer(seed: Optional[int], offset: int):
    return tf.keras.initializers.GlorotUniform(
        seed=None if seed is None else int(seed) + int(offset)
    )


class SpectralDense(tf.keras.layers.Layer):
    """Dense layer with one-step power-iteration spectral normalization."""

    def __init__(
        self,
        units: int,
        *,
        use_bias: bool = True,
        power_iterations: int = 1,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.units = _positive_int("units", units)
        self.use_bias = bool(use_bias)
        self.power_iterations = _positive_int(
            "power_iterations", power_iterations
        )
        self.seed = None if seed is None else int(seed)

    def build(self, input_shape: tf.TensorShape) -> None:
        input_dim = int(input_shape[-1])
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_dim, self.units),
            initializer=_initializer(self.seed, 0),
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.units,),
                initializer="zeros",
                trainable=True,
            )
        else:
            self.bias = None
        self.power_vector = self.add_weight(
            name="power_vector",
            shape=(1, self.units),
            initializer=tf.keras.initializers.RandomNormal(
                stddev=0.02, seed=self.seed
            ),
            trainable=False,
        )
        super().build(input_shape)

    def call(
        self, inputs: tf.Tensor, training: Optional[bool] = None
    ) -> tf.Tensor:
        kernel = tf.cast(self.kernel, self.compute_dtype)
        vector = tf.cast(self.power_vector, self.compute_dtype)
        for _ in range(self.power_iterations):
            left = tf.math.l2_normalize(
                tf.matmul(vector, kernel, transpose_b=True),
                axis=1,
                epsilon=1e-12,
            )
            vector = tf.math.l2_normalize(
                tf.matmul(left, kernel), axis=1, epsilon=1e-12
            )
        sigma = tf.matmul(
            tf.matmul(left, kernel), vector, transpose_b=True
        )
        normalized = kernel / tf.maximum(tf.abs(sigma), 1e-12)
        if training is not False:
            self.power_vector.assign(tf.cast(vector, self.power_vector.dtype))
        output = tf.matmul(inputs, normalized)
        if self.bias is not None:
            output = tf.nn.bias_add(output, tf.cast(self.bias, output.dtype))
        return output

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "use_bias": self.use_bias,
                "power_iterations": self.power_iterations,
                "seed": self.seed,
            }
        )
        return config


class _ResidualBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        units: int,
        *,
        seed: Optional[int],
        spectral_normalization: bool,
        normalize: bool,
        name: str,
    ) -> None:
        super().__init__(name=name)
        dense_type = SpectralDense if spectral_normalization else tf.keras.layers.Dense
        dense_options = (
            {"seed": seed}
            if spectral_normalization
            else {"kernel_initializer": _initializer(seed, 0)}
        )
        second_options = (
            {"seed": None if seed is None else seed + 1}
            if spectral_normalization
            else {"kernel_initializer": _initializer(seed, 1)}
        )
        self.units = units
        self.first = dense_type(units, name="dense_1", **dense_options)
        self.second = dense_type(units, name="dense_2", **second_options)
        self.normalization = (
            tf.keras.layers.LayerNormalization(name="layer_norm")
            if normalize
            else None
        )
        self.spectral_normalization = spectral_normalization
        self.seed = seed
        self.projection: Optional[tf.keras.layers.Layer] = None

    def build(self, input_shape: tf.TensorShape) -> None:
        if int(input_shape[-1]) != self.units:
            if self.spectral_normalization:
                self.projection = SpectralDense(
                    self.units,
                    seed=None if self.seed is None else self.seed + 2,
                    name="projection",
                )
            else:
                self.projection = tf.keras.layers.Dense(
                    self.units,
                    kernel_initializer=_initializer(self.seed, 2),
                    name="projection",
                )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        if self.projection is None:
            residual = inputs
        elif self.spectral_normalization:
            residual = self.projection(inputs, training=training)
        else:
            residual = self.projection(inputs)
        value = (
            self.first(inputs, training=training)
            if self.spectral_normalization
            else self.first(inputs)
        )
        value = tf.nn.leaky_relu(value, alpha=0.2)
        value = (
            self.second(value, training=training)
            if self.spectral_normalization
            else self.second(value)
        )
        if self.normalization is not None:
            value = self.normalization(value, training=training)
        return tf.nn.leaky_relu(residual + value, alpha=0.2)


class _FeatureTokenizer(tf.keras.layers.Layer):
    """Turn each scalar channel into a feature-specific learned token."""

    def __init__(
        self,
        input_width: int,
        token_dim: int,
        *,
        seed: Optional[int],
        name: str = "feature_tokenizer",
    ) -> None:
        super().__init__(name=name)
        self.input_width = _positive_int("input_width", input_width)
        self.token_dim = _positive_int("token_dim", token_dim)
        self.seed = None if seed is None else int(seed)

    def build(self, input_shape: tf.TensorShape) -> None:
        self.scale = self.add_weight(
            name="scale",
            shape=(self.input_width, self.token_dim),
            initializer=_initializer(self.seed, 0),
            trainable=True,
        )
        self.offset = self.add_weight(
            name="offset",
            shape=(self.input_width, self.token_dim),
            initializer=_initializer(self.seed, 1),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return (
            tf.expand_dims(inputs, axis=-1)
            * tf.cast(self.scale, inputs.dtype)
            + tf.cast(self.offset, inputs.dtype)
        )


class _SelfAttention(tf.keras.layers.Layer):
    """Small TensorFlow-2.2-compatible multi-head self-attention layer."""

    def __init__(
        self,
        token_dim: int,
        heads: int,
        *,
        seed: Optional[int],
        name: str = "self_attention",
    ) -> None:
        super().__init__(name=name)
        self.token_dim = _positive_int("token_dim", token_dim)
        self.heads = _positive_int("attention_heads", heads)
        self.key_dim = max(
            1, int(math.ceil(self.token_dim / float(self.heads)))
        )
        self.projected_dim = self.heads * self.key_dim
        self.query = tf.keras.layers.Dense(
            self.projected_dim,
            kernel_initializer=_initializer(seed, 0),
            name="query",
        )
        self.key = tf.keras.layers.Dense(
            self.projected_dim,
            kernel_initializer=_initializer(seed, 1),
            name="key",
        )
        self.value = tf.keras.layers.Dense(
            self.projected_dim,
            kernel_initializer=_initializer(seed, 2),
            name="value",
        )
        self.output_projection = tf.keras.layers.Dense(
            self.token_dim,
            kernel_initializer=_initializer(seed, 3),
            name="output",
        )

    def _split_heads(self, values: tf.Tensor) -> tf.Tensor:
        shape = tf.shape(values)
        values = tf.reshape(
            values,
            [shape[0], shape[1], self.heads, self.key_dim],
        )
        return tf.transpose(values, [0, 2, 1, 3])

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        query = self._split_heads(self.query(inputs))
        key = self._split_heads(self.key(inputs))
        value = self._split_heads(self.value(inputs))
        logits = tf.matmul(query, key, transpose_b=True)
        logits /= tf.sqrt(tf.cast(self.key_dim, logits.dtype))
        weights = tf.nn.softmax(logits, axis=-1)
        attended = tf.matmul(weights, value)
        attended = tf.transpose(attended, [0, 2, 1, 3])
        shape = tf.shape(attended)
        attended = tf.reshape(
            attended,
            [shape[0], shape[1], self.projected_dim],
        )
        return self.output_projection(attended)


class _AttentionBlock(tf.keras.layers.Layer):
    """Pre-normalized feature-token self-attention block."""

    def __init__(
        self,
        token_dim: int,
        heads: int,
        *,
        seed: Optional[int],
        name: str,
    ) -> None:
        super().__init__(name=name)
        self.token_dim = _positive_int("token_dim", token_dim)
        self.heads = _positive_int("attention_heads", heads)
        self.attention_norm = tf.keras.layers.LayerNormalization(
            name="attention_norm"
        )
        self.attention = _SelfAttention(
            self.token_dim,
            self.heads,
            seed=seed,
        )
        self.feed_forward_norm = tf.keras.layers.LayerNormalization(
            name="feed_forward_norm"
        )
        self.feed_forward_1 = tf.keras.layers.Dense(
            2 * self.token_dim,
            activation=lambda value: tf.nn.leaky_relu(value, alpha=0.2),
            kernel_initializer=_initializer(seed, 1),
            name="feed_forward_1",
        )
        self.feed_forward_2 = tf.keras.layers.Dense(
            self.token_dim,
            kernel_initializer=_initializer(seed, 2),
            name="feed_forward_2",
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        normalized = self.attention_norm(inputs, training=training)
        value = inputs + self.attention(normalized)
        normalized = self.feed_forward_norm(value, training=training)
        return value + self.feed_forward_2(
            self.feed_forward_1(normalized)
        )


class _FourierFeatureMap(tf.keras.layers.Layer):
    """Deterministic random Fourier map retaining the original channels."""

    def __init__(
        self,
        input_width: int,
        features: int,
        *,
        scale: float,
        seed: Optional[int],
        name: str = "fourier_features",
    ) -> None:
        super().__init__(name=name)
        self.input_width = _positive_int("input_width", input_width)
        self.features = _positive_int("fourier_features", features)
        self.scale = float(scale)
        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("fourier_scale must be finite and positive")
        self.seed = None if seed is None else int(seed)

    def build(self, input_shape: tf.TensorShape) -> None:
        self.frequencies = self.add_weight(
            name="frequencies",
            shape=(self.input_width, self.features),
            initializer=tf.keras.initializers.RandomNormal(
                stddev=self.scale, seed=self.seed
            ),
            trainable=False,
        )
        self.phase = self.add_weight(
            name="phase",
            shape=(self.features,),
            initializer=tf.keras.initializers.RandomUniform(
                minval=-math.pi,
                maxval=math.pi,
                seed=None if self.seed is None else self.seed + 1,
            ),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        projection = (
            2.0
            * math.pi
            * tf.matmul(inputs, tf.cast(self.frequencies, inputs.dtype))
            + tf.cast(self.phase, inputs.dtype)
        )
        return tf.concat(
            [inputs, tf.math.sin(projection), tf.math.cos(projection)], axis=1
        )


class ConditionalGenerator(tf.keras.Model):
    """Residual MLP with one activation-correct layer per transformer head."""

    def __init__(
        self,
        head_specs: Sequence[Any],
        *,
        noise_dim: int = 128,
        condition_dim: int = 0,
        hidden_dims: Sequence[int] = (256, 256),
        random_state: Optional[int] = None,
        name: str = "conditional_generator",
    ) -> None:
        super().__init__(name=name)
        self.head_specs = _head_specs(head_specs)
        self.noise_dim = _positive_int("noise_dim", noise_dim)
        if (
            isinstance(condition_dim, bool)
            or not isinstance(condition_dim, Integral)
            or int(condition_dim) < 0
        ):
            raise ValueError("condition_dim must be a non-negative integer")
        self.condition_dim = int(condition_dim)
        self.hidden_dims = _hidden_dims(hidden_dims)
        self.random_state = (
            None if random_state is None else int(random_state)
        )
        self.output_dim = self.head_specs[-1].stop

        self.stem = tf.keras.layers.Dense(
            self.hidden_dims[0],
            kernel_initializer=_initializer(self.random_state, 10),
            name="stem",
        )
        self.blocks = [
            _ResidualBlock(
                width,
                seed=(
                    None
                    if self.random_state is None
                    else self.random_state + 100 + index * 10
                ),
                spectral_normalization=False,
                normalize=True,
                name="residual_%d" % index,
            )
            for index, width in enumerate(self.hidden_dims)
        ]
        self.output_layers = [
            tf.keras.layers.Dense(
                head.width,
                kernel_initializer=_initializer(
                    self.random_state, 1000 + index
                ),
                name="head_%d_%s" % (index, head.activation),
            )
            for index, head in enumerate(self.head_specs)
        ]

    def _inputs(
        self, inputs: Any, condition: Optional[Any]
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        if condition is None and isinstance(inputs, (tuple, list)):
            if len(inputs) != 2:
                raise ValueError("generator inputs must be (noise, condition)")
            inputs, condition = inputs
        noise = tf.convert_to_tensor(inputs, dtype=self.compute_dtype)
        if noise.shape.rank != 2:
            raise ValueError("generator noise must be a rank-2 tensor")
        if condition is None and self.condition_dim:
            static_width = noise.shape[-1]
            if static_width == self.noise_dim + self.condition_dim:
                noise, condition = tf.split(
                    noise, [self.noise_dim, self.condition_dim], axis=1
                )
        if condition is None:
            condition_tensor = tf.zeros(
                [tf.shape(noise)[0], self.condition_dim], dtype=noise.dtype
            )
        else:
            condition_tensor = tf.convert_to_tensor(
                condition, dtype=noise.dtype
            )
        tf.debugging.assert_equal(tf.shape(noise)[1], self.noise_dim)
        tf.debugging.assert_equal(
            tf.shape(condition_tensor)[1], self.condition_dim
        )
        tf.debugging.assert_equal(
            tf.shape(noise)[0], tf.shape(condition_tensor)[0]
        )
        return noise, condition_tensor

    def call(
        self,
        inputs: Any,
        condition: Optional[Any] = None,
        training: bool = False,
    ) -> tf.Tensor:
        noise, condition_tensor = self._inputs(inputs, condition)
        value = tf.concat([noise, condition_tensor], axis=1)
        value = self.stem(value)
        value = tf.nn.leaky_relu(value, alpha=0.2)
        for block in self.blocks:
            value = block(value, training=training)
        outputs = []
        for head, layer in zip(self.head_specs, self.output_layers):
            output = layer(value)
            if head.activation == "tanh":
                output = tf.math.tanh(output)
            elif head.activation == "sigmoid":
                output = tf.math.sigmoid(output)
            elif head.activation == "softmax":
                output = tf.nn.softmax(output, axis=1)
            outputs.append(output)
        return tf.concat(outputs, axis=1)

    def split_heads(self, values: Any) -> Dict[Tuple[Any, str], tf.Tensor]:
        tensor = tf.convert_to_tensor(values)
        return {
            (head.column, head.name): tensor[:, head.slice]
            for head in self.head_specs
        }

    def get_config(self) -> Dict[str, Any]:
        return {
            "head_specs": [head.to_dict() for head in self.head_specs],
            "noise_dim": self.noise_dim,
            "condition_dim": self.condition_dim,
            "hidden_dims": list(self.hidden_dims),
            "random_state": self.random_state,
            "name": self.name,
        }


class ConditionalDenoisingEncoder(tf.keras.Model):
    """Encoder used only by masked-denoising generator warmup."""

    def __init__(
        self,
        feature_dim: int,
        noise_dim: int,
        *,
        condition_dim: int = 0,
        hidden_dims: Sequence[int] = (256, 256),
        random_state: Optional[int] = None,
        name: str = "conditional_denoising_encoder",
    ) -> None:
        super().__init__(name=name)
        self.feature_dim = _positive_int("feature_dim", feature_dim)
        self.noise_dim = _positive_int("noise_dim", noise_dim)
        self.condition_dim = _nonnegative_int(
            "condition_dim", condition_dim
        )
        self.hidden_dims = _hidden_dims(hidden_dims)
        self.random_state = (
            None if random_state is None else int(random_state)
        )
        self.hidden_layers = [
            tf.keras.layers.Dense(
                width,
                activation=lambda value: tf.nn.leaky_relu(value, alpha=0.2),
                kernel_initializer=_initializer(
                    self.random_state, 4000 + index
                ),
                name="hidden_%d" % index,
            )
            for index, width in enumerate(self.hidden_dims)
        ]
        self.latent = tf.keras.layers.Dense(
            self.noise_dim,
            kernel_initializer=_initializer(self.random_state, 5000),
            name="latent",
        )

    def call(
        self,
        inputs: Any,
        corruption_mask: Optional[Any] = None,
        condition: Optional[Any] = None,
        training: bool = False,
    ) -> tf.Tensor:
        if corruption_mask is None and isinstance(inputs, (tuple, list)):
            if len(inputs) != 3:
                raise ValueError(
                    "encoder inputs must be (features, mask, condition)"
                )
            inputs, corruption_mask, condition = inputs
        features = tf.convert_to_tensor(inputs, dtype=self.compute_dtype)
        mask_tensor = tf.convert_to_tensor(
            corruption_mask, dtype=features.dtype
        )
        if features.shape.rank != 2 or mask_tensor.shape.rank != 2:
            raise ValueError("encoder features and mask must be rank-2")
        if condition is None:
            condition_tensor = tf.zeros(
                [tf.shape(features)[0], self.condition_dim],
                dtype=features.dtype,
            )
        else:
            condition_tensor = tf.convert_to_tensor(
                condition, dtype=features.dtype
            )
        tf.debugging.assert_equal(tf.shape(features)[1], self.feature_dim)
        tf.debugging.assert_equal(tf.shape(mask_tensor), tf.shape(features))
        tf.debugging.assert_equal(
            tf.shape(condition_tensor)[1], self.condition_dim
        )
        tf.debugging.assert_equal(
            tf.shape(condition_tensor)[0], tf.shape(features)[0]
        )
        value = tf.concat(
            [features, mask_tensor, condition_tensor], axis=1
        )
        for layer in self.hidden_layers:
            value = layer(value)
        return self.latent(value)

    def get_config(self) -> Dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "noise_dim": self.noise_dim,
            "condition_dim": self.condition_dim,
            "hidden_dims": list(self.hidden_dims),
            "random_state": self.random_state,
            "name": self.name,
        }


class ConditionalCritic(tf.keras.Model):
    """Configurable conditional critic producing an unrestricted linear score.

    ``pac`` counts raw rows per critic example. Inputs remain ordinary
    ``(features, condition)`` row tensors; packing is internal and the score
    batch therefore has size ``raw_batch // pac``.
    """

    def __init__(
        self,
        feature_dim: int,
        *,
        condition_dim: int = 0,
        hidden_dims: Sequence[int] = (256, 256),
        spectral_normalization: bool = False,
        random_state: Optional[int] = None,
        architecture: str = "residual",
        pac: int = 1,
        attention_heads: int = 4,
        attention_dim: Optional[int] = None,
        fourier_features: int = 32,
        fourier_scale: float = 1.0,
        parameter_budget: Optional[int] = None,
        name: str = "conditional_critic",
    ) -> None:
        super().__init__(name=name)
        self.feature_dim = _positive_int("feature_dim", feature_dim)
        self.condition_dim = _nonnegative_int(
            "condition_dim", condition_dim
        )
        self.hidden_dims = _hidden_dims(hidden_dims)
        self.spectral_normalization = bool(spectral_normalization)
        self.random_state = (
            None if random_state is None else int(random_state)
        )
        self.architecture = _critic_architecture(architecture)
        self.pac = _positive_int("pac", pac)
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
        self.fourier_scale = float(fourier_scale)
        if not math.isfinite(self.fourier_scale) or self.fourier_scale <= 0.0:
            raise ValueError("fourier_scale must be finite and positive")
        self.parameter_budget = (
            None
            if parameter_budget is None
            else _positive_int("parameter_budget", parameter_budget)
        )
        self.packed_feature_dim = self.pac * self.feature_dim
        self.packed_condition_dim = self.pac * self.condition_dim
        self.input_width = (
            self.packed_feature_dim + self.packed_condition_dim
        )

        layers = len(self.hidden_dims)
        if self.parameter_budget is not None:
            matched = _matched_width(
                self.architecture,
                input_width=self.input_width,
                target=self.parameter_budget,
                layers=layers,
                attention_heads=self.attention_heads,
                fourier_features=self.fourier_features,
            )
            self.resolved_hidden_dims = (matched,) * layers
        else:
            self.resolved_hidden_dims = self.hidden_dims

        self.tokenizer: Optional[tf.keras.layers.Layer] = None
        self.attention_blocks: Sequence[tf.keras.layers.Layer] = ()
        self.fourier_map: Optional[tf.keras.layers.Layer] = None
        self.blocks: Sequence[tf.keras.layers.Layer] = ()

        if self.architecture == "attention":
            token_dim = (
                self.attention_dim
                if self.attention_dim is not None
                else self.resolved_hidden_dims[0]
            )
            # Parameter matching controls the token width. An explicit
            # attention_dim remains authoritative only without a budget.
            if self.parameter_budget is not None:
                token_dim = self.resolved_hidden_dims[0]
            self.resolved_attention_dim = token_dim
            self.tokenizer = _FeatureTokenizer(
                self.input_width,
                token_dim,
                seed=self.random_state,
            )
            self.attention_blocks = [
                _AttentionBlock(
                    token_dim,
                    self.attention_heads,
                    seed=(
                        None
                        if self.random_state is None
                        else self.random_state + 500 + index * 10
                    ),
                    name="attention_%d" % index,
                )
                for index in range(layers)
            ]
            self.stem = None
        else:
            self.resolved_attention_dim = None
            if self.architecture == "fourier":
                self.fourier_map = _FourierFeatureMap(
                    self.input_width,
                    self.fourier_features,
                    scale=self.fourier_scale,
                    seed=(
                        None
                        if self.random_state is None
                        else self.random_state + 400
                    ),
                )
            if self.spectral_normalization:
                self.stem = SpectralDense(
                    self.resolved_hidden_dims[0],
                    seed=self.random_state,
                    name="stem",
                )
            else:
                self.stem = tf.keras.layers.Dense(
                    self.resolved_hidden_dims[0],
                    kernel_initializer=_initializer(
                        self.random_state, 2000
                    ),
                    name="stem",
                )
            self.blocks = [
                _ResidualBlock(
                    width,
                    seed=(
                        None
                        if self.random_state is None
                        else self.random_state + 200 + index * 10
                    ),
                    spectral_normalization=self.spectral_normalization,
                    normalize=False,
                    name="residual_%d" % index,
                )
                for index, width in enumerate(self.resolved_hidden_dims)
            ]

        if self.spectral_normalization:
            self.score = SpectralDense(
                1,
                seed=(
                    None
                    if self.random_state is None
                    else self.random_state + 3000
                ),
                name="score",
            )
        else:
            self.score = tf.keras.layers.Dense(
                1,
                kernel_initializer=_initializer(self.random_state, 3000),
                name="score",
            )

    @property
    def effective_batch_divisor(self) -> int:
        return self.pac

    def _inputs(
        self, inputs: Any, condition: Optional[Any]
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        if condition is None and isinstance(inputs, (tuple, list)):
            if len(inputs) != 2:
                raise ValueError("critic inputs must be (features, condition)")
            inputs, condition = inputs
        features = tf.convert_to_tensor(inputs, dtype=self.compute_dtype)
        if features.shape.rank != 2:
            raise ValueError("critic features must be a rank-2 tensor")
        if condition is None and self.condition_dim:
            static_width = features.shape[-1]
            if static_width == self.feature_dim + self.condition_dim:
                features, condition = tf.split(
                    features,
                    [self.feature_dim, self.condition_dim],
                    axis=1,
                )
        if condition is None:
            condition_tensor = tf.zeros(
                [tf.shape(features)[0], self.condition_dim],
                dtype=features.dtype,
            )
        else:
            condition_tensor = tf.convert_to_tensor(
                condition, dtype=features.dtype
            )
        if condition_tensor.shape.rank != 2:
            raise ValueError("critic condition must be a rank-2 tensor")
        tf.debugging.assert_equal(tf.shape(features)[1], self.feature_dim)
        tf.debugging.assert_equal(
            tf.shape(condition_tensor)[1], self.condition_dim
        )
        tf.debugging.assert_equal(
            tf.shape(features)[0], tf.shape(condition_tensor)[0]
        )
        return features, condition_tensor

    def pack_inputs(
        self, features: Any, condition: Any
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Pack consecutive feature and condition rows for PacGAN."""

        feature_tensor = tf.convert_to_tensor(
            features, dtype=self.compute_dtype
        )
        condition_tensor = tf.convert_to_tensor(
            condition, dtype=feature_tensor.dtype
        )
        static_rows = feature_tensor.shape[0]
        if static_rows is not None and int(static_rows) % self.pac:
            raise ValueError(
                "critic batch size %d is not divisible by pac=%d"
                % (int(static_rows), self.pac)
            )
        rows = tf.shape(feature_tensor)[0]
        tf.debugging.assert_equal(
            tf.math.floormod(rows, self.pac),
            0,
            message="critic batch size must be divisible by pac",
        )
        packs = rows // self.pac
        return (
            tf.reshape(
                feature_tensor, [packs, self.packed_feature_dim]
            ),
            tf.reshape(
                condition_tensor, [packs, self.packed_condition_dim]
            ),
        )

    def _representation(
        self, value: tf.Tensor, training: bool
    ) -> tf.Tensor:
        if self.architecture == "attention":
            if self.tokenizer is None:
                raise RuntimeError("attention tokenizer is unavailable")
            tokens = self.tokenizer(value)
            for block in self.attention_blocks:
                tokens = block(tokens, training=training)
            return tf.reduce_mean(tokens, axis=1)

        if self.architecture == "fourier":
            if self.fourier_map is None:
                raise RuntimeError("Fourier feature map is unavailable")
            value = self.fourier_map(value)
        if self.stem is None:
            raise RuntimeError("critic stem is unavailable")
        value = (
            self.stem(value, training=training)
            if self.spectral_normalization
            else self.stem(value)
        )
        value = tf.nn.leaky_relu(value, alpha=0.2)
        for block in self.blocks:
            value = block(value, training=training)
        return value

    def call(
        self,
        inputs: Any,
        condition: Optional[Any] = None,
        training: bool = False,
        return_features: bool = False,
    ):
        features, condition_tensor = self._inputs(inputs, condition)
        packed_features, packed_condition = self.pack_inputs(
            features, condition_tensor
        )
        value = tf.concat([packed_features, packed_condition], axis=1)
        representation = self._representation(value, training)
        score = (
            self.score(representation, training=training)
            if self.spectral_normalization
            else self.score(representation)
        )
        if return_features:
            return score, representation
        return score

    def extract_features(
        self,
        inputs: Any,
        condition: Optional[Any] = None,
        training: bool = False,
    ) -> tf.Tensor:
        """Return the representation immediately before the linear score."""

        _, representation = self(
            inputs,
            condition=condition,
            training=training,
            return_features=True,
        )
        return representation

    @property
    def estimated_trainable_parameters(self) -> int:
        if self.architecture == "attention":
            return _attention_parameter_count(
                self.input_width,
                int(self.resolved_attention_dim),
                self.attention_heads,
                len(self.hidden_dims),
            )
        if self.architecture == "fourier":
            return _fourier_parameter_count(
                self.input_width,
                self.resolved_hidden_dims,
                self.fourier_features,
            )
        return _residual_parameter_count(
            self.input_width, self.resolved_hidden_dims
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "condition_dim": self.condition_dim,
            "hidden_dims": list(self.hidden_dims),
            "spectral_normalization": self.spectral_normalization,
            "random_state": self.random_state,
            "architecture": self.architecture,
            "pac": self.pac,
            "attention_heads": self.attention_heads,
            "attention_dim": self.attention_dim,
            "fourier_features": self.fourier_features,
            "fourier_scale": self.fourier_scale,
            "parameter_budget": self.parameter_budget,
            "name": self.name,
        }


__all__ = [
    "ConditionalCritic",
    "ConditionalDenoisingEncoder",
    "ConditionalGenerator",
    "SpectralDense",
]
