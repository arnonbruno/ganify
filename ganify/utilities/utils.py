import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.optimizers import Adam, RMSprop

WGAN_CLIP_VALUE = 0.01


class ClipConstraint(Constraint):
    """Clip weights into the symmetric box used by the original WGAN."""

    def __init__(self, clip_value):
        self.clip_value = float(clip_value)

    def __call__(self, weights):
        return tf.clip_by_value(weights, -self.clip_value, self.clip_value)

    def get_config(self):
        return {"clip_value": self.clip_value}


def wasserstein_loss(y_true, y_pred):
    """Critic objective ``mean(y_true * y_pred)`` with Keras argument order."""
    y_pred = tf.reshape(y_pred, (-1,))
    y_true = tf.cast(tf.reshape(y_true, (-1,)), y_pred.dtype)
    return tf.reduce_mean(y_true * y_pred)


def kernel_initializer(seed, index):
    """Glorot init with a distinct seed per layer.

    A fixed normal standard deviation of 0.02 collapses a plain MLP. Glorot
    keeps generator outputs spread through the tanh range.
    """
    if seed is None:
        return GlorotNormal()
    return GlorotNormal(seed=int(seed) + int(index) * 997)


def hidden_width(n_features, multiplier, max_units):
    n_features = int(n_features)
    multiplier = int(multiplier)
    max_units = int(max_units)
    if n_features < 1 or multiplier < 1 or max_units < 1:
        raise ValueError("n_features, multiplier, and max_units must be positive")
    return min(max_units, max(n_features, n_features * multiplier))


def positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    number = int(value)
    if number < 1:
        raise ValueError(f"{name} must be a positive integer")
    return number


def integer_seed(name, value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def scalar_loss(value):
    if hasattr(value, "numpy"):
        value = value.numpy()
    number = float(np.mean(np.asarray(value, dtype=np.float64)))
    if not np.isfinite(number):
        raise RuntimeError(
            "Training produced a non-finite loss. Check for extreme feature "
            "values or use a smaller learning rate."
        )
    return number


def finite_gradient_pairs(gradients, variables):
    pairs = []
    for gradient, variable in zip(gradients, variables):
        if gradient is None:
            continue
        finite = tf.reduce_all(tf.math.is_finite(gradient))
        if not bool(finite.numpy()):
            raise RuntimeError(
                "Training produced a non-finite gradient. Try a smaller "
                "learning rate or check the input scale."
            )
        pairs.append((gradient, variable))
    if not pairs:
        raise RuntimeError("No gradients were computed. The model graph is disconnected.")
    return pairs


def adversary_targets(rng, batch_size, kind, model_type, flip_rate):
    """Labels for one critic or discriminator update.

    WGAN uses -1 for real rows and +1 for generated rows. The classic GAN
    uses 1 and 0. ``flip_rate`` flips individual rows, which is the label
    noise from Improved Techniques for Training GANs.
    """
    if model_type == "wgan":
        labels = np.full(batch_size, -1.0 if kind == "real" else 1.0, dtype=np.float32)
        if flip_rate:
            labels[rng.random(batch_size) < flip_rate] *= -1.0
        return labels
    if model_type == "gan":
        labels = np.full(batch_size, 1.0 if kind == "real" else 0.0, dtype=np.float32)
        if flip_rate:
            flip = rng.random(batch_size) < flip_rate
            labels[flip] = 1.0 - labels[flip]
        return labels
    raise ValueError("model_type must be 'gan' or 'wgan'")


def generator_targets(batch_size, model_type):
    if model_type == "wgan":
        return np.full(batch_size, -1.0, dtype=np.float32)
    if model_type == "gan":
        return np.full(batch_size, 1.0, dtype=np.float32)
    raise ValueError("model_type must be 'gan' or 'wgan'")


def iter_epoch_indices(rng, n_rows, batch_size):
    """Shuffle once and cover every row in an epoch.

    A leftover of one row is absorbed into the previous batch so batch
    normalization never sees a single example.
    """
    n_rows = int(n_rows)
    batch_size = int(batch_size)
    if n_rows < 1 or batch_size < 1:
        raise ValueError("n_rows and batch_size must be positive")
    order = np.asarray(rng.permutation(n_rows), dtype=np.int64)
    batches = []
    start = 0
    while start < n_rows:
        end = min(start + batch_size, n_rows)
        if n_rows - end == 1:
            end = n_rows
        batches.append(order[start:end])
        start = end
    return batches


def _optimizer(cls, learning_rate, **kwargs):
    try:
        return cls(learning_rate=learning_rate, **kwargs)
    except TypeError:
        return cls(lr=learning_rate, **kwargs)


class RobustMinMaxScaler:
    """Scale real columns to [-1, 1] and keep constant columns unchanged.

    The generator uses a tanh output, so every synthetic coordinate is mapped
    back inside the training minimum and maximum. A constant column is stored
    and written back exactly on the inverse transform.
    """

    def fit(self, data):
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError("Scaler expects a 2D array")
        self.data_min_ = data.min(axis=0)
        self.data_max_ = data.max(axis=0)
        span = self.data_max_ - self.data_min_
        self.constant_ = span == 0
        self.span_ = span.copy()
        self.span_[self.constant_] = 1.0
        return self

    def transform(self, data):
        data = np.asarray(data, dtype=np.float64)
        scaled = ((data - self.data_min_) / self.span_) * 2.0 - 1.0
        if np.any(self.constant_):
            scaled = np.array(scaled, copy=True)
            scaled[:, self.constant_] = 0.0
        return scaled

    def inverse_transform(self, data):
        data = np.clip(np.asarray(data, dtype=np.float64), -1.0, 1.0)
        restored = ((data + 1.0) / 2.0) * self.span_ + self.data_min_
        if np.any(self.constant_):
            restored = np.array(restored, copy=True)
            restored[:, self.constant_] = self.data_min_[self.constant_]
        return restored


class QuantileCopulaScaler:
    """Map each column to uniform ``[-1, 1]`` through its empirical CDF.

    Min-max scaling crushes skewed columns: when one column spans 65 to
    535,000, ordinary values land within a hair of -1, where the tanh
    generator saturates and its gradients vanish. The copula scaler instead
    stores the sorted training values of each column and maps a value to its
    mid-rank ``(rank + 0.5) / n`` scaled to ``[-1, 1]``. The training matrix
    the GAN sees is uniform on every column, so heavy tails, spikes at zero,
    and 0/1 flags all spread across the full tanh range. The inverse maps a
    generator output back through the empirical quantile function, which
    reproduces point masses (every value in a tied quantile block inverts to
    the tied value) and keeps every coordinate inside the training range.
    Constant columns are written back exactly, like :class:`RobustMinMaxScaler`.
    """

    def __init__(self, max_quantiles=16384):
        self.max_quantiles = positive_int("max_quantiles", max_quantiles)
        self.sorted_ = None
        self.constant_ = None
        self.data_min_ = None
        self.data_max_ = None

    def fit(self, data):
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError("Scaler expects a 2D array")
        self.data_min_ = data.min(axis=0)
        self.data_max_ = data.max(axis=0)
        span = self.data_max_ - self.data_min_
        self.constant_ = span == 0
        self.sorted_ = []
        self.step_ = []
        for column in range(data.shape[1]):
            ordered = np.sort(data[:, column])
            if len(ordered) > self.max_quantiles:
                index = np.linspace(0, len(ordered) - 1, self.max_quantiles)
                ordered = ordered[np.round(index).astype(int)]
            self.sorted_.append(ordered)
            # Low-cardinality columns invert as a step function so sampling
            # returns only observed values (exact 0/1 flags, exact categories).
            # Continuous columns interpolate between order stats, which draws
            # novel in-between values instead of copies.
            self.step_.append(len(np.unique(ordered)) <= 256)
        self.step_ = np.asarray(self.step_, dtype=bool)
        return self

    def transform(self, data):
        data = np.asarray(data, dtype=np.float64)
        out = np.empty_like(data)
        for column in range(data.shape[1]):
            if self.constant_[column]:
                out[:, column] = 0.0
                continue
            ref = self.sorted_[column]
            count = len(ref)
            low = np.searchsorted(ref, data[:, column], side="left")
            high = np.searchsorted(ref, data[:, column], side="right")
            uniform = (low + 0.5 * (high - low)) / count
            out[:, column] = np.clip(uniform, 0.0, 1.0) * 2.0 - 1.0
        return out

    def inverse_transform(self, data):
        data = np.clip(np.asarray(data, dtype=np.float64), -1.0, 1.0)
        uniform = (data + 1.0) / 2.0
        out = np.empty_like(uniform)
        for column in range(uniform.shape[1]):
            if self.constant_[column]:
                out[:, column] = self.data_min_[column]
                continue
            out[:, column] = self.quantile(uniform[:, column], column)
        return out

    def quantile(self, uniform, column):
        """Map ``uniform`` values in [0, 1] to training quantiles of ``column``."""
        ref = self.sorted_[column]
        count = len(ref)
        # Forward uses mid-ranks (k + 0.5) / n, so pos = u * n - 0.5 maps
        # training points back onto themselves.
        pos = np.clip(np.asarray(uniform, dtype=np.float64) * count - 0.5, 0.0, count - 1)
        if self.step_[column]:
            return ref[np.floor(pos + 0.5).astype(int)]
        low = np.floor(pos).astype(int)
        high = np.minimum(low + 1, count - 1)
        frac = pos - low
        return ref[low] * (1.0 - frac) + ref[high] * frac

    def save_dict(self):
        payload = {
            "data_min": np.asarray(self.data_min_, dtype=np.float64),
            "data_max": np.asarray(self.data_max_, dtype=np.float64),
            "constant": np.asarray(self.constant_, dtype=np.bool_),
            "step": np.asarray(self.step_, dtype=np.bool_),
            "n_cols": np.asarray(len(self.sorted_)),
            "max_quantiles": np.asarray(self.max_quantiles),
        }
        for column, values in enumerate(self.sorted_):
            payload[f"q{column}"] = np.asarray(values, dtype=np.float64)
        return payload

    @classmethod
    def load_dict(cls, archive):
        scaler = cls(max_quantiles=int(archive["max_quantiles"]))
        scaler.data_min_ = np.asarray(archive["data_min"], dtype=np.float64)
        scaler.data_max_ = np.asarray(archive["data_max"], dtype=np.float64)
        scaler.constant_ = np.asarray(archive["constant"]).astype(bool)
        n_cols = int(archive["n_cols"])
        scaler.sorted_ = [
            np.asarray(archive[f"q{column}"], dtype=np.float64) for column in range(n_cols)
        ]
        if "step" in archive:
            scaler.step_ = np.asarray(archive["step"]).astype(bool)
        else:
            # Backward compatibility with format-2 copula archives.
            scaler.step_ = np.asarray(
                [len(np.unique(references)) <= 256 for references in scaler.sorted_],
                dtype=bool,
            )
        return scaler


class EmpiricalCDFCalibrator:
    """Frozen pointwise CDFs for batch-independent marginal calibration.

    Request-batch ranking makes a synthetic value depend on every other row
    requested at the same time: ``sample(1)`` is forced to the median, and ten
    1k calls differ from one 10k call. This calibrator is fitted once from a
    fixed latent reference pool after training. Future generator outputs are
    converted to uniform values through those frozen empirical CDFs, so each
    row is calibrated independently and sample-call prefixes are stable.
    """

    def __init__(self):
        self.sorted_ = None

    def fit(self, generated):
        generated = np.asarray(generated, dtype=np.float64)
        if generated.ndim != 2 or generated.shape[0] < 2:
            raise ValueError("calibration data must have at least two rows")
        if not np.isfinite(generated).all():
            raise ValueError("calibration data contains NaN or infinite values")
        self.sorted_ = [
            np.sort(generated[:, column]) for column in range(generated.shape[1])
        ]
        return self

    def transform(self, generated):
        if self.sorted_ is None:
            raise RuntimeError("calibrator is not fitted")
        generated = np.asarray(generated, dtype=np.float64)
        if generated.ndim != 2 or generated.shape[1] != len(self.sorted_):
            raise ValueError("generated data has the wrong number of columns")
        uniform = np.empty_like(generated)
        for column, references in enumerate(self.sorted_):
            low = np.searchsorted(references, generated[:, column], side="left")
            high = np.searchsorted(references, generated[:, column], side="right")
            # Mid-ranks handle exact ties deterministically. Values outside the
            # reference range map to the open empirical-CDF endpoints.
            uniform[:, column] = (low + 0.5 * (high - low) + 0.5) / (
                len(references) + 1.0
            )
        return np.clip(uniform, 0.0, 1.0)

    def save_dict(self):
        if self.sorted_ is None:
            raise RuntimeError("calibrator is not fitted")
        payload = {"n_cols": np.asarray(len(self.sorted_))}
        for column, values in enumerate(self.sorted_):
            payload[f"c{column}"] = np.asarray(values, dtype=np.float64)
        return payload

    @classmethod
    def load_dict(cls, archive):
        calibrator = cls()
        n_cols = int(archive["n_cols"])
        calibrator.sorted_ = [
            np.asarray(archive[f"c{column}"], dtype=np.float64)
            for column in range(n_cols)
        ]
        return calibrator


def _series_to_float(series):
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
        return series.astype("float64")
    return pd.to_numeric(series, errors="raise")


def to_float_matrix(data, name="x_train"):
    """Convert tabular input to a finite float64 matrix of shape (rows, columns)."""
    columns = None
    index = None
    if isinstance(data, pd.DataFrame):
        if data.columns.duplicated().any():
            raise ValueError(f"{name} has duplicate column names")
        columns = list(data.columns)
        index = data.index
        converted = []
        for column in data.columns:
            try:
                converted.append(_series_to_float(data[column]).to_numpy(dtype=np.float64))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Column {column!r} in {name} must be numeric") from exc
        values = np.column_stack(converted) if converted else np.empty((len(data), 0))
    elif isinstance(data, pd.Series):
        columns = [data.name if data.name is not None else "value"]
        index = data.index
        try:
            values = _series_to_float(data).to_numpy(dtype=np.float64).reshape(-1, 1)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric") from exc
    else:
        values = np.asarray(data)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.ndim != 2:
            raise RuntimeError("This tool is supposed to work with array data, not tensors")
        if np.iscomplexobj(values):
            raise ValueError(f"{name} must be real numeric data")
        try:
            values = np.array(values, dtype=np.float64, copy=True)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric") from exc
    if values.ndim != 2:
        raise RuntimeError("This tool is supposed to work with array data, not tensors")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return values, columns, index


def prepare_target(y_train, n_rows):
    """Return the single class label and the target index, if it has one."""
    index = None
    if isinstance(y_train, pd.Series):
        index = y_train.index
        values = y_train.to_numpy()
    elif isinstance(y_train, pd.DataFrame):
        index = y_train.index
        values = y_train.to_numpy()
    else:
        values = np.asarray(y_train)
    if values.ndim == 0 or values.size == 0:
        raise ValueError("y_train is empty")
    if values.ndim > 2:
        raise ValueError("y_train must be a vector or a 2D one-hot matrix")
    if values.ndim == 2 and values.shape[1] == 0:
        raise ValueError("y_train is empty")
    if len(values) != n_rows:
        raise RuntimeError("Length of data and target are different")
    if np.issubdtype(values.dtype, np.number):
        if np.iscomplexobj(values) or not np.isfinite(np.asarray(values, dtype=np.float64)).all():
            raise ValueError("y_train contains NaN or infinite values")
    if values.ndim == 2 and values.shape[1] > 1:
        classes = np.unique(values, axis=0)
    else:
        classes = np.unique(values)
    if len(classes) != 1:
        raise RuntimeError(
            "More than 1 classes to amplify on target. Ideally, you want to "
            "create fake examples from one single class"
        )
    return classes[0], index


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class Utilities:
    """Optimizers, loss, and scaling helpers used by the networks."""

    def get_optimizer_wgan(self):
        return _optimizer(RMSprop, 0.00005)

    def get_optimizer_wgan_gp(
        self, learning_rate=0.0001, beta_1=0.0, beta_2=0.9
    ):
        # Adam settings from Gulrajani et al., WGAN-GP.
        return _optimizer(
            Adam,
            float(learning_rate),
            beta_1=float(beta_1),
            beta_2=float(beta_2),
        )

    def get_optimizer_gan(
        self, learning_rate=0.0002, beta_1=0.5, beta_2=0.999
    ):
        return _optimizer(
            Adam,
            float(learning_rate),
            beta_1=float(beta_1),
            beta_2=float(beta_2),
        )

    def get_wasserstein_loss(self, y_true, y_pred):
        return wasserstein_loss(y_true, y_pred)

    def get_gan_loss(self):
        return "binary_crossentropy"

    def get_random_dim(self):
        return 100

    def transform_data(self, data):
        scaler = RobustMinMaxScaler().fit(data)
        return scaler.transform(data), scaler
