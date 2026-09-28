import json
import math
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from ganify._version import __version__
from ganify.conditional import ConditionalGANEngine
from ganify.discriminator.critic import Critic
from ganify.discriminator.discriminator import Discriminator
from ganify.generator.generator import Generator
from ganify.preprocessing import TableTransformer
from ganify.schema import TableSchema
from ganify.utilities.utils import (
    EmpiricalCDFCalibrator,
    QuantileCopulaScaler,
    RobustMinMaxScaler,
    Utilities,
    adversary_targets,
    finite_gradient_pairs,
    generator_targets,
    integer_seed,
    iter_epoch_indices,
    json_value,
    positive_int,
    prepare_target,
    scalar_loss,
    to_float_matrix,
    wasserstein_loss,
)

SAVE_FORMAT = 3
SUPPORTED_SAVE_FORMATS = (1, 2, 3)
_NOT_FITTED = "Model not fit to data yet. Please train the model with fit_data first"


class Ganify:
    """Amplify one class of numeric tabular data with a GAN or a WGAN.

    ``fit_data`` learns a generator from rows that all share one target value.
    ``create_bulk`` draws new rows and maps them back into the training range
    of each column. Architecture width is capped, constant columns are
    preserved, and each network is updated on its own so the generator step
    cannot change the critic or discriminator. The WGAN path uses a gradient
    penalty instead of weight clipping, which keeps the critic's learning
    signal from vanishing in a small fully connected network.
    """

    def __init__(
        self,
        random_state=42,
        random_dim=100,
        max_units=512,
        scaler="minmax",
        ema_decay=0.0,
        ema_warmup_epochs=0,
        calibrate_marginals=False,
        calibration_size=16384,
        round_integers=False,
        inequality_pairs=None,
    ):
        """Create an unfitted model.

        scaler:
            ``"minmax"`` (default) scales each column to ``[-1, 1]``.
            ``"copula"`` maps each column through its empirical CDF first,
            which keeps skewed columns, spikes at zero, and 0/1 flags spread
            across the tanh range instead of crushed against -1.
        ema_decay, ema_warmup_epochs:
            Exponential moving average of the generator weights used for
            sampling. ``0.0`` (default) samples from the live weights.
            ``ema_warmup_epochs`` skips the first epochs so immature weights
            do not pollute the average.
        calibrate_marginals:
            Map each generator coordinate through a frozen empirical CDF and
            then to training quantiles. Requires ``scaler="copula"``. The
            generator supplies the joint rank structure; the training table
            supplies marginal shapes, including rare categories and tails.
            Calibration is pointwise, so output does not depend on request
            batch size.
        calibration_size:
            Number of fixed latent rows used to fit the generator-output CDFs.
        round_integers:
            Round columns that are integral in the training data.
        inequality_pairs:
            ``[(small, big), ...]`` column names or indices with
            ``small <= big`` enforced on every sampled row by clipping the
            small side down to the big side.
        """
        self.utils = Utilities()
        self.random_state = integer_seed("random_state", random_state)
        self.seed = self.random_state
        self.random_dim = positive_int("random_dim", random_dim)
        self.max_units = positive_int("max_units", max_units)
        if scaler not in ("minmax", "copula"):
            raise ValueError("scaler must be 'minmax' or 'copula'")
        self.scaler_name = scaler
        self.ema_decay = float(ema_decay)
        if not np.isfinite(self.ema_decay) or not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be a finite number in [0, 1)")
        if (
            isinstance(ema_warmup_epochs, bool)
            or not isinstance(ema_warmup_epochs, (int, np.integer))
            or int(ema_warmup_epochs) < 0
        ):
            raise ValueError("ema_warmup_epochs must be an integer >= 0")
        self.ema_warmup_epochs = int(ema_warmup_epochs)
        self.calibrate_marginals = bool(calibrate_marginals)
        self.calibration_size = positive_int("calibration_size", calibration_size)
        self.round_integers = bool(round_integers)
        self.inequality_pairs = list(inequality_pairs) if inequality_pairs else []
        self._rng = (
            np.random.default_rng(self.random_state)
            if self.random_state is not None
            else np.random.default_rng()
        )
        self._sample_chunk = 4096
        self._ema_generator = None
        self._marginal_calibrator = None
        self.table_transformer = None
        self.schema_ = None
        self._typed_table = False
        self._effective_scaler_name = scaler
        self._typed_inequality_pairs = []
        self._conditional_engine = None
        self._int_mask = None
        self._pair_idx = []
        self._fitted = False
        self.f = False
        self.cols_names = None
        self.y_label_ = None
        self.critic_scores_ = None
        self.adversary_one = None
        self.adversary_two = None
        self.scaler = None
        self.type = None
        self.gan = None
        self._reset_history()

    def _reset_history(self):
        self.d1_hist = []
        self.d2_hist = []
        self.g_hist = []
        self.history_ = {
            "real": self.d1_hist,
            "fake": self.d2_hist,
            "generator": self.g_hist,
        }
        self.stopped_epoch_ = None
        self.epochs_ran_ = 0

    def _require_fitted(self):
        if not self._fitted:
            raise RuntimeError(_NOT_FITTED)

    def _reseed(self):
        if self.random_state is None:
            return
        # Keras dropout and initialization use the backend seed. Restore the
        # global Python and NumPy generators so a fit does not disturb the caller.
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        try:
            tf.keras.utils.set_random_seed(int(self.random_state))
        except AttributeError:
            tf.random.set_seed(int(self.random_state))
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        self._rng = np.random.default_rng(int(self.random_state))

    def _noise(self, rows):
        return self._rng.normal(0.0, 1.0, size=(int(rows), self.random_dim)).astype(np.float32)

    def _objective(self, y_true, y_pred):
        y_pred = tf.reshape(y_pred, (-1,))
        y_true = tf.cast(tf.reshape(y_true, (-1,)), y_pred.dtype)
        if self.type == "wgan":
            return tf.reduce_mean(y_true * y_pred)
        return self._bce(y_true, y_pred)

    def _update(self, model, optimizer, features, labels):
        variables = list(model.trainable_variables)
        features = tf.convert_to_tensor(features)
        labels = tf.convert_to_tensor(labels)
        with tf.GradientTape() as tape:
            predictions = model(features, training=True)
            loss = self._objective(labels, predictions)
        pairs = finite_gradient_pairs(tape.gradient(loss, variables), variables)
        optimizer.apply_gradients(pairs)
        return scalar_loss(loss)

    def _gradient_penalty(self, real, fake):
        batch = int(real.shape[0])
        epsilon = tf.constant(self._rng.random((batch, 1)), dtype=real.dtype)
        interpolated = epsilon * real + (1.0 - epsilon) * fake
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            scores = tf.reshape(self.adversary_two(interpolated, training=True), (-1,))
        gradients = tape.gradient(scores, interpolated)
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1) + 1e-12)
        return tf.reduce_mean(tf.square(slopes - 1.0))

    def _update_critic_wgan(self, real, fake, y_real, y_fake):
        variables = list(self.adversary_two.trainable_variables)
        real = tf.convert_to_tensor(real)
        fake = tf.convert_to_tensor(fake)
        y_real = tf.convert_to_tensor(y_real)
        y_fake = tf.convert_to_tensor(y_fake)
        with tf.GradientTape() as tape:
            loss_real = self._objective(y_real, self.adversary_two(real, training=True))
            loss_fake = self._objective(y_fake, self.adversary_two(fake, training=True))
            penalty = self._gradient_penalty(real, fake)
            loss = loss_real + loss_fake + self.gradient_penalty_ * penalty
        pairs = finite_gradient_pairs(tape.gradient(loss, variables), variables)
        self._adversary_optimizer.apply_gradients(pairs)
        return scalar_loss(loss_real), scalar_loss(loss_fake)

    def _update_generator(self, noise, labels):
        variables = list(self.adversary_one.trainable_variables)
        noise = tf.convert_to_tensor(noise)
        labels = tf.convert_to_tensor(labels)
        with tf.GradientTape() as tape:
            fake = self.adversary_one(noise, training=True)
            predictions = self.adversary_two(fake, training=False)
            loss = self._objective(labels, predictions)
        pairs = finite_gradient_pairs(tape.gradient(loss, variables), variables)
        self._generator_optimizer.apply_gradients(pairs)
        return scalar_loss(loss)

    @staticmethod
    def _needs_typed_transform(x_train, schema=None, schema_overrides=None):
        if schema is not None or schema_overrides:
            return True
        if not isinstance(x_train, pd.DataFrame):
            return False
        return bool(
            x_train.isna().any().any()
            or any(
                not (
                    pd.api.types.is_numeric_dtype(dtype)
                    or pd.api.types.is_bool_dtype(dtype)
                )
                for dtype in x_train.dtypes
            )
        )

    def _prepare_typed_inputs(
        self,
        x_train,
        y_train,
        *,
        schema,
        schema_overrides,
        continuous_transform,
        handle_unknown,
        validation_data,
        validation_fraction,
        patience,
    ):
        if not isinstance(x_train, pd.DataFrame):
            raise TypeError("typed preprocessing requires a pandas DataFrame")
        prepare_target(y_train, len(x_train))
        train_frame = x_train.copy()
        validation_frame = None
        if validation_data is not None:
            validation_frame = (
                validation_data[0]
                if isinstance(validation_data, tuple)
                else validation_data
            )
            if not isinstance(validation_frame, pd.DataFrame):
                raise TypeError(
                    "typed validation_data must be a pandas DataFrame"
                )
        else:
            fraction = float(validation_fraction)
            if patience is not None and fraction == 0.0:
                fraction = 0.1
            if fraction > 0.0:
                if len(train_frame) < 4:
                    raise ValueError(
                        "validation splitting needs at least four training rows"
                    )
                rng = np.random.default_rng(
                    None
                    if self.random_state is None
                    else int(self.random_state) + 104729
                )
                order = rng.permutation(len(train_frame))
                count = max(2, int(round(len(train_frame) * fraction)))
                count = min(count, len(train_frame) - 2)
                validation_frame = train_frame.iloc[order[:count]].copy()
                train_frame = train_frame.iloc[order[count:]].copy()

        transformer = TableTransformer(
            schema=schema,
            continuous_transform=continuous_transform,
            handle_unknown=handle_unknown,
            schema_overrides=schema_overrides,
            random_state=self.random_state,
        )
        transformed_train = transformer.fit_transform(train_frame)
        transformed_validation = (
            transformer.transform(validation_frame)
            if validation_frame is not None
            else None
        )
        target = np.asarray(y_train)
        if target.ndim == 2:
            transformed_target = np.repeat(target[:1], len(train_frame), axis=0)
        else:
            transformed_target = np.repeat(target.reshape(-1)[:1], len(train_frame))
        self.table_transformer = transformer
        self.schema_ = transformer.schema_
        self._typed_table = True
        return (
            np.asarray(transformed_train, dtype=np.float64),
            transformed_target,
            (
                np.asarray(transformed_validation, dtype=np.float64)
                if transformed_validation is not None
                else None
            ),
            0.0,
            list(transformer.schema_.names),
        )

    @staticmethod
    def _auto_conditional(x, y):
        if y is not None and pd.Series(np.asarray(y).reshape(-1)).nunique(
            dropna=False
        ) > 1:
            return True
        if not isinstance(x, pd.DataFrame):
            return False
        for column in x.columns:
            series = x[column]
            if (
                pd.api.types.is_bool_dtype(series.dtype)
                or isinstance(series.dtype, pd.CategoricalDtype)
                or not pd.api.types.is_numeric_dtype(series.dtype)
            ):
                return True
        return False

    def fit(
        self,
        x,
        y=None,
        *,
        schema=None,
        conditional=None,
        target_name=None,
        continuous_bins=0,
        conditional_weight=1.0,
        generator_dims=None,
        critic_dims=None,
        spectral_normalization=False,
        **kwargs,
    ):
        """Fit a full table with optional CTGAN-style conditional training.

        Conditional mode is selected automatically for multiclass targets or
        mixed-type dataframes. Pass ``conditional=False`` to retain the
        single-class compatibility path, or ``conditional=True`` to condition
        numeric tables on discovered bins and targets as well.
        """
        use_conditional = (
            (
                self._auto_conditional(x, y)
                or "dp_config" in kwargs
                or "constraints" in kwargs
            )
            if conditional is None
            else bool(conditional)
        )
        if not use_conditional:
            if y is None:
                y = np.zeros(len(x), dtype=np.float64)
            return self.fit_data(x, y, schema=schema, **kwargs)
        if not isinstance(x, pd.DataFrame):
            if schema is None:
                raise TypeError(
                    "conditional fitting requires a DataFrame or explicit schema"
                )
            x = pd.DataFrame(x, columns=list(schema.names))

        options = dict(kwargs)
        epochs = options.pop("epochs", 10)
        batch_size = options.pop("batch_size", 64)
        n_critic = options.pop("n_critic", 5)
        gradient_penalty = options.pop("gradient_penalty", 10.0)
        verbose = options.pop("verbose", 0)
        compile_steps = options.pop("compile", False)
        generator_learning_rate = options.pop(
            "generator_learning_rate", 0.0001
        )
        critic_learning_rate = options.pop(
            "adversary_learning_rate",
            options.pop("critic_learning_rate", 0.0002),
        )
        beta_1 = options.pop("beta_1", 0.0)
        beta_2 = options.pop("beta_2", 0.9)
        schema_overrides = options.pop("schema_overrides", None)
        continuous_transform = options.pop("continuous_transform", "copula")
        handle_unknown = options.pop("handle_unknown", "error")
        max_modes = options.pop("max_modes", 5)
        constraints = options.pop("constraints", None)
        dp_config = options.pop("dp_config", None)
        preprocessor = options.pop("preprocessor", None)
        critic_architecture = options.pop("critic_architecture", "residual")
        pac = options.pop("pac", 1)
        attention_heads = options.pop("attention_heads", 4)
        attention_dim = options.pop("attention_dim", None)
        fourier_features = options.pop("fourier_features", 64)
        fourier_scale = options.pop("fourier_scale", 1.0)
        match_critic_parameters = options.pop(
            "match_critic_parameters", False
        )
        critic_parameter_budget = options.pop(
            "critic_parameter_budget", None
        )
        warmup_epochs = options.pop("warmup_epochs", 0)
        warmup_mask_probability = options.pop(
            "warmup_mask_probability", 0.15
        )
        warmup_learning_rate = options.pop("warmup_learning_rate", None)
        interaction_weight = options.pop("interaction_weight", 0.0)
        interaction_temperature = options.pop(
            "interaction_temperature", 0.1
        )
        feature_matching_weight = options.pop(
            "feature_matching_weight", 0.0
        )
        swa = options.pop("swa", False)
        swa_start_epoch = options.pop("swa_start_epoch", 1)
        swa_frequency = options.pop("swa_frequency", 1)
        swag = options.pop("swag", False)
        swag_start_epoch = options.pop("swag_start_epoch", 1)
        swag_frequency = options.pop("swag_frequency", 1)
        swag_max_rank = options.pop("swag_max_rank", 20)
        swag_scale = options.pop("swag_scale", 1.0)
        label_flip = float(options.pop("label_flip", 0.0))
        if label_flip:
            raise ValueError("conditional WGAN does not support label flipping")
        unsupported = {
            name
            for name in (
                "patience",
                "min_delta",
                "validation_data",
                "validation_fraction",
                "selection_metric",
                "selection_sample_size",
                "selection_callback",
            )
            if name in options
        }
        if unsupported:
            raise TypeError(
                "conditional fit does not yet accept: "
                + ", ".join(sorted(unsupported))
            )
        if options:
            raise TypeError(
                "unknown conditional fit options: "
                + ", ".join(sorted(options))
            )

        width = min(self.max_units, max(64, 4 * max(1, len(x.columns))))
        engine = ConditionalGANEngine(
            random_state=self.random_state,
            noise_dim=self.random_dim,
            generator_dims=(
                tuple(generator_dims) if generator_dims is not None else (width, width)
            ),
            critic_dims=(
                tuple(critic_dims) if critic_dims is not None else (width, width)
            ),
            batch_size=batch_size,
            epochs=epochs,
            n_critic=n_critic,
            gradient_penalty=gradient_penalty,
            conditional_weight=conditional_weight,
            generator_learning_rate=generator_learning_rate,
            critic_learning_rate=critic_learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            ema_decay=self.ema_decay,
            continuous_transform=continuous_transform,
            continuous_bins=continuous_bins,
            handle_unknown=handle_unknown,
            max_modes=max_modes,
            schema=schema,
            schema_overrides=schema_overrides,
            constraints=constraints,
            spectral_normalization=spectral_normalization,
            compile=compile_steps,
            critic_architecture=critic_architecture,
            pac=pac,
            attention_heads=attention_heads,
            attention_dim=attention_dim,
            fourier_features=fourier_features,
            fourier_scale=fourier_scale,
            match_critic_parameters=match_critic_parameters,
            critic_parameter_budget=critic_parameter_budget,
            warmup_epochs=warmup_epochs,
            warmup_mask_probability=warmup_mask_probability,
            warmup_learning_rate=warmup_learning_rate,
            interaction_weight=interaction_weight,
            interaction_temperature=interaction_temperature,
            feature_matching_weight=feature_matching_weight,
            swa=swa,
            swa_start_epoch=swa_start_epoch,
            swa_frequency=swa_frequency,
            swag=swag,
            swag_start_epoch=swag_start_epoch,
            swag_frequency=swag_frequency,
            swag_max_rank=swag_max_rank,
            swag_scale=swag_scale,
            dp_config=dp_config,
        ).fit(
            x,
            target=y,
            target_name=(
                target_name
                if target_name is not None
                else getattr(y, "name", None)
            ),
            verbose=verbose,
            preprocessor=preprocessor,
        )
        self._conditional_engine = engine
        self._fitted = True
        self.f = True
        self.type = "conditional_wgan"
        self.cols_names = list(x.columns)
        self.schema_ = engine.schema_
        self.table_transformer = engine.transformer_
        self.history_ = engine.history_
        self.epochs_ran_ = engine.epochs_ran_
        self.target_classes_ = engine.target_classes_
        self.target_name_ = engine.target_name_
        self.privacy_report_ = engine.privacy_report_
        self.y_label_ = None
        return self

    def sample(
        self,
        n,
        *,
        conditions=None,
        return_target=False,
        use_ema=True,
        weight_source=None,
        weight_seed=None,
        return_audit=False,
        output="dataframe",
    ):
        """Sample rows, optionally fixing table values or a target class."""
        if self._conditional_engine is not None:
            result = self._conditional_engine.sample(
                n,
                conditions=conditions,
                return_target=return_target,
                use_ema=use_ema,
                weight_source=weight_source,
                weight_seed=weight_seed,
                return_audit=return_audit,
            )
            if output in (1, "dataframe", True):
                return result
            if output in (None, 0, False):
                if return_target and return_audit:
                    frame, target, audit = result
                    return frame.to_numpy(), target.to_numpy(), audit
                if return_target:
                    frame, target = result
                    return frame.to_numpy(), target.to_numpy()
                if return_audit:
                    frame, audit = result
                    return frame.to_numpy(), audit
                return result.to_numpy()
            raise ValueError("output must be None or 'dataframe'")
        if conditions is not None or return_target or return_audit:
            raise ValueError(
                "conditions, return_target, and return_audit require a conditional fit"
            )
        if weight_source is not None or weight_seed is not None:
            raise ValueError(
                "weight_source and weight_seed require a conditional fit"
            )
        return self.create_bulk(length=n, output=output)

    def fit_data(
        self,
        x_train,
        y_train,
        type="wgan",
        cols_names=None,
        schema=None,
        schema_overrides=None,
        continuous_transform="copula",
        handle_unknown="error",
        batch_size=8,
        epochs=1,
        patience=None,
        min_delta=0.0,
        n_critic=5,
        label_flip=0.0,
        gradient_penalty=10.0,
        generator_learning_rate=None,
        adversary_learning_rate=None,
        beta_1=None,
        beta_2=None,
        validation_data=None,
        validation_fraction=0.0,
        selection_metric="auto",
        selection_sample_size=1024,
        selection_callback=None,
        verbose=1,
        compile=False,
    ):
        """Fit a generator on one class of numeric rows.

        Parameters
        ----------
        x_train:
            Array, series, or dataframe of real features. One-dimensional
            input is treated as a single column. Images and other tensors
            are rejected.
        y_train:
            Target used only to confirm that every row belongs to one class.
            A one-hot matrix with a single repeated row is accepted. The
            label is stored on ``y_label_`` and is not generated as a column.
        type:
            ``"wgan"`` (default) or ``"gan"``. Case is ignored.
        cols_names:
            Column names used when ``create_bulk(..., output=1)`` builds a
            dataframe. A dataframe passed as ``x_train`` supplies these names
            when this argument is omitted.
        schema, schema_overrides:
            Optional typed-table description or inference overrides. A typed
            preprocessor is selected automatically for non-numeric or nullable
            dataframes. It handles continuous, count, binary, categorical,
            ordinal, datetime, constant, and missing values.
        continuous_transform, handle_unknown:
            Typed-preprocessor options; see :class:`TableTransformer`.
        batch_size, epochs:
            Mini-batch size and number of full passes. Each epoch shuffles
            the rows and visits each row once. ``batch_size`` larger than the
            table uses one batch per epoch.
        patience, min_delta:
            Optional early stopping on the generator loss at the end of an
            epoch. ``patience`` is the number of epochs without an improvement
            of at least ``min_delta`` before training stops.
        n_critic:
            Adversary updates for every generator update. The default of 5
            follows the original WGAN training ratio.
        label_flip:
            Per-row label noise probability in ``[0, 1]``. Defaults to zero;
            flipping Wasserstein labels changes the critic objective.
        gradient_penalty:
            Weight of the WGAN-GP penalty. Ignored for ``type="gan"``.
        generator_learning_rate, adversary_learning_rate, beta_1, beta_2:
            Optional Adam/TTUR settings. WGAN-GP defaults to the original
            ``1e-4, beta_1=0, beta_2=0.9``; classic GAN defaults to
            ``2e-4, beta_1=0.5, beta_2=0.999``.
        validation_data, validation_fraction:
            Optional validation features, or a deterministic fraction held
            out before fitting transforms. When ``patience`` is set and no
            validation source is supplied, 10% is held out automatically.
        selection_metric:
            ``"auto"`` uses a validation fidelity/coverage score when a
            validation split exists and generator loss otherwise.
            ``"validation"`` requires validation data; ``"generator"`` keeps
            the legacy loss criterion.
        selection_callback:
            Optional ``callback(real_validation, generated) -> float`` used
            for checkpoint selection; lower is better.
        verbose:
            ``0`` silences the progress bar and epoch log.
        compile:
            Run each critic/generator update as one compiled graph instead of
            eager ops. Same objective, far fewer host-device round trips, so a
            GPU is actually faster. Reproducible per device for a fixed seed;
            CPU and GPU numerics may differ in the last decimals.
        """
        self._compile = bool(compile)
        # ``fit_data`` keeps the legacy numeric-coercion contract. The modern
        # ``fit`` method auto-selects typed/conditional preprocessing; callers
        # can opt into typed preprocessing here with schema information.
        typed = schema is not None or bool(schema_overrides)
        original_names = None
        original_pairs = list(self.inequality_pairs)
        if typed:
            (
                x_train,
                y_train,
                validation_data,
                validation_fraction,
                original_names,
            ) = self._prepare_typed_inputs(
                x_train,
                y_train,
                schema=schema,
                schema_overrides=schema_overrides,
                continuous_transform=continuous_transform,
                handle_unknown=handle_unknown,
                validation_data=validation_data,
                validation_fraction=validation_fraction,
                patience=patience,
            )
            cols_names = list(self.table_transformer.get_feature_names_out())
            self._typed_inequality_pairs = original_pairs
            self.inequality_pairs = []
            self._effective_scaler_name = (
                "copula" if self.calibrate_marginals else "minmax"
            )
        else:
            self.table_transformer = None
            self.schema_ = None
            self._typed_table = False
            self._typed_inequality_pairs = []
            self._effective_scaler_name = self.scaler_name
        try:
            self._prepare_fit(
                x_train,
                y_train,
            type=type,
            cols_names=cols_names,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            min_delta=min_delta,
            n_critic=n_critic,
            label_flip=label_flip,
            gradient_penalty=gradient_penalty,
            generator_learning_rate=generator_learning_rate,
            adversary_learning_rate=adversary_learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            validation_data=validation_data,
            validation_fraction=validation_fraction,
            selection_metric=selection_metric,
            selection_sample_size=selection_sample_size,
            selection_callback=selection_callback,
            )
        finally:
            self.inequality_pairs = original_pairs
        if typed:
            self.cols_names = original_names
        if self._compile:
            return self._fit_compiled(
                epochs=self.epochs,
                batch_size=self.batch_size_,
                patience=patience,
                min_delta=float(min_delta),
                n_critic=self.n_critic_,
                verbose=verbose,
            )
        return self._fit_eager(
            epochs=self.epochs,
            batch_size=self.batch_size_,
            patience=patience,
            min_delta=float(min_delta),
            n_critic=self.n_critic_,
            label_flip=self.label_flip_,
            verbose=verbose,
        )

    def _prepare_fit(
        self,
        x_train,
        y_train,
        type="wgan",
        cols_names=None,
        batch_size=8,
        epochs=1,
        patience=None,
        min_delta=0.0,
        n_critic=5,
        label_flip=0.0,
        gradient_penalty=10.0,
        generator_learning_rate=None,
        adversary_learning_rate=None,
        beta_1=None,
        beta_2=None,
        validation_data=None,
        validation_fraction=0.0,
        selection_metric="auto",
        selection_sample_size=1024,
        selection_callback=None,
    ):
        """Validate inputs, build networks, and scale the training matrix."""
        model_type = str(type).strip().lower()
        if model_type not in {"gan", "wgan"}:
            raise RuntimeError("Invalid type of GAN. Choose 'gan' or 'wgan'.")
        epochs = positive_int("epochs", epochs)
        batch_size = positive_int("batch_size", batch_size)
        n_critic = positive_int("n_critic", n_critic)
        if patience is not None:
            patience = positive_int("patience", patience)
        min_delta = float(min_delta)
        if not np.isfinite(min_delta) or min_delta < 0:
            raise ValueError("min_delta must be a finite number >= 0")
        label_flip = float(label_flip)
        if not np.isfinite(label_flip) or not 0.0 <= label_flip <= 1.0:
            raise ValueError("label_flip must be between 0 and 1")
        gradient_penalty = float(gradient_penalty)
        if not np.isfinite(gradient_penalty) or gradient_penalty < 0:
            raise ValueError("gradient_penalty must be a finite number >= 0")
        validation_fraction = float(validation_fraction)
        if not np.isfinite(validation_fraction) or not 0.0 <= validation_fraction < 0.5:
            raise ValueError("validation_fraction must be in [0, 0.5)")
        selection_metric = str(selection_metric).strip().lower()
        if selection_metric not in {"auto", "validation", "generator"}:
            raise ValueError(
                "selection_metric must be 'auto', 'validation', or 'generator'"
            )
        selection_sample_size = positive_int(
            "selection_sample_size", selection_sample_size
        )

        default_lr = 0.0001 if model_type == "wgan" else 0.0002
        default_beta_1 = 0.0 if model_type == "wgan" else 0.5
        default_beta_2 = 0.9 if model_type == "wgan" else 0.999
        generator_learning_rate = (
            default_lr
            if generator_learning_rate is None
            else float(generator_learning_rate)
        )
        adversary_learning_rate = (
            default_lr
            if adversary_learning_rate is None
            else float(adversary_learning_rate)
        )
        beta_1 = default_beta_1 if beta_1 is None else float(beta_1)
        beta_2 = default_beta_2 if beta_2 is None else float(beta_2)
        for name, value in (
            ("generator_learning_rate", generator_learning_rate),
            ("adversary_learning_rate", adversary_learning_rate),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite number > 0")
        for name, value in (("beta_1", beta_1), ("beta_2", beta_2)):
            if not np.isfinite(value) or not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be a finite number in [0, 1)")

        x, x_columns, x_index = to_float_matrix(x_train, "x_train")
        if len(x) < 2:
            raise ValueError("x_train needs at least 2 rows")
        y_label, y_index = prepare_target(y_train, len(x))
        if x_index is not None and y_index is not None and not x_index.equals(y_index):
            raise ValueError("x_train and y_train must have the same index")
        if cols_names is None:
            names = list(x_columns) if x_columns is not None else None
        else:
            names = list(cols_names)
            if len(names) != x.shape[1]:
                raise RuntimeError(
                    "Column names must be an array with the same size as input data"
                )

        validation = None
        if validation_data is not None:
            validation_features = (
                validation_data[0]
                if isinstance(validation_data, tuple)
                else validation_data
            )
            validation, validation_columns, _ = to_float_matrix(
                validation_features, "validation_data"
            )
            if validation.shape[1] != x.shape[1]:
                raise ValueError(
                    "validation_data must have the same number of columns as x_train"
                )
            if (
                x_columns is not None
                and validation_columns is not None
                and list(validation_columns) != list(x_columns)
            ):
                raise ValueError(
                    "validation_data columns must match x_train columns in order"
                )
        else:
            if patience is not None and validation_fraction == 0.0:
                validation_fraction = 0.1
            if validation_fraction > 0.0:
                if len(x) < 4:
                    raise ValueError(
                        "validation splitting needs at least four training rows"
                    )
                split_rng = np.random.default_rng(
                    None
                    if self.random_state is None
                    else int(self.random_state) + 104729
                )
                order = split_rng.permutation(len(x))
                validation_rows = max(2, int(round(len(x) * validation_fraction)))
                validation_rows = min(validation_rows, len(x) - 2)
                validation = x[order[:validation_rows]]
                x = x[order[validation_rows:]]

        self.type = model_type
        self.epochs = epochs
        self.batch_size_ = batch_size
        self.n_critic_ = n_critic
        self.label_flip_ = label_flip
        self.gradient_penalty_ = gradient_penalty
        self.generator_learning_rate_ = generator_learning_rate
        self.adversary_learning_rate_ = adversary_learning_rate
        self.beta_1_ = beta_1
        self.beta_2_ = beta_2
        self.selection_sample_size_ = selection_sample_size
        self.selection_callback_ = selection_callback
        self._validation_raw = validation
        self.selection_metric_ = (
            "validation"
            if selection_metric == "auto" and validation is not None
            else "generator"
            if selection_metric == "auto"
            else selection_metric
        )
        if self.selection_metric_ == "validation" and validation is None:
            raise ValueError(
                "selection_metric='validation' needs validation_data or "
                "validation_fraction > 0"
            )
        self.cols_names = names
        self.y_label_ = y_label
        self.n_features_ = int(x.shape[1])
        self._fitted = False
        self.f = False
        self.critic_scores_ = None
        self._reset_history()
        self._reseed()
        self.validation_history_ = []
        self.best_epoch_ = None

        self.gen = Generator(
            x,
            random_dim=self.random_dim,
            max_units=self.max_units,
            seed=self.random_state,
        )
        self.adversary_one = self.gen.get_generator()
        if (
            self.calibrate_marginals
            and self._effective_scaler_name != "copula"
            and not self._typed_table
        ):
            raise ValueError("calibrate_marginals=True requires scaler='copula'")
        if self._effective_scaler_name == "copula":
            self.scaler = QuantileCopulaScaler().fit(x)
        else:
            self.scaler = RobustMinMaxScaler().fit(x)
        self.x_train = self.scaler.transform(x).astype(np.float32)
        self._validation_scaled = (
            self.scaler.transform(validation).astype(np.float32)
            if validation is not None
            else None
        )
        validation_rng = np.random.default_rng(
            None
            if self.random_state is None
            else int(self.random_state) + 130363
        )
        self._validation_noise = (
            validation_rng.normal(
                0.0,
                1.0,
                size=(
                    min(selection_sample_size, max(256, len(validation))),
                    self.random_dim,
                ),
            ).astype(np.float32)
            if validation is not None
            else None
        )
        if model_type == "wgan":
            self.critic = Critic(x, seed=self.random_state, max_units=self.max_units)
            self.adversary_two = self.critic.get_critic()
            self._adversary_optimizer = self.utils.get_optimizer_wgan_gp(
                adversary_learning_rate, beta_1, beta_2
            )
            self._generator_optimizer = self.utils.get_optimizer_wgan_gp(
                generator_learning_rate, beta_1, beta_2
            )
            self.loss = wasserstein_loss
        else:
            self.discriminator = Discriminator(
                x, seed=self.random_state, max_units=self.max_units
            )
            self.adversary_two = self.discriminator.get_discriminator()
            self._adversary_optimizer = self.utils.get_optimizer_gan(
                adversary_learning_rate, beta_1, beta_2
            )
            self._generator_optimizer = self.utils.get_optimizer_gan(
                generator_learning_rate, beta_1, beta_2
            )
            self.loss = "binary_crossentropy"
        self.opt = self._generator_optimizer
        self._bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.gan = None

        if batch_size > len(self.x_train):
            warnings.warn(
                f"batch_size ({batch_size}) is larger than the number of rows "
                f"({len(self.x_train)}); each epoch will use one batch of "
                f"{len(self.x_train)} rows.",
                UserWarning,
                stacklevel=2,
            )

        self._ema_generator = None
        self._ema_started = False
        if self.ema_decay > 0:
            twin = Generator(
                x,
                random_dim=self.random_dim,
                max_units=self.max_units,
                seed=self.random_state,
            ).get_generator()
            twin.set_weights(self.adversary_one.get_weights())
            self._ema_generator = twin
        self._int_mask = np.array(
            [
                bool(np.all(np.abs(x[:, column] - np.round(x[:, column])) < 1e-9))
                for column in range(self.n_features_)
            ]
        )
        name_to_index = {name: i for i, name in enumerate(names)} if names else {}
        self._pair_idx = []
        for small, big in self.inequality_pairs:
            if isinstance(small, str) or isinstance(big, str):
                if not names:
                    raise ValueError(
                        "inequality_pairs with column names needs a DataFrame "
                        "or cols_names"
                    )
                try:
                    self._pair_idx.append((name_to_index[small], name_to_index[big]))
                except KeyError as exc:
                    raise ValueError(f"inequality_pairs has unknown column {exc}") from exc
            else:
                pair = (int(small), int(big))
                if not all(0 <= i < self.n_features_ for i in pair):
                    raise ValueError("inequality_pairs indices are out of range")
                self._pair_idx.append(pair)

    def _ema_step(self):
        self._ema_generator.set_weights(
            [
                self.ema_decay * ema + (1.0 - self.ema_decay) * live
                for ema, live in zip(
                    self._ema_generator.get_weights(), self.adversary_one.get_weights()
                )
            ]
        )

    def _capture_checkpoint(self):
        return {
            "generator": [np.array(value, copy=True) for value in self.adversary_one.get_weights()],
            "adversary": [np.array(value, copy=True) for value in self.adversary_two.get_weights()],
            "ema": (
                [np.array(value, copy=True) for value in self._ema_generator.get_weights()]
                if self._ema_generator is not None
                else None
            ),
        }

    def _restore_checkpoint(self, checkpoint):
        if checkpoint is None:
            return
        self.adversary_one.set_weights(checkpoint["generator"])
        self.adversary_two.set_weights(checkpoint["adversary"])
        if self._ema_generator is not None and checkpoint["ema"] is not None:
            self._ema_generator.set_weights(checkpoint["ema"])

    @staticmethod
    def _ks_statistic(left, right):
        points = np.unique(np.concatenate([left, right]))
        left_cdf = np.searchsorted(np.sort(left), points, side="right") / len(left)
        right_cdf = np.searchsorted(np.sort(right), points, side="right") / len(right)
        return float(np.max(np.abs(left_cdf - right_cdf)))

    def _validation_score(self, epoch):
        """Return a lower-is-better fidelity/coverage checkpoint score."""
        generated_scaled = self.adversary_one(
            self._validation_noise, training=False
        ).numpy()
        validation_scaled = np.asarray(self._validation_scaled, dtype=np.float64)
        generated_scaled = np.asarray(generated_scaled, dtype=np.float64)
        if self.selection_callback_ is not None:
            validation_raw = self.scaler.inverse_transform(validation_scaled)
            generated_raw = self.scaler.inverse_transform(generated_scaled)
            score = float(self.selection_callback_(validation_raw, generated_raw))
            if not np.isfinite(score):
                raise RuntimeError("selection_callback returned a non-finite score")
            components = {"callback": score}
        else:
            marginal = np.median(
                [
                    self._ks_statistic(
                        validation_scaled[:, column], generated_scaled[:, column]
                    )
                    for column in range(self.n_features_)
                    if not self.scaler.constant_[column]
                ]
                or [0.0]
            )
            validation_std = np.std(validation_scaled, axis=0)
            generated_std = np.std(generated_scaled, axis=0)
            varying = validation_std > 1e-8
            coverage = float(
                np.median(
                    np.abs(
                        np.log(
                            np.maximum(generated_std[varying], 1e-8)
                            / validation_std[varying]
                        )
                    )
                )
                if np.any(varying)
                else 0.0
            )
            if self.n_features_ > 1:
                validation_corr = pd.DataFrame(validation_scaled).corr(
                    method="spearman"
                ).to_numpy()
                generated_corr = pd.DataFrame(generated_scaled).corr(
                    method="spearman"
                ).to_numpy()
                upper = np.triu(np.ones_like(validation_corr, dtype=bool), k=1)
                finite = upper & np.isfinite(validation_corr) & np.isfinite(
                    generated_corr
                )
                dependence = float(
                    np.mean(
                        np.abs(validation_corr[finite] - generated_corr[finite])
                    )
                    if np.any(finite)
                    else 0.0
                )
            else:
                dependence = 0.0
            score = float(dependence + 0.25 * marginal + 0.25 * coverage)
            components = {
                "dependence": dependence,
                "marginal": float(marginal),
                "coverage": coverage,
            }
        self.validation_history_.append(
            {"epoch": int(epoch), "score": score, **components}
        )
        return score

    def _selection_value(self, epoch, epoch_generator_losses):
        if self.selection_metric_ == "validation":
            return self._validation_score(epoch)
        return float(np.mean(epoch_generator_losses))

    def _fit_marginal_calibrator(self):
        self._marginal_calibrator = None
        if not self.calibrate_marginals:
            return
        generator = (
            self._ema_generator
            if self._ema_generator is not None
            else self.adversary_one
        )
        rng = np.random.default_rng(
            None
            if self.random_state is None
            else int(self.random_state) + 15485863
        )
        generated = []
        chunk = max(1, int(self._sample_chunk))
        for start in range(0, self.calibration_size, chunk):
            rows = min(chunk, self.calibration_size - start)
            latent = rng.normal(0.0, 1.0, size=(rows, self.random_dim)).astype(
                np.float32
            )
            generated.append(generator(latent, training=False).numpy())
        self._marginal_calibrator = EmpiricalCDFCalibrator().fit(
            np.concatenate(generated, axis=0)
        )

    def _finalize_fit(self, checkpoint):
        self._restore_checkpoint(checkpoint)
        self._fit_marginal_calibrator()
        return self

    def _fit_eager(self, epochs, batch_size, patience, min_delta, n_critic, label_flip, verbose):
        """Train with one eager update per step (legacy loop, CPU-friendly)."""
        best = math.inf
        best_checkpoint = None
        wait = 0
        track_best = patience is not None or self.selection_metric_ == "validation"
        for epoch in range(1, epochs + 1):
            batches = iter_epoch_indices(self._rng, self.x_train.shape[0], batch_size)
            if verbose:
                batches = tqdm(batches, leave=False, desc=f"Epoch {epoch}/{epochs}")
            if self._ema_generator is not None and epoch == self.ema_warmup_epochs + 1:
                self._ema_generator.set_weights(self.adversary_one.get_weights())
                self._ema_started = True
            epoch_generator_losses = []
            for batch_index in batches:
                real_losses = []
                fake_losses = []
                for _ in range(n_critic):
                    real = self.x_train[batch_index]
                    # Materialize generated rows so the adversary step cannot
                    # update the generator. training=True keeps this forward
                    # pass on the same path as the generator update.
                    fake = self.adversary_one(self._noise(len(batch_index)), training=True).numpy()
                    y_real = adversary_targets(
                        self._rng, len(batch_index), "real", self.type, label_flip
                    )
                    y_fake = adversary_targets(
                        self._rng, len(batch_index), "fake", self.type, label_flip
                    )
                    if self.type == "wgan":
                        real_loss, fake_loss = self._update_critic_wgan(real, fake, y_real, y_fake)
                    else:
                        real_loss = self._update(
                            self.adversary_two, self._adversary_optimizer, real, y_real
                        )
                        fake_loss = self._update(
                            self.adversary_two, self._adversary_optimizer, fake, y_fake
                        )
                    real_losses.append(real_loss)
                    fake_losses.append(fake_loss)
                self.d1_hist.append(float(np.mean(real_losses)))
                self.d2_hist.append(float(np.mean(fake_losses)))
                g_loss = self._update_generator(
                    self._noise(len(batch_index)),
                    generator_targets(len(batch_index), self.type),
                )
                self.g_hist.append(g_loss)
                epoch_generator_losses.append(g_loss)
                if self._ema_generator is not None:
                    if epoch > self.ema_warmup_epochs:
                        self._ema_step()
                    else:
                        self._ema_generator.set_weights(self.adversary_one.get_weights())
            self._fitted = True
            self.f = True
            self.epochs_ran_ = epoch
            metric = self._selection_value(epoch, epoch_generator_losses)
            if verbose:
                print(
                    f"Epoch {epoch}/{epochs} "
                    f"real={self.d1_hist[-1]:.4f} "
                    f"fake={self.d2_hist[-1]:.4f} "
                    f"generator={self.g_hist[-1]:.4f}"
                )
            if track_best and metric < best - min_delta:
                best = metric
                wait = 0
                self.best_epoch_ = epoch
                best_checkpoint = self._capture_checkpoint()
            elif patience is not None:
                wait += 1
                if wait >= patience:
                    self.stopped_epoch_ = epoch
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
        return self._finalize_fit(best_checkpoint)

    def _fit_compiled(self, epochs, batch_size, patience, min_delta, n_critic, verbose):
        """Train with one compiled graph per update (GPU-friendly).

        Same objective as :meth:`_fit_eager`: the training matrix lives on the
        device, noise and label flips come from a seeded ``tf.random.Generator``,
        and each critic/generator update runs as a single graph, so a step costs
        one host sync instead of dozens. Reproducible per device for a fixed
        seed.
        """
        seed = self.random_state if self.random_state is not None else 0
        tgen = tf.random.Generator.from_seed(int(seed) + 999983)
        x_dev = tf.constant(self.x_train)
        flip = float(self.label_flip_)
        gp_weight = float(self.gradient_penalty_)
        is_wgan = self.type == "wgan"
        adv_vars = list(self.adversary_two.trainable_variables)
        gen_vars = list(self.adversary_one.trainable_variables)
        ema_vars = list(self._ema_generator.variables) if self._ema_generator else []
        decay = float(self.ema_decay)

        def apply_finite_gradients(optimizer, gradients, variables, owner):
            pairs = [
                (gradient, variable)
                for gradient, variable in zip(gradients, variables)
                if gradient is not None
            ]
            if not pairs:
                raise RuntimeError(f"No gradients were computed for {owner}")
            for gradient, _ in pairs:
                tf.debugging.assert_all_finite(
                    gradient, f"{owner} produced a non-finite gradient"
                )
            optimizer.apply_gradients(pairs)

        @tf.function
        def critic_step(real):
            rows = tf.shape(real)[0]
            noise = tgen.normal([rows, self.random_dim])
            with tf.GradientTape() as tape:
                fake = self.adversary_one(noise, training=True)
                if is_wgan:
                    y_real = tf.fill([rows], -1.0)
                    y_fake = tf.fill([rows], 1.0)
                    if flip > 0:
                        drop_real = tf.cast(tgen.uniform([rows]) < flip, tf.float32)
                        drop_fake = tf.cast(tgen.uniform([rows]) < flip, tf.float32)
                        y_real = y_real * (1.0 - 2.0 * drop_real)
                        y_fake = y_fake * (1.0 - 2.0 * drop_fake)
                    scored_real = tf.reshape(self.adversary_two(real, training=True), [-1])
                    scored_fake = tf.reshape(self.adversary_two(fake, training=True), [-1])
                    loss_real = tf.reduce_mean(y_real * scored_real)
                    loss_fake = tf.reduce_mean(y_fake * scored_fake)
                    epsilon = tgen.uniform([rows, 1], dtype=scored_real.dtype)
                    with tf.GradientTape() as gp_tape:
                        interpolated = epsilon * real + (1.0 - epsilon) * fake
                        gp_tape.watch(interpolated)
                        mixed = tf.reshape(
                            self.adversary_two(interpolated, training=True), [-1]
                        )
                    slopes = tf.sqrt(
                        tf.reduce_sum(tf.square(gp_tape.gradient(mixed, interpolated)), axis=1)
                        + 1e-12
                    )
                    penalty = tf.reduce_mean(tf.square(slopes - 1.0))
                    loss = loss_real + loss_fake + gp_weight * penalty
                else:
                    y_real = tf.fill([rows], 1.0)
                    y_fake = tf.fill([rows], 0.0)
                    if flip > 0:
                        drop_real = tf.cast(tgen.uniform([rows]) < flip, tf.float32)
                        drop_fake = tf.cast(tgen.uniform([rows]) < flip, tf.float32)
                        y_real = tf.abs(y_real - drop_real)
                        y_fake = tf.abs(y_fake - drop_fake)
                    loss_real = self._bce(y_real, self.adversary_two(real, training=True))
                    loss_fake = self._bce(y_fake, self.adversary_two(fake, training=True))
                    loss = loss_real + loss_fake
            gradients = tape.gradient(loss, adv_vars)
            apply_finite_gradients(
                self._adversary_optimizer,
                gradients,
                adv_vars,
                "adversary",
            )
            return loss_real, loss_fake

        @tf.function
        def generator_step(rows, update_ema):
            noise = tgen.normal([rows, self.random_dim])
            with tf.GradientTape() as tape:
                fake = self.adversary_one(noise, training=True)
                judged = tf.reshape(self.adversary_two(fake, training=False), [-1])
                if is_wgan:
                    loss = tf.reduce_mean(-1.0 * judged)
                else:
                    loss = self._bce(tf.ones_like(judged), judged)
            gradients = tape.gradient(loss, gen_vars)
            apply_finite_gradients(
                self._generator_optimizer,
                gradients,
                gen_vars,
                "generator",
            )
            if ema_vars and update_ema:
                for ema_var, live_var in zip(ema_vars, self.adversary_one.variables):
                    ema_var.assign(decay * ema_var + (1.0 - decay) * live_var)
            return loss

        best = math.inf
        best_checkpoint = None
        wait = 0
        track_best = patience is not None or self.selection_metric_ == "validation"
        for epoch in range(1, epochs + 1):
            batches = iter_epoch_indices(self._rng, x_dev.shape[0], batch_size)
            if verbose:
                batches = tqdm(batches, leave=False, desc=f"Epoch {epoch}/{epochs}")
            if self._ema_generator is not None and epoch == self.ema_warmup_epochs + 1:
                self._ema_generator.set_weights(self.adversary_one.get_weights())
                self._ema_started = True
            ema_active = self._ema_generator is not None and epoch > self.ema_warmup_epochs
            epoch_generator_losses = []
            for batch_index in batches:
                real = tf.gather(x_dev, batch_index)
                total_real = tf.constant(0.0)
                total_fake = tf.constant(0.0)
                for _ in range(n_critic):
                    step_real, step_fake = critic_step(real)
                    total_real = total_real + step_real
                    total_fake = total_fake + step_fake
                gen_loss = generator_step(len(batch_index), ema_active)
                if self._ema_generator is not None and not ema_active:
                    self._ema_generator.set_weights(self.adversary_one.get_weights())
                real_value = float((total_real / n_critic).numpy())
                fake_value = float((total_fake / n_critic).numpy())
                gen_value = float(gen_loss.numpy())
                if not (np.isfinite(real_value) and np.isfinite(fake_value)
                        and np.isfinite(gen_value)):
                    raise RuntimeError(
                        "Training produced a non-finite loss. Check for extreme "
                        "feature values or use a smaller learning rate."
                    )
                self.d1_hist.append(real_value)
                self.d2_hist.append(fake_value)
                self.g_hist.append(gen_value)
                epoch_generator_losses.append(gen_value)
            self._fitted = True
            self.f = True
            self.epochs_ran_ = epoch
            metric = self._selection_value(epoch, epoch_generator_losses)
            if verbose:
                print(
                    f"Epoch {epoch}/{epochs} "
                    f"real={self.d1_hist[-1]:.4f} "
                    f"fake={self.d2_hist[-1]:.4f} "
                    f"generator={self.g_hist[-1]:.4f}"
                )
            if track_best and metric < best - min_delta:
                best = metric
                wait = 0
                self.best_epoch_ = epoch
                best_checkpoint = self._capture_checkpoint()
            elif patience is not None:
                wait += 1
                if wait >= patience:
                    self.stopped_epoch_ = epoch
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
        return self._finalize_fit(best_checkpoint)

    def get_gan(self):
        """Return a stacked generator and adversary model.

        Training itself does not use this model. Gradients are applied to
        one network at a time in ``fit_data``.
        """
        self._require_fitted()
        if self._conditional_engine is not None:
            raise RuntimeError(
                "conditional models expose generator_ and critic_ through "
                "model._conditional_engine"
            )
        flag = self.adversary_two.trainable
        self.adversary_two.trainable = False
        try:
            gan_input = tf.keras.Input(shape=(self.random_dim,))
            generated = self.adversary_one(gan_input)
            judged = self.adversary_two(generated)
            gan = tf.keras.Model(gan_input, judged, name="ganify")
            optimizer = (
                self.utils.get_optimizer_wgan_gp(
                    getattr(self, "generator_learning_rate_", 0.0001),
                    getattr(self, "beta_1_", 0.0),
                    getattr(self, "beta_2_", 0.9),
                )
                if self.type == "wgan"
                else self.utils.get_optimizer_gan(
                    getattr(self, "generator_learning_rate_", 0.0002),
                    getattr(self, "beta_1_", 0.5),
                    getattr(self, "beta_2_", 0.999),
                )
            )
            loss = wasserstein_loss if self.type == "wgan" else "binary_crossentropy"
            gan.compile(optimizer=optimizer, loss=loss)
        finally:
            self.adversary_two.trainable = flag
        self.gan = gan
        return gan

    def create_bulk(self, lenght=None, output=None, length=None):
        """Draw synthetic rows from the fitted generator.

        ``length`` is the number of rows. ``lenght`` remains accepted so
        existing positional calls keep working. ``output=1`` returns a
        dataframe and needs column names from a dataframe input or from
        ``cols_names``. Rows are drawn from the EMA generator when one was
        fitted, then mapped back through the scaler. With
        ``calibrate_marginals``, each column is mapped to training quantiles
        by rank order instead, so rare categories and heavy tails match the
        training proportions exactly. Integer columns are rounded and
        ``inequality_pairs`` enforced when those options are set. Critic or
        discriminator scores for the drawn rows are stored on
        ``critic_scores_``.
        """
        if lenght is not None and length is not None and int(lenght) != int(length):
            raise ValueError("Pass length once. lenght is accepted as a legacy alias of length.")
        count = length if length is not None else lenght
        if count is None:
            raise ValueError("length is required")
        count = positive_int("length", count)
        if self._conditional_engine is not None:
            return self.sample(count, output=output)
        self._require_fitted()
        if output in (None, 0, False):
            as_frame = False
        elif output == 1 or output == "dataframe":
            as_frame = True
        else:
            raise ValueError("output must be None or 1")
        if as_frame and not self.cols_names:
            raise ValueError(
                "output=1 needs column names. Pass a DataFrame to fit_data or set cols_names."
            )

        generator = self._ema_generator if self._ema_generator is not None else self.adversary_one
        latent = self._noise(count)
        generated = []
        scores = []
        chunk = max(1, int(self._sample_chunk))
        for start in range(0, count, chunk):
            part = latent[start : start + chunk]
            fake = generator(part, training=False).numpy()
            generated.append(fake)
            judged = self.adversary_two(fake, training=False).numpy().reshape(-1)
            scores.append(judged)
        fake = np.concatenate(generated, axis=0)
        self.critic_scores_ = np.concatenate(scores, axis=0).astype(np.float64, copy=False)
        if self.calibrate_marginals:
            data = self._calibrate_marginals(fake)
        else:
            data = self.scaler.inverse_transform(fake)
        data = self._postprocess(data)
        if self.table_transformer is not None:
            frame = self.table_transformer.inverse_transform(data)
            for small, big in self._typed_inequality_pairs:
                if isinstance(small, str) and isinstance(big, str):
                    over = frame[small] > frame[big]
                    frame.loc[over, small] = frame.loc[over, big]
            return frame if as_frame else frame.to_numpy()
        if as_frame:
            return pd.DataFrame(data, columns=self.cols_names)
        return data

    def _calibrate_marginals(self, fake):
        """Map generator outputs through frozen CDFs to training quantiles."""
        if self._marginal_calibrator is None:
            raise RuntimeError("marginal calibrator is not fitted")
        uniform = self._marginal_calibrator.transform(fake)
        out = np.empty_like(fake, dtype=np.float64)
        for column in range(fake.shape[1]):
            if self.scaler.constant_[column]:
                out[:, column] = self.scaler.data_min_[column]
                continue
            out[:, column] = self.scaler.quantile(
                uniform[:, column], column
            )
        return out

    def _postprocess(self, data):
        values = np.array(np.asarray(data, dtype=np.float64), copy=True)
        if self.round_integers and self._int_mask is not None:
            values[:, self._int_mask] = np.round(values[:, self._int_mask])
        for small, big in self._pair_idx:
            over = values[:, small] > values[:, big]
            values[over, small] = values[over, big]
        return values

    def plot_performance(self, path=None, show=True):
        """Plot adversary losses on real and synthetic rows, and generator loss."""
        self._require_fitted()
        if self._conditional_engine is not None:
            histories = self._conditional_engine.history_
            if not histories.get("generator"):
                raise RuntimeError("No training history to plot.")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6))
            try:
                for name in ("critic", "generator", "conditional", "gradient_penalty"):
                    values = histories.get(name)
                    if values:
                        ax.plot(values, label=name.replace("_", " ").title())
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.legend()
                if path is not None:
                    path = Path(path)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(path, bbox_inches="tight")
                if show:
                    plt.show()
            finally:
                plt.close(fig)
            return
        if not self.g_hist:
            raise RuntimeError("No training history to plot.")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        try:
            ax.plot(self.d1_hist, color="b", label="Real")
            ax.plot(self.d2_hist, color="r", label="Fake")
            ax.plot(self.g_hist, color="g", label="Generator")
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.legend()
            if path is not None:
                path = Path(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(path, bbox_inches="tight")
            if show:
                plt.show()
        finally:
            plt.close(fig)

    def save(self, directory):
        """Save the generator, adversary, scaler, and training history."""
        self._require_fitted()
        directory = Path(directory)
        if directory.exists() and not directory.is_dir():
            raise ValueError(f"save path must be a directory, got {directory}")
        if self._conditional_engine is not None:
            return self._conditional_engine.save(directory)
        directory.mkdir(parents=True, exist_ok=True)
        generator_path = str(directory / "generator.weights.h5")
        adversary_path = str(directory / "adversary.weights.h5")
        self.adversary_one.save_weights(generator_path)
        self.adversary_two.save_weights(adversary_path)
        if self._ema_generator is not None:
            self._ema_generator.save_weights(str(directory / "ema_generator.weights.h5"))
        if isinstance(self.scaler, QuantileCopulaScaler):
            np.savez(str(directory / "scaler_copula.npz"), **self.scaler.save_dict())
        else:
            np.savez(
                str(directory / "scaler.npz"),
                data_min=self.scaler.data_min_,
                data_max=self.scaler.data_max_,
                span=self.scaler.span_,
                constant=np.asarray(self.scaler.constant_, dtype=np.bool_),
            )
        if self._marginal_calibrator is not None:
            np.savez(
                str(directory / "marginal_calibrator.npz"),
                **self._marginal_calibrator.save_dict(),
            )
        if self.table_transformer is not None:
            self.table_transformer.save(directory / "typed_preprocessor.json")
        metadata = {
            "format": SAVE_FORMAT,
            "version": __version__,
            "type": self.type,
            "random_dim": self.random_dim,
            "max_units": self.max_units,
            "n_features": self.n_features_,
            "random_state": self.random_state,
            "cols_names": json_value(self.cols_names),
            "y_label": json_value(self.y_label_),
            "d1_hist": self.d1_hist,
            "d2_hist": self.d2_hist,
            "g_hist": self.g_hist,
            "stopped_epoch": self.stopped_epoch_,
            "label_flip": self.label_flip_,
            "gradient_penalty": self.gradient_penalty_,
            "n_critic": self.n_critic_,
            "batch_size": self.batch_size_,
            "epochs": self.epochs,
            "epochs_ran": self.epochs_ran_,
            "scaler": self.scaler_name,
            "effective_scaler": self._effective_scaler_name,
            "typed_table": self.table_transformer is not None,
            "typed_inequality_pairs": json_value(
                self._typed_inequality_pairs
            ),
            "ema_decay": self.ema_decay,
            "ema_warmup_epochs": self.ema_warmup_epochs,
            "has_ema": self._ema_generator is not None,
            "calibrate_marginals": self.calibrate_marginals,
            "calibration_size": self.calibration_size,
            "round_integers": self.round_integers,
            "inequality_pairs": json_value(self.inequality_pairs),
            "int_mask": json_value(self._int_mask),
            "pair_idx": json_value(self._pair_idx),
            "generator_learning_rate": getattr(
                self, "generator_learning_rate_", None
            ),
            "adversary_learning_rate": getattr(
                self, "adversary_learning_rate_", None
            ),
            "beta_1": getattr(self, "beta_1_", None),
            "beta_2": getattr(self, "beta_2_", None),
            "selection_metric": getattr(self, "selection_metric_", "generator"),
            "selection_sample_size": getattr(
                self, "selection_sample_size_", 1024
            ),
            "validation_history": getattr(self, "validation_history_", []),
            "best_epoch": getattr(self, "best_epoch_", None),
            "compile": getattr(self, "_compile", False),
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return str(directory)

    @classmethod
    def load(cls, directory):
        """Load a model written by ``save``.

        Both format 2 (this version) and format 1 (1.1.x, min-max scaler,
        no EMA or calibration) are accepted.
        """
        directory = Path(directory)
        meta_path = directory / "metadata.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"No GANify model metadata at {meta_path}")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata.get("format") == "ganify-conditional-engine":
            engine = ConditionalGANEngine.load(directory)
            width = max(engine.generator_dims + engine.critic_dims)
            model = cls(
                random_state=engine.random_state,
                random_dim=engine.noise_dim,
                max_units=width,
                ema_decay=engine.ema_decay,
            )
            model._conditional_engine = engine
            model._fitted = True
            model.f = True
            model.type = "conditional_wgan"
            model.cols_names = list(engine.schema_.names)
            model.schema_ = engine.schema_
            model.table_transformer = engine.transformer_
            model.history_ = engine.history_
            model.epochs_ran_ = engine.epochs_ran_
            model.target_classes_ = engine.target_classes_
            model.target_name_ = engine.target_name_
            model.privacy_report_ = engine.privacy_report_
            model.y_label_ = None
            return model
        if metadata.get("format") not in SUPPORTED_SAVE_FORMATS:
            raise ValueError("Unsupported GANify model format")
        model = cls(
            random_state=metadata.get("random_state"),
            random_dim=metadata["random_dim"],
            max_units=metadata["max_units"],
            scaler=metadata.get("scaler", "minmax"),
            ema_decay=metadata.get("ema_decay", 0.0),
            ema_warmup_epochs=metadata.get("ema_warmup_epochs", 0),
            calibrate_marginals=metadata.get("calibrate_marginals", False),
            calibration_size=metadata.get("calibration_size", 16384),
            round_integers=metadata.get("round_integers", False),
            inequality_pairs=metadata.get("inequality_pairs"),
        )
        n_features = int(metadata["n_features"])
        dummy = np.zeros((2, n_features), dtype=np.float64)
        model.type = metadata["type"]
        model.n_features_ = n_features
        model.gen = Generator(
            dummy,
            random_dim=model.random_dim,
            max_units=model.max_units,
            seed=model.random_state,
        )
        model.adversary_one = model.gen.get_generator()
        if model.type == "wgan":
            model.critic = Critic(dummy, seed=model.random_state, max_units=model.max_units)
            model.adversary_two = model.critic.get_critic()
        elif model.type == "gan":
            model.discriminator = Discriminator(
                dummy, seed=model.random_state, max_units=model.max_units
            )
            model.adversary_two = model.discriminator.get_discriminator()
        else:
            raise ValueError("Unsupported model type in metadata")
        model.adversary_one.load_weights(str(directory / "generator.weights.h5"))
        model.adversary_two.load_weights(str(directory / "adversary.weights.h5"))
        if metadata.get("has_ema"):
            model._ema_generator = Generator(
                dummy,
                random_dim=model.random_dim,
                max_units=model.max_units,
                seed=model.random_state,
            ).get_generator()
            model._ema_generator.load_weights(str(directory / "ema_generator.weights.h5"))
        if (directory / "scaler_copula.npz").is_file():
            with np.load(str(directory / "scaler_copula.npz")) as archive:
                model.scaler = QuantileCopulaScaler.load_dict(archive)
        else:
            scaler_file = np.load(str(directory / "scaler.npz"))
            model.scaler = RobustMinMaxScaler()
            model.scaler.data_min_ = scaler_file["data_min"]
            model.scaler.data_max_ = scaler_file["data_max"]
            model.scaler.span_ = scaler_file["span"]
            model.scaler.constant_ = np.asarray(scaler_file["constant"]).astype(bool)
        calibrator_path = directory / "marginal_calibrator.npz"
        if calibrator_path.is_file():
            with np.load(str(calibrator_path)) as archive:
                model._marginal_calibrator = EmpiricalCDFCalibrator.load_dict(
                    archive
                )
        int_mask = metadata.get("int_mask")
        model._int_mask = np.asarray(int_mask, dtype=bool) if int_mask is not None else None
        pair_idx = metadata.get("pair_idx") or []
        model._pair_idx = [(int(small), int(big)) for small, big in pair_idx]
        typed_path = directory / "typed_preprocessor.json"
        if typed_path.is_file():
            model.table_transformer = TableTransformer.load(typed_path)
            model.schema_ = model.table_transformer.schema_
            model._typed_table = True
        else:
            model.table_transformer = None
            model.schema_ = None
            model._typed_table = False
        model._effective_scaler_name = metadata.get(
            "effective_scaler", metadata.get("scaler", "minmax")
        )
        model._typed_inequality_pairs = metadata.get(
            "typed_inequality_pairs", []
        )
        model.cols_names = metadata.get("cols_names")
        model.y_label_ = metadata.get("y_label")
        model.label_flip_ = metadata.get("label_flip", 0.0)
        model.gradient_penalty_ = metadata.get("gradient_penalty", 10.0)
        model.generator_learning_rate_ = metadata.get(
            "generator_learning_rate", 0.0001
        )
        model.adversary_learning_rate_ = metadata.get(
            "adversary_learning_rate", 0.0001
        )
        model.beta_1_ = metadata.get("beta_1", 0.0)
        model.beta_2_ = metadata.get("beta_2", 0.9)
        model.n_critic_ = metadata.get("n_critic", 5)
        model.batch_size_ = metadata.get("batch_size")
        model.epochs = metadata.get("epochs")
        model.selection_metric_ = metadata.get("selection_metric", "generator")
        model.selection_sample_size_ = metadata.get(
            "selection_sample_size", 1024
        )
        model.selection_callback_ = None
        model.validation_history_ = metadata.get("validation_history", [])
        model.best_epoch_ = metadata.get("best_epoch")
        model._compile = metadata.get("compile", False)
        model._reset_history()
        model.d1_hist.extend(metadata.get("d1_hist", []))
        model.d2_hist.extend(metadata.get("d2_hist", []))
        model.g_hist.extend(metadata.get("g_hist", []))
        model.history_ = {
            "real": model.d1_hist,
            "fake": model.d2_hist,
            "generator": model.g_hist,
        }
        model.stopped_epoch_ = metadata.get("stopped_epoch")
        model.epochs_ran_ = metadata.get("epochs_ran", 0)
        if model.calibrate_marginals and model._marginal_calibrator is None:
            # Format-2 calibrated models did not persist frozen generator CDFs.
            # Rebuild them deterministically from the restored generator.
            model._fit_marginal_calibrator()
        model._fitted = True
        model.f = True
        return model
