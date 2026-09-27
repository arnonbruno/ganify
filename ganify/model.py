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
from ganify.discriminator.critic import Critic
from ganify.discriminator.discriminator import Discriminator
from ganify.generator.generator import Generator
from ganify.utilities.utils import (
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

SAVE_FORMAT = 1
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

    def __init__(self, random_state=42, random_dim=100, max_units=512):
        self.utils = Utilities()
        self.random_state = integer_seed("random_state", random_state)
        self.seed = self.random_state
        self.random_dim = positive_int("random_dim", random_dim)
        self.max_units = positive_int("max_units", max_units)
        self._rng = (
            np.random.default_rng(self.random_state)
            if self.random_state is not None
            else np.random.default_rng()
        )
        self._sample_chunk = 4096
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

    def fit_data(
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
        label_flip=0.05,
        gradient_penalty=10.0,
        verbose=1,
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
            Per-row label noise probability in ``[0, 1]``.
        gradient_penalty:
            Weight of the WGAN-GP penalty. Ignored for ``type="gan"``.
        verbose:
            ``0`` silences the progress bar and epoch log.
        """
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

        self.type = model_type
        self.epochs = epochs
        self.batch_size_ = batch_size
        self.n_critic_ = n_critic
        self.label_flip_ = label_flip
        self.gradient_penalty_ = gradient_penalty
        self.cols_names = names
        self.y_label_ = y_label
        self.n_features_ = int(x.shape[1])
        self._fitted = False
        self.f = False
        self.critic_scores_ = None
        self._reset_history()
        self._reseed()

        self.gen = Generator(
            x,
            random_dim=self.random_dim,
            max_units=self.max_units,
            seed=self.random_state,
        )
        self.adversary_one = self.gen.get_generator()
        self.scaler = RobustMinMaxScaler().fit(x)
        self.x_train = self.scaler.transform(x).astype(np.float32)
        if model_type == "wgan":
            self.critic = Critic(x, seed=self.random_state, max_units=self.max_units)
            self.adversary_two = self.critic.get_critic()
            self._adversary_optimizer = self.utils.get_optimizer_wgan_gp()
            self._generator_optimizer = self.utils.get_optimizer_wgan_gp()
            self.loss = wasserstein_loss
        else:
            self.discriminator = Discriminator(
                x, seed=self.random_state, max_units=self.max_units
            )
            self.adversary_two = self.discriminator.get_discriminator()
            self._adversary_optimizer = self.utils.get_optimizer_gan()
            self._generator_optimizer = self.utils.get_optimizer_gan()
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

        best = math.inf
        wait = 0
        for epoch in range(1, epochs + 1):
            batches = iter_epoch_indices(self._rng, self.x_train.shape[0], batch_size)
            if verbose:
                batches = tqdm(batches, leave=False, desc=f"Epoch {epoch}/{epochs}")
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
            self._fitted = True
            self.f = True
            self.epochs_ran_ = epoch
            metric = float(np.mean(epoch_generator_losses))
            if verbose:
                print(
                    f"Epoch {epoch}/{epochs} "
                    f"real={self.d1_hist[-1]:.4f} "
                    f"fake={self.d2_hist[-1]:.4f} "
                    f"generator={self.g_hist[-1]:.4f}"
                )
            if patience is not None:
                if metric < best - min_delta:
                    best = metric
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        self.stopped_epoch_ = epoch
                        if verbose:
                            print(f"Early stopping at epoch {epoch}")
                        break
        return self

    def get_gan(self):
        """Return a stacked generator and adversary model.

        Training itself does not use this model. Gradients are applied to
        one network at a time in ``fit_data``.
        """
        self._require_fitted()
        flag = self.adversary_two.trainable
        self.adversary_two.trainable = False
        try:
            gan_input = tf.keras.Input(shape=(self.random_dim,))
            generated = self.adversary_one(gan_input)
            judged = self.adversary_two(generated)
            gan = tf.keras.Model(gan_input, judged, name="ganify")
            optimizer = (
                self.utils.get_optimizer_wgan()
                if self.type == "wgan"
                else self.utils.get_optimizer_gan()
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
        ``cols_names``. Critic or discriminator scores for the drawn rows
        are stored on ``critic_scores_``.
        """
        self._require_fitted()
        if lenght is not None and length is not None and int(lenght) != int(length):
            raise ValueError("Pass length once. lenght is accepted as a legacy alias of length.")
        count = length if length is not None else lenght
        if count is None:
            raise ValueError("length is required")
        count = positive_int("length", count)
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

        latent = self._noise(count)
        generated = []
        scores = []
        chunk = max(1, int(self._sample_chunk))
        for start in range(0, count, chunk):
            part = latent[start : start + chunk]
            fake = self.adversary_one(part, training=False).numpy()
            generated.append(fake)
            judged = self.adversary_two(fake, training=False).numpy().reshape(-1)
            scores.append(judged)
        fake = np.concatenate(generated, axis=0)
        self.critic_scores_ = np.concatenate(scores, axis=0).astype(np.float64, copy=False)
        data = self.scaler.inverse_transform(fake)
        if as_frame:
            return pd.DataFrame(data, columns=self.cols_names)
        return data

    def plot_performance(self, path=None, show=True):
        """Plot adversary losses on real and synthetic rows, and generator loss."""
        self._require_fitted()
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
        directory.mkdir(parents=True, exist_ok=True)
        generator_path = str(directory / "generator.weights.h5")
        adversary_path = str(directory / "adversary.weights.h5")
        self.adversary_one.save_weights(generator_path)
        self.adversary_two.save_weights(adversary_path)
        np.savez(
            str(directory / "scaler.npz"),
            data_min=self.scaler.data_min_,
            data_max=self.scaler.data_max_,
            span=self.scaler.span_,
            constant=np.asarray(self.scaler.constant_, dtype=np.bool_),
        )
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
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return str(directory)

    @classmethod
    def load(cls, directory):
        """Load a model written by ``save``."""
        directory = Path(directory)
        meta_path = directory / "metadata.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"No GANify model metadata at {meta_path}")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata.get("format") != SAVE_FORMAT:
            raise ValueError("Unsupported GANify model format")
        model = cls(
            random_state=metadata.get("random_state"),
            random_dim=metadata["random_dim"],
            max_units=metadata["max_units"],
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
        scaler_file = np.load(str(directory / "scaler.npz"))
        model.scaler = RobustMinMaxScaler()
        model.scaler.data_min_ = scaler_file["data_min"]
        model.scaler.data_max_ = scaler_file["data_max"]
        model.scaler.span_ = scaler_file["span"]
        model.scaler.constant_ = np.asarray(scaler_file["constant"]).astype(bool)
        model.cols_names = metadata.get("cols_names")
        model.y_label_ = metadata.get("y_label")
        model.label_flip_ = metadata.get("label_flip", 0.05)
        model.gradient_penalty_ = metadata.get("gradient_penalty", 10.0)
        model.n_critic_ = metadata.get("n_critic", 5)
        model.batch_size_ = metadata.get("batch_size")
        model.epochs = metadata.get("epochs")
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
        model._fitted = True
        model.f = True
        return model
