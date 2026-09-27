import os
import tempfile
import unittest
import warnings
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import numpy as np
import pandas as pd
import tensorflow as tf

import ganify
from ganify import ClipConstraint, Ganify, Utilities
from ganify._version import __version__
from ganify.utilities.utils import (
    RobustMinMaxScaler,
    adversary_targets,
    generator_targets,
    hidden_width,
    iter_epoch_indices,
)

ROOT = Path(__file__).resolve().parents[1]


def _small_model(seed=0):
    return Ganify(random_state=seed, random_dim=16, max_units=32)


def _fit(model, x, y, **kwargs):
    params = dict(epochs=1, batch_size=16, n_critic=1, verbose=0, label_flip=0.0)
    params.update(kwargs)
    return model.fit_data(x, y, **params)


class GanifyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        cls.x = rng.normal(loc=2.0, scale=0.5, size=(32, 3))
        cls.x[:, 2] = 5.0
        cls.y = np.ones(32)
        cls.wgan = _fit(_small_model(0), cls.x, cls.y)
        cls.gan = _fit(_small_model(1), cls.x, cls.y, type="gan")

    def test_version_and_public_api(self):
        self.assertEqual(__version__, "1.1.0")
        self.assertEqual(ganify.__version__, "1.1.0")
        self.assertIn("Ganify", ganify.__all__)
        self.assertNotIn("Adam", ganify.__all__)
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("1.1.0", readme)
        self.assertIn("logo.png", readme)
        self.assertIn("Ganify", readme)
        self.assertGreater((ROOT / "logo.png").stat().st_size, 0)
        self.assertGreater((ROOT / "ganify.gif").stat().st_size, 0)
        setup_text = (ROOT / "setup.py").read_text(encoding="utf-8")
        self.assertIn("_version.py", setup_text)

    def test_init_does_not_reset_global_numpy_generator(self):
        np.random.seed(12345)
        expected = np.random.random()
        np.random.seed(12345)
        Ganify(random_state=7)
        self.assertEqual(expected, np.random.random())

    def test_fit_does_not_consume_global_numpy_generator(self):
        np.random.seed(12345)
        expected = np.random.rand(3)
        np.random.seed(12345)
        _fit(_small_model(3), self.x, self.y)
        np.testing.assert_array_equal(expected, np.random.rand(3))

    def test_helpers(self):
        self.assertEqual(hidden_width(4, 8, 512), 32)
        self.assertEqual(hidden_width(4, 2, 512), 8)
        self.assertEqual(hidden_width(200, 8, 512), 512)
        with self.assertRaises(ValueError):
            hidden_width(0, 2, 512)

        rng = np.random.default_rng(0)
        self.assertTrue(np.all(adversary_targets(rng, 8, "real", "gan", 0.0) == 1))
        self.assertTrue(np.all(adversary_targets(rng, 8, "fake", "gan", 0.0) == 0))
        self.assertTrue(np.all(generator_targets(8, "gan") == 1))
        self.assertTrue(np.all(adversary_targets(rng, 8, "real", "wgan", 0.0) == -1))
        self.assertTrue(np.all(adversary_targets(rng, 8, "fake", "wgan", 0.0) == 1))
        self.assertTrue(np.all(generator_targets(8, "wgan") == -1))
        flipped = adversary_targets(np.random.default_rng(0), 8, "real", "gan", 1.0)
        self.assertTrue(np.all(flipped == 0))

        batches = iter_epoch_indices(np.random.default_rng(0), 32, 8)
        self.assertEqual([len(batch) for batch in batches], [8, 8, 8, 8])
        covered = np.concatenate(batches)
        self.assertEqual(sorted(covered.tolist()), list(range(32)))
        absorbed = iter_epoch_indices(np.random.default_rng(1), 9, 8)
        self.assertEqual([len(batch) for batch in absorbed], [9])
        split = iter_epoch_indices(np.random.default_rng(1), 10, 8)
        self.assertEqual(sorted(len(batch) for batch in split), [2, 8])
        self.assertEqual(sorted(np.concatenate(split).tolist()), list(range(10)))

        constraint = ClipConstraint(0.01)
        clipped = constraint(tf.constant([-1.0, 0.0, 0.2, 0.01])).numpy()
        np.testing.assert_allclose(clipped, [-0.01, 0.0, 0.01, 0.01])
        self.assertEqual(constraint.get_config(), {"clip_value": 0.01})

        data = np.array([[0.0, 3.0], [10.0, 3.0]])
        scaler = RobustMinMaxScaler().fit(data)
        transformed = scaler.transform(data)
        np.testing.assert_allclose(transformed[:, 0], [-1.0, 1.0])
        np.testing.assert_allclose(transformed[:, 1], [0.0, 0.0])
        restored = scaler.inverse_transform(np.array([[0.0, 5.0], [-1.0, -4.0], [1.0, 9.0]]))
        np.testing.assert_allclose(restored, [[5.0, 3.0], [0.0, 3.0], [10.0, 3.0]])
        via_utils, _ = Utilities().transform_data(data)
        np.testing.assert_allclose(via_utils, transformed)
        self.assertIsNotNone(Utilities().get_optimizer_gan())
        self.assertIsNotNone(Utilities().get_optimizer_wgan())

    def test_wgan_training_contract(self):
        self.assertEqual(self.wgan.type, "wgan")
        self.assertEqual(len(self.wgan.g_hist), 2)
        self.assertEqual(len(self.wgan.d1_hist), 2)
        self.assertEqual(len(self.wgan.d2_hist), 2)
        self.assertTrue(np.all(np.isfinite(self.wgan.g_hist)))
        self.assertEqual(float(self.wgan.y_label_), 1.0)
        layer_names = {type(layer).__name__ for layer in self.wgan.adversary_one.layers}
        self.assertNotIn("Dropout", layer_names)
        self.assertNotIn("BatchNormalization", layer_names)
        critic_names = {type(layer).__name__ for layer in self.wgan.adversary_two.layers}
        self.assertNotIn("Dropout", critic_names)
        self.assertNotIn("BatchNormalization", critic_names)
        for weights in self.wgan.adversary_two.get_weights():
            self.assertLessEqual(float(np.max(np.abs(weights))), 0.01 + 1e-5)
        generator_peak = max(float(np.max(np.abs(weights))) for weights in self.wgan.adversary_one.get_weights())
        self.assertGreater(generator_peak, 0.01)

        synthetic = self.wgan.create_bulk(16)
        self.assertEqual(synthetic.shape, (16, 3))
        self.assertEqual(self.wgan.critic_scores_.shape, (16,))
        self.assertTrue(np.all(np.isfinite(synthetic)))
        self.assertTrue(np.all(np.isfinite(self.wgan.critic_scores_)))
        low = self.x.min(axis=0)
        high = self.x.max(axis=0)
        self.assertTrue(np.all(synthetic >= low - 1e-5))
        self.assertTrue(np.all(synthetic <= high + 1e-5))
        np.testing.assert_allclose(synthetic[:, 2], 5.0)
        self.assertGreater(float(np.std(synthetic[:, 0])), 1e-8)
        another = self.wgan.create_bulk(16)
        self.assertFalse(np.allclose(synthetic, another))

        self.wgan._sample_chunk = 2
        chunked = self.wgan.create_bulk(5)
        self.wgan._sample_chunk = 4096
        self.assertEqual(chunked.shape, (5, 3))
        self.assertEqual(len(self.wgan.critic_scores_), 5)

        stacked = self.wgan.get_gan()
        self.assertEqual(int(stacked.input.shape[-1]), 16)
        self.assertTrue(self.wgan.adversary_two.trainable)
        self.assertGreater(len(self.wgan.adversary_two.trainable_variables), 0)

    def test_gan_training_contract(self):
        self.assertEqual(self.gan.type, "gan")
        self.assertEqual(len(self.gan.g_hist), 2)
        self.assertTrue(np.all(np.isfinite(self.gan.history_["generator"])))
        names = {type(layer).__name__ for layer in self.gan.adversary_two.layers}
        self.assertIn("Dropout", names)
        peak = max(float(np.max(np.abs(weights))) for weights in self.gan.adversary_two.get_weights())
        self.assertGreater(peak, 0.01)
        synthetic = self.gan.create_bulk(8)
        self.assertEqual(synthetic.shape, (8, 3))
        self.assertTrue(np.all((self.gan.critic_scores_ >= 0.0) & (self.gan.critic_scores_ <= 1.0)))

    def test_case_insensitive_type_and_default_latent_size(self):
        model = Ganify(random_state=0)
        _fit(model, self.x[:, :2], self.y, type="WGAN")
        self.assertEqual(model.type, "wgan")
        self.assertEqual(model.random_dim, 100)
        self.assertEqual(int(model.adversary_one.input.shape[-1]), 100)
        self.assertEqual(int(model.adversary_one.output.shape[-1]), 2)

    def test_history_counts_steps_not_critic_updates(self):
        model = _fit(_small_model(), self.x, self.y, n_critic=5)
        self.assertEqual(len(model.g_hist), 2)
        self.assertEqual(len(model.d1_hist), 2)

    def test_multi_epoch_updates_weights_and_resets_on_refit(self):
        first = _fit(_small_model(0), self.x, self.y, epochs=1)
        second = _fit(_small_model(0), self.x, self.y, epochs=2)
        self.assertFalse(
            np.array_equal(
                first.adversary_one.get_weights()[0],
                second.adversary_one.get_weights()[0],
            )
        )
        self.assertFalse(
            np.array_equal(
                first.adversary_two.get_weights()[0],
                second.adversary_two.get_weights()[0],
            )
        )
        self.assertEqual(len(second.g_hist), 4)
        second.fit_data(self.x, self.y, epochs=1, batch_size=16, n_critic=1, verbose=0)
        self.assertEqual(len(second.g_hist), 2)
        self.assertIsNone(second.stopped_epoch_)

    def test_reproducible_fit_and_distinct_seeds(self):
        def train(seed, epochs=1):
            return _fit(_small_model(seed), self.x, self.y, epochs=epochs)

        left = train(0)
        right = train(0)
        other = train(1)
        for left_weights, right_weights in zip(
            left.adversary_one.get_weights(), right.adversary_one.get_weights()
        ):
            np.testing.assert_allclose(left_weights, right_weights, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(left.create_bulk(8), right.create_bulk(8), rtol=1e-5, atol=1e-5)
        self.assertFalse(
            np.array_equal(left.adversary_one.get_weights()[0], other.adversary_one.get_weights()[0])
        )

    def test_early_stopping(self):
        model = _fit(
            _small_model(),
            self.x,
            self.y,
            epochs=6,
            patience=1,
            min_delta=1e9,
        )
        self.assertEqual(model.stopped_epoch_, 2)
        self.assertEqual(len(model.g_hist), 4)
        self.assertEqual(model.epochs_ran_, 2)

    def test_dataframe_roundtrip_and_aliases(self):
        frame = pd.DataFrame(self.x, columns=["age", "income", "flag"])
        model = _fit(_small_model(), frame, pd.Series(self.y, name="target"))
        drawn = model.create_bulk(4, 1)
        self.assertIsInstance(drawn, pd.DataFrame)
        self.assertEqual(list(drawn.columns), ["age", "income", "flag"])
        self.assertEqual(model.create_bulk(length=3).shape, (3, 3))
        self.assertEqual(model.create_bulk(lenght=2, length=2).shape, (2, 3))
        with self.assertRaises(ValueError):
            model.create_bulk(lenght=2, length=3)
        with self.assertRaises(ValueError):
            model.create_bulk(0)
        chained = _small_model().fit_data(
            frame, self.y, epochs=1, batch_size=16, n_critic=1, verbose=0
        ).create_bulk(3)
        self.assertEqual(chained.shape, (3, 3))

    def test_numpy_output_requires_names(self):
        with self.assertRaises(ValueError):
            self.wgan.create_bulk(2, 1)
        with self.assertRaises(ValueError):
            self.wgan.create_bulk(2, output=2)

    def test_one_dimensional_input_and_one_hot_target(self):
        column = np.linspace(0.0, 1.0, 16)
        model = _fit(_small_model(), column, np.zeros(16), batch_size=8)
        self.assertEqual(model.create_bulk(4).shape, (4, 1))
        one_hot = np.tile(np.array([1.0, 0.0]), (32, 1))
        model = _fit(_small_model(), self.x, one_hot)
        np.testing.assert_allclose(model.y_label_, [1.0, 0.0])
        mixed = np.vstack([one_hot[:16], np.tile(np.array([0.0, 1.0]), (16, 1))])
        with self.assertRaises(RuntimeError):
            _fit(_small_model(), self.x, mixed)

    def test_constant_table_is_preserved(self):
        table = np.ones((8, 2))
        table[:, 1] = 3.0
        model = _fit(_small_model(), table, np.zeros(8), batch_size=4)
        synthetic = model.create_bulk(5)
        np.testing.assert_allclose(synthetic[:, 0], 1.0)
        np.testing.assert_allclose(synthetic[:, 1], 3.0)

    def test_numeric_strings_and_booleans(self):
        frame = pd.DataFrame({"a": ["1.5", "2.5", "3.5", "4.5"], "b": [True, False, True, False]})
        model = _fit(_small_model(), frame, np.ones(4), batch_size=4)
        self.assertEqual(model.n_features_, 2)
        self.assertEqual(model.create_bulk(2).shape, (2, 2))

    def test_remainder_batches(self):
        rows = np.random.default_rng(0).normal(size=(9, 2))
        model = _fit(_small_model(), rows, np.ones(9), batch_size=8)
        self.assertEqual(len(model.g_hist), 1)
        rows = np.random.default_rng(0).normal(size=(10, 2))
        model = _fit(_small_model(), rows, np.ones(10), batch_size=8)
        self.assertEqual(len(model.g_hist), 2)

    def test_validation_errors(self):
        model = _small_model()
        with self.assertRaises(RuntimeError):
            model.fit_data(np.ones((4, 2, 2)), np.ones(4), verbose=0)
        with self.assertRaises(RuntimeError):
            model.fit_data(self.x, np.array([0, 1] * 16), verbose=0)
        with self.assertRaises(RuntimeError):
            model.fit_data(self.x, np.ones(4), verbose=0)
        with self.assertRaises(RuntimeError):
            model.fit_data(self.x, self.y, cols_names=["only"], verbose=0)
        with self.assertRaises(RuntimeError):
            model.fit_data(self.x, self.y, type="vae", verbose=0)
        dirty = self.x.copy()
        dirty[0, 0] = np.nan
        with self.assertRaises(ValueError):
            model.fit_data(dirty, self.y, verbose=0)
        dirty[0, 0] = np.inf
        with self.assertRaises(ValueError):
            model.fit_data(dirty, self.y, verbose=0)
        with self.assertRaises(ValueError):
            model.fit_data(pd.DataFrame({"a": ["x", "y"], "b": [1.0, 2.0]}), np.ones(2), verbose=0)
        with self.assertRaises(ValueError):
            model.fit_data(np.empty((0, 2)), np.empty((0,)), verbose=0)
        with self.assertRaises(ValueError):
            model.fit_data(np.ones((1, 2)), np.ones(1), verbose=0)
        with self.assertRaises(ValueError):
            model.fit_data(self.x, self.y, epochs=0, verbose=0)
        with self.assertRaises(ValueError):
            model.fit_data(self.x, self.y, batch_size=True, verbose=0)
        with self.assertRaises(ValueError):
            model.fit_data(self.x, self.y, patience=0, verbose=0)
        with self.assertRaises(ValueError):
            model.fit_data(self.x, self.y, min_delta=-0.1, verbose=0)
        with self.assertRaises(ValueError):
            model.fit_data(self.x, self.y, label_flip=1.5, verbose=0)
        frame = pd.DataFrame(self.x, columns=list("abc"))
        shifted = pd.Series(self.y, index=np.arange(1, 33))
        with self.assertRaises(ValueError):
            model.fit_data(frame, shifted, verbose=0)
        duplicated = pd.DataFrame(np.ones((4, 2)), columns=["a", "a"])
        with self.assertRaises(ValueError):
            model.fit_data(duplicated, np.ones(4), verbose=0)
        fresh = Ganify()
        with self.assertRaises(RuntimeError):
            fresh.create_bulk(2)
        with self.assertRaises(RuntimeError):
            fresh.plot_performance(show=False)
        with self.assertRaises(FileNotFoundError):
            Ganify.load(ROOT / "missing-ganify-model")

    def test_large_batch_warns_and_still_trains(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = _fit(_small_model(), self.x[:4], self.y[:4], batch_size=32, epochs=2)
        messages = [
            str(item.message)
            for item in caught
            if issubclass(item.category, UserWarning) and "batch_size" in str(item.message)
        ]
        self.assertTrue(messages)
        self.assertEqual(len(model.g_hist), 2)

    def test_plot_save_load_and_verbose(self):
        model = _fit(_small_model(2), self.x, self.y, verbose=1)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            figure = directory / "nested" / "loss.png"
            model.plot_performance(path=figure, show=False)
            self.assertGreater(figure.stat().st_size, 0)
            destination = directory / "model"
            model.save(destination)
            self.assertTrue((destination / "metadata.json").is_file())
            restored = Ganify.load(destination)
            again = Ganify.load(destination)
            for original, loaded in zip(
                model.adversary_one.get_weights(), restored.adversary_one.get_weights()
            ):
                np.testing.assert_allclose(original, loaded)
            for original, loaded in zip(
                model.adversary_two.get_weights(), restored.adversary_two.get_weights()
            ):
                np.testing.assert_allclose(original, loaded)
            np.testing.assert_allclose(restored.create_bulk(6), again.create_bulk(6))
            self.assertEqual(restored.g_hist, model.g_hist)
            self.assertEqual(float(restored.y_label_), 1.0)
            drawn = restored.create_bulk(4)
            self.assertTrue(np.all(drawn[:, 2] == 5.0))
            with self.assertRaises(ValueError):
                model.save(figure)


if __name__ == "__main__":
    unittest.main()
