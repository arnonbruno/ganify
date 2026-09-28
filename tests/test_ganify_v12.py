import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import numpy as np
import pandas as pd

from ganify import Ganify, QuantileCopulaScaler
from ganify.utilities.utils import EmpiricalCDFCalibrator


def _small_model(seed=0, **kwargs):
    params = dict(random_state=seed, random_dim=8, max_units=32)
    params.update(kwargs)
    return Ganify(**params)


def _fit(model, x, y, **kwargs):
    params = dict(epochs=2, batch_size=16, n_critic=1, verbose=0, label_flip=0.0)
    params.update(kwargs)
    return model.fit_data(x, y, **params)


def _skewed_frame(seed=0, rows=64):
    rng = np.random.default_rng(seed)
    heavy = rng.pareto(1.5, size=rows) * 100.0 + 1.0
    flags = (rng.random(rows) < 0.05).astype(float)
    small = rng.normal(size=rows)
    return pd.DataFrame({"heavy": heavy, "flag": flags, "small": small})


class CopulaScalerTests(unittest.TestCase):
    def test_roundtrip_exact_and_uniform_spread(self):
        frame = _skewed_frame()
        scaler = QuantileCopulaScaler().fit(frame.to_numpy(dtype=float))
        transformed = scaler.transform(frame.to_numpy(dtype=float))
        self.assertTrue(np.all(transformed >= -1.0))
        self.assertTrue(np.all(transformed <= 1.0))
        # Skewed column spreads across the range instead of crushing at -1.
        self.assertGreater(float(np.median(transformed[:, 0])), -0.5)
        np.testing.assert_allclose(
            scaler.inverse_transform(transformed), frame.to_numpy(dtype=float), atol=1e-9
        )

    def test_point_masses_invert_to_tied_values(self):
        column = np.array([0.0] * 90 + [1.0] * 10).reshape(-1, 1)
        scaler = QuantileCopulaScaler().fit(column)
        low = scaler.inverse_transform(np.array([[-0.5]]))[0, 0]
        high = scaler.inverse_transform(np.array([[0.95]]))[0, 0]
        self.assertEqual(low, 0.0)
        self.assertEqual(high, 1.0)

    def test_constant_columns_and_subsampling(self):
        rng = np.random.default_rng(0)
        data = np.column_stack([np.full(50, 3.0), rng.normal(size=50)])
        scaler = QuantileCopulaScaler(max_quantiles=8).fit(data)
        transformed = scaler.transform(data)
        np.testing.assert_allclose(transformed[:, 0], 0.0)
        restored = scaler.inverse_transform(transformed)
        np.testing.assert_allclose(restored[:, 0], 3.0)
        with self.assertRaises(ValueError):
            QuantileCopulaScaler(max_quantiles=0)

    def test_save_dict_roundtrip(self):
        frame = _skewed_frame()
        scaler = QuantileCopulaScaler().fit(frame.to_numpy(dtype=float))
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "copula.npz")
            np.savez(path, **scaler.save_dict())
            with np.load(path) as archive:
                restored = QuantileCopulaScaler.load_dict(archive)
        probe = np.linspace(-1.0, 1.0, 11).reshape(-1, 1)
        probe = np.repeat(probe, 3, axis=1)
        np.testing.assert_allclose(
            restored.inverse_transform(probe), scaler.inverse_transform(probe)
        )
        np.testing.assert_array_equal(restored.step_, scaler.step_)

    def test_frozen_calibrator_is_pointwise_and_persistent(self):
        reference = np.column_stack(
            [np.linspace(-1.0, 1.0, 101), np.linspace(1.0, -1.0, 101)]
        )
        calibrator = EmpiricalCDFCalibrator().fit(reference)
        probe = reference[[5, 25, 75]]
        together = calibrator.transform(probe)
        separate = np.vstack(
            [calibrator.transform(row.reshape(1, -1)) for row in probe]
        )
        np.testing.assert_allclose(together, separate)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "calibrator.npz")
            np.savez(path, **calibrator.save_dict())
            with np.load(path) as archive:
                restored = EmpiricalCDFCalibrator.load_dict(archive)
        np.testing.assert_allclose(restored.transform(probe), together)


class CopulaModelTests(unittest.TestCase):
    def test_copula_fit_respects_ranges(self):
        frame = _skewed_frame()
        model = _fit(_small_model(0, scaler="copula"), frame, np.ones(len(frame)))
        self.assertEqual(len(model.g_hist), 8)
        synthetic = model.create_bulk(32, output=1)
        self.assertEqual(list(synthetic.columns), ["heavy", "flag", "small"])
        self.assertTrue(bool(((synthetic >= frame.min()) & (synthetic <= frame.max())).all().all()))

    def test_invalid_options_raise(self):
        with self.assertRaises(ValueError):
            Ganify(scaler="robust")
        with self.assertRaises(ValueError):
            Ganify(ema_decay=1.0)
        with self.assertRaises(ValueError):
            Ganify(ema_decay=-0.1)
        with self.assertRaises(ValueError):
            Ganify(ema_warmup_epochs=-1)
        with self.assertRaises(ValueError):
            Ganify(ema_warmup_epochs=True)
        frame = _skewed_frame()
        with self.assertRaises(ValueError):
            _fit(_small_model(0, calibrate_marginals=True), frame, np.ones(len(frame)))
        with self.assertRaises(ValueError):
            _fit(
                _small_model(0, inequality_pairs=[("heavy", "missing")]),
                frame,
                np.ones(len(frame)),
            )
        with self.assertRaises(ValueError):
            _fit(
                _small_model(0, inequality_pairs=[(0, 99)]),
                frame.to_numpy(),
                np.ones(len(frame)),
            )

    def test_calibration_matches_binary_rates_exactly(self):
        frame = _skewed_frame()
        model = _fit(
            _small_model(0, scaler="copula", calibrate_marginals=True),
            frame,
            np.ones(len(frame)),
        )
        synthetic = model.create_bulk(200, output=1)
        values = synthetic["flag"].to_numpy()
        self.assertTrue(bool(np.all((values == 0.0) | (values == 1.0))))
        train_rate = float(frame["flag"].mean())
        self.assertLessEqual(abs(float(values.mean()) - train_rate), 1.0 / len(values) + 1e-9)

    def test_rounding_and_inequalities(self):
        rng = np.random.default_rng(1)
        counts = rng.integers(0, 50, size=64).astype(float)
        users = np.minimum(counts, rng.integers(0, 50, size=64).astype(float))
        frame = pd.DataFrame({"events": counts, "users": users})
        model = _fit(
            _small_model(
                1,
                scaler="copula",
                calibrate_marginals=True,
                round_integers=True,
                inequality_pairs=[("users", "events")],
            ),
            frame,
            np.ones(len(frame)),
        )
        synthetic = model.create_bulk(64, output=1)
        self.assertTrue(
            bool(np.all(np.abs(synthetic.to_numpy() - np.round(synthetic.to_numpy())) < 1e-9))
        )
        self.assertTrue(bool(np.all(synthetic["users"] <= synthetic["events"])))

    def test_ema_sampling_and_persistence(self):
        frame = _skewed_frame()
        model = _fit(
            _small_model(2, ema_decay=0.9, ema_warmup_epochs=1),
            frame,
            np.ones(len(frame)),
        )
        self.assertIsNotNone(model._ema_generator)
        first = model.create_bulk(16)
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "model"
            model.save(destination)
            metadata = json.loads((destination / "metadata.json").read_text())
            self.assertEqual(metadata["format"], 3)
            self.assertTrue(metadata["has_ema"])
            restored = Ganify.load(destination)
            again = Ganify.load(destination)
        np.testing.assert_allclose(restored.create_bulk(16), again.create_bulk(16))
        # Restored EMA sampling matches a fresh twin with the same seed.
        twin = _fit(
            _small_model(2, ema_decay=0.9, ema_warmup_epochs=1),
            frame,
            np.ones(len(frame)),
        )
        np.testing.assert_allclose(first, twin.create_bulk(16), rtol=1e-5, atol=1e-5)

    def test_full_feature_save_load_roundtrip(self):
        frame = _skewed_frame()
        model = _fit(
            _small_model(
                3,
                scaler="copula",
                ema_decay=0.9,
                calibrate_marginals=True,
                round_integers=True,
            ),
            frame,
            np.ones(len(frame)),
        )
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "model"
            model.save(destination)
            self.assertTrue((destination / "scaler_copula.npz").is_file())
            self.assertTrue((destination / "ema_generator.weights.h5").is_file())
            restored = Ganify.load(destination)
            again = Ganify.load(destination)
        pd.testing.assert_frame_equal(
            restored.create_bulk(32, output=1), again.create_bulk(32, output=1)
        )
        self.assertEqual(restored.scaler_name, "copula")
        self.assertTrue(restored.calibrate_marginals)
        self.assertTrue(restored.round_integers)

    def test_legacy_format_still_loads(self):
        frame = _skewed_frame()
        model = _fit(_small_model(4), frame, np.ones(len(frame)))
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "model"
            model.save(destination)
            meta_path = destination / "metadata.json"
            metadata = json.loads(meta_path.read_text())
            metadata["format"] = 1
            for key in (
                "scaler", "ema_decay", "ema_warmup_epochs", "has_ema",
                "calibrate_marginals", "round_integers", "inequality_pairs",
                "int_mask", "pair_idx",
            ):
                metadata.pop(key, None)
            meta_path.write_text(json.dumps(metadata))
            restored = Ganify.load(destination)
        self.assertEqual(restored.scaler_name, "minmax")
        self.assertIsNone(restored._ema_generator)
        self.assertEqual(restored.create_bulk(4).shape, (4, 3))

    def test_compiled_loop_contract_and_reproducibility(self):
        frame = _skewed_frame()
        left = _fit(
            _small_model(5, scaler="copula"), frame, np.ones(len(frame)), compile=True
        )
        right = _fit(
            _small_model(5, scaler="copula"), frame, np.ones(len(frame)), compile=True
        )
        self.assertEqual(len(left.g_hist), 8)
        self.assertEqual(len(left.d1_hist), 8)
        for left_weights, right_weights in zip(
            left.adversary_one.get_weights(), right.adversary_one.get_weights()
        ):
            np.testing.assert_allclose(left_weights, right_weights, rtol=1e-5, atol=1e-5)
        synthetic = left.create_bulk(16, output=1)
        self.assertTrue(bool(((synthetic >= frame.min()) & (synthetic <= frame.max())).all().all()))
        gan = _fit(
            _small_model(6), frame, np.ones(len(frame)), type="gan", compile=True, epochs=1
        )
        self.assertEqual(gan.create_bulk(4).shape, (4, 3))

    def test_compiled_ema_warmup(self):
        frame = _skewed_frame()
        model = _fit(
            _small_model(7, ema_decay=0.9, ema_warmup_epochs=5),
            frame,
            np.ones(len(frame)),
            compile=True,
        )
        # Warmup longer than training: EMA tracks the live weights.
        for ema_weights, live_weights in zip(
            model._ema_generator.get_weights(), model.adversary_one.get_weights()
        ):
            np.testing.assert_allclose(ema_weights, live_weights, rtol=1e-5, atol=1e-5)

    def test_calibrated_sampling_is_call_size_invariant(self):
        frame = _skewed_frame(rows=96)
        model = _fit(
            _small_model(
                8,
                scaler="copula",
                calibrate_marginals=True,
                calibration_size=256,
            ),
            frame,
            np.ones(len(frame)),
        )
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "model"
            model.save(destination)
            one_call = Ganify.load(destination)
            chunks = Ganify.load(destination)
        expected = one_call.create_bulk(80)
        actual = np.vstack([chunks.create_bulk(31), chunks.create_bulk(49)])
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_validation_checkpoint_and_optimizer_config(self):
        frame = _skewed_frame(rows=96)
        calls = []

        def score(validation, generated):
            calls.append((validation.shape, generated.shape))
            return [3.0, 1.0, 2.0][len(calls) - 1]

        model = _fit(
            _small_model(9),
            frame,
            np.ones(len(frame)),
            epochs=3,
            validation_fraction=0.2,
            selection_metric="validation",
            selection_callback=score,
            generator_learning_rate=0.0003,
            adversary_learning_rate=0.0001,
            beta_1=0.1,
            beta_2=0.8,
        )
        self.assertEqual(model.best_epoch_, 2)
        self.assertEqual(len(model.validation_history_), 3)
        self.assertEqual(model.generator_learning_rate_, 0.0003)
        self.assertEqual(model.adversary_learning_rate_, 0.0001)
        self.assertEqual(model.beta_1_, 0.1)
        self.assertEqual(model.beta_2_, 0.8)

    def test_wgan_label_flip_defaults_to_zero(self):
        frame = _skewed_frame()
        model = Ganify(random_state=10, random_dim=8, max_units=32)
        model.fit_data(
            frame,
            np.ones(len(frame)),
            epochs=1,
            batch_size=16,
            n_critic=1,
            verbose=0,
        )
        self.assertEqual(model.label_flip_, 0.0)


if __name__ == "__main__":
    unittest.main()
