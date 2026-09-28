"""Fast coverage for architecture and anti-collapse ablations."""

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import numpy as np
import pandas as pd
import tensorflow as tf

from benchmarks.ganify_bench.ablation import (
    AblationBudget,
    AblationRunner,
    load_ablation_matrix,
)
from ganify.conditional import ConditionalCritic, ConditionalGANEngine


def _frame(rows=16):
    return pd.DataFrame(
        {
            "x": np.linspace(-1.0, 1.0, rows),
            "y": np.sin(np.linspace(-1.0, 1.0, rows)),
            "kind": np.resize(np.array(["a", "b"]), rows),
        }
    )


def _options(**updates):
    values = {
        "random_state": 7,
        "noise_dim": 4,
        "generator_dims": (8,),
        "critic_dims": (8,),
        "batch_size": 8,
        "epochs": 1,
        "n_critic": 1,
        "schema_overrides": {
            "x": "continuous",
            "y": "continuous",
            "kind": "categorical",
        },
    }
    values.update(updates)
    return values


class CriticArchitectureTests(unittest.TestCase):
    def test_every_architecture_has_finite_shape_and_gradients(self):
        features = tf.reshape(tf.linspace(-1.0, 1.0, 48), (8, 6))
        condition = tf.reshape(tf.linspace(1.0, -1.0, 16), (8, 2))
        for architecture in ("residual", "attention", "fourier"):
            with self.subTest(architecture=architecture):
                critic = ConditionalCritic(
                    6,
                    condition_dim=2,
                    hidden_dims=(16, 16),
                    architecture=architecture,
                    pac=2,
                    attention_heads=4,
                    fourier_features=8,
                    random_state=3,
                )
                with tf.GradientTape() as tape:
                    tape.watch(features)
                    score = critic(
                        [features, condition], training=True
                    )
                    loss = tf.reduce_mean(score)
                gradients = tape.gradient(
                    loss, [features] + critic.trainable_variables
                )
                self.assertEqual(score.shape, (4, 1))
                self.assertTrue(
                    all(gradient is not None for gradient in gradients)
                )
                self.assertTrue(
                    all(
                        bool(
                            tf.reduce_all(tf.math.is_finite(gradient)).numpy()
                        )
                        for gradient in gradients
                    )
                )

    def test_attention_and_fourier_match_reference_parameter_budget(self):
        reference = ConditionalCritic(
            8,
            condition_dim=4,
            hidden_dims=(64, 64),
            architecture="residual",
        )
        reference(
            [tf.zeros((1, 8)), tf.zeros((1, 4))], training=False
        )
        target = sum(
            int(tf.size(variable))
            for variable in reference.trainable_variables
        )
        for architecture in ("attention", "fourier"):
            with self.subTest(architecture=architecture):
                critic = ConditionalCritic(
                    8,
                    condition_dim=4,
                    hidden_dims=(64, 64),
                    architecture=architecture,
                    parameter_budget=target,
                    attention_heads=4,
                    fourier_features=16,
                )
                critic(
                    [tf.zeros((1, 8)), tf.zeros((1, 4))],
                    training=False,
                )
                actual = sum(
                    int(tf.size(variable))
                    for variable in critic.trainable_variables
                )
                self.assertLess(abs(actual - target) / target, 0.05)

    def test_pacgan_packs_features_and_conditions_and_validates_batch(self):
        critic = ConditionalCritic(
            3, condition_dim=2, hidden_dims=(8,), pac=2
        )
        features = np.arange(18, dtype=np.float32).reshape(6, 3)
        condition = np.arange(12, dtype=np.float32).reshape(6, 2)
        packed_features, packed_condition = critic.pack_inputs(
            features, condition
        )
        np.testing.assert_array_equal(
            packed_features.numpy(), features.reshape(3, 6)
        )
        np.testing.assert_array_equal(
            packed_condition.numpy(), condition.reshape(3, 4)
        )
        self.assertEqual(critic([features, condition]).shape, (3, 1))
        with self.assertRaises(ValueError):
            critic(
                [
                    np.zeros((5, 3), dtype=np.float32),
                    np.zeros((5, 2), dtype=np.float32),
                ]
            )
        with self.assertRaises(ValueError):
            ConditionalGANEngine(batch_size=7, pac=2)


class AntiCollapseTrainingTests(unittest.TestCase):
    def test_warmup_changes_generator_and_sampling_stays_noise_driven(self):
        frame = _frame()
        baseline = ConditionalGANEngine(**_options()).fit(frame)
        warmed = ConditionalGANEngine(
            **_options(warmup_epochs=1, warmup_mask_probability=0.5)
        ).fit(frame)
        self.assertEqual(len(warmed.history["warmup"]), 1)
        self.assertTrue(
            any(
                not np.array_equal(left, right)
                for left, right in zip(
                    baseline.generator_.get_weights(),
                    warmed.generator_.get_weights(),
                )
            )
        )
        # The warmup encoder is not part of generation; inference remains
        # generator(noise, condition).
        warmed.warmup_encoder_ = None
        warmed.set_sampling_seed(10)
        first = warmed.sample(8)
        warmed.set_sampling_seed(11)
        second = warmed.sample(8)
        self.assertFalse(first.equals(second))

    def test_auxiliary_losses_and_swa_swag_are_finite_and_deterministic(self):
        engine = ConditionalGANEngine(
            **_options(
                epochs=2,
                interaction_weight=0.2,
                feature_matching_weight=0.2,
                swa=True,
                swag=True,
                swag_max_rank=2,
            )
        ).fit(_frame())
        for name in (
            "generator",
            "interaction",
            "feature_matching",
        ):
            self.assertTrue(bool(np.isfinite(engine.history[name]).all()))
        self.assertEqual(engine.swa_collections_, 2)
        self.assertEqual(engine.swag_collections_, 2)
        for left, right in zip(engine.swa_weights(), engine.swa_weights()):
            np.testing.assert_array_equal(left, right)
        for left, right in zip(
            engine.sample_swag_weights(23),
            engine.sample_swag_weights(23),
        ):
            np.testing.assert_array_equal(left, right)

        for source in ("swa", "swag"):
            engine.set_sampling_seed(19)
            first = engine.sample(
                6, weight_source=source, weight_seed=29
            )
            engine.set_sampling_seed(19)
            second = engine.sample(
                6, weight_source=source, weight_seed=29
            )
            pd.testing.assert_frame_equal(first, second)

    def test_all_options_and_weight_collection_survive_save_load(self):
        engine = ConditionalGANEngine(
            **_options(
                critic_architecture="fourier",
                fourier_features=4,
                pac=2,
                warmup_epochs=1,
                interaction_weight=0.1,
                feature_matching_weight=0.1,
                swa=True,
                swag=True,
                swag_max_rank=2,
            )
        ).fit(_frame())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model"
            engine.save(path)
            expected = engine.sample(
                6, weight_source="swag", weight_seed=31
            )
            restored = ConditionalGANEngine.load(path)
            actual = restored.sample(
                6, weight_source="swag", weight_seed=31
            )
        self.assertEqual(restored._config_dict(), engine._config_dict())
        self.assertEqual(restored.swa_collections_, engine.swa_collections_)
        self.assertEqual(restored.swag_collections_, engine.swag_collections_)
        pd.testing.assert_frame_equal(expected, actual)


class AblationMatrixTests(unittest.TestCase):
    def test_checked_matrix_contains_every_requested_factor(self):
        configs = load_ablation_matrix()
        names = {config.name for config in configs}
        self.assertTrue(
            {
                "residual",
                "attention",
                "fourier",
                "pacgan",
                "warmup",
                "interaction",
                "swa",
                "swag",
            }.issubset(names)
        )
        for config in configs:
            self.assertFalse(
                {
                    "marginal_moment_weight",
                    "moment_weight",
                    "marginal_loss_weight",
                }.intersection(config.engine)
            )
        runner = AblationRunner(
            configs,
            budget=AblationBudget(
                epochs=1, batch_size=8, n_critic=1
            ),
            model_seeds=(3,),
            sample_seeds=(5,),
        )
        self.assertEqual(len(runner.models), len(configs))


if __name__ == "__main__":
    unittest.main()
