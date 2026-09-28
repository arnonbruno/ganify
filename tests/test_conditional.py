"""Focused tests for the standalone conditional mixed-type GAN engine."""

import json
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

from ganify.conditional import (
    ConditionSampler,
    ConditionalCritic,
    ConditionalGANEngine,
    ConditionalGenerator,
)
from ganify import Ganify
from ganify.preprocessing import TableTransformer


def _mixed_frame(rows=24):
    return pd.DataFrame(
        {
            "amount": np.linspace(-1.25, 3.75, rows),
            "segment": np.resize(np.array(["common", "other", "rare"]), rows),
            "flag": np.resize(np.array([False, True]), rows),
            "nullable": np.resize(
                np.array(["present", None, "present"], dtype=object), rows
            ),
            "constant": ["fixed"] * rows,
        }
    )


class ConditionSamplerTests(unittest.TestCase):
    def test_log_frequency_keeps_rare_mode_and_is_deterministic(self):
        frame = pd.DataFrame(
            {
                "value": np.linspace(0.0, 1.0, 100),
                "mode": ["rare"] + ["common"] * 99,
            }
        )
        transformer = TableTransformer(
            schema_overrides={"value": "continuous", "mode": "categorical"}
        )
        encoded = transformer.fit_transform(frame)
        left = ConditionSampler(random_state=17).fit(encoded, transformer)
        right = ConditionSampler(random_state=17).fit(encoded, transformer)

        left_batch = left.sample_training(300)
        right_batch = right.sample_training(300)
        for left_value, right_value in zip(left_batch, right_batch):
            np.testing.assert_array_equal(left_value, right_value)

        group = left.groups[0]
        chosen = [
            group.labels[int(value)]
            for value in left_batch.value_indices.tolist()
        ]
        self.assertIn("rare", chosen)
        for row, label in zip(left_batch.real_indices, chosen):
            self.assertEqual(frame.iloc[int(row)]["mode"], label)

        # Persistence includes the current RNG state, not merely the seed.
        restored = ConditionSampler.from_json(
            json.dumps(left.to_dict(), allow_nan=False)
        )
        expected = left.sample_training(32)
        actual = restored.sample_training(32)
        for expected_value, actual_value in zip(expected, actual):
            np.testing.assert_array_equal(expected_value, actual_value)

    def test_target_missing_masks_and_quantile_bins_are_groups(self):
        frame = _mixed_frame()
        transformer = TableTransformer(
            schema_overrides={
                "amount": "continuous",
                "segment": "categorical",
                "nullable": "categorical",
            }
        )
        encoded = transformer.fit_transform(frame)
        target = pd.Series(
            np.resize(np.array(["alpha", "beta", "gamma"]), len(frame)),
            name="outcome",
        )
        sampler = ConditionSampler(
            continuous_bins=4, random_state=4
        ).fit(encoded, transformer, target)
        kinds = {group.kind for group in sampler.groups}
        self.assertIn("target", kinds)
        self.assertIn("missing", kinds)
        self.assertIn("continuous_bin", kinds)
        condition = sampler.encode_conditions(
            {"segment": "rare", "outcome": "gamma"}, rows=5
        )
        self.assertEqual(condition.shape, (5, sampler.condition_dim))
        self.assertTrue(np.all(condition.sum(axis=1) >= 2.0))


class ConditionalNetworkTests(unittest.TestCase):
    def test_generator_respects_every_head_activation(self):
        frame = _mixed_frame()
        transformer = TableTransformer(
            schema_overrides={
                "amount": "continuous",
                "segment": "categorical",
                "nullable": "categorical",
            }
        )
        encoded = transformer.fit_transform(frame)
        sampler = ConditionSampler(random_state=2).fit(encoded, transformer)
        generator = ConditionalGenerator(
            transformer.head_metadata,
            noise_dim=6,
            condition_dim=sampler.condition_dim,
            hidden_dims=(16,),
            random_state=2,
        )
        output = generator(
            [
                tf.zeros((7, 6), dtype=tf.float32),
                tf.zeros((7, sampler.condition_dim), dtype=tf.float32),
            ],
            training=False,
        ).numpy()
        self.assertEqual(output.shape, (7, transformer.output_dim_))
        self.assertTrue(bool(np.isfinite(output).all()))
        for head in transformer.head_metadata:
            values = output[:, head.slice]
            if head.activation == "tanh":
                self.assertTrue(bool(np.all(np.abs(values) <= 1.0)))
            elif head.activation == "sigmoid":
                self.assertTrue(
                    bool(np.all((values >= 0.0) & (values <= 1.0)))
                )
            elif head.activation == "softmax":
                np.testing.assert_allclose(
                    values.sum(axis=1), 1.0, atol=1e-6
                )

        critic = ConditionalCritic(
            transformer.output_dim_,
            condition_dim=sampler.condition_dim,
            hidden_dims=(16,),
            spectral_normalization=True,
            random_state=2,
        )
        scores = critic(
            [output, np.zeros((7, sampler.condition_dim), dtype=np.float32)]
        ).numpy()
        self.assertEqual(scores.shape, (7, 1))
        self.assertTrue(bool(np.isfinite(scores).all()))


class ConditionalEngineTests(unittest.TestCase):
    def _fit(self):
        frame = _mixed_frame()
        target = pd.Series(
            np.resize(np.array(["alpha", "beta", "gamma"]), len(frame)),
            name="outcome",
        )
        engine = ConditionalGANEngine(
            random_state=11,
            noise_dim=8,
            generator_dims=(16,),
            critic_dims=(16,),
            batch_size=12,
            epochs=1,
            n_critic=1,
            ema_decay=0.5,
            schema_overrides={
                "amount": "continuous",
                "segment": "categorical",
                "nullable": "categorical",
            },
        )
        engine.fit(frame, target, verbose=0)
        return frame, target, engine

    def test_conditions_and_multiclass_target_are_honored(self):
        _, _, engine = self._fit()
        synthetic, target = engine.sample(
            18,
            conditions={
                "segment": "rare",
                "nullable": None,
                "amount": 0.25,
                "outcome": "gamma",
            },
            return_target=True,
        )
        self.assertEqual(len(synthetic), 18)
        self.assertTrue((synthetic["segment"] == "rare").all())
        self.assertTrue(synthetic["nullable"].isna().all())
        np.testing.assert_allclose(synthetic["amount"], 0.25, atol=1e-10)
        self.assertEqual(set(target), {"gamma"})
        self.assertEqual(target.name, "outcome")
        self.assertEqual(len(engine.history_["generator"]), 1)

    def test_save_load_has_deterministic_sampling_parity(self):
        _, _, engine = self._fit()
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "conditional"
            engine.save(destination)
            expected_frame, expected_target = engine.sample(
                12, return_target=True
            )
            restored = ConditionalGANEngine.load(destination)
            actual_frame, actual_target = restored.sample(
                12, return_target=True
            )
        pd.testing.assert_frame_equal(expected_frame, actual_frame)
        pd.testing.assert_series_equal(expected_target, actual_target)
        self.assertEqual(restored.history_, engine.history_)


class ConditionalGanifyFacadeTests(unittest.TestCase):
    def test_modern_fit_auto_selects_conditional_engine(self):
        frame = _mixed_frame()
        target = pd.Series(
            np.resize(np.array(["alpha", "beta", "gamma"]), len(frame)),
            name="outcome",
        )
        model = Ganify(
            random_state=21,
            random_dim=8,
            max_units=32,
            ema_decay=0.5,
        )
        model.fit(
            frame,
            target,
            epochs=1,
            batch_size=12,
            n_critic=1,
            schema_overrides={
                "amount": "continuous",
                "segment": "categorical",
                "nullable": "categorical",
            },
            verbose=0,
        )
        self.assertEqual(model.type, "conditional_wgan")
        synthetic, labels = model.sample(
            10,
            conditions={"segment": "rare", "outcome": "gamma"},
            return_target=True,
        )
        self.assertTrue((synthetic["segment"] == "rare").all())
        self.assertEqual(set(labels), {"gamma"})
        self.assertEqual(model.create_bulk(4, output=1).shape, (4, 5))

    def test_facade_conditional_save_load(self):
        frame = _mixed_frame()
        target = pd.Series(
            np.resize(np.array(["alpha", "beta"]), len(frame)),
            name="target",
        )
        model = Ganify(random_state=22, random_dim=8, max_units=32)
        model.fit(
            frame,
            target,
            conditional=True,
            epochs=1,
            batch_size=12,
            n_critic=1,
            schema_overrides={
                "amount": "continuous",
                "segment": "categorical",
                "nullable": "categorical",
            },
            verbose=0,
        )
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "facade"
            model.save(destination)
            left = Ganify.load(destination)
            right = Ganify.load(destination)
        left_frame, left_target = left.sample(8, return_target=True)
        right_frame, right_target = right.sample(8, return_target=True)
        pd.testing.assert_frame_equal(left_frame, right_frame)
        pd.testing.assert_series_equal(left_target, right_target)

    def test_facade_forwards_architecture_and_sampling_options(self):
        frame = _mixed_frame()
        model = Ganify(random_state=23, random_dim=8, max_units=32)
        model.fit(
            frame,
            conditional=True,
            epochs=1,
            batch_size=12,
            n_critic=1,
            critic_architecture="fourier",
            fourier_features=4,
            pac=2,
            warmup_epochs=1,
            interaction_weight=0.1,
            feature_matching_weight=0.1,
            swa=True,
            schema_overrides={
                "amount": "continuous",
                "segment": "categorical",
                "nullable": "categorical",
            },
            verbose=0,
        )
        engine = model._conditional_engine
        self.assertEqual(engine.critic_architecture, "fourier")
        self.assertEqual(engine.pac, 2)
        self.assertEqual(engine.warmup_epochs_ran_, 1)
        self.assertEqual(model.sample(6, weight_source="swa").shape, (6, 5))


if __name__ == "__main__":
    unittest.main()
