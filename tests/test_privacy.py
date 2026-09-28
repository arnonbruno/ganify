"""Fast deterministic tests for privacy attacks and optional DP training."""

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import pandas as pd

from ganify import Ganify
from ganify.conditional import ConditionalGANEngine
from ganify.preprocessing import TableTransformer
from ganify.privacy import (
    DPConfig,
    PrivacyAccountant,
    audit_privacy,
    canary_extraction_audit,
    clip_and_noise_gradients,
    evaluate_privacy_gates,
    exact_match_audit,
    export_smoothed_marginals,
    formal_dp_eligible,
    inspect_artifact_risk,
    membership_inference_attack,
)


def _privacy_frame(rows=12):
    return pd.DataFrame(
        {
            "x": np.linspace(-1.0, 1.0, rows),
            "group": np.resize(np.asarray(["a", "b", "c"]), rows),
        }
    )


def _dp_engine(config):
    return ConditionalGANEngine(
        random_state=19,
        noise_dim=4,
        generator_dims=(8,),
        critic_dims=(8,),
        batch_size=4,
        epochs=1,
        n_critic=1,
        gradient_penalty=0.0,
        feature_matching_weight=0.0,
        interaction_weight=0.0,
        warmup_epochs=0,
        dp_config=config,
    )


class AttackAuditTests(unittest.TestCase):
    def test_membership_detects_memorization_and_independent_is_near_chance(self):
        rng = np.random.default_rng(31)
        train = pd.DataFrame(rng.normal(size=(180, 3)), columns=list("abc"))
        holdout = pd.DataFrame(rng.normal(size=(180, 3)), columns=list("abc"))
        independent = pd.DataFrame(
            rng.normal(size=(180, 3)), columns=list("abc")
        )
        memorized = membership_inference_attack(
            train,
            holdout,
            train.copy(),
            k=1,
            bootstrap_replicates=20,
            random_state=4,
        )
        control = membership_inference_attack(
            train,
            holdout,
            independent,
            k=3,
            bootstrap_replicates=20,
            random_state=4,
        )
        self.assertGreater(memorized.roc_auc.value, 0.99)
        self.assertGreater(memorized.tpr_at_1pct_fpr.value, 0.99)
        self.assertLess(abs(control.roc_auc.value - 0.5), 0.15)
        self.assertIsNotNone(control.roc_auc.ci_low)

    def test_exact_matches_adjust_for_holdout_collisions(self):
        train = pd.DataFrame({"x": [1, 1, 1], "kind": ["a"] * 3})
        holdout = train.copy()
        synthetic = train.copy()
        result = exact_match_audit(
            train, holdout, synthetic, bootstrap_replicates=5
        )
        self.assertEqual(result.exact_match_rate.value, 1.0)
        self.assertEqual(result.train_only_exact_match_rate.value, 0.0)
        self.assertEqual(result.collision_adjusted_exact_match_rate, 0.0)

    def test_canary_extraction_and_candidate_exposure(self):
        canary = pd.DataFrame({"x": [999.0], "kind": ["canary"]})
        decoys = pd.DataFrame(
            {"x": [100.0, 200.0, 300.0], "kind": ["d1", "d2", "d3"]}
        )
        synthetic = pd.concat(
            [canary, pd.DataFrame({"x": [0.0], "kind": ["ordinary"]})],
            ignore_index=True,
        )
        result = canary_extraction_audit(
            synthetic,
            canary,
            decoys=decoys,
            categorical_columns=["kind"],
            bootstrap_replicates=5,
        )
        self.assertEqual(result.extraction_rate.value, 1.0)
        self.assertEqual(result.records[0].exact_count, 1)
        self.assertAlmostEqual(result.records[0].exposure_bits, 2.0)

    def test_structured_report_states_attack_limitations(self):
        rng = np.random.default_rng(8)
        train = pd.DataFrame(
            {"x": rng.normal(size=30), "secret": rng.integers(0, 2, 30)}
        )
        holdout = pd.DataFrame(
            {"x": rng.normal(size=30), "secret": rng.integers(0, 2, 30)}
        )
        synthetic = train.sample(
            n=30, replace=True, random_state=3
        ).reset_index(drop=True)
        report = audit_privacy(
            train,
            holdout,
            synthetic,
            sensitive_columns=["secret"],
            categorical_columns=["secret"],
            bootstrap_replicates=3,
            n_singling_attacks=5,
            random_state=2,
        )
        self.assertFalse(report.as_dict()["contains_privacy_guarantee"])
        self.assertIn("not privacy", report.proximity.limitations[0].lower())
        self.assertIn(
            "membership_roc_auc", set(report.to_frame()["metric"])
        )


class ArtifactRiskTests(unittest.TestCase):
    def test_empirical_quantiles_categories_and_row_state_are_detected(self):
        frame = _privacy_frame()
        transformer = TableTransformer(
            schema_overrides={"x": "continuous", "group": "categorical"}
        ).fit(frame)
        state = {
            "transformer": transformer.to_dict(),
            "sampler": {"output_matrix": [[0.0, 1.0]]},
        }
        report = inspect_artifact_risk(state)
        self.assertTrue(report.empirical_quantile_state_detected)
        self.assertTrue(report.category_state_detected)
        self.assertTrue(report.training_record_state_detected)
        self.assertTrue(report.model_artifacts_more_sensitive_than_samples)
        self.assertFalse(report.safe_for_public_release)

    def test_smoothed_export_is_coarse_and_refuses_a_dp_label(self):
        exported = export_smoothed_marginals(
            _privacy_frame(),
            categorical_columns=["group"],
            bins=4,
            probability_quantum=0.01,
        )
        self.assertEqual(
            exported["format"], "ganify-smoothed-quantized-marginals"
        )
        self.assertFalse(exported["privacy"]["differentially_private"])
        self.assertIn("warning", exported["privacy"])


class DPPrimitiveTests(unittest.TestCase):
    def test_global_clipping_noise_and_accounting_are_deterministic(self):
        gradients = (
            np.asarray([[3.0, 4.0], [0.0, 2.0]]),
            np.asarray([[0.0], [0.0]]),
        )
        clipped, norms, factors = clip_and_noise_gradients(
            gradients,
            l2_norm_clip=1.0,
            noise_multiplier=0.0,
            random_state=7,
        )
        np.testing.assert_allclose(norms, [5.0, 2.0])
        np.testing.assert_allclose(factors, [0.2, 0.5])
        np.testing.assert_allclose(clipped[0], [0.3, 0.9])
        noisy_left = clip_and_noise_gradients(
            gradients,
            l2_norm_clip=1.0,
            noise_multiplier=1.0,
            random_state=9,
        )[0]
        noisy_right = clip_and_noise_gradients(
            gradients,
            l2_norm_clip=1.0,
            noise_multiplier=1.0,
            random_state=9,
        )[0]
        for left, right in zip(noisy_left, noisy_right):
            np.testing.assert_array_equal(left, right)

        accountant = PrivacyAccountant(2.0, orders=(2.0, 4.0, 8.0))
        accountant.record_step(sample_rate=0.1)
        first = accountant.epsilon(1e-5)
        accountant.record_step(sample_rate=0.1)
        self.assertGreater(accountant.epsilon(1e-5), first)
        restored = PrivacyAccountant.from_state_dict(
            accountant.state_dict()
        )
        self.assertEqual(restored.steps, 2)
        self.assertAlmostEqual(
            restored.epsilon(1e-5), accountant.epsilon(1e-5)
        )


class DPEngineTests(unittest.TestCase):
    def test_incompatible_training_options_raise(self):
        with self.assertRaisesRegex(ValueError, "gradient_penalty"):
            ConditionalGANEngine(dp_config=DPConfig())
        with self.assertRaisesRegex(ValueError, "pac must be 1"):
            ConditionalGANEngine(
                gradient_penalty=0.0,
                pac=2,
                batch_size=4,
                dp_config=DPConfig(),
            )

    def test_private_preprocessing_refuses_formal_claim(self):
        frame = _privacy_frame(8)
        engine = _dp_engine(
            DPConfig(
                noise_multiplier=2.0,
                delta=1e-3,
                noise_seed=17,
            )
        )
        engine.fit(
            frame,
            schema_overrides=None,
            verbose=0,
        )
        report = engine.privacy_report_
        self.assertEqual(report["scope"], "training_only_dp")
        self.assertFalse(report["formal_dp"])
        self.assertIsNone(report["epsilon_claim"])
        self.assertIsNotNone(report["accountant"]["epsilon"])
        self.assertFalse(formal_dp_eligible(report))
        self.assertEqual(engine.sampler_.output_matrix_.shape[0], 1)

    def test_public_fixed_preprocessor_and_accountant_persist(self):
        frame = _privacy_frame(8)
        preprocessor = TableTransformer(
            schema_overrides={"x": "continuous", "group": "categorical"}
        ).fit(frame)
        engine = _dp_engine(
            DPConfig(
                noise_multiplier=2.0,
                delta=1e-3,
                preprocessing_public=True,
            )
        )
        engine.fit(frame, preprocessor=preprocessor, verbose=0)
        self.assertTrue(formal_dp_eligible(engine.privacy_report_))
        formal_gate = evaluate_privacy_gates(
            None,
            {
                "rules": [
                    {
                        "id": "formal",
                        "metric": "formal_dp",
                        "operator": "==",
                        "threshold": 1.0,
                        "pillar": "privacy_dp",
                    }
                ]
            },
            dp_report=engine.privacy_report_,
        )
        self.assertTrue(formal_gate.passed)
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "dp"
            engine.save(destination)
            expected = engine.sample(4)
            restored = ConditionalGANEngine.load(destination)
            actual = restored.sample(4)
        pd.testing.assert_frame_equal(expected, actual)
        self.assertEqual(
            restored.privacy_accountant_.steps,
            engine.privacy_accountant_.steps,
        )
        self.assertTrue(formal_dp_eligible(restored.privacy_report_))

    def test_ganify_facade_exposes_dp_boundary_report(self):
        frame = _privacy_frame(8)
        preprocessor = TableTransformer(
            schema_overrides={"x": "continuous", "group": "categorical"}
        ).fit(frame)
        model = Ganify(random_state=20, random_dim=4, max_units=8)
        model.fit(
            frame,
            conditional=True,
            epochs=1,
            batch_size=4,
            n_critic=1,
            gradient_penalty=0.0,
            dp_config=DPConfig(
                noise_multiplier=2.0,
                delta=1e-3,
                preprocessing_public=True,
            ),
            preprocessor=preprocessor,
            generator_dims=(8,),
            critic_dims=(8,),
            verbose=0,
        )
        self.assertTrue(formal_dp_eligible(model.privacy_report_))
        self.assertEqual(model.privacy_report_["scope"], "end_to_end")


class PrivacyGateTests(unittest.TestCase):
    def test_missing_attack_and_unverified_formal_dp_fail_closed(self):
        records = pd.DataFrame(
            [
                {
                    "pillar": "privacy_attack",
                    "metric": "membership_roc_auc",
                    "value": 0.51,
                },
                {
                    "pillar": "privacy_dp",
                    "metric": "formal_dp",
                    "value": 1.0,
                },
            ]
        )
        config = {
            "rules": [
                {
                    "id": "membership",
                    "metric": "membership_roc_auc",
                    "operator": "<=",
                    "threshold": 0.6,
                    "pillar": "privacy_attack",
                },
                {
                    "id": "canary",
                    "metric": "canary_extraction_rate",
                    "operator": "==",
                    "threshold": 0.0,
                    "pillar": "privacy_attack",
                },
                {
                    "id": "formal",
                    "metric": "formal_dp",
                    "operator": "==",
                    "threshold": 1.0,
                    "pillar": "privacy_dp",
                },
            ]
        }
        result = evaluate_privacy_gates(records, config)
        self.assertFalse(result.passed)
        decisions = {rule.rule_id: rule for rule in result.rules}
        self.assertIn("fail closed", decisions["canary"].message)
        self.assertIn("fail closed", decisions["formal"].message)


if __name__ == "__main__":
    unittest.main()
