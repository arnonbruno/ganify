"""Fast, offline tests for the final release-gate protocol."""

import unittest

import numpy as np
import pandas as pd

from benchmarks.ganify_bench import (
    BootstrapAdapter,
    GaussianCopulaAdapter,
    IndependentMarginalsAdapter,
    ReleaseDecision,
    ReleaseRunner,
    generate_controlled_panel,
    load_release_config,
)


class _DeterministicAdapter:
    """Small test adapter that samples fitted rows without external packages."""

    def __init__(self, name="mock"):
        self.name = name
        self.train = None

    def fit(self, train, *, seed, target=None):
        del target
        self.fit_seed = int(seed)
        self.train = train.reset_index(drop=True).copy(deep=True)
        return self

    def sample(self, n_rows, *, seed):
        if self.train is None:
            raise RuntimeError("not fitted")
        rng = np.random.default_rng(int(seed))
        positions = rng.integers(0, len(self.train), size=int(n_rows))
        return self.train.iloc[positions].reset_index(drop=True).copy(deep=True)

    def get_config(self):
        return {"adapter": self.name, "test_double": True}

    def clone(self):
        return type(self)(self.name)


def _suite(sample_seeds=(0, 1)):
    return {
        "smoke": {
            "version": "test",
            "datasets": ["controlled_panel"],
            "split_seeds": [0, 1],
            "model_seeds": [0],
            "sample_seeds": list(sample_seeds),
            "max_rows": 96,
        }
    }


def _metric_evaluator(real_train, real_test, synthetic, context):
    del real_train, real_test, synthetic
    loss = 0.10 if context["model_name"] == "candidate" else 0.70
    return pd.DataFrame(
        [
            {
                "stage": "raw",
                "pillar": "fidelity",
                "metric": "structure_loss",
                "value": loss,
            }
        ]
    )


def _privacy_evaluator(real_train, real_test, synthetic, context):
    del real_train, real_test, synthetic, context
    return pd.DataFrame(
        [
            {
                "stage": "raw",
                "pillar": "privacy_attack",
                "metric": "membership_roc_auc_ci_high",
                "value": 0.55,
            }
        ]
    )


class ControlledPanelTests(unittest.TestCase):
    def test_panel_is_deterministic_and_truth_known(self):
        left = generate_controlled_panel(rows=512, seed=17)
        right = generate_controlled_panel(rows=512, seed=17)
        pd.testing.assert_frame_equal(left.frame, right.frame)
        pd.testing.assert_frame_equal(left.truth_frame, right.truth_frame)
        self.assertEqual(left.manifest(), right.manifest())
        self.assertEqual(left.frame.shape, (512, 100))
        self.assertEqual(
            left.frame["target"].to_numpy().tolist(),
            np.logical_xor(
                left.frame["xor_left"], left.frame["xor_right"]
            )
            .astype(np.int8)
            .tolist(),
        )
        self.assertGreater(
            left.frame["zipf_category"].value_counts().iloc[0],
            left.frame["zipf_category"].value_counts().iloc[-1],
        )
        self.assertGreater(
            float(left.truth_frame["mar_probability"].std()), 0.05
        )
        self.assertGreater(
            np.corrcoef(
                left.truth_frame["mnar_complete"],
                left.truth_frame["mnar_probability"],
            )[0, 1],
            0.5,
        )
        for constraint in left.constraints:
            self.assertTrue(bool(constraint.evaluate(left.frame).all()))

    def test_different_seed_changes_data_not_protocol(self):
        left = generate_controlled_panel(rows=128, seed=1)
        right = generate_controlled_panel(rows=128, seed=2)
        self.assertNotEqual(left.data_hash, right.data_hash)
        self.assertEqual(
            [item.name for item in left.constraints],
            [item.name for item in right.constraints],
        )


class ControlAdapterTests(unittest.TestCase):
    def test_controls_expose_memorization_and_dependence(self):
        rng = np.random.default_rng(9)
        latent = rng.normal(size=800)
        train = pd.DataFrame(
            {
                "x": latent,
                "y": 2.0 * latent + rng.normal(scale=0.08, size=len(latent)),
            }
        )
        bootstrap = BootstrapAdapter().fit(train, seed=0)
        copied = bootstrap.sample(400, seed=3)
        source_rows = set(map(tuple, train.to_numpy()))
        self.assertTrue(
            all(tuple(row) in source_rows for row in copied.to_numpy())
        )

        independent = IndependentMarginalsAdapter().fit(train, seed=0)
        independent_sample = independent.sample(3000, seed=3)
        copula = GaussianCopulaAdapter().fit(train, seed=0)
        copula_sample = copula.sample(3000, seed=3)
        independent_correlation = abs(
            float(independent_sample.corr().iloc[0, 1])
        )
        copula_correlation = abs(float(copula_sample.corr().iloc[0, 1]))
        self.assertLess(independent_correlation, 0.12)
        self.assertGreater(copula_correlation, 0.90)
        self.assertGreater(copula_correlation, independent_correlation + 0.7)


class ReleaseRunnerTests(unittest.TestCase):
    def test_failed_external_adapter_is_retained_without_retry(self):
        config = {
            "version": "runner-test",
            "candidate": "candidate",
            "candidate_model": "bootstrap",
            "sealed_test": True,
            "suites": {
                "smoke": {
                    "datasets": ["controlled_panel"],
                    "split_seeds": [0],
                    "model_seeds": [0],
                    "sample_seeds": [0, 1],
                    "max_rows": 96,
                }
            },
            "models": [
                {"name": "bootstrap", "adapter": "bootstrap"},
                {"name": "ctgan", "adapter": "ctgan"},
            ],
        }
        result = ReleaseRunner(
            config, evaluator=_metric_evaluator
        ).run()
        self.assertEqual(len(result.manifests), 4)
        failed = result.manifests.loc[
            result.manifests["model_name"] == "ctgan"
        ]
        self.assertEqual(len(failed), 2)
        self.assertTrue(failed["status"].eq("failed").all())
        self.assertTrue(failed["attempt"].eq(1).all())
        self.assertTrue(failed["failure_phase"].eq("fit").all())
        self.assertIn("pip install ctgan", failed.iloc[0]["error_message"])
        self.assertEqual(len(result.reliability), 4)
        self.assertNotIn("NaN", result.manifest_json())

    def test_cheap_controlled_run_needs_no_network_or_ml_stack(self):
        config = {
            "version": "offline-test",
            "candidate": "bootstrap",
            "candidate_model": "bootstrap",
            "sealed_test": True,
            "cheap_rows": 96,
            "suites": _suite(sample_seeds=(0,)),
            "models": [
                {"name": "bootstrap", "adapter": "bootstrap"},
                {
                    "name": "independent_marginals",
                    "adapter": "independent_marginals",
                },
                {
                    "name": "gaussian_copula",
                    "adapter": "gaussian_copula",
                },
            ],
        }
        result = ReleaseRunner(config).run_controlled_panel(rows=96)
        self.assertTrue(result.manifests["status"].eq("completed").all())
        self.assertIn("dependence", set(result.metrics["pillar"]))
        self.assertIn("constraint", set(result.metrics["pillar"]))
        self.assertFalse(result.metrics.attrs["contains_blended_score"])

    def test_manifests_are_byte_deterministic(self):
        config = {
            "version": "determinism-test",
            "candidate": "candidate",
            "candidate_model": "candidate",
            "sealed_test": True,
            "suites": _suite(sample_seeds=(0,)),
            "models": [{"name": "candidate", "adapter": "mock"}],
        }
        adapters = {"candidate": _DeterministicAdapter("candidate")}
        first = ReleaseRunner(
            config,
            adapters=adapters,
            evaluator=_metric_evaluator,
        ).run()
        second = ReleaseRunner(
            config,
            adapters=adapters,
            evaluator=_metric_evaluator,
        ).run()
        self.assertEqual(first.manifest_json(), second.manifest_json())
        pd.testing.assert_frame_equal(first.manifests, second.manifests)


class ReleaseDecisionTests(unittest.TestCase):
    def _config(self):
        ordinary = {
            "version": "ordinary-test",
            "rules": [
                {
                    "id": "candidate_fidelity",
                    "metric": "structure_loss",
                    "pillar": "fidelity",
                    "model_name": "candidate",
                    "operator": "<=",
                    "threshold": 0.2,
                    "aggregation": "maximum",
                    "minimum_count": 4,
                }
            ],
        }
        privacy = {
            "version": "privacy-test",
            "rules": [
                {
                    "id": "membership",
                    "metric": "membership_roc_auc_ci_high",
                    "pillar": "privacy_attack",
                    "operator": "<=",
                    "threshold": 0.60,
                    "aggregation": "maximum",
                    "minimum_count": 4,
                }
            ],
        }
        evidence = {
            "confidence": 0.95,
            "bootstrap_replicates": 50,
            "comparisons": [
                {
                    "id": "paired_fidelity",
                    "metric": "structure_loss",
                    "filters": {
                        "stage": "raw",
                        "pillar": "fidelity",
                        "suite": "smoke",
                    },
                    "baseline": "baseline",
                    "direction": "lower",
                    "minimum_pairs": 4,
                    "minimum_effect": 0.0,
                }
            ],
        }
        return {
            "version": "decision-test",
            "candidate": "candidate",
            "candidate_model": "candidate",
            "suites": _suite(),
            "mandatory_baselines": ["baseline"],
            "frontier_baselines": ["ctgan"],
            "ordinary_gates": ordinary,
            "privacy_gates": privacy,
            "claims": [
                {
                    "id": "broad_sota",
                    "claim": "Candidate is broadly state of the art.",
                    "scope": "broad_sota",
                    "priority": 100,
                    "required_suites": ["smoke"],
                    "required_models": ["candidate", "baseline"],
                    "frontier_baselines": ["ctgan"],
                    "statistical_evidence": evidence,
                },
                {
                    "id": "narrow_registered",
                    "claim": "Candidate improves controlled structure loss.",
                    "scope": "narrow_controlled",
                    "priority": 10,
                    "required_suites": ["smoke"],
                    "required_models": ["candidate", "baseline"],
                    "statistical_evidence": evidence,
                },
            ],
        }

    def _result(self, config):
        adapters = {
            "candidate": _DeterministicAdapter("candidate"),
            "baseline": _DeterministicAdapter("baseline"),
        }
        runner_config = {
            "version": config["version"],
            "candidate": config["candidate"],
            "candidate_model": config["candidate_model"],
            "sealed_test": True,
            "suites": config["suites"],
            "models": [
                {"name": "candidate", "adapter": "mock"},
                {"name": "baseline", "adapter": "mock"},
            ],
        }
        return ReleaseRunner(
            runner_config,
            adapters=adapters,
            evaluator=_metric_evaluator,
            privacy_evaluator=_privacy_evaluator,
        ).run()

    def test_missing_core_stress_and_frontier_deny_broad_sota(self):
        config = self._config()
        report = ReleaseDecision(config).evaluate(self._result(config))
        broad = next(
            claim for claim in report.claims if claim.claim_id == "broad_sota"
        )
        self.assertFalse(broad.passed)
        reason = " ".join(broad.reasons)
        self.assertIn("core", reason)
        self.assertIn("stress", reason)
        self.assertIn("ctgan", reason)

    def test_complete_mock_earns_only_preregistered_narrow_claim(self):
        config = self._config()
        report = ReleaseDecision(config).evaluate(self._result(config))
        self.assertTrue(report.approved)
        self.assertEqual(report.claim_id, "narrow_registered")
        self.assertEqual(report.scope, "narrow_controlled")
        self.assertNotEqual(report.claim_id, "broad_sota")
        self.assertFalse(report.contains_blended_score)

    def test_vnext_names_all_suites_and_controls(self):
        config = load_release_config("vnext")
        self.assertEqual(set(config["suites"]), {"smoke", "core", "stress"})
        self.assertEqual(
            set(config["mandatory_baselines"]),
            {"bootstrap", "independent_marginals", "gaussian_copula"},
        )


if __name__ == "__main__":
    unittest.main()
