"""Fast, offline tests for evaluation controls and benchmark invariants."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.ganify_bench import (
    DatasetUnavailableError,
    deterministic_split,
    evaluate_gates,
    get_dataset_manifest,
    load_dataset,
    load_suite,
    split_frame,
)
from ganify.evaluation import (
    InequalityConstraint,
    aggregate_table_report,
    bootstrap_control,
    categorical_total_variation,
    classification_utility,
    constraint_validity,
    c2st_auc,
    dependence_matrix_errors,
    independently_permuted_columns,
    marginal_metrics,
    nearest_neighbor_metrics,
    regression_utility,
)


class MetricControlTests(unittest.TestCase):
    def test_shuffled_columns_keep_marginals_but_lose_dependence(self):
        rng = np.random.default_rng(12)
        latent = rng.normal(size=800)
        real = pd.DataFrame(
            {
                "a": latent,
                "b": 2.0 * latent + rng.normal(scale=0.03, size=len(latent)),
                "c": -latent + rng.normal(scale=0.03, size=len(latent)),
            }
        )
        shuffled = independently_permuted_columns(real, random_state=99)

        marginals = marginal_metrics(real, shuffled)
        np.testing.assert_allclose(marginals["ks_statistic"], 0.0)
        np.testing.assert_allclose(marginals["wasserstein_distance"], 0.0)
        np.testing.assert_allclose(marginals["zero_rate_error"], 0.0)
        self.assertGreater(
            dependence_matrix_errors(real, shuffled).spearman_mae, 0.8
        )
        self.assertEqual(
            dependence_matrix_errors(real, real).spearman_mae, 0.0
        )

    def test_bootstrap_rows_are_detected_as_exact_matches(self):
        rng = np.random.default_rng(7)
        real = pd.DataFrame(
            {
                "x": np.arange(120, dtype=float),
                "y": rng.normal(size=120),
                "kind": ["even" if index % 2 == 0 else "odd" for index in range(120)],
            }
        )
        bootstrap = bootstrap_control(real, n_rows=200, random_state=4)
        metrics = nearest_neighbor_metrics(
            real,
            bootstrap,
            real_holdout=real.iloc[:30],
            categorical_columns=["kind"],
        )
        self.assertEqual(metrics.exact_match_rate, 1.0)
        self.assertEqual(metrics.authenticity, 0.0)

    def test_constraint_validity_reports_individual_and_joint_rates(self):
        frame = pd.DataFrame({"users": [0, 2, 5], "events": [1, 2, 4]})
        report = constraint_validity(
            frame,
            [
                InequalityConstraint(
                    "users", "<=", "events", name="users_not_above_events"
                )
            ],
        )
        individual = report.loc[
            report["constraint"] == "users_not_above_events"
        ].iloc[0]
        self.assertAlmostEqual(individual["valid_rate"], 2.0 / 3.0)
        self.assertEqual(set(report["constraint"]), {"users_not_above_events", "__all__"})

    def test_c2st_and_tstr_trtr_public_apis(self):
        rng = np.random.default_rng(11)
        features = pd.DataFrame(
            {
                "x": rng.normal(size=160),
                "group": np.where(rng.random(160) < 0.5, "a", "b"),
            }
        )
        classification_target = (features["x"].to_numpy() > 0.0).astype(int)
        classification = classification_utility(
            features.iloc[:100],
            classification_target[:100],
            features.iloc[100:],
            classification_target[100:],
            features.iloc[:100].copy(),
            classification_target[:100].copy(),
            categorical_columns=["group"],
        )
        self.assertAlmostEqual(
            classification["trtr"]["balanced_accuracy"],
            classification["tstr"]["balanced_accuracy"],
        )

        regression_target = (
            3.0 * features["x"].to_numpy() + rng.normal(scale=0.05, size=160)
        )
        regression = regression_utility(
            features.iloc[:100],
            regression_target[:100],
            features.iloc[100:],
            regression_target[100:],
            features.iloc[:100].copy(),
            regression_target[:100].copy(),
            categorical_columns=["group"],
        )
        self.assertAlmostEqual(
            regression["trtr"]["rmse"], regression["tstr"]["rmse"]
        )
        self.assertTrue(
            np.isfinite(
                c2st_auc(
                    features.iloc[:80],
                    features.iloc[80:].reset_index(drop=True),
                    categorical_columns=["group"],
                    folds=2,
                )
            )
        )
        self.assertEqual(
            categorical_total_variation(["a", "a", "b"], ["b", "a", "a"]),
            0.0,
        )

    def test_aggregate_report_never_merges_generation_stages(self):
        rng = np.random.default_rng(20)
        train = pd.DataFrame(rng.normal(size=(80, 3)), columns=["a", "b", "c"])
        test = pd.DataFrame(rng.normal(size=(40, 3)), columns=train.columns)
        raw = pd.DataFrame(rng.normal(size=(40, 3)), columns=train.columns)
        calibrated = independently_permuted_columns(test, random_state=2)
        projected = calibrated.clip(-2.0, 2.0)
        report = aggregate_table_report(
            train,
            test,
            {
                "raw": raw,
                "calibrated": calibrated,
                "projected": projected,
            },
            c2st_folds=2,
        )
        self.assertEqual(
            set(report["stage"]), {"raw", "calibrated", "projected"}
        )
        self.assertFalse(report.attrs["contains_blended_score"])
        self.assertNotIn("quality_score", set(report["metric"]))


class BenchmarkProtocolTests(unittest.TestCase):
    def test_split_hash_is_order_independent_and_has_no_leakage(self):
        ids = ["row-%03d" % index for index in range(101)]
        first = deterministic_split(ids, seed=42)
        second = deterministic_split(reversed(ids), seed=42)
        self.assertEqual(first.split_hash, second.split_hash)
        self.assertEqual(first.train, second.train)
        first.assert_no_leakage()
        self.assertFalse(set(first.train) & set(first.validation))
        self.assertFalse(set(first.train) & set(first.test))
        self.assertFalse(set(first.validation) & set(first.test))

        frame = pd.DataFrame({"row_id": ids, "value": np.arange(len(ids))})
        partitions = split_frame(frame, first, id_column="row_id")
        self.assertEqual(sum(map(len, partitions.values())), len(frame))
        self.assertEqual(
            set().union(*(set(part["row_id"]) for part in partitions.values())),
            set(ids),
        )

    def test_gate_logic_passes_conjunctively_and_fails_closed(self):
        records = pd.DataFrame(
            [
                {"stage": "raw", "metric": "dependence", "value": 0.2},
                {"stage": "raw", "metric": "dependence", "value": 0.4},
                {"stage": "raw", "metric": "utility", "value": 0.93},
            ]
        )
        config = {
            "version": "test",
            "rules": [
                {
                    "id": "dependence",
                    "metric": "dependence",
                    "stage": "raw",
                    "operator": "<=",
                    "threshold": 0.5,
                    "aggregation": "median",
                },
                {
                    "id": "utility",
                    "metric": "utility",
                    "stage": "raw",
                    "operator": ">=",
                    "threshold": 0.9,
                },
            ],
        }
        passing = evaluate_gates(records, config)
        self.assertTrue(passing.passed)
        self.assertNotIn("quality_score", passing.to_frame().columns)

        failing_records = records.copy()
        failing_records.loc[failing_records["metric"] == "utility", "value"] = 0.8
        self.assertFalse(evaluate_gates(failing_records, config).passed)
        self.assertFalse(
            evaluate_gates(
                records.loc[records["metric"] != "utility"], config
            ).passed
        )

    def test_suite_metadata_and_deferred_loader_error_are_actionable(self):
        smoke = load_suite("smoke")
        self.assertIn("adult", {item.dataset_id for item in smoke.datasets})
        self.assertIn(
            "kuairand_pure", {item.dataset_id for item in smoke.datasets}
        )
        adult = get_dataset_manifest("adult")
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(DatasetUnavailableError) as caught:
                load_dataset(adult, data_root=Path(temporary))
        message = str(caught.exception)
        self.assertIn("adult", message)
        self.assertIn(adult.source_url, message)
        self.assertIn(adult.local_path, message)


if __name__ == "__main__":
    unittest.main()
