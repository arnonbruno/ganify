"""Fast offline tests for reproducible cross-version comparison tooling."""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.experiments.version_comparison import (
    HistoricalArtifactError,
    VersionRefusalError,
    aggregate_version_comparison,
    deterministic_row_split,
    discover_kuairand_artifacts,
    evaluate_stages,
    probe_source,
    run_protocol,
)


FAKE_SOURCE = '''
import numpy as np
import pandas as pd

__version__ = {version!r}


class Ganify:
    def __init__(self, random_state=42, **kwargs):
        self.random_state = int(random_state)
        self._rng = np.random.default_rng(self.random_state)

    def fit_data(self, x_train, y_train, **kwargs):
        {failure}
        self._train = pd.DataFrame(x_train).reset_index(drop=True).copy()
        return self

    def create_bulk(self, length=None, lenght=None, output=None):
        rows = int(length if length is not None else lenght)
        indices = self._rng.integers(0, len(self._train), size=rows)
        sampled = self._train.iloc[indices].reset_index(drop=True).copy()
        return sampled if output in (1, "dataframe") else sampled.to_numpy()
'''


def make_fake_source(
    root: Path, *, version: str = "2.0.0", fail_fit: bool = False
) -> Path:
    package = root / "ganify"
    package.mkdir(parents=True)
    failure = (
        "raise RuntimeError('intentional fake fit failure')"
        if fail_fit
        else "pass"
    )
    (package / "__init__.py").write_text(
        FAKE_SOURCE.format(version=version, failure=failure),
        encoding="utf-8",
    )
    return root


def numeric_fixture(rows: int = 45) -> pd.DataFrame:
    index = np.arange(rows)
    return pd.DataFrame(
        {
            "row_id": ["row-%03d" % value for value in index],
            "x": np.sin(index / 4.0) + index / 20.0,
            "z": np.cos(index / 5.0) - index / 30.0,
            "target": 2.0 * index / rows + np.sin(index / 3.0),
        }
    )


def classification_fixture(rows: int = 45) -> pd.DataFrame:
    index = np.arange(rows)
    labels = np.asarray(["red", "green", "blue"])[index % 3]
    return pd.DataFrame(
        {
            "row_id": ["class-%03d" % value for value in index],
            "x": index.astype(float) / rows,
            "z": (index % 7).astype(float),
            "target": labels,
        }
    )


def protocol_for(
    *,
    dataset_path: Path,
    source_root: Path,
    output_dir: Path,
    lane: str,
    fail_name: str = "fake-v2",
    sample_rows: int = 18,
) -> dict:
    return {
        "protocol_version": "1",
        "output_dir": str(output_dir),
        "datasets": [
            {
                "id": "tiny",
                "path": str(dataset_path),
                "row_id": "row_id",
                "target": "target",
                "lane": lane,
                "task": (
                    "regression"
                    if lane == "numeric_regression"
                    else "classification"
                ),
                "test_size": 0.2,
            }
        ],
        "models": [
            {
                "name": fail_name,
                "adapter": "v2_numeric_compatibility",
                "source_root": str(source_root),
                "expected_version": "2.0.0",
                "sample_rows": sample_rows,
                "config": {"fit": {"epochs": 1, "verbose": 0}},
            }
        ],
        "split_seeds": [42],
        "fit_seeds": [101],
        "sample_seeds": [7],
        "stages": {"raw": {}},
        "evaluation": {
            "controls": False,
            "c2st_folds": 2,
            "privacy": {"enabled": False},
        },
        "aggregation": {"bootstrap_replicates": 20},
    }


class SourceIsolationTests(unittest.TestCase):
    def test_worker_refuses_exact_version_mismatch_without_parent_contamination(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = make_fake_source(Path(temporary) / "fake", version="9.9.9")
            previous = sys.modules.get("ganify")
            contaminated = types.ModuleType("ganify")
            contaminated.__version__ = "parent-process-version"
            sys.modules["ganify"] = contaminated
            try:
                with self.assertRaisesRegex(VersionRefusalError, "expected.*1.1.0"):
                    probe_source(root, "1.1.0")
                accepted = probe_source(root, "9.9.9")
            finally:
                if previous is None:
                    sys.modules.pop("ganify", None)
                else:
                    sys.modules["ganify"] = previous
        self.assertEqual(
            accepted["provenance"]["observed_version"], "9.9.9"
        )
        self.assertTrue(
            accepted["provenance"]["imported_module"].startswith(str(root))
        )


class SplitCacheAndFailureTests(unittest.TestCase):
    def test_split_and_completed_cell_cache_are_deterministic(self):
        ids = ["id-%02d" % value for value in range(31)]
        first = deterministic_row_split(ids, seed=42, validation_size=0.1)
        second = deterministic_row_split(
            reversed(ids), seed=42, validation_size=0.1
        )
        self.assertEqual(first.as_dict(), second.as_dict())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = make_fake_source(root / "source")
            data_path = root / "numeric.csv"
            numeric_fixture().to_csv(data_path, index=False)
            protocol = protocol_for(
                dataset_path=data_path,
                source_root=source,
                output_dir=root / "output",
                lane="numeric_regression",
            )
            protocol_path = root / "protocol.json"
            protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
            initial = run_protocol(protocol_path)
            sample_path = Path(initial.samples.iloc[0]["path"])
            original_mtime = sample_path.stat().st_mtime_ns
            resumed = run_protocol(protocol_path)
            resumed_mtime = sample_path.stat().st_mtime_ns

        self.assertEqual(initial.splits, resumed.splits)
        self.assertFalse(bool(initial.manifests.iloc[0]["cache_hit"]))
        self.assertTrue(bool(resumed.manifests.iloc[0]["cache_hit"]))
        self.assertEqual(int(resumed.manifests.iloc[0]["attempt"]), 1)
        self.assertEqual(original_mtime, resumed_mtime)

    def test_failed_cell_is_retained_and_not_silently_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = make_fake_source(root / "source", fail_fit=True)
            data_path = root / "numeric.csv"
            numeric_fixture().to_csv(data_path, index=False)
            protocol = protocol_for(
                dataset_path=data_path,
                source_root=source,
                output_dir=root / "output",
                lane="numeric_regression",
                fail_name="failing",
            )
            initial = run_protocol(protocol)
            resumed = run_protocol(protocol)

        self.assertEqual(initial.manifests.iloc[0]["status"], "failed")
        self.assertEqual(resumed.manifests.iloc[0]["status"], "failed")
        self.assertEqual(int(resumed.manifests.iloc[0]["attempt"]), 1)
        self.assertTrue(bool(resumed.manifests.iloc[0]["failure_reused"]))
        self.assertFalse(bool(resumed.reliability["run_success"].iloc[0]))
        self.assertIn(
            "intentional fake fit failure",
            resumed.reliability["error_message"].iloc[0],
        )


class LaneAndEvaluationTests(unittest.TestCase):
    def test_v11_unsupported_stage_is_an_explicit_capability_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = make_fake_source(root / "source", version="1.1.0")
            data_path = root / "numeric.csv"
            numeric_fixture().to_csv(data_path, index=False)
            protocol = protocol_for(
                dataset_path=data_path,
                source_root=source,
                output_dir=root / "output",
                lane="numeric_regression",
            )
            protocol["models"][0].update(
                {
                    "adapter": "v1.1_legacy_wgan",
                    "expected_version": "1.1.0",
                }
            )
            protocol["stages"] = {"raw": {}, "calibrated": {}}
            result = run_protocol(protocol)
        failure = result.manifests.iloc[0]
        self.assertEqual(failure["status"], "failed")
        self.assertEqual(failure["failure_kind"], "capability")
        self.assertIn("calibrated stage is unavailable", failure["error_message"])

    def test_classwise_synthesis_has_exact_deterministic_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = make_fake_source(root / "source")
            data_path = root / "classes.csv"
            classification_fixture().to_csv(data_path, index=False)
            protocol = protocol_for(
                dataset_path=data_path,
                source_root=source,
                output_dir=root / "output",
                lane="multiclass_classwise",
                sample_rows=17,
            )
            result = run_protocol(protocol)
            sample_record = result.samples.iloc[0]
            generated = pd.read_csv(sample_record["path"])
            recorded = json.loads(sample_record["class_counts"])

        observed = generated["target"].value_counts().to_dict()
        expected = {item["value"]: item["count"] for item in recorded}
        self.assertEqual(observed, expected)
        self.assertEqual(sum(observed.values()), 17)
        self.assertEqual(set(observed), {"red", "green", "blue"})

    def test_evaluator_keeps_raw_calibrated_projected_stages_separate(self):
        rng = np.random.default_rng(11)
        train = pd.DataFrame(
            rng.normal(size=(36, 3)), columns=["a", "b", "c"]
        )
        test = pd.DataFrame(
            rng.normal(size=(18, 3)), columns=train.columns
        )
        raw = pd.DataFrame(
            rng.normal(loc=0.5, size=(18, 3)), columns=train.columns
        )
        calibrated = raw * 0.8
        projected = calibrated.clip(-1.0, 1.0)
        result = evaluate_stages(
            train,
            test,
            {
                "raw": raw,
                "calibrated": calibrated,
                "projected": projected,
            },
            constraints=[
                {"type": "range", "column": "a", "min": -1.0, "max": 1.0}
            ],
            c2st_folds=2,
            privacy={"enabled": False},
            include_controls=False,
        )
        self.assertEqual(
            set(result.metrics["stage"]),
            {"raw", "calibrated", "projected"},
        )
        self.assertEqual(
            result.metadata["stage_order"],
            ["raw", "calibrated", "projected"],
        )
        self.assertIn("constraint", set(result.metrics["pillar"]))
        self.assertEqual(len(result.reliability), 3)

    def test_evaluator_runs_bounded_privacy_and_all_controls(self):
        rng = np.random.default_rng(22)
        train = pd.DataFrame(
            rng.normal(size=(24, 3)), columns=["a", "b", "c"]
        )
        test = pd.DataFrame(
            rng.normal(size=(16, 3)), columns=train.columns
        )
        synthetic = pd.DataFrame(
            rng.normal(size=(16, 3)), columns=train.columns
        )
        result = evaluate_stages(
            train,
            test,
            {"raw": synthetic},
            c2st_folds=2,
            privacy={
                "enabled": True,
                "max_rows": 12,
                "bootstrap_replicates": 2,
                "n_singling_attacks": 3,
            },
            include_controls=True,
        )
        self.assertEqual(
            set(result.controls),
            {"bootstrap", "independent", "gaussian_copula"},
        )
        self.assertIn("privacy_attack", set(result.privacy["pillar"]))
        self.assertTrue(bool(result.reliability["run_success"].all()))


class AggregationTests(unittest.TestCase):
    def test_sample_seeds_are_averaged_before_fit_statistics_and_columns_stay_separate(self):
        rows = []
        values = {
            ("A", 1, "a"): [0.0, 2.0],
            ("A", 2, "a"): [10.0, 14.0],
            ("A", 1, "b"): [100.0, 104.0],
            ("A", 2, "b"): [120.0, 124.0],
            ("B", 1, "a"): [1.0, 3.0],
            ("B", 2, "a"): [12.0, 16.0],
            ("B", 1, "b"): [99.0, 103.0],
            ("B", 2, "b"): [119.0, 123.0],
        }
        for (version, fit_seed, column), samples in values.items():
            for sample_seed, value in zip((7, 8), samples):
                rows.append(
                    {
                        "dataset_id": "tiny",
                        "version": version,
                        "adapter": version,
                        "split_seed": 42,
                        "fit_seed": fit_seed,
                        "sample_seed": sample_seed,
                        "stage": "raw",
                        "pillar": "marginal",
                        "metric": "ks_statistic",
                        "column": column,
                        "detail": "numeric",
                        "value": value,
                        "success": True,
                    }
                )
        # One incomplete fitted model must remain a failure instead of being
        # silently reduced to its one successful sample.
        rows.extend(
            [
                {
                    **rows[0],
                    "fit_seed": 3,
                    "sample_seed": 7,
                    "value": 4.0,
                },
                {
                    **rows[0],
                    "fit_seed": 3,
                    "sample_seed": 8,
                    "value": np.nan,
                    "success": False,
                },
            ]
        )
        result = aggregate_version_comparison(
            rows,
            reference_version="A",
            expected_sample_seeds=[7, 8],
            n_bootstrap=50,
            random_state=3,
        )
        summary = result.summary
        a_column = summary.loc[
            (summary["version"] == "A") & (summary["column"] == "a")
        ].iloc[0]
        self.assertAlmostEqual(a_column["median"], 6.5)
        self.assertEqual(int(a_column["fit_count"]), 3)
        self.assertEqual(int(a_column["failed_fits"]), 1)
        self.assertEqual(
            set(summary.loc[summary["version"] == "A", "column"]),
            {"a", "b"},
        )
        delta = result.paired_deltas.loc[
            result.paired_deltas["column"] == "a"
        ].iloc[0]
        self.assertAlmostEqual(delta["median_delta"], 1.5)
        self.assertEqual(int(delta["successful_pairs"]), 2)


class HistoricalArtifactTests(unittest.TestCase):
    def test_historical_artifact_absence_is_actionable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(
                HistoricalArtifactError, "artifact directory is absent"
            ):
                discover_kuairand_artifacts(
                    root / "missing-v11", root / "missing-v12"
                )

    def test_v12_discovery_is_explicitly_artifact_level(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            v11 = root / "kuairand"
            v12 = root / "kuairand_v2"
            v11.mkdir()
            v12.mkdir()
            pd.DataFrame({"x": [1.0]}).to_csv(
                v11 / "video_statistics_raw_synthetic.csv", index=False
            )
            pd.DataFrame({"x": [2.0]}).to_csv(
                v12 / "final_video_synthetic.csv", index=False
            )
            artifacts = discover_kuairand_artifacts(v11, v12)
        current = next(
            artifact
            for artifact in artifacts
            if artifact.historical_version == "1.2.0"
        )
        self.assertEqual(current.label, "v1.2_historical_artifact")
        self.assertEqual(current.evidence_level, "artifact_level")
        self.assertFalse(current.source_available)


if __name__ == "__main__":
    unittest.main()

