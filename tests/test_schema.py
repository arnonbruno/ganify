"""Tests for typed schema inference and reversible table preprocessing."""

import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pandas as pd

from ganify import (
    ColumnSpec,
    ColumnType,
    Ganify,
    TableSchema,
    TableTransformer,
    infer_schema,
)


def _mixed_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "continuous": [0.125, 2.5, np.nan, -4.0, 9.25, 1.5],
            "count": pd.Series([0, 3, 7, None, 11, 4], dtype="Int64"),
            "binary": pd.Series(
                [True, False, None, True, False, True], dtype="boolean"
            ),
            "category": pd.Series(
                pd.Categorical(
                    ["red", "blue", None, "green", "red", "blue"],
                    categories=["blue", "green", "red"],
                )
            ),
            "ordinal": pd.Series(
                pd.Categorical(
                    ["low", "high", "mid", None, "low", "mid"],
                    categories=["low", "mid", "high"],
                    ordered=True,
                )
            ),
            "timestamp": pd.Series(
                [
                    pd.Timestamp("2024-01-01T01:02:03.123456789Z"),
                    pd.Timestamp("2024-02-10T12:00:00Z"),
                    pd.NaT,
                    pd.Timestamp("2024-03-15T23:59:59Z"),
                    pd.Timestamp("2024-05-01T06:30:00Z"),
                    pd.Timestamp("2024-06-30T08:45:00Z"),
                ],
                dtype="datetime64[ns, UTC]",
            ),
            "constant": ["fixed"] * 6,
        }
    )
    frame.index = pd.Index([10, 20, 30, 40, 50, 60], name="row")
    return frame


class SchemaTests(unittest.TestCase):
    def test_schema_is_immutable_and_json_serializable(self):
        spec = ColumnSpec(
            "level",
            ColumnType.ORDINAL,
            nullable=True,
            dtype="category",
            categories=("low", "mid", "high"),
        )
        schema = TableSchema((spec,))
        with self.assertRaises(FrozenInstanceError):
            spec.kind = "categorical"
        with self.assertRaises(FrozenInstanceError):
            schema.columns = ()
        payload = json.loads(schema.to_json())
        self.assertEqual(TableSchema.from_dict(payload), schema)

    def test_inference_uses_dtype_ratio_integer_nature_and_overrides(self):
        rows = 100
        frame = pd.DataFrame(
            {
                "continuous": np.linspace(-0.25, 1.75, rows),
                "count": np.arange(rows),
                "integer_codes": np.tile([10, 20, 30, 40], rows // 4),
                "flag": np.tile([0, 1], rows // 2),
                "label": pd.Categorical(np.tile(["a", "b"], rows // 2)),
                "ordered": pd.Categorical(
                    np.tile(["low", "high"], rows // 2),
                    categories=["low", "high"],
                    ordered=True,
                ),
                "when": pd.date_range("2024-01-01", periods=rows, freq="h"),
                "constant": 7,
            }
        )
        schema = infer_schema(frame)
        self.assertEqual(schema["continuous"].kind, "continuous")
        self.assertEqual(schema["count"].kind, "count")
        self.assertEqual(schema["integer_codes"].kind, "ordinal")
        self.assertEqual(schema["flag"].kind, "binary")
        self.assertEqual(schema["label"].kind, "categorical")
        self.assertEqual(schema["ordered"].kind, "ordinal")
        self.assertEqual(schema["when"].kind, "datetime")
        self.assertEqual(schema["constant"].kind, "constant")

        overridden = infer_schema(
            frame,
            overrides={
                "integer_codes": "count",
                "count": {
                    "kind": "categorical",
                    "categories": tuple(range(rows)),
                },
            },
        )
        self.assertEqual(overridden["integer_codes"].kind, "count")
        self.assertEqual(overridden["count"].kind, "categorical")
        with self.assertRaisesRegex(ValueError, "unknown columns"):
            infer_schema(frame, overrides={"missing": "continuous"})
        with self.assertRaisesRegex(ValueError, "negative"):
            infer_schema(
                pd.DataFrame({"n": [0, -1, 2]}),
                overrides={"n": "count"},
            )


class TableTransformerTests(unittest.TestCase):
    def test_all_supported_types_missingness_and_roundtrip(self):
        frame = _mixed_frame()
        transformer = TableTransformer(
            schema_overrides={"count": "count"}
        )
        encoded = transformer.fit_transform(frame)

        self.assertIsInstance(encoded, np.ndarray)
        self.assertEqual(encoded.shape, (len(frame), transformer.output_dim_))
        self.assertTrue(bool(np.isfinite(encoded).all()))
        self.assertEqual(
            set(transformer.mask_slices_),
            {
                "continuous",
                "count",
                "binary",
                "category",
                "ordinal",
                "timestamp",
            },
        )
        self.assertTrue(
            all(head.activation for head in transformer.head_metadata)
        )

        restored = transformer.inverse_transform(encoded)
        self.assertTrue(restored.index.equals(frame.index))
        np.testing.assert_allclose(
            restored["continuous"],
            frame["continuous"],
            atol=1e-12,
            equal_nan=True,
        )
        pd.testing.assert_series_equal(restored["count"], frame["count"])
        pd.testing.assert_series_equal(restored["binary"], frame["binary"])
        pd.testing.assert_series_equal(restored["category"], frame["category"])
        pd.testing.assert_series_equal(restored["ordinal"], frame["ordinal"])
        self.assertEqual(restored["constant"].tolist(), frame["constant"].tolist())
        difference = (
            restored["timestamp"].dropna().astype("int64")
            - frame["timestamp"].dropna().astype("int64")
        ).abs()
        self.assertLessEqual(int(difference.max()), 2)
        self.assertTrue(restored["timestamp"].isna().equals(frame["timestamp"].isna()))

    def test_unknown_categories_have_configurable_clear_behavior(self):
        train = pd.DataFrame(
            {
                "category": ["alpha", "beta", "alpha", "beta"],
                "binary": ["no", "yes", "no", "yes"],
            }
        )
        overrides = {"category": "categorical", "binary": "binary"}
        probe = pd.DataFrame({"category": ["new"], "binary": ["maybe"]})

        rejecting = TableTransformer(schema_overrides=overrides).fit(train)
        with self.assertRaisesRegex(ValueError, "outside its vocabulary"):
            rejecting.transform(probe)

        ignoring = TableTransformer(
            schema_overrides=overrides, handle_unknown="ignore"
        ).fit(train)
        ignored = ignoring.inverse_transform(ignoring.transform(probe))
        self.assertEqual(ignored.loc[0, "category"], "__unknown__")
        self.assertEqual(ignored.loc[0, "binary"], "__unknown__")

        encoding = TableTransformer(
            schema_overrides=overrides,
            handle_unknown="use_encoded_value",
            unknown_value="<UNK>",
        ).fit(train)
        restored = encoding.inverse_transform(encoding.transform(probe))
        self.assertEqual(restored.loc[0, "category"], "<UNK>")
        self.assertEqual(restored.loc[0, "binary"], "<UNK>")

    def test_serialization_parity_and_deterministic_output(self):
        frame = _mixed_frame()
        first = TableTransformer(
            schema_overrides={"count": "count"}, random_state=99
        ).fit(frame)
        second = TableTransformer(
            schema_overrides={"count": "count"}, random_state=1
        ).fit(frame)
        first_values = first.transform(frame)
        np.testing.assert_array_equal(first_values, second.transform(frame))

        payload = json.loads(json.dumps(first.to_dict()))
        restored = TableTransformer.from_dict(payload)
        np.testing.assert_array_equal(first_values, restored.transform(frame))
        pd.testing.assert_frame_equal(
            first.inverse_transform(first_values),
            restored.inverse_transform(restored.transform(frame)),
        )

        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "typed_preprocessor"
            written = first.save(destination)
            self.assertEqual(written.name, "preprocessor.json")
            loaded = TableTransformer.load(destination)
        np.testing.assert_array_equal(first_values, loaded.transform(frame))

    def test_mode_aware_numeric_transform_is_reversible(self):
        frame = pd.DataFrame(
            {
                "value": np.r_[
                    np.linspace(-8.0, -5.0, 30),
                    np.linspace(4.0, 9.0, 30),
                ],
                "count": np.arange(60),
            }
        )
        transformer = TableTransformer(
            continuous_transform="mode-aware",
            schema_overrides={"count": "count"},
            max_modes=4,
        )
        encoded = transformer.fit_transform(frame)
        restored = transformer.inverse_transform(encoded)
        self.assertGreater(encoded.shape[1], len(frame.columns))
        np.testing.assert_allclose(restored["value"], frame["value"], atol=1e-10)
        np.testing.assert_array_equal(restored["count"], frame["count"])

    def test_constant_and_discrete_values_are_exact(self):
        frame = pd.DataFrame(
            {
                "count": 10**18 + np.tile(np.arange(10, dtype=np.int64), 4),
                "binary": np.tile(["off", "on"], 20),
                "category": np.tile(["x", "y", "z", "x"], 10),
                "constant": [10**18 + 7] * 40,
            }
        )
        transformer = TableTransformer(
            schema_overrides={
                "count": "count",
                "binary": "binary",
                "category": "categorical",
            }
        )
        restored = transformer.inverse_transform(transformer.fit_transform(frame))
        pd.testing.assert_frame_equal(restored, frame)
        reloaded = TableTransformer.from_json(transformer.to_json())
        pd.testing.assert_frame_equal(
            reloaded.inverse_transform(reloaded.transform(frame)), frame
        )


class TypedGanifyIntegrationTests(unittest.TestCase):
    def test_fit_and_sample_mixed_nullable_table(self):
        frame = pd.concat([_mixed_frame()] * 6, ignore_index=True)
        model = Ganify(random_state=12, random_dim=8, max_units=32)
        model.fit(
            frame,
            conditional=False,
            schema_overrides={"count": "count"},
            epochs=1,
            batch_size=12,
            n_critic=1,
            verbose=0,
        )
        synthetic = model.sample(20)
        self.assertIsInstance(synthetic, pd.DataFrame)
        self.assertEqual(list(synthetic.columns), list(frame.columns))
        self.assertEqual(len(synthetic), 20)
        self.assertIsNotNone(model.table_transformer)
        self.assertEqual(model.schema_.names, tuple(frame.columns))
        self.assertTrue(
            set(synthetic["category"].dropna()).issubset(
                set(frame["category"].dropna())
            )
        )
        self.assertTrue(
            set(synthetic["binary"].dropna()).issubset({True, False})
        )
        self.assertTrue((synthetic["constant"] == "fixed").all())

    def test_typed_model_save_load_sampling_parity(self):
        frame = pd.concat([_mixed_frame()] * 5, ignore_index=True)
        model = Ganify(
            random_state=13,
            random_dim=8,
            max_units=32,
            ema_decay=0.9,
        )
        model.fit(
            frame,
            conditional=False,
            schema_overrides={"count": "count"},
            epochs=1,
            batch_size=10,
            n_critic=1,
            verbose=0,
        )
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "typed_model"
            model.save(destination)
            self.assertTrue(
                (destination / "typed_preprocessor.json").is_file()
            )
            left = Ganify.load(destination)
            right = Ganify.load(destination)
        pd.testing.assert_frame_equal(left.sample(12), right.sample(12))
        self.assertTrue(left._typed_table)
        self.assertEqual(left.schema_.names, tuple(frame.columns))

    def test_typed_validation_split_and_constraints(self):
        frame = pd.DataFrame(
            {
                "events": pd.Series(np.arange(40), dtype="Int64"),
                "users": pd.Series(np.arange(40) // 2, dtype="Int64"),
                "group": np.tile(["a", "b"], 20),
            }
        )
        model = Ganify(
            random_state=14,
            random_dim=8,
            max_units=32,
            inequality_pairs=[("users", "events")],
        )
        model.fit(
            frame,
            conditional=False,
            schema_overrides={"events": "count", "users": "count"},
            epochs=2,
            batch_size=10,
            n_critic=1,
            validation_fraction=0.2,
            selection_metric="validation",
            verbose=0,
        )
        synthetic = model.sample(20)
        self.assertEqual(len(model.validation_history_), 2)
        self.assertTrue((synthetic["users"] <= synthetic["events"]).all())


if __name__ == "__main__":
    unittest.main()
