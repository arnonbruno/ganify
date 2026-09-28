"""Fast tests for typed structural constraints and engine integration."""

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import numpy as np
import pandas as pd

from ganify import Ganify
from ganify.conditional import ConditionalGANEngine
from ganify.constraints import (
    BoundConstraint,
    ConstraintSet,
    DomainConstraint,
    FixedSumConstraint,
    ImplicationConstraint,
    LinearEquality,
    LinearInequality,
    PairInequality,
    SimplexConstraint,
    StructuralConstraintTransformer,
    VariableSumConstraint,
    constraint_audit,
)


class ConstraintTypeTests(unittest.TestCase):
    def test_every_type_serializes_and_reports_margins(self):
        frame = pd.DataFrame(
            {
                "small": [0.0, 1.0, 2.0],
                "big": [1.0, 2.0, 4.0],
                "kind": ["a", "b", "a"],
                "p": [0.2, 0.5, 0.7],
                "q": [0.8, 0.5, 0.3],
                "u": [1.0, 2.0, 3.0],
                "v": [2.0, 3.0, 4.0],
                "total": [3.0, 5.0, 7.0],
                "flag": [False, True, True],
                "required": [2.0, 0.0, 0.0],
            }
        )
        constraints = ConstraintSet(
            [
                BoundConstraint("small", minimum=0.0, maximum=2.0),
                DomainConstraint("kind", allowed_values=("a", "b")),
                PairInequality("small", "big"),
                LinearInequality(
                    {"small": 1.0, "big": -1.0}, rhs=0.0
                ),
                LinearEquality({"p": 1.0, "q": 1.0}, rhs=1.0),
                FixedSumConstraint(("p", "q"), total=1.0),
                VariableSumConstraint(("u", "v"), total_column="total"),
                ImplicationConstraint(
                    "flag", True, "required", 0.0
                ),
            ]
        )
        report = constraints.validate(frame)
        self.assertTrue(report.valid)
        self.assertEqual(
            set(report.violations["constraint"]),
            {constraint.label for constraint in constraints} | {"__all__"},
        )
        self.assertEqual(len(report.margins), len(constraints))
        restored = ConstraintSet.from_json(constraints.to_json())
        self.assertEqual(restored.to_dict(), constraints.to_dict())

    def test_pair_gap_folding_avoids_clipping_boundary_pileup(self):
        frame = pd.DataFrame(
            {
                "small": np.zeros(4),
                "big": [0.25, 0.5, 1.0, 2.0],
            }
        )
        constraints = ConstraintSet([PairInequality("small", "big")])
        transformer = StructuralConstraintTransformer(constraints).fit(frame)
        coordinates = transformer.transform(frame)
        signed = np.asarray([-2.0, -1.0, 0.0, 0.5])
        coordinates["big"] = signed
        reconstructed = transformer.inverse_transform(coordinates)
        self.assertEqual(constraints.violation_rate(reconstructed), 0.0)
        structural_boundary = float(
            np.mean(
                np.isclose(
                    reconstructed["small"], reconstructed["big"]
                )
            )
        )
        clipped_boundary = float(np.mean(np.maximum(signed, 0.0) == 0.0))
        self.assertLess(structural_boundary, clipped_boundary)

    def test_fixed_and_variable_simplexes_reconstruct_exact_sums(self):
        frame = pd.DataFrame(
            {
                "a": [0.2, 0.0, 0.6],
                "b": [0.3, 0.5, 0.2],
                "c": [0.5, 0.5, 0.2],
                "u": [1.0, 0.0, 3.0],
                "v": [2.0, 4.0, 2.0],
                "total": [3.0, 4.0, 5.0],
            }
        )
        constraints = ConstraintSet(
            [
                SimplexConstraint(("a", "b", "c")),
                VariableSumConstraint(
                    ("u", "v"), total_column="total"
                ),
            ]
        )
        transformer = StructuralConstraintTransformer(constraints).fit(frame)
        coordinates = transformer.transform(frame)
        coordinates["b"] = [-8.0, 2.0, 12.0]
        coordinates["c"] = [3.0, -5.0, 1.0]
        coordinates["u"] = [6.0, -3.0, 0.0]
        reconstructed = transformer.inverse_transform(coordinates)
        np.testing.assert_allclose(
            reconstructed[["a", "b", "c"]].sum(axis=1),
            1.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            reconstructed[["u", "v"]].sum(axis=1),
            reconstructed["total"],
            atol=1e-12,
        )
        self.assertEqual(constraints.violation_rate(reconstructed), 0.0)

    def test_iterative_linear_projection_is_explicit_and_feasible(self):
        frame = pd.DataFrame({"x": [4.0, -2.0], "y": [3.0, 4.0]})
        constraints = ConstraintSet(
            [
                LinearInequality({"x": 1.0, "y": 1.0}, rhs=2.0),
                LinearInequality({"x": -1.0}, rhs=0.0),
                LinearEquality({"x": 1.0, "y": -1.0}, rhs=0.0),
            ]
        )
        projected, audit = constraints.project(frame, return_audit=True)
        self.assertEqual(constraints.violation_rate(projected), 0.0)
        self.assertTrue(audit.converged)
        self.assertGreaterEqual(audit.iterations, 1)
        self.assertTrue(
            all("maximum_adjustment" in step for step in audit.steps)
        )

    def test_implication_reconstruction_and_condition_safety(self):
        frame = pd.DataFrame(
            {
                "flag": [False, True, False, True],
                "value": [3.0, 0.0, 2.0, 0.0],
            }
        )
        constraints = ConstraintSet(
            [ImplicationConstraint("flag", True, "value", 0.0)]
        )
        transformer = StructuralConstraintTransformer(constraints).fit(frame)
        coordinates = transformer.transform(frame)
        coordinates["flag"] = True
        coordinates["value"] = 5.0
        reconstructed = transformer.inverse_transform(coordinates)
        self.assertEqual(constraints.violation_rate(reconstructed), 0.0)
        self.assertTrue((reconstructed["value"] == 0.0).all())
        with self.assertRaisesRegex(ValueError, "unsafe"):
            transformer.transform_conditions({"value": 5.0}, 2)


class ConstraintEngineTests(unittest.TestCase):
    @staticmethod
    def _fixture(rows=18):
        small = np.linspace(0.0, 3.0, rows)
        gap = np.resize(np.asarray([0.0, 0.25, 1.0]), rows)
        a = np.linspace(0.1, 0.7, rows)
        b = (1.0 - a) * 0.4
        c = 1.0 - a - b
        flag = np.resize(np.asarray([False, True]), rows)
        value = np.where(flag, 0.0, np.linspace(1.0, 2.0, rows))
        return pd.DataFrame(
            {
                "small": small,
                "big": small + gap,
                "a": a,
                "b": b,
                "c": c,
                "flag": flag,
                "value": value,
            }
        )

    def test_audit_completeness_and_engine_save_load_parity(self):
        frame = self._fixture()
        constraints = ConstraintSet(
            [
                PairInequality("small", "big"),
                SimplexConstraint(("a", "b", "c")),
                ImplicationConstraint("flag", True, "value", 0.0),
            ]
        )
        engine = ConditionalGANEngine(
            random_state=31,
            noise_dim=4,
            generator_dims=(8,),
            critic_dims=(8,),
            batch_size=9,
            epochs=1,
            n_critic=1,
            constraints=constraints,
            schema_overrides={
                name: "continuous"
                for name in ("small", "big", "a", "b", "c", "value")
            },
        ).fit(frame)
        encoded = engine.preprocessor.transform(frame)
        roundtrip = engine.preprocessor.inverse_transform(encoded)
        pd.testing.assert_frame_equal(
            frame,
            roundtrip,
            check_exact=False,
            atol=1e-10,
            rtol=1e-10,
        )
        sampled, audit = engine.sample(12, return_audit=True)
        self.assertEqual(constraints.violation_rate(sampled), 0.0)
        required = {
            "raw_violation_rate",
            "pre_projection_violation_rate",
            "post_violation_rate",
            "changed_row_fraction",
            "normalized_intervention_magnitude",
            "equality_boundary_mass",
            "constraint_margins",
        }
        self.assertTrue(required.issubset(audit))
        self.assertEqual(
            set(audit["constraint_margins"]),
            {constraint.label for constraint in constraints},
        )
        requested = frame.iloc[2].to_dict()
        conditioned = engine.sample(3, conditions=requested)
        for column, value in requested.items():
            if isinstance(value, (bool, np.bool_)):
                self.assertTrue((conditioned[column] == bool(value)).all())
            else:
                np.testing.assert_allclose(
                    conditioned[column], float(value), atol=1e-8
                )
        with self.assertRaisesRegex(ValueError, "dependent column"):
            engine.sample(2, conditions={"big": 2.0})

        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "constrained"
            engine.save(destination)
            expected = engine.sample(10)
            restored = ConditionalGANEngine.load(destination)
            actual = restored.sample(10)
        pd.testing.assert_frame_equal(expected, actual)
        self.assertEqual(restored.constraints.to_dict(), constraints.to_dict())
        self.assertEqual(constraints.violation_rate(actual), 0.0)

    def test_standalone_audit_records_projection_intervention(self):
        constraints = ConstraintSet(
            [LinearInequality({"x": 1.0, "y": 1.0}, rhs=1.0)]
        )
        raw = pd.DataFrame({"x": [2.0, 0.2], "y": [2.0, 0.3]})
        projected, projection = constraints.project(
            raw, return_audit=True
        )
        audit = constraint_audit(
            constraints,
            raw,
            projected,
            numeric_scales={"x": 1.0, "y": 1.0},
            projection=projection,
        )
        self.assertGreater(audit["raw_violation_rate"], 0.0)
        self.assertEqual(audit["post_violation_rate"], 0.0)
        self.assertGreater(audit["changed_row_fraction"], 0.0)
        self.assertGreater(
            audit["normalized_intervention_magnitude"], 0.0
        )

    def test_ganify_facade_forwards_constraints_and_audit(self):
        frame = self._fixture()
        constraints = ConstraintSet(
            [
                PairInequality("small", "big"),
                SimplexConstraint(("a", "b", "c")),
            ]
        )
        model = Ganify(random_state=32, random_dim=4, max_units=16)
        model.fit(
            frame,
            conditional=True,
            constraints=constraints,
            epochs=1,
            batch_size=9,
            n_critic=1,
            schema_overrides={
                name: "continuous"
                for name in ("small", "big", "a", "b", "c", "value")
            },
            verbose=0,
        )
        sampled, audit = model.sample(10, return_audit=True)
        self.assertEqual(constraints.violation_rate(sampled), 0.0)
        self.assertEqual(audit["post_violation_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
