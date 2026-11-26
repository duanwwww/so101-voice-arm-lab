"""Independent physical scoring regressions; no ASR, network or model calls."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from scripts.evaluate_reasoner_flow10 import (
    CONFIG, check_runtime_unchanged, physical_checks, prepare, record_failure,
)


TOLERANCES = {"move_mm": 5, "wrist_deg": 2, "lift_mm": 20}


def snapshot(xyz=(100, 50, 120), wrist=0.0):
    return {"tcp_mm": list(xyz), "joints": [0, 0, 0, wrist, 0, 0]}


def row(before=None, after=None, frames=None, status="success"):
    return {"before": before or snapshot(), "after": after or snapshot(),
            "frames": frames or [], "receipt": {"status": status,
                "measurement": {"actual_deg": 2.5, "error_deg": 0, "error_mm": 0}}}


class Flow10ScoringTests(unittest.TestCase):
    def test_wrist_uses_joint_delta_instead_of_receipt_or_absolute_angle(self):
        start = .8
        sample = row(snapshot(wrist=start), snapshot(wrist=start+np.deg2rad(-2.5)))
        checks = physical_checks({"goal": "wrist", "wrist_deg": -2.5}, sample, TOLERANCES)
        self.assertTrue(all(checks.values()), checks)
        self.assertAlmostEqual(sample["independent_wrist_deg"], -2.5)
        self.assertAlmostEqual(sample["independent_wrist_error_deg"], 0)
        # Persisted scores must use Python JSON values, never NumPy scalar bools.
        self.assertEqual(json.loads(json.dumps(checks)), checks)

    def test_success_receipt_cannot_make_a_stationary_small_movement_pass(self):
        sample = row()
        checks = physical_checks({"goal": "move", "delta_mm": [0, 3, 0]}, sample, TOLERANCES)
        self.assertTrue(checks["move_within_tolerance"])
        self.assertFalse(checks["move_nonzero_correct_direction"])
        self.assertFalse(all(checks.values()))

    def test_opposite_small_movement_fails_even_inside_distance_tolerance(self):
        sample = row(after=snapshot((100, 49, 120)))
        checks = physical_checks({"goal": "move", "delta_mm": [0, 1, 0]}, sample, TOLERANCES)
        self.assertTrue(checks["move_within_tolerance"])
        self.assertFalse(checks["move_nonzero_correct_direction"])

    def test_wrong_axis_cannot_satisfy_nonzero_required_direction(self):
        sample = row(after=snapshot((101, 50, 120)))
        checks = physical_checks({"goal": "move", "delta_mm": [0, 1, 0]}, sample, TOLERANCES)
        self.assertTrue(checks["move_within_tolerance"])
        self.assertFalse(checks["move_nonzero_correct_direction"])

    def test_stationary_and_wrong_sign_wrist_do_not_pass_a_loose_tolerance(self):
        for degrees in (0, -1):
            with self.subTest(actual_degrees=degrees):
                sample = row(after=snapshot(wrist=np.deg2rad(degrees)))
                checks = physical_checks({"goal": "wrist", "wrist_deg": 1}, sample, TOLERANCES)
                self.assertTrue(checks["wrist_within_tolerance"])
                self.assertFalse(checks["wrist_nonzero_correct_direction"])

    def test_move_then_wrist_uses_boundary_to_separate_two_physical_actions(self):
        before = snapshot(wrist=.2)
        boundary = snapshot((100, 55, 120), wrist=.2)
        # Wrist rotation can change TCP; do not charge this to the translation.
        after = snapshot((110, 60, 125), wrist=.2+np.deg2rad(2.5))
        sample = row(before, after, [{"phase": "before_step_2", "physics": boundary}])
        checks = physical_checks({"goal": "move_wrist", "delta_mm": [0, 5, 0], "wrist_deg": 2.5},
                                 sample, TOLERANCES)
        self.assertTrue(all(checks.values()), checks)
        self.assertEqual(sample["independent_displacement_mm"], [0, 5, 0])
        self.assertAlmostEqual(sample["independent_wrist_deg"], 2.5)

    def test_missing_step_boundary_does_not_fall_back_to_success_receipt(self):
        sample = row(after=snapshot((100, 55, 120), wrist=np.deg2rad(2.5)))
        checks = physical_checks({"goal": "move_wrist", "delta_mm": [0, 5, 0], "wrist_deg": 2.5},
                                 sample, TOLERANCES)
        self.assertFalse(checks["move_within_tolerance"])
        self.assertFalse(checks["wrist_within_tolerance"])

    def test_nonfinite_measurement_and_failed_receipt_cannot_pass(self):
        for sample in (row(after=snapshot(wrist=float("nan"))),
                       row(after=snapshot(wrist=np.deg2rad(2.5)), status="unverified")):
            with self.subTest(sample=sample):
                checks = physical_checks({"goal": "wrist", "wrist_deg": 2.5}, sample, TOLERANCES)
                self.assertFalse(all(checks.values()))

    def test_declared_asr_must_match_runtime_client(self):
        spec = json.loads(CONFIG.read_text(encoding="utf-8"))
        spec["asr_model"] = "different-model"
        with TemporaryDirectory() as directory:
            config = Path(directory)/"config.json"
            config.write_text(json.dumps(spec), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ASR model"):
                prepare(config)

    def test_changed_runtime_is_detected(self):
        with patch("scripts.evaluate_reasoner_flow10.runtime_fingerprints", return_value={"planner": "changed"}):
            with self.assertRaisesRegex(RuntimeError, "Runtime code changed"):
                check_runtime_unchanged({"runtime_sha256": {"planner": "original"}})

    def test_scoring_exception_is_recorded_and_reraised(self):
        report = {"state": "started", "rows": [{"id": "trial"}]}
        saved = []
        with self.assertRaisesRegex(KeyError, "missing evidence"):
            with record_failure(report, lambda: saved.append(deepcopy(report))):
                raise KeyError("missing evidence")
        self.assertEqual(report["state"], "aborted")
        self.assertEqual(saved[-1]["rows"][0]["harness_error_type"], "KeyError")


if __name__ == "__main__":
    unittest.main()
