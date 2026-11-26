import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from arm_lab.motion_guard import KeepoutBox, MotionGuard
from arm_lab.scene import OPEN, STATIONS, make_scene, taught_pose
from arm_lab.simulator import GRIP_Z, PARK, TRAVEL_Z
from arm_lab.kinematics import ArmModel


class MotionGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = make_scene()
        cls.guard = MotionGuard(cls.model)
        arm = ArmModel()
        cls.taught = {
            (name, height): taught_pose(arm, xy, height, OPEN)
            for name, xy in STATIONS.items()
            for height in (GRIP_Z, TRAVEL_Z)
        }

    def test_all_default_stations_and_interstation_routes_clear(self):
        poses = [PARK, *self.taught.values()]
        for start in poses:
            for end in poses:
                self.guard.check_path(start, end)

    def test_stationary_lower_arm_hits_table_while_tcp_inside_workcell(self):
        q = [0.7523, 1.7318, -1.146, -1.5058, 0.9183, 1.1572]
        with self.assertRaisesRegex(ValueError, "lower_arm 与 桌面"):
            self.guard.check_path(q, q)

    def test_non_adjacent_shoulder_and_forearm_can_hit(self):
        q = [-0.1481, -1.0485, 1.594, -0.0997, -1.8972, 0.5695]
        with self.assertRaisesRegex(ValueError, "shoulder 与 lower_arm"):
            self.guard.check_path(q, q)

    def test_sweep_finds_fixture_missed_by_ends_and_midpoint(self):
        fix = KeepoutBox("支架", (0.27641, 0.04733, 0.09096), (0.002,)*3)
        guard = MotionGuard(self.model, keepouts=[fix])
        a = self.taught["A", TRAVEL_Z]
        b = self.taught["B", TRAVEL_Z]
        for q in (a, (a+b)*0.5, b):
            guard.check_path(q, q)
        with self.assertRaisesRegex(ValueError, "支架"):
            guard.check_path(a, b)

    def test_limits_and_invalid_measurements_are_failures(self):
        for q in (np.array([np.nan]*6), PARK[:5], PARK+5):
            with self.subTest(q=q), self.assertRaises(ValueError):
                self.guard.check_path(PARK, q)
        outside = PARK.copy()
        outside[0] = 1.8
        with self.assertRaisesRegex(ValueError, "工作区"):
            self.guard.check_path(outside, outside)

    def test_invalid_fixture_is_rejected_at_configuration(self):
        for center, half in [((0,0,np.nan), (0.01,)*3), ((0,0,0), (0.01,-0.01,0.01))]:
            with self.subTest(center=center, half=half), self.assertRaises(ValueError):
                MotionGuard(self.model, keepouts=[KeepoutBox("支架", center, half)])

    def test_example_fixture_blocks_one_of_two_approaches(self):
        config = Path(__file__).resolve().parents[1] / "configs" / "keepouts.example.json"
        guard = MotionGuard.from_config(self.model, config)
        guard.check_path(PARK, self.taught["B", TRAVEL_Z])
        with self.assertRaisesRegex(ValueError, "仿真示例相机支架"):
            guard.check_path(PARK, self.taught["A", TRAVEL_Z])

    def test_payload_hits_post_that_the_arm_clears(self):
        from scripts.evaluate_motion_guard import payload_ablation
        result = payload_ablation()
        self.assertTrue(result["bare_arm_sweep_clear_at_commanded_and_measured_jaw"])
        self.assertTrue(result["reference_intersections"])
        self.assertEqual(result["guarded_pick_status"], "success")
        self.assertEqual(result["guarded_place_status"], "rejected")
        self.assertIn("所持木块", result["reason"])
        self.assertEqual(result["max_joint_change_rad_on_refusal"], 0)
        self.assertTrue(result["still_holding"])

    def test_config_rejects_unlabelled_units_and_invalid_box_values(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixtures.json"
            for row in ({"boxes":[]}, {"frame":"camera","units":"m","boxes":[]},
                        {"frame":"arm_base","units":"mm","boxes":[]},
                        {"frame":"arm_base","units":"m","boxes":[
                            {"name":"x","center":[0,0,False],"half_size":[1,1,1]}]}):
                with self.subTest(row=row):
                    path.write_text(json.dumps(row), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        MotionGuard.from_config(self.model, path)
