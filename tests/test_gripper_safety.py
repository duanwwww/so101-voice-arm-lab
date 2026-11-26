from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
import mujoco

from arm_lab.flow import execute
from arm_lab.gripper import FeedbackStop, HOLD_TORQUE
from arm_lab.scene import OPEN
from arm_lab.simulator import Simulator, MotionError


class GripperSafetyTests(unittest.TestCase):
    def setUp(self):
        self.sim = Simulator("A")
        self.addCleanup(self.sim.close)

    def assert_latched(self):
        sim = self.sim
        self.assertEqual(sim.jaw_mode, "stopped")
        np.testing.assert_allclose(sim.data.ctrl[sim.actuators], sim.joints_now(), atol=1e-8)
        time = sim.data.time
        def unexpected_planner(*args):
            self.fail("A latched command reached the planner")
        receipt = execute(sim, "向前移动一厘米", planner=unexpected_planner)
        self.assertEqual((receipt["status"], receipt["plan"]), ("unverified", []))
        self.assertEqual(time, sim.data.time)

    def test_verified_contact_holds_without_further_closing_position_commands(self):
        sampled = []
        def sample(label):
            if label in ("close", "test_lift", "lift", "carry"):
                s = self.sim
                sampled.append((label, s.data.ctrl[s.jaw], s.model.actuator_gainprm[s.jaw, 0],
                                s.data.actuator_force[s.jaw], s.holding()))
        self.sim.pick("A", sample)
        self.sim.place("C", sample)
        self.assertEqual([row[0] for row in sampled], ["close", "test_lift", "lift", "carry"])
        self.assertTrue(all(command == sampled[0][1] and gain == 0 and abs(torque) <= HOLD_TORQUE
                            for _, command, gain, torque, _ in sampled))
        self.assertTrue(all(row[-1] for row in sampled[1:]))
        self.assertEqual(self.sim.jaw_mode, "open")
        self.assertEqual(self.sim.camera()[0]["objects"][0]["slot"], "C")

    def test_no_bilateral_contact_never_lifts(self):
        for simulated_fingers in (0, 1):
            with self.subTest(simulated_fingers=simulated_fingers):
                self.sim.reset("edge_grasp")
                reader = self.sim.read_gripper
                def partial():
                    return replace(reader(), fingers=simulated_fingers)
                phases = []
                with patch.object(self.sim, "read_gripper", side_effect=partial):
                    with self.assertRaisesRegex(FeedbackStop, "未确认双侧接触"):
                        # Exercise the actuator's independent protection even if
                        # an upstream client bypasses the 8 mm camera gate.
                        self.sim.pick("A", phases.append)
                self.assertNotIn("test_lift", phases)
                self.assert_latched()

    def test_bad_feedback_cancels_and_locks_before_lifting(self):
        for fault in ("absent", "stale", "nan", "alarm", "stuck"):
            with self.subTest(fault=fault):
                self.sim.reset("A")
                reader = self.sim.read_gripper
                def inject():
                    sample = reader()
                    if self.sim.jaw_mode != "closing":
                        return sample
                    if fault == "absent":
                        return None
                    if fault == "stale":
                        return replace(sample, time=sample.time-0.1)
                    if fault == "nan":
                        return replace(sample, position=float("nan"))
                    if fault == "alarm":
                        return replace(sample, alarm=True)
                    return replace(sample, position=OPEN, fingers=0)
                with patch.object(self.sim, "read_gripper", side_effect=inject):
                    receipt = execute(self.sim, "向前抓取木块")
                self.assertEqual(receipt["status"], "unverified")
                self.assertNotIn("test_lift", receipt["phases"])
                self.assert_latched()

    def test_lost_grip_during_carry_and_stop_button_latch(self):
        s = self.sim
        s.pick("A", lambda _: None)
        reader = s.read_gripper
        with patch.object(s, "read_gripper", side_effect=lambda: replace(reader(), fingers=1)):
            with self.assertRaisesRegex(FeedbackStop, "接触丢失"):
                s.translate("y", 5)
        self.assert_latched()
        s.reset("B")
        s.stop_event.set()
        receipt = execute(s, "向前移动一厘米")
        self.assertIn("操作者已停止", receipt["message"])
        self.assert_latched()
        s.reset("B")
        self.assertEqual(execute(s, "向前移动一厘米")["status"], "success")

    def test_operator_stop_with_suspended_block_preserves_capped_hold(self):
        s = self.sim
        s.pick("A", lambda phase: None)
        s.stop_event.set()
        self.assertEqual(execute(s, "放到 B 区")["status"], "unverified")
        self.assert_latched()
        self.assertEqual(s.stop_response, "bounded_hold")
        self.assertEqual(s.model.actuator_gainprm[s.jaw, 0], 0)
        # Independent physics advance, even though application motions are locked.
        mujoco.mj_step(s.model, s.data, nstep=round(.2 / s.model.opt.timestep))
        mujoco.mj_forward(s.model, s.data)
        self.assertTrue(s.holding())
        self.assertLessEqual(abs(s.data.actuator_force[s.jaw]), HOLD_TORQUE)

    def test_reject_offset_without_downward_motion(self):
        self.sim.reset("edge_grasp")
        before = self.sim.joints_now()
        receipt = execute(self.sim, "把木块放到 C 区")
        self.assertEqual(receipt["status"], "unverified")
        self.assertGreater(receipt["measurement"]["observed_offset_mm"], 8)
        self.assertEqual(receipt["plan"], [])
        np.testing.assert_array_equal(self.sim.joints_now(), before)

    def test_alignment_is_bounded_and_offset_rejection_leaves_arm_untouched(self):
        s = self.sim
        before = s.joints_now()
        initial_time = s.data.time
        # A taught mark may project up to 2 mm from its calibrated centre.
        # Chasing that discrepancy displaced the next block in a B→A→B relay.
        s._pick_route("A", [215.3, 77.5])
        self.assertEqual(s.last_alignment["correction_mm"], [0, 0])
        s._pick_route("A", [221, 81])  # sqrt(6²+4²) from taught centre.
        delta = s.last_alignment["correction_mm"]
        self.assertAlmostEqual(np.linalg.norm(delta), 6, places=3)
        self.assertAlmostEqual(delta[0]/delta[1], 1.5, places=3)
        s.visual_alignment = False
        s._pick_route("A", [221, 81])
        self.assertEqual(s.last_alignment["correction_mm"], [0, 0])
        with self.assertRaises(MotionError):
            s._pick_route("A", [224, 77])
        np.testing.assert_array_equal(s.joints_now(), before)
        self.assertEqual(s.data.time, initial_time)

    def test_pregrasp_requires_fresh_unique_target_at_original_station(self):
        s = self.sim
        camera = s.camera
        calls = 0
        def changed_frame():
            nonlocal calls
            calls += 1
            scene, frame = camera()
            if calls >= 2:
                scene["objects"][0]["slot"] = "B"
            return scene, frame
        before = s.joints_now()
        with patch.object(s, "camera", side_effect=changed_frame):
            receipt = execute(s, "向前抓取木块")
        self.assertEqual((receipt["status"], receipt["plan"]), ("unverified", []))
        self.assertIn("pregrasp_observation", receipt["phases"])
        np.testing.assert_array_equal(s.joints_now(), before)

    def test_blocked_trial_lift_is_found_before_descent(self):
        s = self.sim
        real_check = s.guard.check_path
        def block_lift(start, end, *, carrying=False):
            if carrying:
                raise ValueError("试抬路径被障碍占据")
            return real_check(start, end)
        initial = s.data.time
        with patch.object(s.guard, "check_path", side_effect=block_lift):
            receipt = execute(s, "向前抓取木块")
        self.assertEqual(receipt["status"], "rejected")
        self.assertNotIn("pregrasp", receipt["phases"])
        self.assertEqual(initial, s.data.time)
