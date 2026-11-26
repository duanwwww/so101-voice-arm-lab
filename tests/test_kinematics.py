import os
import unittest
from unittest.mock import patch

import numpy as np

from arm_lab.kinematics import SimulationKinematics, _select_backend


class ShareableKinematicsTests(unittest.TestCase):
    def test_local_translation_preserves_opening_and_respects_joint_limits(self):
        model = SimulationKinematics()
        start = np.array([0.0, -0.28, 0.65, 0.8, 0.0, 0.55])
        goal = model.fk(start) + [0.003, -0.002, 0.001]
        pose, error, success = model.ik(goal, start)
        self.assertTrue(success)
        self.assertAlmostEqual(error, np.linalg.norm(model.fk(pose) - goal))
        self.assertLessEqual(error, 0.0002)
        self.assertEqual(pose[-1], start[-1])
        self.assertTrue(((model.limits[:, 0] <= pose) & (pose <= model.limits[:, 1])).all())

        _, residual, success = model.ik([2, 2, 2], start)
        self.assertFalse(success)
        self.assertGreater(residual, 1.0)

    def test_missing_course_file_selects_standalone_unless_history_is_requested(self):
        with patch("arm_lab.kinematics.find_spec", return_value=None):
            with patch.dict(os.environ, VOICE_ARM_KINEMATICS="auto"):
                self.assertIs(_select_backend(), SimulationKinematics)
            with patch.dict(os.environ, VOICE_ARM_KINEMATICS="course"):
                with self.assertRaises(ImportError):
                    _select_backend()
