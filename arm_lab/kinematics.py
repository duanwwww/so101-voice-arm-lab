"""Kinematics for the SO-101 simulation. This module has no device interface."""
from importlib.util import find_spec
import os
from pathlib import Path

import mujoco
from mujoco import minimize
import numpy as np


JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
MODEL_XML = Path(__file__).resolve().parents[1] / "vendor" / "so101_course" / "SO101" / "so101_new_calib.xml"


class SimulationKinematics:
    """Evaluate the official MJCF and solve XYZ with MuJoCo's bounded least squares."""

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
        self.kin_data = mujoco.MjData(self.model)
        joint_ids = [self.model.joint(name).id for name in JOINTS]
        self._positions = self.model.jnt_qposadr[joint_ids]
        self.limits = self.model.jnt_range[joint_ids].copy()
        self._tip = self.model.site("gripperframe").id

    def fk(self, q):
        """World-frame gripper site in metres, given six simulated joint angles in radians."""
        angles = np.asarray(q, dtype=float)
        if angles.shape != (len(JOINTS),) or not np.isfinite(angles).all():
            raise ValueError("Expected six finite joint angles")
        self.kin_data.qpos[self._positions] = angles
        mujoco.mj_kinematics(self.model, self.kin_data)
        return self.kin_data.site_xpos[self._tip].copy()

    def ik(self, target, seed, tolerance=0.0002, iterations=100):
        """Return (joint angles, XYZ error in metres, within tolerance). Preserve jaw opening."""
        goal = np.asarray(target, dtype=float)
        if goal.shape != (3,) or not np.isfinite(goal).all():
            raise ValueError("Expected a finite XYZ target in metres")
        pose = np.asarray(seed, dtype=float).copy()
        self.fk(pose)
        lower, upper = self.limits[:-1].T
        origin = np.clip(pose[:-1], lower, upper)

        def residual(candidates):
            # MuJoCo batches the base point with finite-difference samples in columns.
            errors = np.empty((3, candidates.shape[1]))
            for col, arm in enumerate(candidates.T):
                pose[:-1] = arm
                errors[:, col] = self.fk(pose) - goal
            # XYZ alone leaves two free axes. Prefer a small change from the current posture.
            posture = 0.0005 * (candidates - origin[:, None])
            return np.vstack((errors, posture))

        solution, _ = minimize.least_squares(
            origin, residual, bounds=(lower, upper), max_iter=iterations, verbose=0
        )
        pose[:-1] = solution.reshape(-1)
        error_m = float(np.linalg.norm(self.fk(pose) - goal))
        return pose, error_m, error_m <= tolerance


def _select_backend():
    mode = os.getenv("VOICE_ARM_KINEMATICS", "auto").strip().lower()
    if mode not in {"auto", "standalone", "course"}:
        raise ValueError("VOICE_ARM_KINEMATICS must be auto, standalone or course")
    if mode != "standalone" and find_spec("vendor.so101_course.so101_core") is not None:
        from vendor.so101_course.so101_core import ArmModel as CourseArm
        return CourseArm
    if mode == "course":
        raise ImportError("Historical course kinematics requested but so101_core.py is unavailable")
    return SimulationKinematics


ArmModel = _select_backend()
