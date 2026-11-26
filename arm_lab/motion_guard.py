"""Kinematic preflight for the calibrated SO-101 simulation workcell, in metres."""
from dataclasses import dataclass
import json
from pathlib import Path

import mujoco
import numpy as np

from .kinematics import JOINTS
from .gripper import PICK_RADIUS_MM
from .scene import GRASP_CENTER, HALF_BLOCK


@dataclass(frozen=True)
class KeepoutBox:
    """Axis-aligned fixture or forbidden volume measured in the arm base frame."""

    name: str
    center: tuple[float, float, float]
    half_size: tuple[float, float, float]


class MotionGuard:
    """Check the joint interpolation using a separate forward-kinematics data object."""

    TCP_MIN = np.array([0.12, -0.26, 0.035])
    TCP_MAX = np.array([0.32, 0.18, 0.25])
    MARGIN = 0.001  # 1 mm nominal clearance in the simulator, not a hardware tolerance.
    # A 24 mm cube in any orientation fits its circumsphere. Allow the full
    # admitted pick offset and another 3 mm for centre/calibration uncertainty.
    PAYLOAD_RADIUS = np.sqrt(3) * HALF_BLOCK + PICK_RADIUS_MM / 1000 + 0.003
    LINKS = ("base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper",
             "moving_jaw_so101_v1")

    @classmethod
    def from_config(cls, model, path):
        """Load base-frame boxes. Require explicit units to avoid mm/m mistakes."""
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"无法读取障碍配置 {path}：{exc}") from exc
        if not isinstance(payload, dict) or set(payload) != {"frame", "units", "boxes"}:
            raise ValueError("障碍配置必须包含 frame、units、boxes")
        if payload["frame"] != "arm_base" or payload["units"] != "m":
            raise ValueError("障碍配置须使用 arm_base 坐标系和米（m）")
        if not isinstance(payload["boxes"], list):
            raise ValueError("boxes 应是障碍盒列表")
        boxes = []
        names = set()
        for row in payload["boxes"]:
            if not isinstance(row, dict) or set(row) != {"name", "center", "half_size"}:
                raise ValueError("障碍盒必须包含 name、center、half_size")
            name = row["name"]
            if not isinstance(name, str) or not name.strip() or name in names:
                raise ValueError("障碍盒名称应非空且唯一")
            names.add(name)
            for key in ("center", "half_size"):
                xyz = row[key]
                if (not isinstance(xyz, list) or len(xyz) != 3
                        or not all(type(x) in (float, int) for x in xyz)):
                    raise ValueError(f"{name} 的 {key} 应是三个数值")
            boxes.append(KeepoutBox(name, tuple(row["center"]), tuple(row["half_size"])))
        return cls(model, keepouts=boxes)

    def __init__(self, model, *, keepouts=()):
        self.model = model
        self.data = mujoco.MjData(model)
        self.jids = [model.joint(name).id for name in JOINTS]
        self.qpos = model.jnt_qposadr[self.jids]
        self.limits = model.jnt_range[self.jids]
        self.tcp = model.site("gripperframe").id
        self.tcp_levers = self._levers(model.site_bodyid[self.tcp],
                                       np.linalg.norm(model.site_pos[self.tcp]))
        self.gripper_body = model.body("gripper").id
        self.payload_levers = self._levers(self.gripper_body, np.linalg.norm(GRASP_CENTER))
        self.desk = model.geom("desk").id
        pads = {model.geom("fixed_pad").id, model.geom("moving_pad").id}
        links = []
        for index, name in enumerate(self.LINKS):
            # The base has only visual meshes. The other links have named group-3
            # collision copies even though their physics collision masks are disabled.
            group = 2 if index == 0 else 3
            geoms = [g for g in range(model.ngeom)
                     if model.geom_bodyid[g] == model.body(name).id
                     and (model.geom_group[g] == group or g in pads)]
            if not geoms:
                raise ValueError(f"缺少碰撞检测使用的臂段几何体：{name}")
            links.append(geoms)

        self.robot_geoms = np.array([g for link in links for g in link])
        self.levers = np.zeros((model.ngeom, len(self.jids)))
        for g in self.robot_geoms:
            radius = np.linalg.norm(model.geom_pos[g]) + model.geom_rbound[g]
            self.levers[g] = self._levers(model.geom_bodyid[g], radius)

        # The base and the shoulder's rotary mounting flange pass through the
        # desktop in the course asset. They are tested against fixtures and self.
        pairs = [(g, self.desk) for link in links[2:] for g in link]
        for i, link in enumerate(links):
            # Parent/child surfaces overlap at designed hinge mounts.
            for other in links[i+2:]:
                pairs.extend((g, h) for g in link for h in other)
        self.first, self.second = np.asarray(pairs, dtype=int).T
        a = self.levers[self.first]
        b = self.levers[self.second]
        # A common upstream rotation has no effect on two links' relative distance.
        self.relative_levers = np.where((a > 0) & (b > 0), 0, a + b)
        self.keepouts = []
        for box in keepouts:
            center = np.asarray(box.center, dtype=float)
            half = np.asarray(box.half_size, dtype=float)
            if (center.shape != (3,) or half.shape != (3,) or not np.all(half > 0)
                    or not np.all(np.isfinite([center, half]))):
                raise ValueError("障碍盒坐标或半尺寸无效")
            self.keepouts.append((box.name, center-half, center+half))

    def _levers(self, body, radius):
        # For a rotation dq, the travel of any point bounded by radius R is
        # at most R*abs(dq). Sum link lengths to cover all intervening poses.
        radii = np.zeros(len(self.jids))
        while body:
            for n, jid in enumerate(self.jids):
                if self.model.jnt_bodyid[jid] == body:
                    radii[n] = radius + np.linalg.norm(self.model.jnt_pos[jid])
            radius += np.linalg.norm(self.model.body_pos[body])
            body = self.model.body_parentid[body]
        return radii

    def check_path(self, start, target, *, carrying=False):
        """Raise before dispatch if the complete linear joint segment cannot be cleared."""
        start = np.asarray(start, dtype=float)
        target = np.asarray(target, dtype=float)
        for pose in (start, target):
            if pose.shape != (len(self.jids),) or not np.all(np.isfinite(pose)):
                raise ValueError("路径包含无效关节读数")
            if np.any(pose < self.limits[:, 0]) or np.any(pose > self.limits[:, 1]):
                raise ValueError("路径超出关节范围")

        pending = [(start, target, 0)]
        probes = 0
        while pending:
            left, right, depth = pending.pop()
            middle = (left + right) * 0.5
            travel = np.abs(right-left) * 0.5
            probes += 1
            if self._interval_clear(middle, travel, carrying):
                continue
            if depth >= 16 or probes >= 4096:
                raise ValueError("路径过于接近边界，无法确认整段净空")
            pending.append((middle, right, depth+1))
            pending.append((left, middle, depth+1))

    def _interval_clear(self, middle, travel, carrying):
        m, d = self.model, self.data
        d.qpos[self.qpos] = middle
        mujoco.mj_kinematics(m, d)
        tcp = d.site_xpos[self.tcp]
        workspace_gap = np.min(np.minimum(tcp-self.TCP_MIN, self.TCP_MAX-tcp))
        if workspace_gap <= 0:
            raise ValueError("末端超出当前标定工作区")
        clear = workspace_gap > self.tcp_levers @ travel

        rot = d.geom_xmat.reshape(-1, 3, 3)
        center = d.geom_xpos + np.einsum("ijk,ik->ij", rot, m.geom_aabb[:, :3])
        half = np.einsum("ijk,ik->ij", np.abs(rot), m.geom_aabb[:, 3:])
        low, high = center-half, center+half
        a, b = self.first, self.second
        delta = np.maximum(low[a]-high[b], low[b]-high[a])
        gaps = np.linalg.norm(np.maximum(delta, 0), axis=1)
        needed = self.MARGIN + self.relative_levers @ travel
        # AABB separation clears most pairs. Query the convex CAD hulls for
        # close ones; the margin plus the swept-motion bound covers the interval.
        for k in np.flatnonzero(gaps <= needed):
            first, second = int(a[k]), int(b[k])
            dist = mujoco.mj_geomDistance(m, d, first, second, needed[k] + 1e-6, None)
            if not np.isfinite(dist) or dist <= self.MARGIN:
                lhs = m.body(m.geom_bodyid[first]).name
                rhs = "桌面" if second == self.desk else m.body(m.geom_bodyid[second]).name
                raise ValueError(f"路径净空不足：{lhs} 与 {rhs}")
            if dist <= needed[k]:
                clear = False

        for name, box_low, box_high in self.keepouts:
            g = self.robot_geoms
            delta = np.maximum(low[g]-box_high, box_low-high[g])
            gaps = np.linalg.norm(np.maximum(delta, 0), axis=1)
            if np.any(gaps <= self.MARGIN):
                raise ValueError(f"臂段包围盒接近已登记障碍：{name}")
            if np.any(gaps <= self.MARGIN + self.levers[g] @ travel):
                clear = False
            if carrying:
                # Use jaw-frame geometry and a declared payload only. Runtime
                # never reads free-block position/orientation from MuJoCo.
                body = self.gripper_body
                center = d.xpos[body] + d.xmat[body].reshape(3, 3) @ GRASP_CENTER
                outside = np.maximum(np.maximum(box_low-center, center-box_high), 0)
                payload_gap = np.linalg.norm(outside) - self.PAYLOAD_RADIUS
                if payload_gap <= self.MARGIN:
                    raise ValueError(f"所持木块安全包络接近已登记障碍：{name}")
                if payload_gap <= self.MARGIN + self.payload_levers @ travel:
                    clear = False
        return clear
