"""All motions step the MuJoCo actuators. The free block is initialized only on reset."""
import time
from threading import Event

import mujoco
import numpy as np

from .kinematics import ArmModel, JOINTS
from .scene import (make_scene, taught_pose, STATIONS, LAYOUTS, HALF_BLOCK, TABLE_Z,
                    WIDTH, HEIGHT, OPEN, CLOSED)
from .vision import detect, station_rois, annotate, tabletop_xy_mm
from .motion_guard import MotionGuard
from .gripper import (JawFeedback, FeedbackStop, CLOSE_RATE, CLOSE_FOLLOW_LIMIT,
                      CONTROL_PERIOD, CLOSE_TORQUE_LIMIT, HOLD_TORQUE, HOLD_SETTLE_SECONDS, TEST_LIFT,
                      PICK_RADIUS_MM, MAX_ALIGNMENT_MM, ALIGNMENT_DEADBAND_MM)

PARK = np.array([.82, -.65, .62, .42, 0., OPEN])
TRAVEL_Z = .082
GRIP_Z = TABLE_Z + HALF_BLOCK + .001


class MotionError(Exception):
    pass


class Simulator:
    def __init__(self, layout="A"):
        self.model = make_scene()
        self.data = mujoco.MjData(self.model)
        self.arm = ArmModel()
        self.guard = MotionGuard(self.model)
        self.preview = None
        self.playback_speed = 0
        self.renderer = mujoco.Renderer(self.model, height=HEIGHT, width=WIDTH)
        self.joints = [self.model.joint(n).qposadr[0] for n in JOINTS]
        self.actuators = [self.model.actuator(n).id for n in JOINTS]
        self.tcp = self.model.site("gripperframe").id
        self.block_geom = self.model.geom("red_block").id
        self.grip_pads = {self.model.geom(name).id for name in ("fixed_pad","moving_pad")}
        self.desk_geom = self.model.geom("desk").id
        self.jaw = self.actuators[-1]
        self.jaw_gain = self.model.actuator_gainprm[self.jaw].copy()
        self.jaw_bias = self.model.actuator_biasprm[self.jaw].copy()
        self.jaw_range = self.model.actuator_forcerange[self.jaw].copy()
        self.stop_event = Event()
        self.pause_event = Event()
        self.visual_alignment = True
        self.taught = {(name, z): taught_pose(self.arm, xy, z, OPEN)
                       for name, xy in STATIONS.items() for z in (TRAVEL_Z, GRIP_Z, GRIP_Z+TEST_LIFT)}
        self.reset(layout)

    def reset(self, layout):
        if layout not in LAYOUTS:
            raise ValueError("未知的初始摆位")
        self.layout = layout
        self.stop_event.clear()
        self.pause_event.clear()
        self.pause_applied = False
        self.external_stop_reason = None
        self.fault_reason = None
        self.stop_response = None
        self.last_alignment = None
        self.hold_position = None
        self.jaw_mode = "open"
        self.jaw_log = []
        self._restore_jaw_drive()
        self.model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
        setup = LAYOUTS[layout]
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.joints] = PARK
        self.data.ctrl[self.actuators] = PARK
        i = self.model.joint("block_free").qposadr[0]
        half_angle = np.deg2rad(setup.yaw)/2
        self.data.qpos[i:i+7] = [*setup.xy, TABLE_Z+HALF_BLOCK+.001,np.cos(half_angle),0,0,np.sin(half_angle)]
        self.model.geom_rgba[self.model.geom("block_band").id,3] = float(setup.reflective_band)
        for n,(x,y,color) in enumerate(setup.props):
            name = f"marker_{n}"
            self.data.mocap_pos[self.model.body(name).mocapid[0]] = [x,y,TABLE_Z+HALF_BLOCK]
            self.model.geom_rgba[self.model.geom(name+"_visual").id] = color
        if setup.cover:
            eye = self.model.cam_pos[self.model.camera("overhead").id]
            plane = np.array([*STATIONS[setup.cover],TABLE_Z+2*HALF_BLOCK])
            cover_z = TABLE_Z+.14
            center = eye+(plane-eye)*((cover_z-eye[2])/(plane[2]-eye[2]))
            self.data.mocap_pos[self.model.body("view_cover").mocapid[0]] = center
        self.settle(.45)
        self.rois = station_rois(self.model,self.data)

    def close(self):
        self.renderer.close()

    def _restore_jaw_drive(self):
        self.model.actuator_gainprm[self.jaw] = self.jaw_gain
        self.model.actuator_biasprm[self.jaw] = self.jaw_bias
        self.model.actuator_forcerange[self.jaw] = self.jaw_range

    def lock_motion(self, reason):
        if self.fault_reason is None:
            self.fault_reason = reason
            self._jaw_event("stop")
            preserve_grasp = self.jaw_mode == "hold" and self.holding()
            if not preserve_grasp:
                self._restore_jaw_drive()
            self.model.actuator_forcerange[self.jaw] = [-HOLD_TORQUE, HOLD_TORQUE]
            # Cancel all queued servo targets at the present encoder angles. No
            # automatic release, further closure, home movement or retry follows.
            current = self.joints_now()
            if np.isfinite(current).all():
                self.data.ctrl[self.actuators] = current
                self.stop_response = "bounded_hold" if preserve_grasp else "cancel_at_encoder"
            else:
                self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
                self.stop_response = "disable_simulated_actuation"
            self.jaw_mode = "stopped"
        return self.fault_reason

    def stop(self, reason):
        raise FeedbackStop(self.lock_motion(reason))

    def apply_pause(self):
        """Cancel the trajectory in the worker, leaving the simulated scene at this step."""
        if not self.pause_event.is_set() or self.pause_applied or self.fault_reason or self.stop_event.is_set():
            return
        sample = self._feedback()
        actual = self.joints_now()
        if not np.isfinite(actual).all():
            self.stop("暂停时关节读数无效，已锁止。")
        self.data.ctrl[self.actuators] = actual
        if self.jaw_mode == "hold":
            if sample.fingers != 2:
                self.stop("暂停时夹持反馈丢失，已锁止。")
            # The jaw's position servo stays disabled. A later place() must
            # use the measured opening rather than the unused old target.
            self.hold_position = sample.position
        self._jaw_event("pause")
        self.pause_applied = True

    def resume_here(self):
        """Unlock fresh commands only; neither step physics nor recover an interrupted trajectory."""
        if self.stop_event.is_set() or self.fault_reason:
            raise MotionError("急停或故障已锁止，不能用恢复口令解除。")
        if not self.pause_event.is_set():
            raise MotionError("当前没有暂停的任务。")
        self.apply_pause()
        sample = self._feedback()
        if self.jaw_mode == "hold":
            if sample.fingers != 2 or sample.on_table:
                raise MotionError("当前夹持或离桌状态未确认，请核对夹爪；仿真保持暂停。")
        elif sample.fingers:
            raise MotionError("当前夹爪仍有接触，无法确认负载；仿真保持暂停。")
        else:
            self.jaw_mode = "open"
        self._jaw_event("resume")
        self.pause_applied = False
        self.pause_event.clear()

    def check_ready(self):
        if self.stop_event.is_set():
            self.stop(self.external_stop_reason or "操作者已停止仿真，请检查现场后重置。")
        if self.fault_reason:
            raise FeedbackStop(self.fault_reason)
        if self.pause_event.is_set():
            self.apply_pause()
            raise FeedbackStop("任务已暂停，剩余步骤已取消；现场状态已保留。")

    def apply_pending_stop(self):
        """Consume an idle stop on the simulator's owning worker, without stepping."""
        if self.stop_event.is_set() and not self.fault_reason:
            try:
                self.check_ready()
            except FeedbackStop:
                pass
        return self.gripper_status()

    def _tick(self, steps):
        self.check_ready()
        self._feedback()
        mujoco.mj_step(self.model, self.data, nstep=steps)
        mujoco.mj_forward(self.model, self.data)
        actual = self.joints_now()
        if (not np.isfinite(actual).all() or
                np.any(np.abs(actual[:-1] - self.data.ctrl[self.actuators[:-1]]) > np.deg2rad(12))):
            self.stop("机械臂关节反馈无效或跟随误差过大，已取消轨迹。")
        try:
            self.guard.check_path(actual, actual, carrying=self.jaw_mode == "hold")
        except ValueError as exc:
            self.stop(f"运动实际姿态超出几何保护范围：{exc}")
        if self.jaw_mode == "hold":
            reading = self._feedback()
            if reading.fingers != 2:
                self.stop("握持时指尖接触丢失，已停止后续运动。")

    def settle(self, seconds):
        remaining = round(seconds/self.model.opt.timestep)
        batch = max(1, round(CONTROL_PERIOD/self.model.opt.timestep))
        while remaining > 0:
            count = min(batch, remaining)
            self._tick(count)
            remaining -= count

    def camera(self):
        self.renderer.update_scene(self.data, camera="overhead")
        rgb = self.renderer.render()
        gain = LAYOUTS[self.layout].gain
        if gain != 1:
            rgb = np.rint(rgb*gain).astype(np.uint8)
        scene = detect(rgb, self.rois)
        scene["holding"] = self.holding()
        if not scene["holding"]:
            for obj in scene["objects"]:
                obj["xy_mm"] = tabletop_xy_mm(self.model, self.data, obj["pixel"])
        return scene, annotate(rgb, self.rois, scene)

    def read_gripper(self):
        """Virtual tactile pads and joint encoder; no pose of the payload is used."""
        touching = set()
        for contact in self.data.contact:
            if contact.geom1 == self.block_geom:
                touching.add(contact.geom2)
            elif contact.geom2 == self.block_geom:
                touching.add(contact.geom1)
        return JawFeedback(float(self.data.time), float(self.data.qpos[self.joints[-1]]),
                           len(self.grip_pads & touching), self.desk_geom in touching)

    def _feedback(self):
        sample = self.read_gripper()
        if sample is None or not sample.valid_at(self.data.time):
            self.stop("夹爪反馈缺失、过期或无效，已取消运动目标。")
        if sample.alarm:
            self.stop("夹爪反馈报告驱动报警，已停止并锁定。")
        return sample

    def holding(self):
        frame = self.read_gripper()
        return bool(frame and frame.valid_at(self.data.time) and not frame.alarm
                    and frame.fingers == 2 and not frame.on_table)

    def _jaw_event(self, event):
        self.jaw_log.append({"event": event, "time_s": round(float(self.data.time), 4),
                             "encoder_rad": round(float(self.data.qpos[self.joints[-1]]), 5),
                             "command_rad": round(float(self.data.ctrl[self.jaw]), 5)})

    def gripper_status(self):
        return {"mode": self.jaw_mode, "fault": self.fault_reason,
                "stop_response": self.stop_response,
                "alignment": self.last_alignment,
                "hold_torque_limit_Nm_sim": HOLD_TORQUE,
                "events": list(self.jaw_log)}

    def joints_now(self):
        return self.data.qpos[self.joints].copy()

    def position(self):
        return self.data.site_xpos[self.tcp].copy()

    def move(self, target):
        self.check_ready()
        if (not np.isfinite(target).all() or np.any(target < self.arm.limits[:,0])
                or np.any(target > self.arm.limits[:,1])):
            raise MotionError("关节目标超限")
        start = self.data.ctrl[self.actuators].copy()
        self._preflight(start, target)
        # 10 ms servo targets with 2 ms contact integration. Smooth both endpoints.
        ticks = max(25, int(np.max(np.abs(target-start)) / .009))
        started = time.monotonic()
        for i, f in enumerate(np.linspace(0, 1, ticks)):
            self.check_ready()
            self.data.ctrl[self.actuators] = start + (target-start)*f*f*(3-2*f)
            self._tick(5)
            if self.preview and i % 8 == 0:
                self.preview()
                if self.playback_speed > 0:
                    delay = (i+1)*.01/self.playback_speed - (time.monotonic()-started)
                    if delay > 0:
                        time.sleep(delay)
        self.settle(.22)

    def _preflight(self, start, target):
        actual = self.joints_now()
        carrying = self.jaw_mode == "hold"
        try:
            self.guard.check_path(actual, actual, carrying=carrying)
            self.guard.check_path(start, target, carrying=carrying)
            if carrying:
                # Torque hold has a measured opening distinct from its latched
                # inactive position command. Check the carry at this opening too.
                grip_start, grip_end = start.copy(), target.copy()
                grip_start[-1] = grip_end[-1] = actual[-1]
                self.guard.check_path(grip_start, grip_end, carrying=True)
        except ValueError as exc:
            raise MotionError(str(exc)) from exc

    def station(self, name, z, opening):
        target = self.taught[name,z].copy()
        target[-1] = opening
        self.move(target)

    def _pick_route(self, source, xy_mm):
        """Solve the complete vertical grip path and check it before leaving park."""
        if source not in STATIONS:
            raise MotionError("取料工位无效")
        center_mm = np.asarray(STATIONS[source])*1000
        correction = np.zeros(2)
        if xy_mm is not None:
            observed = np.asarray(xy_mm, dtype=float)
            if observed.shape != (2,) or not np.isfinite(observed).all():
                raise MotionError("抓取中心观测无效")
            delta = observed - center_mm
            span = np.linalg.norm(delta)
            if span > PICK_RADIUS_MM:
                raise MotionError("抓取中心超出允许范围")
            if self.visual_alignment and span > ALIGNMENT_DEADBAND_MM:
                correction = delta * min(1.0, MAX_ALIGNMENT_MM/span)
        xy = (center_mm + correction)/1000
        self.last_alignment = {"source": source, "observation_mm": None if xy_mm is None else observed.tolist(),
                               "correction_mm": correction.round(3).tolist(),
                               "target_xy_mm": (xy*1000).round(3).tolist()}
        try:
            poses = {z: (taught_pose(self.arm, xy, z, OPEN) if np.any(correction)
                         else self.taught[source, z].copy())
                     for z in (TRAVEL_Z, GRIP_Z, GRIP_Z+TEST_LIFT)}
            prior = self.data.ctrl[self.actuators].copy()
            for z, opening, carried in ((TRAVEL_Z, OPEN, False), (GRIP_Z, OPEN, False),
                                        (GRIP_Z, CLOSED, False), (GRIP_Z+TEST_LIFT, CLOSED, True),
                                        (TRAVEL_Z, CLOSED, True)):
                waypoint = poses[z].copy()
                waypoint[-1] = opening
                self.guard.check_path(prior, waypoint, carrying=carried)
                prior = waypoint
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise MotionError(f"抓取路径预检失败：{exc}") from exc
        return poses

    def pick(self, source, phase, xy_mm=None):
        self.check_ready()
        self.jaw_log.clear()
        poses = self._pick_route(source, xy_mm)
        self.move(poses[TRAVEL_Z])
        phase("pregrasp")
        self.move(poses[GRIP_Z])
        phase("descend")
        self._close_on_contact()
        phase("close")
        test_lift = poses[GRIP_Z+TEST_LIFT].copy()
        test_lift[-1] = self.hold_position
        self.move(test_lift)
        phase("test_lift")
        if not self.holding():
            self.stop("试抬后未确认持续双侧夹持和离桌，已停止搬运。")
        lift = poses[TRAVEL_Z].copy()
        lift[-1] = self.hold_position
        self.move(lift)
        phase("lift")

    def _close_on_contact(self):
        start = self.data.ctrl[self.actuators].copy()
        closed = start.copy()
        closed[-1] = CLOSED
        self._preflight(start, closed)
        self.jaw_mode = "closing"
        self._jaw_event("closure_start")
        self.model.actuator_forcerange[self.jaw] = [-CLOSE_TORQUE_LIMIT, CLOSE_TORQUE_LIMIT]
        command = float(self.data.ctrl[self.jaw])
        while command > CLOSED:
            self.check_ready()
            sample = self._feedback()
            if sample.fingers == 2:
                self.hold_position = sample.position
                self.data.ctrl[self.jaw] = sample.position
                self._jaw_event("bilateral_contact")
                # Disengage position tracking; supply a bounded, damped holding
                # effort. No new closing position is queued during lift or carry.
                self.model.actuator_gainprm[self.jaw, 0] = 0
                self.model.actuator_biasprm[self.jaw, :3] = [-HOLD_TORQUE, 0, -0.2]
                self.model.actuator_forcerange[self.jaw] = [-HOLD_TORQUE, HOLD_TORQUE]
                self.jaw_mode = "hold"
                self.settle(HOLD_SETTLE_SECONDS)
                return
            if abs(sample.position-command) > CLOSE_FOLLOW_LIMIT:
                self.stop("合爪跟随误差过大且未确认双侧接触，已停止。")
            command = max(CLOSED, command-CLOSE_RATE*CONTROL_PERIOD)
            self.data.ctrl[self.jaw] = command
            self.settle(CONTROL_PERIOD)
        self.stop("合爪行程内未确认双侧接触，已停止，未试抬。")

    def place(self, target, phase):
        if self.jaw_mode != "hold" or not self.holding():
            self.stop("搬运前无法确认夹持状态。")
        self.station(target,TRAVEL_Z,self.hold_position)
        phase("carry")
        self.station(target,GRIP_Z,self.hold_position)
        self._jaw_event("release")
        self.jaw_mode = "releasing"
        self._restore_jaw_drive()
        self.data.ctrl[self.jaw] = self.joints_now()[-1]
        self.station(target,GRIP_Z,OPEN)
        self.jaw_mode = "open"
        self.hold_position = None
        phase("release")
        self.station(target,TRAVEL_Z,OPEN)
        self.home()
        phase("retreat")

    def home(self):
        self.move(PARK)

    def translate(self, axis, mm):
        self.check_ready()
        if axis not in ("x", "y", "z") or not np.isfinite(mm) or not 0 < abs(mm) <= 50:
            raise MotionError("每次平移应在 0 到 50 mm 之间")
        index = "xyz".index(axis)
        origin = self.position()
        q = self.joints_now()
        waypoints = []
        for delta in np.linspace(0, mm/1000, max(2, int(np.ceil(abs(mm)/5))+1))[1:]:
            point = origin.copy()
            point[index] += delta
            if point[2] < TABLE_Z + .022:
                raise MotionError("目标低于定量微调的桌面净空")
            q, _, ok = self.arm.ik(point, q)
            if not ok:
                raise MotionError("该段路径的 IK 未收敛")
            q[-1] = self.data.ctrl[self.actuators[-1]]
            waypoints.append(q.copy())
        # Reject a blocked later segment before executing an earlier segment of
        # this same quantitative command. Each move still checks live feedback.
        previous = self.data.ctrl[self.actuators].copy()
        for waypoint in waypoints:
            self._preflight(previous, waypoint)
            previous = waypoint
        for q in waypoints:
            self.move(q)
        observed = (self.position()-origin)*1000
        desired = np.zeros(3)
        desired[index] = mm
        return {"requested_mm": mm, "displacement_mm": observed.round(3).tolist(),
                "error_mm": round(float(np.linalg.norm(observed-desired)), 3)}

    def rotate_wrist(self, degrees):
        self.check_ready()
        if not np.isfinite(degrees) or not 0 < abs(degrees) <= 15:
            raise MotionError("每次转腕应在 0 到 15 度之间")
        before = self.joints_now()
        target = before.copy()
        target[-1] = self.data.ctrl[self.actuators[-1]]
        target[3] += np.deg2rad(degrees)
        if min(self.arm.fk(q)[2] for q in np.linspace(before,target,12)) < TABLE_Z+.022:
            raise MotionError("转腕路径低于桌面净空")
        self.move(target)
        actual = float(np.rad2deg(self.joints_now()[3]-before[3]))
        return {"requested_deg": degrees, "actual_deg": round(actual,3),
                "error_deg": round(abs(actual-degrees),3)}
