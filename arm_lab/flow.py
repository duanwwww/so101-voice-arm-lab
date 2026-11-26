"""Observe, plan, validate, execute each step and report what the sensors confirmed."""
from dataclasses import asdict
import time
import math
import re

from .model_access import DEEPSEEK_MODEL
from .planner import (Intent, actions, ask, check_steps, classify_control_text, local_intent,
                      missing_motion_unit, offline_reply, source_slot)
from .simulator import MotionError
from .gripper import FeedbackStop, PICK_RADIUS_MM
from .scene import STATIONS


def interaction_intent(text):
    phrase = re.sub(r"[\s，,。.!！?？]", "", text).translate(
        str.maketrans("復請暫繼續執況現狀態務進還麼當問", "复请暂继续执况现状态务进还么当问"))
    if re.fullmatch(r"(?:请|麻烦)?(?:恢复控制|恢复操作|解除暂停|继续接收指令)", phrase):
        return Intent("resume")
    if re.fullmatch(r"(?:请问)?(?:现在|当前|目前)?(?:任务)?(?:状态|什么状态|执行情况|进度(?:怎么样|如何)?|进行到哪一步了?|执行到哪一步了?|完成了吗|在做什么|在干什么|还在执行吗|暂停了吗|是不是暂停了)", phrase):
        return Intent("status")
    directive = classify_control_text(text)
    if directive:
        return directive
    if missing_motion_unit(text):
        return ask("请补充距离或角度的单位，例如毫米、厘米或度。")
    return None


def interaction_receipt(text, scene, status, message):
    receipt = {"text":text, "planner":"local_control", "status":status, "message":message,
               "before":scene, "after":scene, "steps":[], "plan":[], "phases":[], "measurement":{}}
    if status == "clarify" and missing_motion_unit(text):
        receipt["clarification"] = {"command":text, "question":message}
    return receipt


def run_step(sim, item, scene, capture, log):
    issue = check_steps(item, scene)
    if issue:
        return *issue, {}
    source, reason = source_slot(scene)
    held = scene["holding"]
    if item.kind == "observe":
        message = "夹爪已持有木块。" if held else (f"相机看到木块在 {source} 区。" if source else reason)
        log.append("observe()")
        return "success", message, {}
    if item.kind in ("pick","transfer") and not held:
        if item.kind == "transfer" and source == item.target:
            return "already_done", f"红块已在 {source} 区。", {}
        fresh = capture("pregrasp_observation")
        fresh_slot, reason = source_slot(fresh)
        if fresh["holding"] or fresh_slot != source:
            return "unverified", "下爪前画面已变化，请重新确认工位。" + reason, {}
        obj = fresh["objects"][0]
        xy = obj.get("xy_mm")
        if not xy or len(xy) != 2 or not all(math.isfinite(v) for v in xy):
            return "unverified", "当前无法确认木块坐标，机械臂保持原位。", {}
        offset = math.dist(xy, [v*1000 for v in STATIONS[source]])
        if offset > PICK_RADIUS_MM:
            return ("unverified", f"红块偏离抓取中心 {offset:.1f} mm，超过 {PICK_RADIUS_MM:g} mm 范围，未下爪。",
                    {"observed_offset_mm": round(offset, 2)})
        log.append(f"pick({source})")
        sim.pick(source, capture, xy_mm=xy)
        if not sim.holding():
            return "unverified", "提升后未检测到双侧夹持，后续动作已停止。", {}
        if item.kind == "pick":
            return ("success", f"已从 {source} 区抓起木块，夹爪双侧接触已确认。",
                    {"observed_offset_mm": round(offset, 2)})
    if item.kind in ("place","transfer"):
        log.append(f"place({item.target})")
        sim.place(item.target, capture)
        log.append(f"verify({item.target})")
        from .wrist_view import confirm_drop
        confirmations = []
        views = []
        for i in range(3):
            if i:
                sim.settle(.5)
            seen = capture(f"verify_{i+1}")
            confirmed, view = confirm_drop(sim, item.target, seen)
            confirmations.append(confirmed)
            views.append(view)
        if all(confirmations):
            return "success", f"相机连续确认红块已在 {item.target} 区。", {"drop_views": views}
        return "unverified", f"释放后未能确认木块稳定在 {item.target} 区。", {"drop_views": views}
    if item.kind == "home":
        log.append("home()")
        sim.home()
        return "success", "机械臂已回到待机位。", {}
    if item.kind == "move":
        log.append(f"move_delta({item.axis},{item.amount:g} mm)")
        measured = sim.translate(item.axis, item.amount)
        ok = measured["error_mm"] <= 5
        message = "已测量末端平移。" if ok else "平移跟随误差超过 5 mm。"
    elif item.kind == "wrist":
        log.append(f"wrist_delta({item.amount:g} deg)")
        measured = sim.rotate_wrist(item.amount)
        ok = measured["error_deg"] <= 2
        message = "已测量腕关节转角。" if ok else "转角跟随误差超过 2 度。"
    else:
        raise MotionError("未知动作")
    if held and not sim.holding():
        return "unverified", "运动后未确认持续夹持，后续动作已停止。", measured
    return "success" if ok else "unverified", message, measured


def execute(sim, text, *, planner=None, observer=None, previous=None):
    started = time.perf_counter()
    phases, log, outcomes = [], [], []

    def capture(label):
        scene, image = sim.camera()
        phases.append(label)
        if observer:
            observer(label, scene, image)
        return scene

    before = capture("before")
    directive = interaction_intent(text)
    if directive:
        status = {"ignore":"ignored","ask":"clarify","pause":"paused",
                  "resume":"resumed","status":"observed"}[directive.kind]
        message = directive.message
        if directive.kind == "pause":
            if sim.fault_reason or sim.stop_event.is_set():
                status, message = "rejected", "急停或故障已锁止。"
            else:
                sim.pause_event.set()
                sim.apply_pause()
                message = "任务已暂停，现场状态已保留。"
        elif directive.kind == "resume":
            try:
                sim.resume_here()
                message = "现场已恢复接收新指令，已中断的步骤不会重放。"
            except (FeedbackStop, MotionError) as exc:
                status, message = "rejected", str(exc)
        elif directive.kind == "status":
            message = ("已锁止。" if sim.fault_reason or sim.stop_event.is_set() else
                       "已暂停。" if sim.pause_event.is_set() else "当前空闲。")
        return interaction_receipt(text, before, status, message)
    latch = None
    try:
        sim.check_ready()
    except FeedbackStop as exc:
        latch = str(exc)
    if latch:
        intent = ask(latch)
    elif planner is not None:
        context = {**before, "dialogue": previous} if previous else before
        intent = planner(text, context)
    else:
        intent = offline_reply(previous, text) if previous else local_intent(text)
    issue = (("unverified", latch) if latch else
             ("clarify", intent.message) if intent.kind == "ask" else check_steps(intent, before))
    status, message = issue or ("success", "")
    measurement = {}
    steps = actions(intent)
    if not issue:
        for i, item in enumerate(steps):
            scene = before if i == 0 else capture(f"before_step_{i+1}")
            step_start = sim.data.time
            try:
                status, message, measurement = run_step(sim, item, scene, capture, log)
            except MotionError as exc:
                if sim.data.time > step_start:
                    reason = sim.lock_motion(f"动作中途遇到运动限制，已锁止：{exc}")
                    status, message, measurement = "unverified", reason, {}
                else:
                    status, message, measurement = "rejected", str(exc), {}
            except FeedbackStop as exc:
                soft_pause = sim.pause_event.is_set() and not (sim.fault_reason or sim.stop_event.is_set())
                status, message, measurement = "paused" if soft_pause else "unverified", str(exc), {}
            outcomes.append({"step":i+1, "kind":item.kind, "status":status,
                             "message":message, "measurement":measurement})
            if item.kind in ("pick", "transfer", "place"):
                outcomes[-1]["gripper"] = sim.gripper_status()
            if status not in ("success","already_done"):
                break
        if len(steps) > 1:
            complete = sum(o["status"] in ("success","already_done") for o in outcomes)
            message = f"已完成 {complete}/{len(steps)} 步。{message}"
            if complete == len(steps):
                status = "success"
    result = {"text":text, "planner":"interlock" if latch else DEEPSEEK_MODEL if planner is not None else "rules",
              "intent":asdict(intent), "before":before, "plan":log, "steps":outcomes,
              "status":status, "message":message, "after":capture("after"), "phases":phases,
              "gripper":sim.gripper_status(),
              "measurement":measurement, "wall_seconds":round(time.perf_counter()-started,3)}
    if status == "clarify":
        command = previous["command"] + "；补充：" + text if previous else text
        result["clarification"] = {"command":command[-400:], "question":message}
    return result
