"""Simulation is owned by a single worker (including its OpenGL context)."""
import asyncio
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
import math
from pathlib import Path
from threading import Lock
import time
from typing import Literal
from uuid import UUID
from urllib.parse import urlsplit

import cv2
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field, field_validator

from .flow import execute, interaction_intent, interaction_receipt
from .gripper import FeedbackStop
from .model_access import DEEPSEEK_MODEL, PLAN_PROMPT_SHA256, ModelAccess, ModelDenied, ModelFailure, campus_api_key
from .scene import LAYOUTS
from .simulator import MotionError, Simulator
from .wrist_view import observe_wrist


class CommandText(BaseModel):
    text: str = Field(min_length=1,max_length=600,pattern=r"\S")


class Command(CommandText):
    id: UUID
    revision: int = Field(ge=0)
    issued_at_ms: int
    use_deepseek: bool = False
    reply_to: UUID | None = None
    watchdog_token: UUID | None = None
    # An older client can still request local rules. Its Qwen opt-in cannot authorize a different service.
    use_qwen: Literal[False] = Field(default=False, exclude=True, deprecated=True)


class Reset(BaseModel):
    layout: str

    @field_validator("layout")
    @classmethod
    def known_layout(cls, value):
        if value not in LAYOUTS:
            raise ValueError("未知仿真场景")
        return value


class Heartbeat(BaseModel):
    id: UUID
    watchdog_token: UUID


@dataclass
class LinkLease:
    id: UUID
    token: UUID
    deadline: float
    timer: asyncio.TimerHandle | None = None
    active: bool = True
    expired: bool = False


def local_console(request):
    local = {"localhost", "127.0.0.1", "::1"}
    return bool(request.client and request.client.host in local and request.url.hostname in local)


def one_request_consent(request, kind):
    supplied = request.headers.get("X-Arm-Approval")
    if not supplied:
        return False
    origin = urlsplit(request.headers.get("Origin", ""))
    if (supplied != kind or not local_console(request) or origin.scheme != request.url.scheme
            or origin.netloc != request.url.netloc):
        raise ModelDenied("单次模型批准须来自本机控制台的同源页面。")
    return True


class CameraState:
    def __init__(self):
        self.lock = Lock()
        self.jpeg = b""
        self.wrist_jpeg = b""
        self.details = {}

    def publish(self, sim, stage, scene, frame):
        ok, jpg = cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY,85])
        _, wrist = observe_wrist(sim)
        wrist_ok, wrist_jpg = cv2.imencode(".jpg", wrist, [cv2.IMWRITE_JPEG_QUALITY,80])
        if not (ok and wrist_ok):
            raise RuntimeError("相机编码失败")
        details = {"phase":stage,"scene":scene,"layout":sim.layout,"tcp_mm":(sim.position()*1000).round(1).tolist(),
                   "sim_seconds":round(sim.data.time,2)}
        with self.lock:
            self.details = details
            self.jpeg = jpg.tobytes()
            self.wrist_jpeg = wrist_jpg.tobytes()


def create_app(access=None, layout="A", playback_speed=2, keepouts=None, watchdog_seconds=3.0):
    if not math.isfinite(watchdog_seconds) or watchdog_seconds <= 0:
        raise ValueError("心跳期限须为有限正数")
    access = access or ModelAccess()
    pool = ThreadPoolExecutor(max_workers=1,thread_name_prefix="arm-sim")
    camera = CameraState()
    receipts = OrderedDict()
    pending = None
    control_transition = False
    last_finished_ms = 0
    revision = 0
    stop_sequence = 0
    lease = None
    link_fault = None
    last_stop_reason = None
    latest = None
    try:
        key_ready = bool(campus_api_key())
    except ModelFailure:
        key_ready = False

    @asynccontextmanager
    async def lifespan(app):
        def start():
            sim = Simulator(layout)
            if keepouts is not None:
                from .motion_guard import MotionGuard
                try:
                    sim.guard = MotionGuard.from_config(sim.model, keepouts)
                except Exception:
                    sim.close()
                    raise
            if playback_speed > 0:
                sim.playback_speed = playback_speed
                sim.preview = lambda: camera.publish(sim,"moving",*sim.camera())
            camera.publish(sim,"ready",*sim.camera())
            return sim
        loop = asyncio.get_running_loop()
        try:
            app.state.sim = await loop.run_in_executor(pool,start)
            yield
        finally:
            if lease and lease.timer:
                lease.timer.cancel()
            if getattr(app.state,"sim",None) is not None:
                app.state.sim.stop_event.set()
                await loop.run_in_executor(pool,app.state.sim.close)
            pool.shutdown(wait=True)

    app = FastAPI(title="SO-101 桌面仿真", lifespan=lifespan)

    def request_stop(reason=None):
        """Always set the interlock before scheduling work on the simulation thread."""
        nonlocal stop_sequence, revision, last_stop_reason, link_fault
        stop_sequence += 1
        revision += 1
        if lease:
            lease.active = False
            if lease.timer:
                lease.timer.cancel()
        sim = app.state.sim
        resetting = pending is not None and pending[0] == "reset"
        if resetting or not sim.stop_event.is_set():
            last_stop_reason = reason
            sim.external_stop_reason = reason
            if resetting:
                link_fault = reason
        sim.stop_event.set()
        return asyncio.get_running_loop().run_in_executor(pool, sim.apply_pending_stop)

    def expire_link(owner):
        nonlocal link_fault
        if lease is not owner or not owner.active:
            return
        if app.state.sim.stop_event.is_set() or app.state.sim.fault_reason:
            owner.active = False
            return
        owner.expired = True
        link_fault = "监控画面心跳超时，后续指令已锁止。请恢复连接并核对现场。"
        request_stop(link_fault)

    def check_deadline(owner):
        # Heartbeats change only the deadline. A callback left over from a
        # previous request cannot act on a new command, even if IDs are replayed.
        if lease is not owner or not owner.active:
            return
        remaining = owner.deadline - time.monotonic()
        if remaining <= 0:
            expire_link(owner)
        else:
            owner.timer = asyncio.get_running_loop().call_later(remaining, check_deadline, owner)

    @app.exception_handler(ModelDenied)
    async def denied(request,exc):
        return JSONResponse({"detail":str(exc)},status_code=403)

    @app.exception_handler(ModelFailure)
    async def failed(request,exc):
        return JSONResponse({"detail":str(exc)},status_code=502)

    @app.get("/")
    async def home():
        return FileResponse(Path(__file__).with_name("index.html"),media_type="text/html",headers={"Cache-Control":"no-store"})

    @app.get("/api/state")
    async def state(request:Request):
        with camera.lock:
            details = camera.details
        with access.lock:
            quotas = dict(access.remaining)
        sim = app.state.sim
        active = bool(lease and lease.active)
        return {**details,"backend":"MuJoCo","revision":revision,"busy":pending is not None,
                "paused":sim.pause_event.is_set() and not (sim.stop_event.is_set() or bool(sim.fault_reason)),
                "stopped":sim.stop_event.is_set() or bool(sim.fault_reason), "fault":link_fault or sim.fault_reason,
                "watchdog_seconds":watchdog_seconds, "watchdog_active":active,
                "watchdog_remaining_ms":max(0,round((lease.deadline-time.monotonic())*1000)) if active else None,
                "watchdog_expired":link_fault is not None,
                "quota":quotas, "local_approval":local_console(request), "campus_key":key_ready,
                "planner_prompt_sha256":PLAN_PROMPT_SHA256,
                "asr_engine":access.asr_engine,"asr_model":access.asr_model,"asr_ready":access.voice_ready(),
                "whisper_ready":all((access.whisper_dir / n).is_file() for n in ("model.bin","tokenizer.json","config.json")),
                "last":latest,"server_ms":int(time.time()*1000)}

    @app.get("/api/scenarios")
    async def scenarios():
        return [{"id":key, "label":cfg.label,"description":cfg.description,"command":cfg.command}
                for key,cfg in LAYOUTS.items()]

    @app.get("/camera.jpg")
    async def frame():
        with camera.lock:
            data = camera.jpeg
        return Response(data,media_type="image/jpeg",headers={"Cache-Control":"no-store"})

    @app.post("/api/reset")
    async def reset(body:Reset):
        nonlocal pending, revision, latest
        if pending is not None:
            raise HTTPException(409,"命令执行中。")
        sim = app.state.sim
        stop_at_request = stop_sequence
        def work():
            sim.reset(body.layout)
            camera.publish(sim,"ready",*sim.camera())
        async def job():
            nonlocal latest,pending,link_fault
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(pool,work)
                if stop_sequence != stop_at_request:
                    raise HTTPException(409,"重置期间收到停止信号，控制保持锁止。")
                latest = None
                link_fault = None
            except FeedbackStop as exc:
                raise HTTPException(409,str(exc)) from exc
            finally:
                if stop_sequence != stop_at_request:
                    sim.external_stop_reason = last_stop_reason
                    sim.stop_event.set()
                    await loop.run_in_executor(pool,sim.apply_pending_stop)
                pending = None
        revision += 1
        task = asyncio.create_task(job())
        pending = ("reset",None,task)
        await asyncio.shield(task)
        return {"revision":revision,"layout":body.layout}

    @app.get("/camera/wrist.jpg")
    async def wrist_frame():
        with camera.lock:
            data = camera.wrist_jpeg
        return Response(data, media_type="image/jpeg", headers={"Cache-Control":"no-store"})

    @app.post("/api/stop")
    async def stop():
        handled = request_stop()
        if pending is None:
            await asyncio.shield(handled)
        return {"stopped":True,"revision":revision}

    @app.post("/api/heartbeat")
    async def heartbeat(body:Heartbeat):
        if (lease is None or not lease.active or body.id != lease.id
                or body.watchdog_token != lease.token):
            raise HTTPException(409,"没有与该心跳匹配的活动指令。")
        now = time.monotonic()
        if now >= lease.deadline:
            expire_link(lease)
            raise HTTPException(409,"心跳已过期，控制已锁止。")
        if app.state.sim.stop_event.is_set() or app.state.sim.fault_reason:
            raise HTTPException(409,"控制已锁止，不能续期。")
        lease.deadline = now + watchdog_seconds
        return {"accepted":True,"watchdog_seconds":watchdog_seconds}

    @app.post("/api/command-kind")
    async def command_kind(body:CommandText):
        intent = interaction_intent(body.text)
        return {"kind":intent.kind if intent else "motion"}

    def remember(key, signature, receipt):
        receipts[key] = signature, receipt
        if len(receipts) > 128:
            receipts.popitem(last=False)
        return receipt

    async def local_control(body, signature, directive):
        nonlocal revision, latest, pending, control_transition
        sim = app.state.sim
        key = str(body.id)
        kind = directive.kind
        with camera.lock:
            snapshot = camera.details
        locked = sim.stop_event.is_set() or bool(sim.fault_reason)
        if kind in ("pause","resume"):
            if locked:
                raise HTTPException(409,"急停或故障已锁止。请核对回执，恢复口令不能解锁。")
            if control_transition or (pending and pending[0] == "reset"):
                raise HTTPException(409,"控制状态正在切换，请查看最新回执。")
        def receipt(status, message):
            nonlocal latest
            result = interaction_receipt(body.text, snapshot["scene"], status, message)
            result.update(id=key, revision=revision)
            if pending is not None:
                result.pop("clarification", None)
            # A question about progress must not consume a pending clarification
            # or replace the authoritative receipt of an ongoing trajectory.
            if pending is None and (kind in ("pause","resume") or "clarification" in result or not (latest and latest.get("clarification"))):
                latest = result
            return remember(key, signature, result)
        if kind == "status":
            condition = ("已锁止" if locked else "已暂停" if sim.pause_event.is_set() else
                         "正在执行任务" if pending else "空闲")
            held = "最新反馈已确认持物" if snapshot["scene"]["holding"] else "最新反馈未确认持物"
            return receipt("observed", f"当前{condition}，{held}。")
        if kind in ("ignore","ask"):
            return receipt("ignored" if kind=="ignore" else "clarify", directive.message)
        if kind == "resume" and pending:
            raise HTTPException(409,"等待当前任务返回暂停回执后，再恢复控制。")
        if kind == "resume" and not sim.pause_event.is_set():
            raise HTTPException(409,"控制没有暂停；不会重播任何任务。")
        if kind == "pause":
            if not sim.pause_event.is_set():
                sim.pause_event.set()
                revision += 1
            if pending:
                return receipt("paused","暂停信号已送达，等待当前任务中止；剩余步骤不会重放。")
        async def transition():
            nonlocal control_transition, pending, revision
            try:
                method = sim.apply_pause if kind=="pause" else sim.resume_here
                await asyncio.get_running_loop().run_in_executor(pool, method)
                if sim.stop_event.is_set() or sim.fault_reason:
                    raise HTTPException(409,"急停或故障已锁止。")
                if kind == "resume":
                    revision += 1
            except (FeedbackStop, MotionError) as exc:
                raise HTTPException(409,str(exc)) from exc
            finally:
                control_transition = False
                pending = None
            if kind == "pause":
                return receipt("paused","控制已暂停，已保留现场。")
            return receipt("resumed","现场已恢复接收新指令，已中断的步骤不会重放。")
        control_transition = True
        task = asyncio.create_task(transition())
        pending = (key,signature,task)
        return await asyncio.shield(task)

    @app.post("/api/commands")
    async def command(body:Command, request:Request):
        nonlocal pending,revision,latest,lease
        key = str(body.id)
        signature = (body.text,body.revision,body.issued_at_ms,body.use_deepseek,body.reply_to,body.watchdog_token)
        if key in receipts:
            original, receipt = receipts[key]
            if original != signature:
                raise HTTPException(409,"同一命令编号对应不同内容。")
            return receipt
        if pending is not None and pending[0] == key:
            if pending[1] == signature:
                return await asyncio.shield(pending[2])
            raise HTTPException(409,"同一命令编号对应不同内容。")
        if body.revision!=revision or abs(body.issued_at_ms-time.time()*1000)>30000:
            raise HTTPException(409,"指令或画面已过期，请查看当前画面后重新发送。")
        directive = interaction_intent(body.text)
        if directive:
            return await local_control(body,signature,directive)
        if pending is not None:
            raise HTTPException(409,"当前任务仍在执行。可以询问状态或发出暂停；新动作未排队，请核对新画面后重新发送。")
        if app.state.sim.stop_event.is_set() or app.state.sim.fault_reason:
            raise HTTPException(409,"控制已锁止。查看回执，重置仿真后才能再次运动。")
        if app.state.sim.pause_event.is_set():
            raise HTTPException(409,"已暂停，请核对当前画面并说“恢复控制”，再发新动作。")
        previous = None
        if body.reply_to:
            if latest is None or latest.get("id") != str(body.reply_to) or not latest.get("clarification"):
                raise HTTPException(409,"这条追问已过期，请重新描述任务。")
            previous = latest["clarification"]
        once = one_request_consent(request, "deepseek")
        if not local_console(request) and body.watchdog_token is None:
            raise HTTPException(422,"远程控制指令须提供独立的监控心跳标识。")
        if body.watchdog_token == body.id:
            raise HTTPException(422,"监控心跳标识与命令编号须独立生成。")
        if body.use_deepseek and not once:
            access.require_approval("deepseek")
        if body.issued_at_ms < last_finished_ms:
            raise HTTPException(409,"该指令早于最近的任务回执，请核对现场并重新发送。")
        sim = app.state.sim
        event_loop = asyncio.get_running_loop()
        owner = (LinkLease(body.id, body.watchdog_token, time.monotonic()+watchdog_seconds)
                 if body.watchdog_token is not None else None)
        lease = owner
        if owner:
            owner.timer = event_loop.call_later(watchdog_seconds, check_deadline, owner)

        def guard_planner():
            # A synchronous model call can outlive the lease. Gate its result
            # on the worker too, before it can reach the actuator.
            if owner and owner.active and time.monotonic() >= owner.deadline:
                # The event loop serializes a racing heartbeat, manual stop and
                # expiry. Keep the model worker still until that decision is made.
                async def check_on_loop():
                    if time.monotonic() >= owner.deadline:
                        expire_link(owner)
                asyncio.run_coroutine_threadsafe(check_on_loop(), event_loop).result()
            sim.check_ready()

        def work():
            def planner(text, scene):
                guard_planner()
                intent = access.plan(text, scene, approved_once=once)
                guard_planner()
                return intent
            try:
                return execute(sim,body.text,planner=planner if body.use_deepseek else None, previous=previous,
                               observer=lambda phase,scene,img:camera.publish(sim,phase,scene,img))
            except (ModelFailure, FeedbackStop) as exc:
                scene, frame = sim.camera()
                camera.publish(sim,"after",scene,frame)
                return {"text":body.text,"planner":DEEPSEEK_MODEL,
                        "status":"unverified" if isinstance(exc, FeedbackStop) else "model_error","message":str(exc),
                        "before":scene,"after":scene,"steps":[],"plan":[],"measurement":{}}
        async def job():
            nonlocal latest,pending,lease,last_finished_ms
            try:
                receipt = await asyncio.get_running_loop().run_in_executor(pool,work)
                if owner and owner.active and time.monotonic() >= owner.deadline:
                    expire_link(owner)
                if owner and owner.expired:
                    if link_fault not in receipt["message"]:
                        receipt["message"] = f"{link_fault} {receipt['message']}"
                    receipt.update(status="unverified", watchdog_expired=True)
                elif sim.pause_event.is_set() and not (sim.stop_event.is_set() or sim.fault_reason):
                    if receipt["status"] in ("unverified","model_error"):
                        receipt.update(status="paused", message="任务已暂停，剩余步骤已取消；现场已保留。")
                last_finished_ms = int(time.time()*1000)
                receipt.update(id=key,revision=revision)
                latest = receipt
                return remember(key, signature, receipt)
            finally:
                if owner and owner.timer:
                    owner.timer.cancel()
                    owner.active = False
                lease = None
                pending = None
        revision += 1
        task = asyncio.create_task(job())
        pending = (key,signature,task)
        return await asyncio.shield(task)

    asr_lock = asyncio.Lock()

    @app.post("/api/transcribe")
    async def transcribe(request:Request, file:UploadFile=File(...)):
        once = one_request_consent(request,"asr")
        if not once:
            access.require_approval("asr")
        audio = await file.read(4*1024*1024+1)
        if not audio or len(audio)>4*1024*1024:
            raise HTTPException(400,"请选择 1 字节到 4 MiB 的短口令录音。")
        async with asr_lock:
            txt = await asyncio.to_thread(access.transcribe,audio,approved_once=once)
        return {"text":txt,"model":access.asr_model,"engine":access.asr_engine}

    @app.post("/api/speak")
    async def speak(request:Request, receipt_id:UUID|None=None):
        once = one_request_consent(request,"tts")
        if not once:
            access.require_approval("tts")
        if latest is None:
            raise HTTPException(409,"还没有可播报的回执。")
        if receipt_id and str(receipt_id) != latest["id"]:
            raise HTTPException(409,"回执已更新，请重新选择。")
        message = latest["message"]
        if latest.get("feedback"):
            message += " " + latest["feedback"]
        return Response(await access.speak(message,approved_once=once),media_type="audio/mpeg")

    feedback_lock = asyncio.Lock()

    @app.post("/api/feedback")
    async def feedback(request:Request, receipt_id:UUID):
        once = one_request_consent(request,"deepseek")
        if not once:
            access.require_approval("deepseek")
        receipt = latest
        if not receipt or str(receipt_id) != receipt["id"]:
            raise HTTPException(409,"请选择当前执行回执。")
        async with feedback_lock:
            if receipt is not latest:
                raise HTTPException(409,"回执已更新。")
            if not receipt.get("feedback"):
                speech = await asyncio.to_thread(access.feedback,receipt,approved_once=once)
                if receipt is not latest:
                    raise HTTPException(409,"回执已更新。")
                receipt["feedback"] = speech
        return {"speech":receipt["feedback"]}
    return app
