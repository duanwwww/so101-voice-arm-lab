"""Model clients: bounded experiment calls and direct user-triggered web calls."""
import asyncio
import hashlib
from io import BytesIO
import json
import os
import re
from pathlib import Path
import sys
from threading import Lock

import httpx
from dotenv import dotenv_values

from .planner import ask, check_command_match, check_steps, classify_control_text, decode_model_plan, local_intent, missing_motion_unit
from .sensevoice import DEFAULT_MODEL_DIR, MODEL_ID as SENSEVOICE_MODEL_ID, SenseVoiceRecognizer, model_ready, prepare_model


class ModelDenied(Exception):
    pass


class ModelFailure(Exception):
    pass


DEEPSEEK_BASE = "https://models.sjtu.edu.cn/api/v1"
DEEPSEEK_ENDPOINT = DEEPSEEK_BASE + "/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
PLAN_MAX_TOKENS = 160
WEB_PLANNER_MODEL = "deepseek-reasoner"
REASONER_MAX_TOKENS = 4096
PLAN_INSTRUCTIONS = (Path(__file__).resolve().parents[1] / "configs" / "planner100_finalist.txt").read_text(encoding="utf-8").strip()
PLAN_PROMPT_SHA256 = hashlib.sha256(PLAN_INSTRUCTIONS.encode("utf-8")).hexdigest()


def campus_api_key():
    key = os.getenv("SJTU_DEEPSEEK_API_KEY", "").strip()
    if key:
        return key
    course_file = os.getenv("SJTU_DEEPSEEK_ENV_FILE", "").strip()
    settings = os.environ
    if course_file:
        try:
            with Path(course_file).expanduser().open(encoding="utf-8-sig") as handle:
                settings = dotenv_values(stream=handle, interpolate=False)
        except (OSError, UnicodeError, ValueError) as exc:
            raise ModelFailure("无法读取 SJTU_DEEPSEEK_ENV_FILE 指定的本机配置文件。") from exc
    key = (settings.get("SJTU_API_KEY") or "").strip()
    base = (settings.get("SJTU_API_BASE_URL") or "").strip().rstrip("/")
    if key and base != DEEPSEEK_BASE:
        raise ModelFailure("已有 SJTU_API_KEY，但配套的 SJTU_API_BASE_URL 与上海交大官方地址不一致。")
    return key


def planning_task(text, scene, *, station_only=False):
    objects = []
    for observation in scene["objects"]:
        obj = {"id":"red_1" if len(scene["objects"])==1 else "ambiguous", "slot":observation["slot"]}
        if not station_only and "xy_mm" in observation:
            obj["xy_mm"] = observation["xy_mm"]
        objects.append(obj)
    task = {"command":text,"objects":objects}
    if scene.get("holding"):
        task["holding"] = True
    if scene.get("dialogue"):
        task["dialogue"] = scene["dialogue"]
    return task


def campus_payload(instructions, task, *, model=DEEPSEEK_MODEL, max_tokens=PLAN_MAX_TOKENS):
    return {"model":model, "temperature":0, "max_tokens":max_tokens, "stream":False,
            "messages":[{"role":"system","content":instructions},
                        {"role":"user","content":json.dumps(task,ensure_ascii=False)}]}


class ModelAccess:
    def __init__(self, *, planner_model=DEEPSEEK_MODEL, max_tokens=None, timeout=None):
        if planner_model not in (DEEPSEEK_MODEL, WEB_PLANNER_MODEL):
            raise ValueError("Unsupported campus planner model")
        self.planner_model = planner_model
        reasoning = planner_model == WEB_PLANNER_MODEL
        self.planner_max_tokens = max_tokens if max_tokens is not None else (REASONER_MAX_TOKENS if reasoning else PLAN_MAX_TOKENS)
        self.planner_timeout = timeout if timeout is not None else (120 if reasoning else 30)
        if (type(self.planner_max_tokens) is not int or not 1 <= self.planner_max_tokens <= 8192
                or not 1 <= self.planner_timeout <= 180):
            raise ValueError("Invalid planner output budget or timeout")
        self.remaining = {"asr":0,"deepseek":0,"tts":0}
        self.used = dict(self.remaining)
        self.lock = Lock()
        self.asr_lock = Lock()
        self.asr_engine = "sensevoice"
        self.sensevoice_dir = DEFAULT_MODEL_DIR
        self.sensevoice = None
        self.whisper_dir = Path(__file__).resolve().parents[1] / "outputs" / "models" / "faster-whisper-small"
        self.whisper = None

    def authorize(self, kind, count, data_scope):
        if not 1 <= count <= 20:
            raise ValueError("每项授权次数须为 1–20")
        if not sys.stdin.isatty():
            raise ModelDenied("模型调用只接受控制端交互终端的本次授权。")
        phrase = f"批准 {kind} {count}"
        print(f"{data_scope}\n本次进程最多 {count} 次，失败的尝试也计数。")
        if input(f"确认请输入「{phrase}」，直接回车保持关闭：").strip() == phrase:
            self.remaining[kind] = count

    def require_approval(self, kind):
        if self.remaining[kind] <= 0:
            raise ModelDenied("此模型未获本次授权，或已用完授权次数。")

    def consume(self, kind, approved_once=False):
        with self.lock:
            if not approved_once:
                self.require_approval(kind)
                self.remaining[kind] -= 1
            self.used[kind] += 1

    def plan(self, text, scene, *, approved_once=False, station_only=False):
        if control := classify_control_text(text):
            return control
        if missing_motion_unit(text):
            return ask("请补充距离或角度的单位，例如毫米、厘米或度。")
        if not approved_once:
            self.require_approval("deepseek")
        key = campus_api_key()
        if not key:
            raise ModelFailure("控制端缺少 SJTU_DEEPSEEK_API_KEY。")
        task = planning_task(text, scene, station_only=station_only)
        body = self.chat_json(PLAN_INSTRUCTIONS, task, key, approved_once)
        try:
            stated = local_intent(text)
            if stated.kind != "ask" and (issue := check_steps(stated,scene)) and issue[0]=="clarify":
                return ask(issue[1])
            result = decode_model_plan(body,scene)
            check_command_match(text, result, scene)
            return result
        except (ValueError, TypeError, OverflowError) as exc:
            raise ModelFailure(f"动作未执行，规划校验失败：{exc}") from exc

    def feedback(self, receipt, *, approved_once=False):
        if not approved_once:
            self.require_approval("deepseek")
        key = campus_api_key()
        if not key:
            raise ModelFailure("未找到交大 DeepSeek 凭据。")
        facts = {"command":receipt["text"], "status":receipt["status"], "message":receipt["message"],
                 "steps":[{"action":s["kind"],"status":s["status"]} for s in receipt.get("steps",[])],
                 "slots":[o["slot"] for o in receipt.get("after",{}).get("objects",[])]}
        body = self.chat_json('你是机械臂的语音反馈模块。依据回执和相机观测，用一句中文解释已经发生的结果或需要补充的信息。'
                              '保留失败和未确认状态，不发出动作。只返回 {"speech":"中文"}。', facts, key, approved_once)
        if not isinstance(body,dict) or set(body) != {"speech"} or not isinstance(body["speech"],str) or not 1 <= len(body["speech"].strip()) <= 160:
            raise ModelFailure("语音反馈格式有误，仍可播报原始回执。")
        return body["speech"].strip()

    def chat_json(self, instructions, task, key, approved_once):
        payload = campus_payload(instructions, task, model=self.planner_model, max_tokens=self.planner_max_tokens)
        self.consume("deepseek", approved_once)
        try:
            with httpx.Client(timeout=self.planner_timeout,follow_redirects=False) as client:
                response = client.post(DEEPSEEK_ENDPOINT,headers={"Authorization":"Bearer "+key},json=payload)
        except (httpx.HTTPError, UnicodeError) as exc:
            raise ModelFailure("连接交大 DeepSeek 失败，请检查校园网或交大 VPN。") from exc
        if response.status_code != 200:
            raise ModelFailure(f"交大 DeepSeek 返回 HTTP {response.status_code}，未执行动作。")
        try:
            choice = response.json()["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ValueError("incomplete model reply")
            content = choice["message"]["content"].strip()
            fenced = re.fullmatch(r"```(?:json)?\s*([\s\S]+?)\s*```", content, flags=re.I)
            body = json.loads(fenced[1] if fenced else content)
            return body
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            raise ModelFailure("交大 DeepSeek 回复不完整或未返回有效 JSON。") from exc

    def voice_ready(self):
        if self.asr_engine == "sensevoice":
            return model_ready(self.sensevoice_dir)
        return all((self.whisper_dir / name).is_file() for name in ("model.bin","tokenizer.json","config.json"))

    @property
    def asr_model(self):
        return SENSEVOICE_MODEL_ID if self.asr_engine == "sensevoice" else "Systran/faster-whisper-small"

    def transcribe(self, audio, *, approved_once=False):
        self.consume("asr", approved_once)
        with self.asr_lock:
            if self.asr_engine == "sensevoice":
                try:
                    if self.sensevoice is None:
                        prepare_model(self.sensevoice_dir, allow_download=approved_once)
                        self.sensevoice = SenseVoiceRecognizer(self.sensevoice_dir)
                    return self.sensevoice.transcribe(audio)
                except ValueError as exc:
                    raise ModelFailure(str(exc)) from exc
                except Exception as exc:
                    raise ModelFailure("SenseVoice 转写失败，请检查模型文件、录音和 voice 依赖。") from exc
            if self.asr_engine != "whisper":
                raise ModelFailure("未知的语音识别引擎。")
            return self._transcribe_whisper(audio, approved_once)

    def _transcribe_whisper(self, audio, approved_once):
        path = self.whisper_dir
        if not self.voice_ready():
            if not approved_once:
                raise ModelFailure("请指定本地 Whisper 权重，或在本机网页批准首次下载及转写。")
            try:
                from faster_whisper.utils import download_model
                download_model("Systran/faster-whisper-small", output_dir=str(path), use_auth_token=False)
            except Exception as exc:
                raise ModelFailure("Whisper 权重下载失败，请检查 Hugging Face 连接或指定本地权重。") from exc
        try:
            from faster_whisper import WhisperModel
            if self.whisper is None:
                self.whisper = WhisperModel(str(path), device="cpu", compute_type="int8", local_files_only=True)
            segments, _ = self.whisper.transcribe(BytesIO(audio),language="zh",beam_size=1,vad_filter=False)
            text = "".join(segment.text for segment in segments).strip()
            if text:
                return text
        except Exception as exc:
            raise ModelFailure("本地音频转写失败，请检查权重、音频格式和 voice 依赖。") from exc
        raise ModelFailure("这段录音没有识别到文字，请重新录音。")

    async def speak(self, feedback, *, approved_once=False):
        self.consume("tts", approved_once)
        try:
            import edge_tts
            async with asyncio.timeout(25):
                chunks = [part["data"] async for part in edge_tts.Communicate(
                    feedback,"zh-CN-XiaoxiaoNeural").stream() if part["type"]=="audio"]
            if not chunks:
                raise ValueError("empty speech")
            return b"".join(chunks)
        except Exception as exc:
            raise ModelFailure("微软在线语音合成失败。文字回执仍可读取。") from exc
