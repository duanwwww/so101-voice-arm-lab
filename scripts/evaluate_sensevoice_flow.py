"""Run approved local speech through the normal HTTP control entry points."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch
from uuid import uuid4

import cv2
from fastapi.testclient import TestClient
import httpx
import numpy as np

from arm_lab.evaluate import block_state
from arm_lab.flow import execute
from arm_lab.model_access import ModelAccess, ModelFailure, PLAN_PROMPT_SHA256
from arm_lab.web import create_app
from scripts.evaluate_prompt_trial import save, exclusive

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/safety/sensevoice"


class RecordedAccess(ModelAccess):
    def __init__(self, report, persist, cloud_limit):
        super().__init__()
        self.asr_engine = "sensevoice"
        self.report, self.persist, self.cloud_limit = report, persist, cloud_limit
        self.remaining["asr"] = 4
        self.remaining["deepseek"] = cloud_limit

    def chat_json(self, instructions, task, key, approved_once):
        if len(self.report["campus_attempts"]) >= self.cloud_limit:
            raise ModelFailure("This run's campus allowance is exhausted.")
        row = {"task":task, "state":"started"}
        self.report["campus_attempts"].append(row)
        self.persist()
        original = httpx.Client
        def capture(response):
            row["http_status"] = response.status_code
            if response.status_code == 200:
                response.read()
                try:
                    choice = response.json()["choices"][0]
                    row["raw"] = choice["message"]["content"]
                    row["finish_reason"] = choice.get("finish_reason")
                except (ValueError, KeyError, IndexError, TypeError):
                    row["malformed"] = True
        def client(**kwargs):
            return original(event_hooks={"response":[capture]}, **kwargs)
        try:
            with patch("arm_lab.model_access.httpx.Client", side_effect=client):
                result = super().chat_json(instructions, task, key, approved_once)
            row["state"] = "answered"
            return result
        except ModelFailure as exc:
            row["state"] = "failed"
            row["error"] = str(exc)
            raise
        finally:
            self.persist()


def physical_snapshot(sim):
    return {"block":block_state(sim), "joints":sim.joints_now().tolist(),
            "tcp_mm":(sim.data.site_xpos[sim.tcp]*1000).tolist()}


def run(out, cloud_limit):
    if out.exists():
        raise ValueError("Existing run retained. Select a new output file for an authorized repeat.")
    report = {"mode":"real_sensevoice_to_http_to_" + ("live_deepseek" if cloud_limit else "local_rules"),
              "synthetic_audio_source":"Existing four Windows offline TTS recordings",
              "asr_engine":"sensevoice", "planner_prompt_sha256":PLAN_PROMPT_SHA256,
              "initial_layout":"A", "scene_resets":0, "campus_attempts":[], "rows":[],
              "tts":"text receipts only", "camera":"OpenCV HSV, original simulator renderer",
              "created_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out.parent.mkdir(parents=True, exist_ok=True)
    persist = lambda:save(out, report)
    access = RecordedAccess(report, persist, cloud_limit)
    cases = {c["id"]:c for c in json.loads((ROOT/"configs/prompt_trial.json").read_bytes())["cases"]}
    frames = out.with_suffix(".frames")
    frames.mkdir(exist_ok=True)
    with exclusive(out.with_suffix(".lock")), ExitStack() as stack:
        app = create_app(access,layout="A",playback_speed=0,allow_local_rules=True)
        client = stack.enter_context(TestClient(app, base_url="http://127.0.0.1",client=("127.0.0.1",55555)))
        worker = stack.enter_context(ThreadPoolExecutor(1))
        report["initial_state"] = client.get("/api/state").json()
        persist()
        for name in ("transfer","move","negative","clarify"):
            case = cases[name]
            wav = ROOT/"results/safety/prompt_trial_audio"/case["wav"]
            before = physical_snapshot(app.state.sim)
            row = {"id":name,"reference":case["reference"],
                   "audio_sha256":hashlib.sha256(wav.read_bytes()).hexdigest(),"before":before,"frames":[]}
            report["rows"].append(row)
            persist()
            started = time.perf_counter()
            response = client.post("/api/transcribe",files={"file":(wav.name,wav.read_bytes(),"audio/wav")})
            row["asr_seconds"] = round(time.perf_counter()-started,3)
            row["asr_http_status"] = response.status_code
            row["asr_response"] = response.json()
            if response.status_code!=200:
                row["passed"] = False
                persist()
                continue
            state = client.get("/api/state").json()
            command_id, token = str(uuid4()),str(uuid4())
            body = {"id":command_id,"text":response.json()["text"],"revision":state["revision"],
                    "issued_at_ms":int(time.time()*1000),"watchdog_token":token,
                    "use_deepseek":bool(cloud_limit)}
            row["command"] = body
            def observer_execute(sim, text, **kwargs):
                previous = kwargs.get("observer")
                def observe(phase, scene, frame):
                    if previous:
                        previous(phase,scene,frame)
                    if phase in ("before","after","lift") or phase.startswith("verify_"):
                        filename = f"{name}-{phase}.jpg"
                        cv2.imwrite(str(frames/filename),frame)
                        row["frames"].append({"phase":phase,"image":filename,"scene":scene,
                                              "physics":physical_snapshot(sim)})
                kwargs["observer"] = observe
                return execute(sim,text,**kwargs)
            started = time.perf_counter()
            row["successful_monitor_renewals"] = 0
            with patch("arm_lab.web.execute", side_effect=observer_execute):
                future = worker.submit(client.post,"/api/commands",json=body)
                while not future.done():
                    current = client.get("/api/state")
                    frame = client.get("/camera.jpg")
                    if current.status_code==200 and frame.status_code==200:
                        decoded = cv2.imdecode(np.frombuffer(frame.content,np.uint8),cv2.IMREAD_COLOR)
                        if decoded is not None:
                            beat = client.post("/api/heartbeat",json={"id":command_id,"watchdog_token":token})
                            row["successful_monitor_renewals"] += beat.status_code==200
                    time.sleep(.1)
                motion = future.result()
            row["control_seconds"] = round(time.perf_counter()-started,3)
            row["control_http_status"] = motion.status_code
            row["receipt"] = motion.json()
            row["after"] = physical_snapshot(app.state.sim)
            row["state_after"] = client.get("/api/state").json()
            unchanged = (before["block"]["time_s"]==row["after"]["block"]["time_s"]
                         and np.allclose(before["joints"],row["after"]["joints"],atol=1e-10,rtol=0))
            status = row["receipt"].get("status")
            if name=="transfer":
                confirmations = [f["physics"]["block"] for f in row["frames"] if f["phase"].startswith("verify_")]
                passed = status=="success" and len(confirmations)==3 and all(
                    b["inside"]=="B" and "desk" in b["contacts"] for b in confirmations)
            elif name=="move":
                delta = np.subtract(row["after"]["tcp_mm"],before["tcp_mm"])
                row["independent_displacement_mm"] = delta.tolist()
                row["error_mm"] = float(np.linalg.norm(delta-[0,12,0]))
                passed = status=="success" and row["error_mm"]<=5
            else:
                passed = unchanged and status==("ignored" if name=="negative" else "clarify")
            row["passed"] = bool(passed)
            persist()
        report["summary"] = {"passed":sum(r["passed"] for r in report["rows"]),
                             "total":len(report["rows"]),"asr_calls":access.used["asr"],
                             "cloud_calls":len(report["campus_attempts"]),
                             "same_scene":report["scene_resets"]==0}
        persist()
    return report


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--approve-local-asr",action="store_true")
    cli.add_argument("--approved-campus-attempts",type=int,default=0)
    cli.add_argument("--out",type=Path,default=OUT/"flow_local.json")
    args = cli.parse_args()
    if not args.approve_local_asr or not 0<=args.approved_campus_attempts<=8:
        cli.error("Approve local speech and specify only an authorized campus allowance, up to 8.")
    print(json.dumps(run(args.out,args.approved_campus_attempts)["summary"],ensure_ascii=False))
