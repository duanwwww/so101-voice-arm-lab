"""Ten raw audio inputs through the default web API, one scene, no model retries.

Default invocation only validates the preregistered inputs. --run performs the
real local ASR and campus reasoning requests. All source WAVs are synthetic.
The separate typed auxiliary command starts genuine motion for the final pause
test. It consumes a model request, but is not an eleventh audio observation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
from threading import RLock
import time
from unittest.mock import patch
from uuid import uuid4

import cv2
from fastapi.testclient import TestClient
import httpx
import numpy as np

from arm_lab.asr_benchmark import anchors, cer_text, edit_distance
from arm_lab.flow import execute
from arm_lab.kinematics import ArmModel
from arm_lab.model_access import ModelAccess, ModelFailure, PLAN_PROMPT_SHA256
from arm_lab.planner_scoring import score_response
from arm_lab.web import create_app
from scripts.evaluate_prompt_trial import exclusive, save
from scripts.evaluate_sensevoice_flow import physical_snapshot

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/reasoner_flow10.json"


def prepare(config=CONFIG):
    raw = config.read_bytes()
    spec = json.loads(raw)
    cases = spec["cases"]
    if len(cases) != 10 or len({c["id"] for c in cases}) != 10:
        raise ValueError("Exactly ten independently identified audio cases are required.")
    if spec["planner_model"] != "deepseek-reasoner":
        raise ValueError("This experiment requires the reasoner planner.")
    inputs = []
    for case in cases:
        path = (ROOT / case["audio"]).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("Audio must be within the project.")
        inputs.append({"id":case["id"], "audio":case["audio"], "exists":path.is_file(),
                       "audio_sha256":hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None})
    return {"specification":spec, "config_sha256":hashlib.sha256(raw).hexdigest(),
            "inputs":inputs, "all_audio_ready":all(item["exists"] for item in inputs)}


class RecordedAccess(ModelAccess):
    """Observe actual HTTP responses without storing credentials or reasoning text."""
    def __init__(self, report, persist, cloud_limit):
        super().__init__(planner_model="deepseek-reasoner")
        self.report, self.persist, self.cloud_limit = report, persist, cloud_limit
        self.label = None

    def transcribe(self, audio, *, approved_once=False):
        if len(self.report["asr_attempts"]) >= 10:
            raise ModelFailure("Experiment's ten audio attempts are exhausted.")
        row = {"case_id":self.label, "state":"started", "model":self.asr_model,
               "audio_sha256":hashlib.sha256(audio).hexdigest()}
        self.report["asr_attempts"].append(row)
        self.persist()
        started = time.perf_counter()
        try:
            result = super().transcribe(audio, approved_once=approved_once)
            row.update(state="answered", raw_transcript=result)
            return result
        except Exception as exc:
            row.update(state="failed", error_type=type(exc).__name__)
            raise
        finally:
            row["seconds"] = round(time.perf_counter()-started, 3)
            self.persist()

    def chat_json(self, instructions, task, key, approved_once):
        if len(self.report["campus_attempts"]) >= self.cloud_limit:
            raise ModelFailure("Experiment's campus request cap is exhausted.")
        row = {"case_id":self.label, "state":"started", "task":task,
               "requested_model":self.planner_model, "max_tokens":self.planner_max_tokens,
               "prompt_sha256":hashlib.sha256(instructions.encode()).hexdigest()}
        self.report["campus_attempts"].append(row)
        self.persist()
        original_client = httpx.Client
        def capture(response):
            row["http_status"] = response.status_code
            response.read()
            if response.status_code == 200:
                try:
                    payload = response.json()
                    choice = payload["choices"][0]
                    message = choice["message"]
                    row.update(raw=message.get("content"), finish_reason=choice.get("finish_reason"),
                               returned_model=payload.get("model"), usage=payload.get("usage"),
                               reasoning_chars=len(message.get("reasoning_content") or ""))
                except (ValueError, KeyError, IndexError, TypeError):
                    row["malformed_response"] = True
        def client(**kwargs):
            return original_client(event_hooks={"response":[capture]}, **kwargs)
        started = time.perf_counter()
        try:
            with patch("arm_lab.model_access.httpx.Client", side_effect=client):
                result = super().chat_json(instructions, task, key, approved_once)
            row["state"] = "answered"
            return result
        except Exception as exc:
            row.update(state="failed", error_type=type(exc).__name__)
            raise
        finally:
            row["seconds"] = round(time.perf_counter()-started, 3)
            self.persist()


def stationary(before, after):
    return (before["block"]["time_s"] == after["block"]["time_s"]
            and np.allclose(before["joints"], after["joints"], atol=1e-10, rtol=0)
            and np.allclose(before["block"]["xyz_m"], after["block"]["xyz_m"], atol=1e-8, rtol=0))


def physical_checks(case, row, tolerances):
    """Scores MuJoCo state, separately from vision and controller self-reports."""
    before, after = row["before"], row["after"]
    status = row.get("receipt", {}).get("status")
    goal = case["goal"]
    checks = {}
    if goal == "transfer":
        lifts = [f["physics"]["block"] for f in row["frames"] if f["phase"] == "lift"]
        verifies = [f for f in row["frames"] if f["phase"].startswith("verify_")]
        checks = {
            "receipt_success":status == "success",
            "source_matches_gold":before["block"]["inside"] == case["source"],
            "both_fingers_lift_over_threshold":bool(len(lifts) == 1
                and set(lifts[0]["contacts"]) == {"fixed_pad", "moving_pad"}
                and (lifts[0]["xyz_m"][2]-before["block"]["xyz_m"][2])*1000 > tolerances["lift_mm"]),
            "three_verification_phases":[f["phase"] for f in verifies] == ["verify_1", "verify_2", "verify_3"],
            "all_three_landings_correct":len(verifies) == 3 and all(
                f["physics"]["block"]["inside"] == case["target"]
                and f["physics"]["block"]["contacts"] == ["desk"] for f in verifies),
            "released_at_target":after["block"]["inside"] == case["target"] and after["block"]["contacts"] == ["desk"],
        }
    elif goal in ("move", "move_wrist"):
        boundary = next((f["physics"] for f in row["frames"] if f["phase"] == "before_step_2"), None)
        moved = after if goal == "move" else boundary
        error = None
        if moved is not None:
            delta = np.subtract(moved["tcp_mm"], before["tcp_mm"])
            error = float(np.linalg.norm(delta-case["delta_mm"]))
            row["independent_displacement_mm"] = delta.tolist()
        row["independent_move_error_mm"] = error
        checks = {"receipt_success":status == "success", "move_within_tolerance":error is not None and error <= tolerances["move_mm"]}
        if goal == "move_wrist":
            degrees = float(np.rad2deg(after["joints"][3]-boundary["joints"][3])) if boundary else None
            row["independent_wrist_deg"] = degrees
            checks["wrist_within_tolerance"] = degrees is not None and abs(degrees-case["wrist_deg"]) <= tolerances["wrist_deg"]
    elif goal in ("ignore", "ask"):
        checks = {"expected_receipt":status == ("ignored" if goal == "ignore" else "clarify"),
                  "no_physical_change":stationary(before, after)}
    return checks


class Harness:
    def __init__(self, app, client, worker, access, report, persist, frame_dir):
        self.app, self.client, self.worker = app, client, worker
        self.access, self.report, self.persist, self.frame_dir = access, report, persist, frame_dir

    def state(self):
        return self.client.get("/api/state").json()

    def monitor(self, row):
        state = self.state()
        response = self.client.get("/camera.jpg")
        if response.status_code == 200 and cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR) is not None:
            body = row["command"]
            heartbeat = self.client.post("/api/heartbeat", json={"id":body["id"], "watchdog_token":body["watchdog_token"]})
            row["monitor_renewals"] = row.get("monitor_renewals", 0) + (heartbeat.status_code == 200)
        return state

    def observed(self, row):
        def observed_execute(sim, text, **kwargs):
            previous = kwargs.get("observer")
            def observe(phase, scene, image):
                if previous:
                    previous(phase, scene, image)
                # The second-step boundary separates translation from wrist rotation.
                if phase in ("before", "after", "lift", "before_step_2") or phase.startswith("verify_"):
                    filename = f'{row["id"]}-{phase}.jpg'
                    cv2.imwrite(str(self.frame_dir/filename), image)
                    row["frames"].append({"phase":phase, "image":filename, "scene":scene,
                                          "physics":physical_snapshot(sim)})
            kwargs["observer"] = observe
            return execute(sim, text, **kwargs)
        return patch("arm_lab.web.execute", side_effect=observed_execute)

    def transcribe(self, case, row):
        self.access.label = case["id"]
        path = ROOT/case["audio"]
        started = time.perf_counter()
        response = self.client.post("/api/transcribe", files={"file":(path.name, path.read_bytes(), "audio/wav")})
        row.update(asr_seconds=round(time.perf_counter()-started, 3), asr_http_status=response.status_code,
                   asr_response=response.json())
        if response.status_code != 200:
            return None
        text = response.json()["text"]
        ref, got = cer_text(case["reference"]), cer_text(text)
        row["asr_comparison"] = {"reference_anchors":anchors(case["reference"]), "transcript_anchors":anchors(text),
                                 "anchors_exact":anchors(case["reference"]) == anchors(text),
                                 "orthographic_cer":edit_distance(ref, got)/max(1, len(ref)),
                                 "note":"CER counts Chinese versus Arabic numeral spelling differences; anchors normalize quantities. Neither establishes complete semantic correctness."}
        return text

    def body(self, row, text, reply_to=None):
        body = {"id":str(uuid4()), "text":text, "revision":self.state()["revision"],
                "issued_at_ms":int(time.time()*1000), "watchdog_token":str(uuid4())}
        if reply_to:
            body["reply_to"] = reply_to
        row["command"] = body
        self.access.label = row["id"]
        return body

    def finish(self, future, row, started):
        while not future.done():
            self.monitor(row)
            time.sleep(.05)
        response = future.result()
        row.update(control_seconds=round(time.perf_counter()-started, 3), control_http_status=response.status_code,
                   receipt=response.json(), after=physical_snapshot(self.app.state.sim), state_after=self.state())

    def command(self, row, text, reply_to=None):
        body = self.body(row, text, reply_to)
        started = time.perf_counter()
        with self.observed(row):
            future = self.worker.submit(self.client.post, "/api/commands", json=body)
            self.finish(future, row, started)

    def pause_trial(self, case, row, specification):
        auxiliary = {**specification["auxiliary"], "frames":[], "before":physical_snapshot(self.app.state.sim)}
        self.report["auxiliary"] = auxiliary
        # Only playback pacing changes; the continuous physical scene is preserved.
        self.app.state.sim.playback_speed = .25
        body = self.body(auxiliary, auxiliary["text"])
        started = time.perf_counter()
        with self.observed(auxiliary):
            future = self.worker.submit(self.client.post, "/api/commands", json=body)
            trigger = None
            while not future.done():
                current = self.monitor(auxiliary)
                displacement = float(np.linalg.norm(np.subtract(current["tcp_mm"], auxiliary["before"]["tcp_mm"])))
                if current.get("phase") == "moving" and current.get("busy") and displacement >= .5:
                    trigger = {"state":current, "tcp_displacement_mm":displacement}
                    break
                time.sleep(.02)
            row["moving_trigger"] = trigger
            # Real ASR runs after observed physical movement, with heartbeats continuing.
            audio_future = self.worker.submit(self.transcribe, case, row)
            while not audio_future.done():
                self.monitor(auxiliary)
                time.sleep(.02)
            text = audio_future.result()
            row["motion_pending_when_asr_returned"] = not future.done()
            if text is not None:
                pause_body = self.body(row, text)
                response = self.client.post("/api/commands", json=pause_body)
                row.update(control_http_status=response.status_code, receipt=response.json())
            self.finish(future, auxiliary, started)
        row["after"] = physical_snapshot(self.app.state.sim)
        row["state_after"] = self.state()
        time.sleep(.2)
        row["settled_after"] = physical_snapshot(self.app.state.sim)
        row["physical_checks"] = {
            "observed_actual_motion":trigger is not None,
            "pause_arrived_before_auxiliary_completed":row["motion_pending_when_asr_returned"],
            "audio_pause_acknowledged":row.get("receipt", {}).get("status") == "paused",
            "auxiliary_motion_interrupted":auxiliary.get("receipt", {}).get("status") == "paused",
            "remaining_steps_not_executed":len(auxiliary.get("receipt", {}).get("steps", [])) < 2,
            "pause_latched":row["state_after"].get("paused") is True,
            "stationary_after_pause":stationary(row["after"], row["settled_after"]),
            "simulation_clock_not_reset":row["after"]["block"]["time_s"] >= auxiliary["before"]["block"]["time_s"],
            "block_preserved":np.allclose(row["after"]["block"]["xyz_m"], auxiliary["before"]["block"]["xyz_m"], atol=.0005, rtol=0),
        }


def score_planning(case, row, report):
    attempts = [item for item in report["campus_attempts"] if item["case_id"] == row["id"]]
    row["campus_attempt_indices"] = [i for i, item in enumerate(report["campus_attempts"]) if item["case_id"] == row["id"]]
    if case.get("route") == "local":
        expected_status = {"ignore":"ignored", "ask":"clarify", "pause":"paused"}[case["expected"]["local"]]
        row["local_routing_correct"] = not attempts and row.get("receipt", {}).get("status") == expected_status
    elif attempts:
        attempt = attempts[-1]
        row["raw_planner_score"] = score_response({**case, "route":"planner", "scene":attempt["task"]},
            attempt.get("raw"), finish_reason=attempt.get("finish_reason"), http_status=attempt.get("http_status"))
    else:
        row["raw_planner_score"] = {"semantic_correct":False, "raw_json_valid":False,
                                    "failure_tags":["expected_model_response_missing"]}


def run(out, *, config=CONFIG, cloud_limit=12):
    frozen = prepare(config)
    if not frozen["all_audio_ready"]:
        raise ValueError("Generate the six new offline WAVs first; input recordings are missing.")
    if out.exists() or not 1 <= cloud_limit <= 12:
        raise ValueError("Choose a new output file and a campus attempt cap in 1..12; no automatic resume/retry.")
    out.parent.mkdir(parents=True, exist_ok=True)
    spec = frozen["specification"]
    report = {**frozen, "state":"started", "mode":"real_sensevoice_raw_http_reasoner_continuous_simulation",
              "created_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "planner_prompt_sha256":PLAN_PROMPT_SHA256, "kinematics_backend":f"{ArmModel.__module__}.{ArmModel.__name__}",
              "scene_resets":0, "automatic_retries":0, "cloud_limit":cloud_limit,
              "playback_pacing":{"main_commands":"unpaced", "auxiliary_speed":0.25},
              "asr_attempts":[], "campus_attempts":[], "rows":[]}
    write_lock = RLock()
    def persist():
        with write_lock:
            save(out, report)
    access = RecordedAccess(report, persist, cloud_limit)
    frame_dir = out.with_suffix(".frames")
    frame_dir.mkdir(exist_ok=True)
    with exclusive(out.with_suffix(".lock")), ExitStack() as stack:
        if out.exists():
            raise ValueError("Another run has already created this report.")
        persist()
        # Positive startup pacing installs the normal live-camera publisher.
        # Unpace the first nine tasks only; retain camera updates during motion.
        app = create_app(access, layout=spec["initial_layout"], playback_speed=1)
        client = stack.enter_context(TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 55555)))
        app.state.sim.playback_speed = 0
        worker = stack.enter_context(ThreadPoolExecutor(2))
        harness = Harness(app, client, worker, access, report, persist, frame_dir)
        report["initial_state"] = harness.state()
        for case in spec["cases"]:
            row = {**case, "frames":[], "before":physical_snapshot(app.state.sim)}
            report["rows"].append(row)
            persist()
            print(json.dumps({"case":case["id"], "state":"started"}), flush=True)
            if case["goal"] == "pause":
                harness.pause_trial(case, row, spec)
                score_planning({**spec["auxiliary"], "route":"planner"}, report["auxiliary"], report)
            else:
                text = harness.transcribe(case, row)
                if text is not None:
                    prior = next((r for r in report["rows"] if r["id"] == case.get("reply_to")), None)
                    reply_to = prior.get("receipt", {}).get("id") if prior else None
                    if case.get("reply_to") and not reply_to:
                        row["receipt"] = {"status":"dependency_failed", "message":"The expected preceding clarification has no command receipt."}
                    else:
                        harness.command(row, text, reply_to)
                row.setdefault("after", physical_snapshot(app.state.sim))
                row["physical_checks"] = physical_checks(case, row, spec["tolerances"])
            score_planning(case, row, report)
            row["task_passed"] = bool(row["physical_checks"]) and all(bool(v) for v in row["physical_checks"].values())
            row["full_chain_passed"] = row["task_passed"] and row.get("local_routing_correct", row.get("raw_planner_score", {}).get("semantic_correct", False))
            persist()
            print(json.dumps({"case":case["id"], "status":row.get("receipt", {}).get("status"),
                              "task_passed":row["task_passed"], "cloud_calls":len(report["campus_attempts"])}), flush=True)
        scores = [r["raw_planner_score"] for r in report["rows"] if "raw_planner_score" in r]
        report["state"] = "completed"
        report["summary"] = {"audio_cases":len(report["rows"]), "asr_calls":len(report["asr_attempts"]),
            "campus_calls":len(report["campus_attempts"]),
            "auxiliary_campus_calls":sum(a["case_id"] == spec["auxiliary"]["id"] for a in report["campus_attempts"]),
            "task_passed":sum(r["task_passed"] for r in report["rows"]),
            "full_chain_passed":sum(r["full_chain_passed"] for r in report["rows"]),
            "main_planner_cases":len(scores), "raw_planner_semantic_correct":sum(s["semantic_correct"] for s in scores),
            "raw_json_valid":sum(s["raw_json_valid"] for s in scores),
            "asr_anchor_exact":sum(r.get("asr_comparison", {}).get("anchors_exact", False) for r in report["rows"]),
            "same_scene":report["scene_resets"] == 0, "source":"synthetic_audio_only_not_human_accuracy"}
        persist()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--out", type=Path, default=ROOT/"results/safety/reasoner/flow10.json")
    parser.add_argument("--cloud-limit", type=int, default=12)
    parser.add_argument("--run", action="store_true", help="Execute real local ASR, campus planning and MuJoCo; default only inventories inputs.")
    args = parser.parse_args()
    result = run(args.out, config=args.config, cloud_limit=args.cloud_limit)["summary"] if args.run else prepare(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
