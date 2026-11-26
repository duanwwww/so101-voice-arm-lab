"""Live paired planning/execution on a saved, unedited SenseVoice transcription.

The four pair members reuse one recorded ASR result; reference controls are typed
text, not additional speech samples. Every trial starts in its declared scene.
The approval argument covers this script plus the source run, including failures.
"""
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
import numpy as np

from arm_lab.flow import execute
from arm_lab.kinematics import ArmModel
from arm_lab.model_access import PLAN_PROMPT_SHA256
from arm_lab.planner_scoring import score_response
from arm_lab.web import create_app
from scripts.evaluate_prompt_trial import exclusive, save
from scripts.evaluate_sensevoice_flow import OUT, ROOT, RecordedAccess, physical_snapshot


def prepare(source, approved_total):
    """Freeze provenance, inputs and expected outputs before any model request."""
    raw = source.read_bytes()
    main = json.loads(raw)
    if (main.get("mode") != "real_sensevoice_to_http_to_live_deepseek"
            or main.get("asr_engine") != "sensevoice"
            or main.get("planner_prompt_sha256") != PLAN_PROMPT_SHA256
            or main.get("summary", {}).get("total") != 4):
        raise ValueError("Source must be a completed live SenseVoice run with the current frozen prompt.")
    prior = len(main["campus_attempts"])
    if main["summary"].get("cloud_calls") != prior:
        raise ValueError("Source attempt count disagrees with its completed summary.")
    if not 1 <= approved_total <= 8 or approved_total - prior < 4:
        raise ValueError("Approval must leave four attempts after charging the source run; total cap is eight.")
    transfer = next(row for row in main["rows"] if row["id"] == "transfer")
    case = next(row for row in json.loads((ROOT / "configs/prompt_trial.json").read_bytes())["cases"]
                if row["id"] == "transfer")
    wav = ROOT / "results/safety/prompt_trial_audio" / case["wav"]
    audio_hash = hashlib.sha256(wav.read_bytes()).hexdigest()
    transcript = transfer["asr_response"]["text"]
    if (transfer.get("asr_http_status") != 200
            or transfer["asr_response"].get("engine") != "sensevoice"
            or transfer["audio_sha256"] != audio_hash
            or transfer["command"]["text"] != transcript
            or transfer["reference"] != case["reference"]
            or "右干" not in transcript
            or not any(a["task"]["command"] == transcript for a in main["campus_attempts"])):
        raise ValueError("Saved raw ASR, submitted command, audio hash or reference failed provenance checks.")
    # Equal terminal punctuation keeps the reference pair focused on the word error.
    reference = case["reference"].rstrip("。.!?！？") + "。"
    trials = []
    for layout, variant in (("A", "noisy"), ("A", "reference"),
                            ("C", "reference"), ("C", "noisy")):
        trials.append({"id": f"{layout}_{variant}", "layout": layout, "variant": variant,
                       "text": transcript if variant == "noisy" else reference,
                       "input_source": "reused_raw_asr" if variant == "noisy" else "typed_reference_control",
                       "expected": {"steps": [{"action": "transfer", "object": "red_1",
                                                "from": layout, "to": "B"}]}})
    return {"source_file": source.name, "source_sha256": hashlib.sha256(raw).hexdigest(),
            "audio_sha256": audio_hash, "source_campus_attempts": prior,
            "approved_cumulative_attempts": approved_total, "trials": trials}


def physical_checks(row):
    """Independent measurements; an unavailable measurement is null, never success."""
    before = row["before"]["block"]
    lifts = [frame["physics"]["block"] for frame in row["frames"] if frame["phase"] == "lift"]
    verifies = [frame for frame in row["frames"] if frame["phase"].startswith("verify_")]
    return {
        "receipt_success": row["receipt"].get("status") == "success",
        "initial_source_matches": before["inside"] == row["layout"],
        "single_lift_with_both_pads_and_over_20mm": (
            len(lifts) == 1 and set(lifts[0]["contacts"]) == {"fixed_pad", "moving_pad"}
            and lifts[0]["xyz_m"][2] - before["xyz_m"][2] > .020) if lifts else None,
        "verification_phases": [frame["phase"] for frame in verifies] == ["verify_1", "verify_2", "verify_3"],
        "all_three_landings_inside_B_only_desk_contact": (
            len(verifies) == 3 and all(frame["physics"]["block"]["inside"] == "B"
                                      and frame["physics"]["block"]["contacts"] == ["desk"]
                                      for frame in verifies)) if verifies else None,
        "released_after": row["after"]["block"]["contacts"] == ["desk"]
                          and row["after"]["block"]["inside"] == "B",
    }


def run(source, out, approved_total, *, dry_run=False):
    frozen = prepare(source, approved_total)
    if dry_run:
        return frozen
    if out.exists():
        raise ValueError("Existing run retained; this script does not resume or retry model requests.")
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {"mode": "saved_raw_sensevoice_and_typed_reference_to_live_deepseek_http_simulation",
              **frozen, "planner_prompt_sha256": PLAN_PROMPT_SHA256,
              "kinematics_backend": f"{ArmModel.__module__}.{ArmModel.__name__}",
              "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "asr_calls": 0, "automatic_retries": 0, "fresh_scene_per_trial": True,
              "campus_attempts": [], "rows": [], "state": "started"}
    persist = lambda: save(out, report)
    remaining = approved_total - frozen["source_campus_attempts"]
    access = RecordedAccess(report, persist, remaining)
    access.remaining["asr"] = 0
    frames = out.with_suffix(".frames")
    frames.mkdir(exist_ok=True)
    # Hold the source lock as well: concurrent follow-ups must not spend its allowance.
    with exclusive(source.with_suffix(".lock")), exclusive(out.with_suffix(".lock")):
        if out.exists():
            raise ValueError("Another process has already started this output.")
        if hashlib.sha256(source.read_bytes()).hexdigest() != frozen["source_sha256"]:
            raise ValueError("Source report changed after the trial plan was frozen.")
        persist()
        for trial in frozen["trials"]:
            row = {**trial, "frames": [], "state": "started"}
            report["rows"].append(row)
            persist()
            with ExitStack() as stack:
                app = create_app(access, layout=trial["layout"], playback_speed=0)
                client = stack.enter_context(TestClient(app, base_url="http://127.0.0.1",
                                                        client=("127.0.0.1", 55555)))
                worker = stack.enter_context(ThreadPoolExecutor(1))
                row["before"] = physical_snapshot(app.state.sim)
                row["initial_state"] = client.get("/api/state").json()
                command_id, token = str(uuid4()), str(uuid4())
                body = {"id": command_id, "text": trial["text"],
                        "revision": row["initial_state"]["revision"],
                        "issued_at_ms": int(time.time() * 1000), "watchdog_token": token,
                        "use_deepseek": True}
                row["command"] = body
                attempt_start = len(report["campus_attempts"])

                def observed_execute(sim, text, **kwargs):
                    original_observer = kwargs.get("observer")
                    def observe(phase, scene, image):
                        if original_observer:
                            original_observer(phase, scene, image)
                        if phase in ("before", "after", "lift") or phase.startswith("verify_"):
                            filename = f"{trial['id']}-{phase}.jpg"
                            if not cv2.imwrite(str(frames / filename), image):
                                raise OSError("Failed to save a simulation evidence frame.")
                            row["frames"].append({"phase": phase, "image": filename, "scene": scene,
                                                  "physics": physical_snapshot(sim)})
                    kwargs["observer"] = observe
                    return execute(sim, text, **kwargs)

                started = time.perf_counter()
                row["successful_monitor_renewals"] = 0
                with patch("arm_lab.web.execute", side_effect=observed_execute):
                    future = worker.submit(client.post, "/api/commands", json=body)
                    while not future.done():
                        state = client.get("/api/state")
                        frame = client.get("/camera.jpg")
                        if state.status_code == 200 and frame.status_code == 200:
                            decoded = cv2.imdecode(np.frombuffer(frame.content, np.uint8), cv2.IMREAD_COLOR)
                            if decoded is not None:
                                beat = client.post("/api/heartbeat", json={"id": command_id, "watchdog_token": token})
                                row["successful_monitor_renewals"] += beat.status_code == 200
                        time.sleep(.1)
                    response = future.result()
                row.update(control_seconds=round(time.perf_counter() - started, 3),
                           control_http_status=response.status_code, receipt=response.json(),
                           after=physical_snapshot(app.state.sim))
                row["campus_attempt_indices"] = list(range(attempt_start, len(report["campus_attempts"])))
                attempts = report["campus_attempts"][attempt_start:]
                call = attempts[0] if len(attempts) == 1 else {}
                row["raw_scores"] = score_response(
                    {"route": "planner", "expected": trial["expected"],
                     "scene": call.get("task", {"objects": []})}, call.get("raw"),
                    finish_reason=call.get("finish_reason"), http_status=call.get("http_status"))
                row["physical_checks"] = physical_checks(row)
                row["task_completed"] = all(value is True for value in row["physical_checks"].values())
                row["state"] = "complete"
                persist()
                print(json.dumps({"id": row["id"], "raw_strict_exact": row["raw_scores"]["strict_exact"],
                                  "task_completed": row["task_completed"], "status": row["receipt"].get("status")},
                                 ensure_ascii=False), flush=True)
        summary = {"total": len(report["rows"]), "asr_calls": 0,
                   "cloud_calls": len(report["campus_attempts"]),
                   "cumulative_cloud_attempts": frozen["source_campus_attempts"] + len(report["campus_attempts"])}
        for variant in ("noisy", "reference"):
            rows = [row for row in report["rows"] if row["variant"] == variant]
            summary[variant] = {"total": len(rows),
                                "raw_strict_exact": sum(row["raw_scores"]["strict_exact"] for row in rows),
                                "raw_semantic_correct": sum(row["raw_scores"]["semantic_correct"] for row in rows),
                                "task_completed": sum(row["task_completed"] for row in rows)}
        report.update(summary=summary, state="complete")
        persist()
    return report


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source", type=Path, default=OUT / "flow_live.json")
    cli.add_argument("--out", type=Path, default=OUT / "pairs_live.json")
    cli.add_argument("--approved-campus-attempts", type=int, required=True,
                     help="Cumulative explicit approval, including source-run attempts; at most eight.")
    cli.add_argument("--dry-run", action="store_true", help="Validate provenance and print fixed inputs; no model calls.")
    args = cli.parse_args()
    result = run(args.source, args.out, args.approved_campus_attempts, dry_run=args.dry_run)
    print(json.dumps(result.get("summary", result), ensure_ascii=False, indent=2))
