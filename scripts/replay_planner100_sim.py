"""Offline replay of a preregistered set of held-out planner replies with the actual simulation."""
import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import cv2
import httpx
import numpy as np

from arm_lab.evaluate import block_state
from arm_lab.flow import execute
from arm_lab.model_access import DEEPSEEK_ENDPOINT, ModelAccess, ModelFailure, planning_task
from arm_lab.planner_scoring import score_response
from arm_lab.simulator import Simulator
from scripts.evaluate_planner100 import DATA, OUT, RUNS
from scripts.evaluate_prompt_trial import save


CASES = ("P042", "P045", "P048", "P053", "P059", "P065", "P066", "P070", "P072", "P083", "P085", "P090")
GOLD_SHA256 = "79d77367c2f22e61fc943b1b09e8ca120f1ab03ee6529be84c915e314bcdd9c3"


def snapshot(sim):
    return {"tcp_mm": (1000 * sim.data.site_xpos[sim.tcp]).tolist(),
            "joints_rad": sim.data.qpos[sim.joints].tolist(), "block": block_state(sim)}


def scene_match(expected, actual):
    observed = actual["objects"]
    gold = expected["objects"]
    if bool(expected.get("holding")) != bool(actual.get("holding")) or len(gold) != len(observed):
        return False
    return all(g["slot"] == o["slot"] and o.get("xy_mm") is not None
               and np.linalg.norm(np.subtract(g["xy_mm"], o["xy_mm"])) <= 2
               for g, o in zip(gold, observed))


def physical_checks(case, initial, final, frames, receipt):
    expected = case["expected"]
    elapsed = final["block"]["time_s"] - initial["block"]["time_s"]
    joint_change = float(np.max(np.abs(np.subtract(final["joints_rad"], initial["joints_rad"]))))
    observation = {"elapsed_sim_seconds": elapsed, "max_joint_change_rad": joint_change}
    status = receipt["status"]
    if expected.get("action") == "ask":
        wording = json.dumps({"action":"ask","question":receipt.get("message","")},ensure_ascii=False)
        checks = {"clarified": status == "clarify", "empty_plan": not receipt.get("plan"),
                  "clarification_focus":score_response(case,wording)["ask_focus_correct"] is True,
                  "frozen": elapsed == 0 and joint_change == 0}
        return checks, observation

    steps = expected["steps"]
    wanted_plan, targets = [], []
    picks = 0
    for step in steps:
        action = step["action"]
        if action in ("pick", "transfer"):
            wanted_plan.append(f"pick({step['from']})")
            picks += 1
        if action in ("place", "transfer"):
            wanted_plan.extend((f"place({step['to']})", f"verify({step['to']})"))
            targets.append(step["to"])
        if action == "move":
            wanted_plan.append(f"move_delta({step['axis']},{step['mm']:g} mm)")
        if action == "wrist":
            wanted_plan.append(f"wrist_delta({step['deg']:g} deg)")

    checks = {"success_receipt": status == "success", "requested_actions_in_order": receipt.get("plan") == wanted_plan}
    if picks:
        lifts = [s["state"]["block"] for s in frames if s["phase"] == "lift"]
        checks["physical_lifts"] = len(lifts) == picks and all(
            set(b["contacts"]) == {"fixed_pad", "moving_pad"}
            and b["xyz_m"][2] - initial["block"]["xyz_m"][2] > .020 for b in lifts)
        if not targets:
            checks["still_held"] = set(final["block"]["contacts"]) == {"fixed_pad", "moving_pad"}
    if targets:
        verify = [s for s in frames if s["phase"].startswith("verify_")]
        checks["three_physical_landings_per_target"] = (
            [f["phase"] for f in verify] == ["verify_1", "verify_2", "verify_3"] * len(targets)
            and all(f["state"]["block"]["inside"] == targets[i//3] and f["state"]["block"]["contacts"] == ["desk"]
                    for i, f in enumerate(verify)))
    if any(s["action"] in ("move", "wrist") for s in steps):
        boundaries = [initial] + [s["state"] for s in frames if s["phase"].startswith("before_step_")] + [final]
        observations = []
        for i, (step, begin, end) in enumerate(zip(steps, boundaries, boundaries[1:])):
            if step["action"] == "move":
                axis = "xyz".index(step["axis"])
                vector = np.subtract(end["tcp_mm"], begin["tcp_mm"])
                wanted = np.zeros(3)
                wanted[axis] = step["mm"]
                err = float(np.linalg.norm(vector - wanted))
                checks[f"displacement_{i+1}"] = err <= 5 and vector[axis] * step["mm"] > 0
                observations.append({"step": i+1, "delta_xyz_mm": vector.tolist(), "error_3d_mm": err})
            elif step["action"] == "wrist":
                angle = float(np.rad2deg(end["joints_rad"][3] - begin["joints_rad"][3]))
                checks[f"rotation_{i+1}"] = abs(angle - step["deg"]) <= 2 and angle * step["deg"] > 0
                observations.append({"step": i+1, "delta_wrist_deg": angle, "error_deg": abs(angle-step["deg"])})
        checks["all_step_boundaries"] = len(boundaries) == 1+len(steps)
        observation["metric_steps"] = observations
    return {k:bool(v) for k,v in checks.items()}, observation


def replay_one(sim, case, source, frame_dir):
    scene = case["scene"]
    layout = scene["objects"][0]["slot"] if scene["objects"] else "empty"
    sim.reset(layout)
    actual, first_frame = sim.camera()
    initial = snapshot(sim)
    row = {"id":case["id"], "command":case["command"], "layout":layout, "expected":case["expected"],
           "raw":source["raw"], "finish_reason":source["finish_reason"],
           "raw_score":score_response(case,source["raw"],finish_reason=source["finish_reason"]),
           "actual_scene":actual, "initial_physics":initial,
           "initial_scene_matches":bool(scene_match(scene, actual)), "frames":[], "sent_to_transport":0}

    def frame_file(tag,image):
        filename = f"{case['id']}_{tag}.jpg"
        if not cv2.imwrite(str(frame_dir/filename), image):
            raise OSError(f"Cannot save {filename}")
        return filename

    row["initial_frame"] = frame_file("before",first_frame)
    if not row["initial_scene_matches"]:
        return {**row,"status":"scene_mismatch","end_to_end":False}
    observation = planning_task(case["command"], {**actual, "dialogue":scene.get("dialogue")})

    def respond(request):
        if str(request.url) != DEEPSEEK_ENDPOINT:
            raise AssertionError("Unexpected replay endpoint")
        sent = json.loads(json.loads(request.content)["messages"][1]["content"])
        if sent != observation:
            raise AssertionError("The planner did not receive the current camera and conversation")
        row["sent_to_transport"] += 1
        return httpx.Response(200,json={"choices":[{"message":{"content":source["raw"]},
                                                    "finish_reason":source["finish_reason"]}]})

    def record(phase, camera, frame):
        if phase == "lift" or phase.startswith(("verify_", "before_step_")):
            row["frames"].append({"phase":phase, "camera":camera, "state":snapshot(sim)})

    original_client = httpx.Client
    def local_client(**kwargs):
        return original_client(transport=httpx.MockTransport(respond), **kwargs)

    # Replay the campus envelope through the real response parser and command/scene checks.
    # The only HTTP client constructed in this scope has an in-memory transport.
    access = ModelAccess()
    with patch("arm_lab.model_access.campus_api_key", return_value="offline"), \
         patch("arm_lab.model_access.httpx.Client", side_effect=local_client):
        try:
            receipt = execute(sim,case["command"],planner=lambda cmd,s:access.plan(cmd,s,approved_once=True),
                              previous=scene.get("dialogue"),observer=record)
        except ModelFailure as exc:
            receipt = {"status":"planner_rejected", "message":str(exc), "plan":[]}
    row["receipt"] = receipt
    row["final_scene"], last_frame = sim.camera()
    row["final_frame"] = frame_file("after",last_frame)
    row["final_physics"] = snapshot(sim)
    checks, measured = physical_checks(case, initial, row["final_physics"], row["frames"], receipt)
    row.update(checks=checks, measurements=measured, status=receipt["status"], end_to_end=all(checks.values()))
    return row


def run(version, out, fixture):
    source_digest = None
    data = DATA.read_bytes()
    if hashlib.sha256(data).hexdigest() != GOLD_SHA256:
        raise ValueError("Dataset differs from preregistered replay")
    cases = {r["id"]:r for r in json.loads(data)["cases"]}
    if fixture:
        sources = {key:{"raw":json.dumps(cases[key]["expected"],ensure_ascii=False),"finish_reason":"stop"}
                   for key in CASES}
    else:
        raw = RUNS.read_bytes()
        source_digest = hashlib.sha256(raw).hexdigest()
        book = json.loads(raw)
        if book["dataset_sha256"] != GOLD_SHA256 or version not in book.get("test_versions",[]):
            raise ValueError("Responses do not match the frozen data or held-out version")
        sources = {r["id"]:r for r in book["attempts"] if r.get("http_status")==200
                   and r.get("state")=="answered" and r["version"]==version and r["split"]=="test"}
    frames = out.with_suffix(".frames")
    frames.mkdir(parents=True, exist_ok=True)
    report = {"source_type":"gold_fixture" if fixture else "recorded_heldout_reply",
              "version":version, "source_sha256":source_digest, "gold_sha256":GOLD_SHA256,
              "case_ids":CASES, "new_model_calls":0,"backend":"MuJoCo + OpenCV", "rows":[]}
    with ExitStack() as stack:
        sim = Simulator("A")
        stack.callback(sim.close)
        for key in CASES:
            if key not in sources:
                report["rows"].append({"id":key,"status":"no_recorded_reply","end_to_end":False})
                continue
            report["rows"].append(replay_one(sim, cases[key], sources[key], frames))
    rows = report["rows"]
    report["summary"] = {"selected":len(rows), "available":sum(r["status"]!="no_recorded_reply" for r in rows),
                         "initial_scene_match":sum(r.get("initial_scene_matches",False) for r in rows),
                         "raw_semantic_correct":sum(r.get("raw_score",{}).get("semantic_correct",False) for r in rows),
                         "end_to_end":sum(r["end_to_end"] for r in rows), "by_status":dict(Counter(r["status"] for r in rows))}
    save(out,report)
    print(json.dumps(report["summary"],ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--version", choices=("baseline","candidate","revised","finalist"))
    group.add_argument("--fixture",action="store_true",help="Use the gold answer only to verify the offline harness")
    args = parser.parse_args()
    file = Path("outputs/planner100/simulation_fixture.json") if args.fixture else OUT/f"simulation_{args.version}.json"
    run(args.version or "fixture",file,args.fixture)
