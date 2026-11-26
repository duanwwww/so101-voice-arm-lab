"""Offline before/after validator replay; no ASR or cloud inference.

Read frozen original replies and original guard decisions. Current validation is
replayed without modifying those sources. Seven previously rejected, correct
Reasoner replies are also executed in independent MuJoCo scenes. These are
selected regression cases, not new blind inputs or a new audio-chain trial.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import cv2
import httpx

from arm_lab.flow import execute
from arm_lab.kinematics import ArmModel
from arm_lab.model_access import DEEPSEEK_ENDPOINT, ModelAccess, ModelFailure, planning_task
from arm_lab.planner_scoring import has_motion, score_response
from arm_lab.simulator import Simulator
from scripts.evaluate_prompt_trial import save
from scripts.evaluate_reasoner import guard_result
from scripts.evaluate_reasoner_noise import assess
from scripts.replay_planner100_guard import replay_choice
from scripts.replay_planner100_sim import physical_checks, scene_match, snapshot

ROOT = Path(__file__).resolve().parents[1]
SIMULATION_CASES = ("P013", "P017", "P058", "P059", "P066", "P075", "P081")
SOURCES = {
    "paired": "results/safety/reasoner/paired.json",
    "noise": "results/safety/reasoner/noise.json",
    "history_replies": "results/safety/planner100/responses.json",
    "history_guard": "results/safety/planner100/guard_replay.json",
    "history_cases": "configs/planner100.json",
}


def load_sources():
    sources, fingerprints = {}, {}
    for name, relative in SOURCES.items():
        blob = (ROOT / relative).read_bytes()
        sources[name] = json.loads(blob)
        fingerprints[name] = {"path": relative, "sha256": hashlib.sha256(blob).hexdigest()}
    old = sources["history_guard"]
    if (old["responses_sha256"] != fingerprints["history_replies"]["sha256"]
            or old["dataset_sha256"] != fingerprints["history_cases"]["sha256"]):
        raise ValueError("Historical replies or gold dataset no longer match the original guard report.")
    return sources, fingerprints


def tally(rows, field):
    """Separate raw correctness, correct motion admission, and incorrect admission."""
    def admitted(row):
        return row[field]["outcome"] == "admitted_motion"
    def correct(row):
        return row[field].get("goal_matches_reference", False)
    wanted_actions = [r for r in rows if r["expected_motion"]]
    raw_correct_actions = [r for r in wanted_actions if r["raw_semantic_correct"]]
    return {
        "replies": len(rows),
        "expected_action_cases": len(wanted_actions),
        "raw_semantic_correct": sum(r["raw_semantic_correct"] for r in rows),
        "raw_correct_action_cases": len(raw_correct_actions),
        "correct_motion_admitted": sum(admitted(r) and correct(r) for r in rows),
        "raw_correct_actions_admitted": sum(admitted(r) and correct(r) for r in raw_correct_actions),
        "raw_correct_actions_not_admitted": sum(not (admitted(r) and correct(r)) for r in raw_correct_actions),
        "wrong_motion_admitted": sum(admitted(r) and not correct(r) for r in rows),
        "outcomes": dict(Counter(r[field]["outcome"] for r in rows)),
        "wrong_motion_admitted_ids": [r["id"] for r in rows if admitted(r) and not correct(r)],
    }


def comparison(rows):
    old, new = tally(rows, "before_guard"), tally(rows, "after_guard")
    changed = [r for r in rows if r["before_guard"] != r["after_guard"]]
    return {"before": old, "after": new,
            "changed_ids": [r["id"] for r in changed],
            "newly_correct_motion_admitted_ids": [r["id"] for r in changed
                if r["after_guard"]["outcome"] == "admitted_motion"
                and r["after_guard"].get("goal_matches_reference")
                and not (r["before_guard"]["outcome"] == "admitted_motion"
                         and r["before_guard"].get("goal_matches_reference"))],
            "new_wrong_motion_admitted_ids": [r["id"] for r in changed
                if r["after_guard"].get("wrong_motion_admitted")
                and not r["before_guard"].get("wrong_motion_admitted")]}


def replay_paired(book):
    cases = {c["id"]: c for c in book["manifest"]["cases"]}
    attempts = {a["index"]: a for a in book["attempts"]}
    rows = []
    for old in book["scores"]:
        case, source = cases[old["case_id"]], attempts[old["attempt_index"]]
        raw_score = score_response(case, source["raw"], finish_reason=source["finish_reason"],
                                   http_status=source["http_status"])
        if raw_score != old["scores"]:
            raise ValueError(f"Raw planner score changed: {case['id']}/{old['model']}")
        rows.append({"id": case["id"], "model": old["model"], "command": case["command"],
                     "expected_motion": has_motion(case["expected"]),
                     "raw_semantic_correct": raw_score["semantic_correct"],
                     "raw_scores_unchanged": True, "raw_scores": raw_score,
                     "before_guard": old["guard"], "after_guard": guard_result(case, source)})
    return {"raw_scores_unchanged": True,
            "summary": {model: comparison([r for r in rows if r["model"] == model])
                        for model in ("deepseek-reasoner", "deepseek-chat")}, "rows": rows}


def replay_noise(book):
    cases = {c["id"]: c for c in book["manifest"]["cases"]}
    attempts = {a["case_id"]: a for a in book["attempts"] if a["state"] == "answered"}
    rows = []
    for old in book["rows"]:
        case = cases[old["id"]]
        new = assess(case, attempts[case["id"]])
        if new["raw_scores"] != old["raw_scores"]:
            raise ValueError(f"Raw noise score changed: {case['id']}")
        rows.append({"id": case["id"], "group": case["group"], "command": case["command"],
                     "expected_motion": has_motion(case["expected"]),
                     "raw_semantic_correct": old["raw_scores"]["semantic_correct"],
                     "raw_scores_unchanged": True, "before_guard": old["guard"], "after_guard": new["guard"],
                     "before_original_intent_risk_admitted": old["relative_to_original_intent_risk_admitted_by_guard"],
                     "after_original_intent_risk_admitted": new["relative_to_original_intent_risk_admitted_by_guard"]})
    groups = {}
    for group in ("recoverable", "clarify", "unrecoverable_from_text"):
        part = [r for r in rows if r["group"] == group]
        groups[group] = {**comparison(part),
            "before_original_intent_risk_admitted": sum(r["before_original_intent_risk_admitted"] for r in part),
            "after_original_intent_risk_admitted": sum(r["after_original_intent_risk_admitted"] for r in part)}
    return {"raw_scores_unchanged": True, "summary": groups, "rows": rows,
            "limitation": "Literal-text correctness does not imply agreement with hidden original intent. Lost negation, numbers or units cannot be reconstructed by this validator."}


def history_guard(row):
    return {"outcome": row["guard_outcome"], "accepted_plan": row["accepted_plan"],
            "reason": row["reason"], "goal_matches_reference": row["guard_goal_match"],
            "wrong_motion_admitted": row["any_wrong_motion_admitted"]}


def replay_history(sources):
    cases = {c["id"]: c for c in sources["history_cases"]["cases"]}
    book = sources["history_replies"]
    attempts = {(a["id"], a["version"]): a for a in book["attempts"] if a["state"] == "answered"}
    rows = []
    for old in sources["history_guard"]["rows"]:
        case = cases[old["id"]]
        source = attempts[(old["id"], old["version"])]
        new = replay_choice(case, source, book["prompts"][old["version"]])
        score_fields = ("raw_strict_schema", "raw_semantic_correct", "raw_failure_tags",
                        "format_adjustments_for_diagnosis", "correct_meaning_in_bare_action", "wrong_motion_proposed")
        if any(new[k] != old[k] for k in score_fields):
            raise ValueError(f"Raw historical score changed: {old['id']}/{old['version']}")
        rows.append({"id": old["id"], "version": old["version"], "split": old["split"],
                     "command": case["command"], "expected_motion": has_motion(case["expected"]),
                     "raw_semantic_correct": old["raw_semantic_correct"], "raw_scores_unchanged": True,
                     "correct_bare_action_meaning": old["correct_meaning_in_bare_action"],
                     "before_guard": history_guard(old), "after_guard": history_guard(new)})
    return {"raw_scores_unchanged": True, "summary": {
        split: {version: comparison([r for r in rows if r["split"] == split and r["version"] == version])
                for version in book["prompts"] if any(r["split"] == split and r["version"] == version for r in rows)}
        for split in ("dev", "test")}, "rows": rows,
        "definition": "Raw correctness keeps the original strict protocol score. Bare-action replies accepted by the historical parser may be semantically correct after wrapping and are labeled separately; they do not increase raw correctness."}


def simulate_one(case, source, prompt, frame_dir):
    scene = case["scene"]
    layout = scene["objects"][0]["slot"] if scene["objects"] else "empty"
    sim = Simulator(layout)
    try:
        actual, image = sim.camera()
        initial = snapshot(sim)
        row = {"id": case["id"], "command": case["command"], "source_model": "deepseek-reasoner",
               "source_attempt_index": source["index"], "source_raw": source["raw"],
               "layout": layout, "expected": case["expected"], "dialogue": scene.get("dialogue"),
               "initial_physics": initial, "initial_observation": actual,
               "initial_scene_matches": bool(scene_match(scene, actual)), "frames": [],
               "mock_http_responses_used": 0}
        def frame(name, img):
            relative = f"{case['id']}-{name}.jpg"
            if not cv2.imwrite(str(frame_dir / relative), img):
                raise OSError("Could not save simulation image")
            return relative
        row["initial_image"] = frame("before", image)
        if not row["initial_scene_matches"]:
            row.update(status="scene_mismatch", physical_passed=False)
            return row
        observed_task = planning_task(case["command"], {**actual, "dialogue": scene.get("dialogue")})
        def respond(request):
            sent = json.loads(request.content)
            if (str(request.url) != DEEPSEEK_ENDPOINT or sent["model"] != "deepseek-reasoner"
                    or sent["messages"][0]["content"] != prompt
                    or json.loads(sent["messages"][1]["content"]) != observed_task):
                raise AssertionError("Replay did not use the expected prompt, command and current observation")
            row["mock_http_responses_used"] += 1
            return httpx.Response(200, json={"model": "offline-replay-of-deepseek-reasoner",
                "choices": [{"message": {"content": source["raw"]}, "finish_reason": source["finish_reason"]}]})
        original_client = httpx.Client
        def local_client(**kwargs):
            return original_client(transport=httpx.MockTransport(respond), **kwargs)
        def observer(phase, observed, img):
            if phase == "lift" or phase.startswith(("verify_", "before_step_")):
                row["frames"].append({"phase": phase, "state": snapshot(sim), "camera": observed,
                                      "image": frame(phase, img)})
        access = ModelAccess(planner_model="deepseek-reasoner")
        def planner(text, observation):
            return access.plan(text, observation, approved_once=True)
        planner.planner_model = "offline-replay-of-deepseek-reasoner"
        with patch("arm_lab.model_access.campus_api_key", return_value="offline-placeholder"), \
                patch("arm_lab.model_access.PLAN_INSTRUCTIONS", prompt), \
                patch("arm_lab.model_access.httpx.Client", side_effect=local_client):
            try:
                receipt = execute(sim, case["command"], planner=planner,
                                  previous=scene.get("dialogue"), observer=observer)
            except ModelFailure as exc:
                receipt = {"status": "planner_rejected", "message": str(exc), "plan": []}
        final_scene, image = sim.camera()
        final = snapshot(sim)
        checks, measurements = physical_checks(case, initial, final, row["frames"], receipt)
        row.update(receipt=receipt, final_physics=final, final_scene=final_scene,
                   final_image=frame("after", image), checks=checks, measurements=measurements,
                   status=receipt["status"], physical_passed=all(checks.values()))
        return row
    finally:
        sim.close()


def run(directory):
    sources, fingerprints = load_sources()
    directory.mkdir(parents=True, exist_ok=False)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "mode": "frozen raw reply replay through revised guard; independent simulation cases",
              "new_asr_calls": 0, "new_cloud_calls": 0, "sources": fingerprints,
              "validator_sha256": hashlib.sha256((ROOT / "arm_lab/planner.py").read_bytes()).hexdigest(),
              "kinematics_backend": f"{ArmModel.__module__}.{ArmModel.__name__}",
              "scope": "Known regression inputs; no prompt changes, new model replies or audio inputs. Original reports remain unchanged.",
              "paired": replay_paired(sources["paired"]), "noise": replay_noise(sources["noise"]),
              "historical": replay_history(sources), "simulation": {"rows": []}}
    output = directory / "report.json"
    save(output, report)
    frames = directory / "frames"
    frames.mkdir()
    paired = sources["paired"]
    cases = {c["id"]: c for c in paired["manifest"]["cases"]}
    replies = {a["case_id"]: a for a in paired["attempts"]
               if a["model"] == "deepseek-reasoner" and a["state"] == "answered"}
    for key in SIMULATION_CASES:
        row = simulate_one(cases[key], replies[key], paired["manifest"]["prompt"], frames)
        report["simulation"]["rows"].append(row)
        save(output, report)
        print(json.dumps({"id": key, "physical_passed": row["physical_passed"], "status": row["status"]}), flush=True)
    rows = report["simulation"]["rows"]
    report["simulation"]["summary"] = {"selected": len(SIMULATION_CASES), "tested": len(rows),
        "initial_scene_matches": sum(r["initial_scene_matches"] for r in rows),
        "physical_passed": sum(r["physical_passed"] for r in rows),
        "by_status": dict(Counter(r["status"] for r in rows)),
        "setup": "One newly initialized MuJoCo instance per selected case, not ten continuous audio tasks.",
        "tolerances": {"translation_error_mm": 5, "wrist_error_deg": 2,
                       "nonzero_correct_direction_required": True}}
    report["new_wrong_motion_admissions"] = [
        {"source": source_name, "id": row["id"], "model_or_version": row.get("model", row.get("version")),
         "command": row["command"], "after_guard": row["after_guard"]}
        for source_name in ("paired", "noise", "historical") for row in report[source_name]["rows"]
        if row["after_guard"].get("wrong_motion_admitted")
        and not row["before_guard"].get("wrong_motion_admitted")]
    # Guard source integrity even if the implementation or harness fails halfway.
    if load_sources()[1] != fingerprints:
        raise ValueError("A frozen input report changed during evaluation")
    report["state"] = "completed"
    save(output, report)
    return report


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", type=Path, default=ROOT / "results/safety/guard_revision")
    cli.add_argument("--run", action="store_true", help="Execute offline guard and MuJoCo replay; no live inference.")
    args = cli.parse_args()
    if args.run:
        result = run(args.out)
        print(json.dumps({"paired": result["paired"]["summary"], "noise": result["noise"]["summary"],
                          "simulation": result["simulation"]["summary"],
                          "new_wrong_motion_admissions": result["new_wrong_motion_admissions"]},
                         ensure_ascii=False, indent=2))
    else:
        sources, hashes = load_sources()
        print(json.dumps({"sources": hashes, "paired_replies": len(sources["paired"]["scores"]),
                          "noise_replies": len(sources["noise"]["rows"]),
                          "historical_replies": len(sources["history_guard"]["rows"]),
                          "simulation_case_ids": SIMULATION_CASES,
                          "new_model_calls": 0, "note": "Use --run for offline replay; output directory must be new."},
                         ensure_ascii=False, indent=2))
