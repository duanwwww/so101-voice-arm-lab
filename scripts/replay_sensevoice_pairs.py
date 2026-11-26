"""Check public-clone kinematics with saved model replies; makes no model calls."""
import argparse
import hashlib
import json
from pathlib import Path

from arm_lab.flow import execute
from arm_lab.kinematics import ArmModel
from arm_lab.planner import check_command_match, decode_model_plan
from arm_lab.simulator import Simulator
from scripts.evaluate_prompt_trial import save
from scripts.evaluate_sensevoice_flow import OUT, physical_snapshot
from scripts.evaluate_sensevoice_pairs import physical_checks


def run(source, out):
    if out.exists():
        raise ValueError("Choose a new output to preserve previous evidence.")
    if ArmModel.__module__ != "arm_lab.kinematics":
        raise ValueError("Set VOICE_ARM_KINEMATICS=standalone before starting Python.")
    blob = source.read_bytes()
    original = json.loads(blob)
    if original.get("state") != "complete":
        raise ValueError("Source paired run must be complete.")
    report = {"mode": "offline_saved_live_json_replay_public_kinematics",
              "source_file": source.name, "source_sha256": hashlib.sha256(blob).hexdigest(),
              "kinematics_backend": f"{ArmModel.__module__}.{ArmModel.__name__}",
              "asr_calls": 0, "cloud_calls": 0, "rows": []}
    for original_row in original["rows"]:
        indices = original_row["campus_attempt_indices"]
        if len(indices) != 1:
            raise ValueError("Replay requires exactly one archived model response per trial.")
        call = original["campus_attempts"][indices[0]]
        if call.get("http_status") != 200 or call.get("finish_reason") != "stop":
            raise ValueError("Incomplete archived reply cannot be replayed.")
        raw = call["raw"]
        row = {"id": original_row["id"], "layout": original_row["layout"],
               "raw_model_reply": raw, "frames": []}
        sim = Simulator()
        try:
            sim.reset(row["layout"])
            row["before"] = physical_snapshot(sim)

            def planner(text, scene):
                proposed = decode_model_plan(json.loads(raw), scene)
                check_command_match(text, proposed, scene)
                return proposed

            def observer(phase, scene, frame):
                if phase in ("before", "after", "lift") or phase.startswith("verify_"):
                    row["frames"].append({"phase": phase, "scene": scene,
                                          "physics": physical_snapshot(sim)})

            row["receipt"] = execute(sim, original_row["text"], planner=planner, observer=observer)
            row["after"] = physical_snapshot(sim)
            row["physical_checks"] = physical_checks(row)
            row["passed"] = all(value is True for value in row["physical_checks"].values())
            report["rows"].append(row)
            save(out, report)
        finally:
            sim.close()
    report["summary"] = {"passed": sum(r["passed"] for r in report["rows"]),
                         "total": len(report["rows"]), "new_model_calls": 0}
    save(out, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=OUT / "pairs_live.json")
    parser.add_argument("--out", type=Path, default=OUT / "pairs_standalone.json")
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.out)["summary"], ensure_ascii=False))
