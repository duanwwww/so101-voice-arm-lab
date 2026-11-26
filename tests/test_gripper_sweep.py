"""Repeat the held-effort comparison on a fixed set of four simulated layouts."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arm_lab.evaluate import block_state
from arm_lab.flow import execute
from arm_lab.simulator import Simulator


CASES = (("A", "C"), ("B", "A"), ("C", "B"), ("rotated", "B"))


def run(output):
    report = {"hardware": False, "model_calls": 0, "close_torque_limit_Nm": 0.80,
              "visual_alignment": False, "placement_policy": "same taught grasp point for all torque variants",
              "required_lift_mm": 20, "required_stable_landings": 3, "variants": []}
    sim = Simulator()
    # Hold this ablation to its original taught grasp points; the separate
    # placement benchmark varies visual alignment as its independent variable.
    sim.visual_alignment = False
    try:
        for effort in (0.08, 0.30, 0.50, 0.75, 0.80):
            trials = []
            with patch("arm_lab.simulator.HOLD_TORQUE", effort):
                for layout, target in CASES:
                    sim.reset(layout)
                    observations = {}
                    def observe(phase, scene, image):
                        observations[phase] = block_state(sim)
                    receipt = execute(sim, f"把木块放到 {target} 区", observer=observe)
                    base_z = observations["before"]["xyz_m"][2]
                    lift = observations.get("lift")
                    height = round((lift["xyz_m"][2] - base_z) * 1000, 2) if lift else None
                    both_pads = bool(lift and set(lift["contacts"]) == {"fixed_pad", "moving_pad"})
                    landings = [observations[p] for p in ("verify_1", "verify_2", "verify_3")
                                if p in observations]
                    stable = len(landings) == 3 and all(
                        v["inside"] == target and v["contacts"] == ["desk"] for v in landings)
                    trials.append({"layout": layout, "goal": target, "status": receipt["status"],
                                   "message": receipt["message"], "phases": receipt["phases"],
                                   "lift_mm": height, "lift_touches_only_both_pads": both_pads,
                                   "target_stable_three_times": stable,
                                   "success": bool(receipt["status"] == "success" and both_pads
                                                   and height > 20 and stable),
                                   "simulated_elapsed_s": round(sim.data.time-.45, 3),
                                   "gripper": receipt["gripper"]})
            passed = sum(r["success"] for r in trials)
            print(f"{effort:.2f} Nm: {passed}/{len(trials)} stable transfers")
            report["variants"].append({"hold_effort_Nm": effort, "successes": passed, "trials": trials})
    finally:
        sim.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", type=Path, default=Path("outputs/grip_sweep.json"))
    run(cli.parse_args().out)
