"""Compare endpoint-only and full-path clearance on the taught simulation routes."""
import argparse
from itertools import product
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from arm_lab.flow import execute
from arm_lab.motion_guard import KeepoutBox, MotionGuard
from arm_lab.scene import HALF_BLOCK, OPEN, STATIONS, make_scene, taught_pose
from arm_lab.simulator import GRIP_Z, PARK, TRAVEL_Z, Simulator
from arm_lab.kinematics import ArmModel


ROOT = Path(__file__).resolve().parents[1]


def refusal(guard, a, b):
    try:
        guard.check_path(a, b)
    except ValueError as err:
        return str(err)
    return None


def compare(guard, poses):
    tip_clear = {
        label: refusal(guard, q, q) is None for label, q in poses.items()
    }
    routes = []
    for start, end in product(poses, repeat=2):
        a, b = poses[start], poses[end]
        t0 = perf_counter()
        reason = refusal(guard, a, b)
        elapsed_ms = (perf_counter()-t0) * 1000
        sampled_clear = all(
            refusal(guard, q, q) is None for q in np.linspace(a, b, 51)
        )
        routes.append({
            "start": start, "end": end,
            "endpoints_clear": tip_clear[start] and tip_clear[end],
            "sweep_clear": reason is None, "sweep_refusal": reason,
            "grid_51_clear": sampled_clear, "sweep_ms": round(elapsed_ms, 3),
        })
    times = [row["sweep_ms"] for row in routes]
    return {
        "route_count": len(routes),
        "endpoints_clear": sum(row["endpoints_clear"] for row in routes),
        "sweeps_clear": sum(row["sweep_clear"] for row in routes),
        "unsafe_sweeps_missed_by_endpoints": sum(
            row["endpoints_clear"] and not row["sweep_clear"] for row in routes
        ),
        "disagreements_with_51_sample_grid": sum(
            row["sweep_clear"] != row["grid_51_clear"] for row in routes
        ),
        "sweep_ms_median": round(float(np.median(times)), 3),
        "sweep_ms_p95": round(float(np.percentile(times, 95)), 3),
        "routes": routes,
    }


def payload_ablation():
    # A 2 x 2 x 31 mm fixed post from the desk to z=66 mm. The witness point
    # is inside the post. Containment in the measured cube proves overlap.
    post = KeepoutBox("标定挡针", (0.207705, 0.056987, 0.0505), (0.001, 0.001, 0.0155))
    witness = np.array([0.207705, 0.056987, 0.064885])
    assert np.all(np.abs(witness-post.center) < post.half_size)
    sim = Simulator("B")
    try:
        baseline_pick = execute(sim, "向前抓取木块")
        if baseline_pick["status"] != "success":
            raise RuntimeError("对照回合未能抓起测试木块")
        guarded = MotionGuard(sim.model, keepouts=[post])
        commanded = sim.data.ctrl[sim.actuators].copy()
        target = sim.taught["A", TRAVEL_Z].copy()
        target[-1] = sim.hold_position
        empty_geometry_clear = refusal(guarded, commanded, target) is None
        # Independently repeat the arm-only calculation at the actual jaw opening.
        measured_start, measured_end = commanded.copy(), target.copy()
        measured_start[-1] = measured_end[-1] = sim.joints_now()[-1]
        empty_geometry_clear &= refusal(guarded, measured_start, measured_end) is None
        occupied = []
        frames = 0
        tick = sim._tick

        def record(steps):
            nonlocal frames
            tick(steps)
            frames += 1
            geom = sim.data.geom("red_block")
            relative = geom.xmat.reshape(3, 3).T @ (witness-geom.xpos)
            if np.max(np.abs(relative)) < HALF_BLOCK:
                occupied.append({"frame":frames, "probe_in_cube_frame_m":relative.tolist()})

        sim._tick = record
        sim.station("A", TRAVEL_Z, sim.hold_position)
    finally:
        sim.close()

    sim = Simulator("B")
    try:
        sim.guard = MotionGuard(sim.model, keepouts=[post])
        pick = execute(sim, "向前抓取木块")
        before = sim.joints_now()
        place = execute(sim, "放到 A 区")
        delta = float(np.max(np.abs(sim.joints_now()-before)))
        return {
            "post_center_m": post.center, "post_half_size_m": post.half_size,
            "probe_point_m": witness.tolist(), "safety_envelope_radius_m": MotionGuard.PAYLOAD_RADIUS,
            "bare_arm_sweep_clear_at_commanded_and_measured_jaw": bool(empty_geometry_clear),
            "reference": "Open workcell; independently read free-block geometry after each "
                         "simulator control tick along the normal B to A carry segment. "
                         "Point inside both volumes proves spatial overlap with the proposed post.",
            "reference_control_ticks": frames, "reference_intersections": occupied,
            "guarded_pick_status": pick["status"], "guarded_place_status": place["status"],
            "reason": place["message"], "max_joint_change_rad_on_refusal": delta,
            "still_holding": bool(sim.holding()),
        }
    finally:
        sim.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"outputs"/"route_ablation.json")
    args = parser.parse_args()
    arm = ArmModel()
    model = make_scene()
    poses = {"PARK": PARK}
    for label, xy in STATIONS.items():
        for phase, z in (("travel", TRAVEL_Z), ("grip", GRIP_Z)):
            poses[f"{label}-{phase}"] = taught_pose(arm, xy, z, OPEN)
    report = {
        "hardware_connected": False, "model_calls": 0,
        "method": "Each ordered pair of the 7 poses, including equal endpoints. Open jaw, empty arm. "
                  "Compare endpoints, a 51-point linear joint grid and the adaptive swept clearance guard. "
                  "All three use the same course CAD geometry. The grid is a comparator, not hardware truth.",
        "time_basis": "One run on this host, preflight wall time in ms; excludes motion and the reference grid.",
        "fixture": "configs/keepouts.example.json", "poses_rad": {k:v.tolist() for k,v in poses.items()},
        "conditions": {
            "clear_table": compare(MotionGuard(model), poses),
            "registered_fixture": compare(MotionGuard.from_config(model, ROOT/"configs"/"keepouts.example.json"), poses),
        },
        "held_cube_vs_registered_fixture": payload_ablation(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for name, row in report["conditions"].items():
        print(f"{name}: {row['endpoints_clear']} endpoints clear, {row['sweeps_clear']} swept paths clear, "
              f"{row['disagreements_with_51_sample_grid']} disagreements with grid; "
              f"median/P95 {row['sweep_ms_median']}/{row['sweep_ms_p95']} ms.")
    held = report["held_cube_vs_registered_fixture"]
    print("Held cube:", held["guarded_place_status"], held["reason"],
          f"{len(held['reference_intersections'])} reference ticks overlap the post.")


if __name__ == "__main__":
    main()
