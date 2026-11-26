"""Independent physical ground truth; simulator control and vision never import this module."""
import json
from pathlib import Path

import cv2
import numpy as np

from .flow import execute
from .scene import STATIONS, LAYOUTS, TABLE_Z, HALF_BLOCK, HALF_ZONE
from .simulator import Simulator


CASES = [
    ("pick_A","A","把红色木块放到 C 区","success"),
    ("pick_B","B","把红色木块放到 C 区","success"),
    ("empty","empty","把红色木块放到 C 区","clarify"),
    ("already_C","C","把红块放到C","already_done"),
    ("forward_50","A","向前移动五厘米","success"),
    ("wrist_down_5","A","手腕向下转五度","success"),
    ("ambiguous","A","向下几度","clarify"),
    ("too_far","A","向前移动五十厘米","rejected"),
    ("outside","outside","把红块放到C","clarify"),
    ("miss","A","把红块放到C","unverified"),
    ("A_to_B_sequence","A","把前面的木块夹起来，再放到右边的工位","success"),
    ("B_to_A","B","把B区的红色积木送到A区","success"),
    ("C_to_A","C","把木块放到左边的工位","success"),
    ("C_to_B","C","把桌上的木块从C挪到B区","success"),
    ("pick_hold","B","向前抓取木块","success"),
    ("metric_sequence","A","先向左挪12毫米，再把手腕向下转3度","success"),
    ("bad_second","A","向左移动一厘米，然后向前移动六十厘米","rejected"),
    ("typed_homophone","A","把A区的红快放到C","success"),
    ("typed_source_conflict","C","把A区的红快放到C","clarify"),
]


def block_state(sim):
    m, d = sim.model, sim.data
    cube = m.geom("red_block").id
    contacts = set()
    for c in d.contact:
        if cube in (c.geom1,c.geom2):
            other = c.geom1 if c.geom2==cube else c.geom2
            contacts.add(m.geom(other).name)
    xyz = d.geom_xpos[cube]
    # Project all box half-extents, including yaw/tilt, before evaluating containment.
    extent = np.abs(d.geom_xmat[cube].reshape(3,3)) @ np.full(3,HALF_BLOCK)
    inside = next((name for name, xy in STATIONS.items()
                   if np.all(np.abs(xyz[:2]-xy)+extent[:2] <= HALF_ZONE)),None)
    return {"xyz_m":xyz.round(6).tolist(), "contacts":sorted(contacts),
            "inside_C":inside=="C","inside":inside,"time_s":round(float(d.time),4)}


# Expected detections are independent of the renderer. Each named destination must be
# occupied by the whole free block and have only desk contact on all three verification frames.
STRESS_CASES = {
    "clutter": ("success",["B"],["C","A"]),
    "two_red": ("clarify",["A","B"],[]),
    "cover_source": ("clarify",[],[]),
    "cover_goal": ("success",["B"],["C"]),
    "cover_both": ("unverified",["B"],["C"]),
    "dim": ("success",["B"],["A"]),
    "rotated": ("success",["A"],["B"]),
    "edge_grasp": ("unverified",["A"],[]),
    "reflective_band": ("clarify",["C","C"],[]),
}


def stress_cases(sim,out):
    records, cards = [], []

    def card(image,title):
        thumb = cv2.resize(image,(384,240),interpolation=cv2.INTER_AREA)
        thumb = cv2.copyMakeBorder(thumb,0,30,0,0,cv2.BORDER_CONSTANT,value=(34,48,43))
        cv2.putText(thumb,title,(8,260),cv2.FONT_HERSHEY_SIMPLEX,.52,(245,248,245),1,cv2.LINE_AA)
        return thumb

    extras = []
    for name,(expected,initial,targets) in STRESS_CASES.items():
        sim.reset(name)
        start_joints = sim.joints_now()
        observations = []
        def observe(phase,scene,frame):
            truth = block_state(sim)
            observations.append({"phase":phase,"camera_slots":[obj["slot"] for obj in scene["objects"]],
                                 "gripper_holding":scene["holding"],"truth":truth})
            if phase=="before":
                cards.append(card(frame,name+" / before"))
            if phase=="after" and name in ("clutter","cover_goal"):
                extras.append(card(frame,name+" / after"))

        result = execute(sim,LAYOUTS[name].command,observer=observe)
        checks = {"status":result["status"]==expected, "initial_count_and_slots":sorted(
                  observations[0]["camera_slots"],key=str)==sorted(initial,key=str)}
        lifts = [o for o in observations if o["phase"]=="lift"]
        verifies = [o for o in observations if o["phase"].startswith("verify_")]
        grounded_lifts = [set(o["truth"]["contacts"])=={"fixed_pad","moving_pad"} and
                          o["truth"]["xyz_m"][2]-observations[0]["truth"]["xyz_m"][2]>.020 for o in lifts]
        if targets:
            checks["each_lift_clears_table_and_both_pads_touch"] = grounded_lifts==[True]*len(targets)
            checks["verification_phases_in_order"] = [o["phase"] for o in verifies]==["verify_1","verify_2","verify_3"]*len(targets)
            checks["three_stable_physical_landings_per_transfer"] = len(verifies)==3*len(targets) and all(
                o["truth"]["inside"]==targets[i//3] and o["truth"]["contacts"]==["desk"] for i,o in enumerate(verifies))
            checks["camera_agrees_or_reports_blocked_view"] = len(verifies)==3*len(targets) and all(
                o["camera_slots"]==([] if name in ("cover_goal","cover_both") else [targets[i//3]]) for i,o in enumerate(verifies))
            expected_view = {"cover_goal":"wrist","cover_both":"unconfirmed"}.get(name,"overhead")
            checks["three_verifications_from_expected_camera"] = [
                view for step in result["steps"] for view in step["measurement"].get("drop_views", [])
            ] == [expected_view]*(3*len(targets))
        elif expected=="clarify":
            checks["no_joint_motion"] = np.array_equal(start_joints,sim.joints_now())
            checks["no_actions"] = not result["plan"]
        elif name=="edge_grasp":
            checks["first_move_measured_then_failed_pick"] = [(s["kind"],s["status"]) for s in result["steps"]]==[
                ("move","success"),("transfer","unverified")] and result["steps"][0]["measurement"]["displacement_mm"][1]>9
            checks["stopped_before_approach_and_block_stayed_on_desk"] = (
                not any(o["phase"] in ("pregrasp","descend","lift") for o in observations)
                and observations[-1]["truth"]["contacts"] == ["desk"])
            checks["no_release_or_final_wrist_step"] = not verifies and result["plan"]==["move_delta(y,10 mm)"]
        records.append({"case":name,"expected":expected,"passed":all(checks.values()),"checks":checks,
                        "observations":observations,"receipt":result})
    while len(cards)%4:
        cards.append(np.zeros_like(cards[0]))
    gallery = np.vstack([np.hstack(cards[i:i+4]) for i in range(0,len(cards),4)])
    if not cv2.imwrite(str(out/"scene_gallery.jpg"),gallery):
        raise OSError("Cannot save scene gallery")
    if not cv2.imwrite(str(out/"relay_and_blind_goal_after.jpg"),np.hstack(extras)):
        raise OSError("Cannot save final camera observations")
    return records


def evaluate(out):
    out = Path(out)
    out.mkdir(parents=True,exist_ok=True)
    sim = Simulator()
    records = []
    try:
        for name,layout,text,expected in CASES:
            sim.reset(layout)
            if name=="miss":
                # Deliberate lateral placement error within A. Used only to initialize this fixture.
                import mujoco
                addr = sim.model.joint("block_free").qposadr[0]
                sim.data.qpos[addr+1] -= .019
                mujoco.mj_forward(sim.model,sim.data)
            snapshots = {}
            def record(phase, scene, frame):
                snapshots[phase] = block_state(sim)
                if phase in ("before","lift","after") and name in ("pick_A","pick_B","miss"):
                    cv2.imwrite(str(out/f"{name}_{phase}.jpg"),frame)
            q0 = sim.joints_now()
            result = execute(sim,text,observer=record)
            passed = result["status"] == expected
            physics = None
            if name=="miss":
                passed &= np.array_equal(q0,sim.joints_now()) and snapshots["after"]["contacts"]==["desk"]
                passed &= "pregrasp" not in snapshots and not result["plan"]
            if "lift" in snapshots:
                lift = snapshots["lift"]
                raised = (lift["xyz_m"][2]-snapshots["before"]["xyz_m"][2])*1000
                destination = {"A_to_B_sequence":"B","B_to_A":"A","C_to_A":"A","C_to_B":"B"}.get(name,"C")
                released_stable = all(f"verify_{i}" in snapshots and snapshots[f"verify_{i}"]["inside"]==destination and
                                      snapshots[f"verify_{i}"]["contacts"]==["desk"] for i in (1,2,3))
                held = {"fixed_pad","moving_pad"}.issubset(lift["contacts"]) and "desk" not in lift["contacts"]
                physics = {"lift_mm":round(raised,2),"two_pad_grasp":held}
                if name=="pick_hold":
                    physically_moved = raised>20 and held and set(snapshots["after"]["contacts"])=={"fixed_pad","moving_pad"}
                else:
                    physics.update(target=destination,released_stable_in_target=released_stable)
                    physically_moved = raised>20 and held and released_stable
                passed &= physically_moved == (expected=="success")
            if name=="metric_sequence":
                first,second = (s["measurement"] for s in result["steps"])
                passed &= first["displacement_mm"][1]>11 and first["error_mm"]<1 and abs(second["actual_deg"]-3)<.2
            if expected in ("clarify","rejected","already_done"):
                passed &= bool(np.max(np.abs(sim.joints_now()-q0)) < 1e-7)
            records.append({"case":name,"expected":expected,"passed":bool(passed),"physics":physics,
                            "observations":snapshots,"receipt":result})
        records.extend(stress_cases(sim,out))
    finally:
        sim.close()
    data={"backend":"MuJoCo 4.0.0 / CPU rules / OpenCV","ai_calls":0,"hardware_connected":False,
          "passed":sum(row["passed"] for row in records),"total":len(records),"cases":records}
    (out/"report.json").write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
    for row in records:
        print(f"{row['case']:15} {row['receipt']['status']:14} {'PASS' if row['passed'] else 'FAIL'}")
    print(f"{data['passed']} / {data['total']} | {out/'report.json'}")
    return data
