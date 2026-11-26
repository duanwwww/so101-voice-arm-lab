"""Repeatable ablations and geometry/accuracy measurements on the offline controller."""
import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

from .evaluate import block_state
from .flow import execute
from .motion_guard import MotionGuard
from .simulator import Simulator
from .wrist_view import observe_wrist


def positions(sim):
    rows = []
    # Freeze both grids before running the correction. The second grid uses distinct
    # diagonal displacements and yaws, including poses outside the admission radius.
    setups = [("lateral",0,dy,yaw) for dy in (-12,-8,-4,0,4,8,12) for yaw in (0,12)]
    setups += [("diagonal",dx,dy,yaw) for dx,dy,yaw in (
        (-3,4,-10),(4,-3,17),(-4,-4,-15),(5,2,20),(6,-2,9),(8,8,10),(-8,7,-10))]
    for source, target in (("A","C"), ("B","A"), ("C","B")):
        for group,dx,dy,yaw in setups:
            for alignment in (False,True):
                sim.reset(source)
                sim.visual_alignment = alignment
                addr = sim.model.joint("block_free").qposadr[0]
                sim.data.qpos[addr:addr+2] += np.array((dx,dy)) / 1000
                angle = np.deg2rad(yaw)/2
                sim.data.qpos[addr+3:addr+7] = [np.cos(angle),0,0,np.sin(angle)]
                mujoco.mj_forward(sim.model, sim.data)
                sim.settle(.1)
                rgb = sim.camera()[0]["objects"]
                measured = rgb[0]["xy_mm"] if len(rgb)==1 else None
                q0 = sim.joints_now()
                r = execute(sim, f"把木块放到 {target} 区")
                truth = block_state(sim)
                landed = truth["inside"]==target and truth["contacts"]==["desk"]
                rows.append(dict(group=group,alignment=alignment,source=source,target=target,
                                 offset_x_mm=dx,offset_y_mm=dy,yaw_deg=yaw,
                                 pixel_xy_mm=measured,approached="pregrasp" in r["phases"],
                                 stayed_at_start=np.array_equal(q0,sim.joints_now()),
                                 status=r["status"],truth_landed=landed,reason=r["message"]))
    sim.visual_alignment = True
    return rows


def numeric(sim):
    rows = []
    for setup in (None, "向左移动一厘米", "向前移动一厘米"):
        for text in ("向前移动十毫米","向后移动十毫米","向左移动十毫米","向右移动十毫米",
                     "向上移动五毫米","向下移动五毫米","手腕向上转三度","手腕向下转三度"):
            sim.reset("A")
            if setup and execute(sim, setup)["status"] != "success":
                raise RuntimeError("Starting configuration could not be reached")
            r = execute(sim,text)
            rows.append(dict(setup=setup,command=text,status=r["status"],measurement=r["measurement"],
                             reason=r["message"]))
    return rows


def cameras(sim, out):
    rows = []
    for layout in ("B","cover_goal","cover_both"):
        for enabled in (False,True):
            sim.reset(layout)
            sim.wrist_enabled = enabled
            r = execute(sim,"把木块放到 C 区")
            truth = block_state(sim)
            rows.append(dict(layout=layout,wrist=enabled,status=r["status"],
                             views=r["measurement"].get("drop_views",[]),
                             truth_landed=truth["inside"]=="C" and truth["contacts"]==["desk"]))
            if layout=="cover_goal" and enabled:
                cv2.imwrite(str(out/"wrist_after.jpg"), observe_wrist(sim)[1])
                cv2.imwrite(str(out/"overhead_after.jpg"), sim.camera()[1])
    sim.wrist_enabled = True
    return rows


def obstacle(sim):
    default = sim.guard
    path = Path(__file__).resolve().parents[1]/"configs/keepouts.example.json"
    rows = []
    for enabled in (False,True):
        sim.reset("A")
        sim.guard = MotionGuard.from_config(sim.model,path) if enabled else default
        start = sim.joints_now()
        r = execute(sim,"把木块放到 B 区")
        rows.append(dict(registered_box=enabled,status=r["status"],reason=r["message"],
                         no_joint_change=np.array_equal(start,sim.joints_now())))
    sim.guard = default
    return rows


def summarize(values):
    return dict(n=len(values),median=round(float(np.median(values)),3),
                max=round(max(values),3)) if values else dict(n=0)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out",type=Path,default=Path("outputs/control_benchmark"))
    out=cli.parse_args().out
    out.mkdir(exist_ok=True,parents=True)
    sim=Simulator()
    try:
        placements=positions(sim)
        moves=numeric(sim)
        views=cameras(sim,out)
        boxes=obstacle(sim)
    finally:
        sim.close()
    grouped={}
    for group in ("lateral","diagonal"):
        for align in (False,True):
            rows=[r for r in placements if r["alignment"]==align and r["group"]==group]
            grouped[f"{group}_{'aligned' if align else 'taught'}"]=dict(n=len(rows),
                approached=sum(r["approached"] for r in rows),landed=sum(r["truth_landed"] for r in rows),
                no_motion=sum(r["stayed_at_start"] for r in rows),
                false_success=sum(r["status"]=="success" and not r["truth_landed"] for r in rows))
    summary=dict(placement_runs=len(placements),placement_groups=grouped,
                 moves=len(moves),move_success=sum(r["status"]=="success" for r in moves),
                 linear_mm=summarize([r["measurement"]["error_mm"] for r in moves if "error_mm" in r["measurement"]]),
                 wrist_deg=summarize([r["measurement"]["error_deg"] for r in moves if "error_deg" in r["measurement"]]))
    result=dict(backend="MuJoCo",hardware_connected=False,model_calls=0,
                summary=summary,placements=placements,motions=moves,cameras=views,obstacles=boxes)
    (out/"control_benchmark.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False))
    for row in views+boxes:
        print(row)


if __name__=="__main__":
    main()
