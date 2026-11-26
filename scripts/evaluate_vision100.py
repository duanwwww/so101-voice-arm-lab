"""Evaluate unchanged production RGB perception on real, external simulated and calibrated local views."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from hashlib import sha1, sha256
import json
from pathlib import Path
from unittest.mock import patch

import cv2
import httpx
import numpy as np

from arm_lab.real_data import VISION_REPO, VISION_REV, asset, iou
from arm_lab.evaluate import block_state
from arm_lab.gripper import PICK_RADIUS_MM
from arm_lab.scene import BLUE, RED, STATIONS, LAYOUTS, Layout
from arm_lab.simulator import Simulator
from arm_lab.vision import detect
from arm_lab.wrist_view import observe_wrist
from scripts.evaluate_real_candidates import BoxFilter, split_of


ROOT = Path(__file__).resolve().parents[1]
TREE = f"https://huggingface.co/api/datasets/{VISION_REPO}/tree/{VISION_REV}"


def publisher_index(cache, client):
    path = cache / VISION_REPO / VISION_REV / "publisher_vision_index.json"
    if not path.exists():
        if client is None:
            raise FileNotFoundError("Run with --download once to cache the publisher file index.")
        entries = []
        for directory in ("real", "real/images", "sim", "sim/images"):
            response = client.get(f"{TREE}/{directory}", params={"limit": 500})
            response.raise_for_status()
            if "next" in response.links:
                raise ValueError("Unexpected pagination at pinned revision")
            entries.extend(e for e in response.json() if e["type"] == "file")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({e["path"]:e.get("lfs",{}).get("oid",e["oid"]) for e in entries}),encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))


def verified(client, cache, index, name):
    contents = asset(client, cache, VISION_REPO, VISION_REV, name).read_bytes()
    oid = index[name]
    hashed = sha256(contents).hexdigest() if len(oid)==64 else sha1(
        f"blob {len(contents)}\0".encode() + contents).hexdigest()
    if hashed != oid:
        raise ValueError(f"Publisher digest mismatch for {name}")
    return contents


def polygon_box(poly):
    points = np.asarray(poly,dtype=float)
    lo, hi = points.min(axis=0), points.max(axis=0)
    # The publisher supplies raster contours in pixel coordinates, endpoints inclusive.
    return [*lo.tolist(), *(hi-lo+1).tolist()]


def assess(boxes, truth):
    """Pinned subsets have at most one labeled red cube per image; count extra boxes as FP."""
    if len(truth)>1:
        raise ValueError("Scorer requires at most one red-cube annotation")
    overlaps = [iou(box,truth[0]) for box in boxes] if truth else []
    hit = bool(overlaps and max(overlaps)>=.5)
    unique = len(boxes)==1
    return {"tp":int(hit), "fp":len(boxes)-int(hit), "fn":len(truth)-int(hit),
            "best_iou":round(max(overlaps),5) if overlaps else None,
            "unique_correct":unique and hit,"unique_wrong":unique and not hit}


def stats(rows):
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp","fp","fn"))
    def ratio(n,d):
        return round(n/d,4) if d else None
    return dict(frames=len(rows), tp=tp, fp=fp, fn=fn, precision=ratio(tp,tp+fp),
                recall=ratio(tp,tp+fn), f1=ratio(2*tp,2*tp+fp+fn),
                empty=sum(not r["boxes"] for r in rows),
                multiple=sum(len(r["boxes"])>1 for r in rows),
                unique_correct=sum(r["unique_correct"] for r in rows),
                unique_wrong=sum(r["unique_wrong"] for r in rows))


def score_subset(client,cache,index,subset):
    classes = verified(client,cache,index,f"{subset}/classes.txt").decode().splitlines()
    red = classes.index("red cube")
    metadata = verified(client,cache,index,f"{subset}/images/metadata.jsonl")
    labels = [json.loads(line) for line in metadata.splitlines() if line.strip()]
    if len(labels) != {"real":100,"sim":264}[subset] or len({a["file_name"] for a in labels})!=len(labels):
        raise ValueError("Unexpected sample set in the pinned manifest")

    def one(label):
        filename = label["file_name"]
        if Path(filename).name!=filename:
            raise ValueError("Invalid dataset filename")
        blob = verified(client,cache,index,f"{subset}/images/{filename}")
        bgr = cv2.imdecode(np.frombuffer(blob,dtype=np.uint8),cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape[:2] != (label["height"],label["width"]):
            raise ValueError(f"Invalid shape for {filename}")
        boxes = [o["box"] for o in detect(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),{})["objects"]]
        objects = label["objects"]
        geometry = objects["bbox"] if subset=="real" else objects["polygons"]
        all_boxes = [g if subset=="real" else polygon_box(g) for g in geometry]
        labeled = list(zip(objects["categories"],all_boxes,strict=True))
        truth = [box for category,box in labeled if category==red]
        score = assess(boxes,truth)
        wrong_overlaps = []
        if score["unique_wrong"]:
            wrong_overlaps = [classes[category] for category,box in labeled if category!=red and iou(boxes[0],box)>=.5]
        return {"file":filename, "sha256":sha256(blob).hexdigest(), "boxes":boxes, "truth":truth,
                "other_annotations_matching_wrong_unique":wrong_overlaps, **score}

    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(one,labels))
    by_group = {k:stats([r for r in rows if bool(r["truth"])==flag]) for k,flag in
                (("with_red_label",True),("without_red_label",False))}
    if subset=="real":
        by_group.update({s:stats([r for r in rows if split_of(r["file"])==s]) for s in ("development","gap","holdout")})
    result = {"metadata_sha256":sha256(metadata).hexdigest(),"publisher_hashes_verified":True,
              "summary":stats(rows),"groups":by_group,"rows":rows}
    if subset=="sim":
        prior = json.loads((ROOT/"results/safety/vision_ablation.json").read_text(encoding="utf-8"))["selected_rule"]
        rule = BoxFilter(**prior)
        filtered=[]
        for row,label in zip(rows,labels,strict=True):
            kept = [b for b in row["boxes"] if rule.accepts(b,label["width"]*label["height"])]
            filtered.append({"boxes":kept,"truth":row["truth"],**assess(kept,row["truth"])})
        result["previously_selected_filter"]={"rule":prior, "summary":stats(filtered),
            "without_red_label":stats([r for r in filtered if not r["truth"]])}
    return result


CONDITIONS = ("normal","dim","dark","boundary","offset","blue_distractor","two_red",
              "camera_cover","reflective_band","off_camera")
SCENE_SEED = 20261126


def new_layouts():
    rng = np.random.default_rng(SCENE_SEED)
    stations = list(STATIONS)
    for g, condition in enumerate(CONDITIONS):
        for i in range(10):
            slot = stations[(i+g)%3]
            other = stations[(i+g+1)%3]
            angle = rng.uniform(0,2*np.pi)
            low,high = {"offset":(6.5,10), "boundary":(16,29)}.get(condition,(0.2,6))
            offset = rng.uniform(low,high)*np.array([np.cos(angle),np.sin(angle)])/1000
            xy = tuple(np.array(STATIONS[slot])+offset)
            settings = {}
            if condition in ("dim","dark"):
                settings["gain"] = .45 if condition=="dim" else .18
            elif condition in ("blue_distractor","two_red"):
                settings["props"] = ((*STATIONS[other],RED if condition=="two_red" else BLUE),)
            elif condition=="camera_cover":
                settings["cover"] = slot
            elif condition=="reflective_band":
                settings["reflective_band"] = True
            elif condition=="off_camera":
                xy = tuple(np.array([.62,.45])+offset)
            layout = Layout(condition,xy,"",yaw=rng.uniform(-36,36),**settings)
            yield f"{condition}-{i:02d}",condition,layout


def camera_result(objects, expected_slot, truth_xy, is_single):
    slot = objects[0]["slot"] if len(objects)==1 else None
    reply = {"objects":objects, "reported_slot":slot,
             "correct_slot":slot is not None and slot==expected_slot,
             "false_slot":slot is not None and slot!=expected_slot,
             "withheld":slot is None}
    if slot == expected_slot and slot is not None and is_single and "xy_mm" in objects[0]:
        reply["xy_error_mm"] = round(float(np.linalg.norm(np.array(objects[0]["xy_mm"])-truth_xy)),4)
    if slot is not None and "xy_mm" in objects[0]:
        reply["pick_radius_accepts"] = bool(np.linalg.norm(np.array(objects[0]["xy_mm"])-1000*np.array(STATIONS[slot]))<=PICK_RADIUS_MM)
    return reply


def summarize_local(rows):
    summary = {"trials":len(rows),"physically_unique_in_slot":sum(r["expected_slot"] is not None for r in rows)}
    for view in ("overhead","wrist"):
        summaries = {k:sum(r[view][k] for r in rows) for k in ("correct_slot","false_slot","withheld")}
        errors = [r[view]["xy_error_mm"] for r in rows if "xy_error_mm" in r[view]]
        if errors:
            summaries.update(xy_n=len(errors), xy_median_mm=round(float(np.median(errors)),3),
                             xy_p95_mm=round(float(np.percentile(errors,95)),3),xy_max_mm=max(errors))
        summary[view] = summaries
    summary["radius_gate"] = {
        "physically_eligible":sum(r["radius_eligible"] for r in rows),
        "accept_correct":sum(bool(r["overhead"].get("pick_radius_accepts")) and r["radius_eligible"]
                             and r["overhead"]["correct_slot"] for r in rows),
        "accept_wrong":sum(bool(r["overhead"].get("pick_radius_accepts")) and
                           not (r["radius_eligible"] and r["overhead"]["correct_slot"]) for r in rows),
    }
    return summary


def score_local():
    rows=[]
    sim=Simulator("A")
    try:
        for key,condition,layout in new_layouts():
            with patch.dict(LAYOUTS,{key:layout}):
                sim.reset(key)
                scene,_ = sim.camera()
                wrist,_ = observe_wrist(sim)
                # World-space geometry enters evaluation only after both camera outputs exist.
                truth = block_state(sim)
                true_xy = np.array(truth["xyz_m"][:2])*1000
                unique = condition!="two_red"
                expected_slot = truth["inside"] if unique else None
                eligible = expected_slot is not None and bool(np.linalg.norm(
                    true_xy-1000*np.array(STATIONS[expected_slot]))<=PICK_RADIUS_MM)
                rows.append({"id":key,"condition":condition,"target_xyz_m":truth["xyz_m"],
                             "geometric_slot":truth["inside"],"expected_slot":expected_slot,
                             "yaw_deg":round(layout.yaw,4),"radius_eligible":eligible,
                             "overhead":camera_result(scene["objects"],expected_slot,true_xy,unique),
                             "wrist":camera_result(wrist,expected_slot,true_xy,unique)})
    finally:
        sim.close()
    thresholds = []
    for mm in (8,7,6):
        accepted = []
        for row in rows:
            overhead = row["overhead"]
            slot = overhead["reported_slot"]
            if slot and np.linalg.norm(np.array(overhead["objects"][0]["xy_mm"])-1000*np.array(STATIONS[slot]))<=mm:
                accepted.append(row)
        correct = sum(r["radius_eligible"] and r["overhead"]["correct_slot"] for r in accepted)
        thresholds.append({"detected_radius_mm":mm,"truth_radius_mm":PICK_RADIUS_MM,
                           "correct_accept":correct,"false_accept":len(accepted)-correct,
                           "false_ids":[r["id"] for r in accepted if not(r["radius_eligible"] and r["overhead"]["correct_slot"])]})
    return {"seed":SCENE_SEED,"summary":summarize_local(rows), "posthoc_radius_sensitivity":thresholds,
            "conditions":{name:summarize_local([r for r in rows if r["condition"]==name]) for name in CONDITIONS},
            "rows":rows}


def radius_followup():
    """New near-boundary scenes following the initial 100. Compare only the frozen 7/8 mm gates."""
    seed = 20261127
    rng = np.random.default_rng(seed)
    rows = []
    gates = (8,7)
    sim = Simulator("A")
    try:
        for station, origin in STATIONS.items():
            center = 1000*np.array(origin)
            for distance in (6.5,7.3,7.8,8.2,8.7,9.5):
                for sector in range(8):
                    angle = 2*np.pi*(sector+rng.uniform(.1,.9))/8
                    xy = np.array(origin) + distance/1000*np.array([np.cos(angle),np.sin(angle)])
                    pose = Layout(station,tuple(xy),"",yaw=rng.uniform(-40,40))
                    with patch.dict(LAYOUTS,{"radius_probe":pose}):
                        sim.reset("radius_probe")
                        scene,_ = sim.camera()
                    objects = scene["objects"]
                    truth = block_state(sim)
                    real_radius = np.linalg.norm(1000*np.array(truth["xyz_m"][:2])-center)
                    measured = None
                    if len(objects)==1 and objects[0]["slot"]==station:
                        measured = np.linalg.norm(np.array(objects[0]["xy_mm"])-center)
                    rows.append({"station":station,"sector":sector,"set_radius_mm":distance,
                                 "actual_radius_mm":round(float(real_radius),5),
                                 "detected_radius_mm":round(float(measured),5) if measured is not None else None,
                                 "eligible":truth["inside"]==station and bool(real_radius<=PICK_RADIUS_MM),
                                 "accepted":[g for g in gates if measured is not None and measured<=g]})
    finally:
        sim.close()
    results = {}
    for g in gates:
        accepted = [r for r in rows if g in r["accepted"]]
        results[str(g)] = {"correct_accept":sum(r["eligible"] for r in accepted),
                           "false_accept":sum(not r["eligible"] for r in accepted)}
    return {"seed":seed,"physical_threshold_mm":PICK_RADIUS_MM,"frames":len(rows),
            "eligible":sum(r["eligible"] for r in rows),"thresholds":results,"rows":rows}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",type=Path,default=ROOT/"outputs"/"public-data")
    parser.add_argument("--download",action="store_true",help="Fetch any missing data files from the pinned publisher revision.")
    parser.add_argument("--out",type=Path,default=ROOT/"results"/"safety"/"vision100.json")
    parser.add_argument("--examples",type=Path,help="Draw a PNG from the saved --out report and cached original frames only.")
    args=parser.parse_args()
    if args.examples:
        if args.examples.suffix.lower()!=".png" or args.examples.resolve()==args.out.resolve():
            parser.error("--examples must name a separate PNG file")
        draw_examples(args.cache,json.loads(args.out.read_text(encoding="utf-8")),args.examples)
        return
    with (httpx.Client(follow_redirects=True,timeout=45) if args.download else nullcontext(None)) as client:
        index=publisher_index(args.cache,client)
        public={subset:score_subset(client,args.cache,index,subset) for subset in ("real","sim")}
    local=score_local()
    report={"dataset":VISION_REPO,"revision":VISION_REV,"iou_threshold":0.5,"hardware_connected":False,
            "model_calls":0,"opencv":cv2.__version__,"public":public,"local":local,
            "radius_followup":radius_followup()}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n", encoding="utf-8")
    print({k:v["summary"] for k,v in public.items()})
    print("local",local["summary"])


def draw_examples(cache,report,destination):
    """Render selected verified original images with frozen predictions; no detector rerun."""
    from io import BytesIO
    from PIL import Image, ImageDraw, ImageFont
    if report["revision"]!=VISION_REV or report["dataset"]!=VISION_REPO:
        raise ValueError("Example images and measured dataset revision must agree")
    samples = [
        ("real","image_20260515_223222_446.jpg","实拍 · 唯一候选命中红块"),
        ("real","image_20260515_224838_915.jpg","实拍 · 一帧出现 15 个候选"),
        ("real","image_20260515_223457_927.jpg","实拍 · 唯一候选没有框全红块"),
        ("sim","d1_base_camera_rgb_20260414_102823.png","外部仿真 · 苹果被识别成红块"),
    ]
    font_dir = Path("C:/Windows/Fonts")
    def font(size,bold=False):
        return ImageFont.truetype(str(font_dir/("msyhbd.ttc" if bold else "msyh.ttc")),size)
    canvas=Image.new("RGB",(1048,980),"#eff2f4")
    draw=ImageDraw.Draw(canvas)
    ink="#192a34"; subtle="#445560"; green="#47f49c"; magenta="#ff46cc"
    draw.text((24,12),"OpenCV 红块检测与发布者标注",font=font(32,True),fill=ink)
    draw.line((27,73,65,73),fill="#07824e",width=5)
    draw.text((75,58),"实线：发布者红块框",font=font(20),fill=ink)
    for x in (332,344,356):
        draw.line((x,73,x+7,73),fill="#bc138f",width=4)
    draw.text((378,58),"虚线：OpenCV 预测框",font=font(20),fill=ink)
    draw.text((765,58),"匹配阈值  IoU ≥ 0.50",font=font(20),fill=ink)
    for i,(split,name,title) in enumerate(samples):
        row = next(r for r in report["public"][split]["rows"] if r["file"]==name)
        blob = (cache/VISION_REPO/VISION_REV/split/"images"/name).read_bytes()
        if sha256(blob).hexdigest()!=row["sha256"]:
            raise ValueError(f"File differs from the scored frame: {name}")
        photo = Image.open(BytesIO(blob)).convert("RGB")
        size=photo.size
        photo.thumbnail((475,302),Image.Resampling.LANCZOS)
        x=24+(i%2)*508; y=102+(i//2)*416
        draw.rectangle((x,y,x+491,y+404),fill="white")
        draw.text((x+12,y+8),title,fill=ink,font=font(21,True))
        left=x+(492-photo.width)//2; top=y+45
        canvas.paste(photo,(left,top))
        for boxes,color,dashed in ((row["truth"],green,False),(row["boxes"],magenta,True)):
            for bx,by,bw,bh in boxes:
                x1,x2 = [left+round(z*photo.width/size[0]) for z in (bx,bx+bw-1)]
                y1,y2 = [top+round(z*photo.height/size[1]) for z in (by,by+bh-1)]
                if not dashed:
                    draw.rectangle((x1,y1,x2,y2),outline="#123b2f",width=4)
                    draw.rectangle((x1,y1,x2,y2),outline=color,width=2)
                    continue
                for a in range(x1,x2+1,8):
                    for b in (y1,y2):
                        draw.line((a,b,min(a+4,x2),b),fill=color,width=2)
                for b in range(y1,y2+1,8):
                    for a in (x1,x2):
                        draw.line((a,b,a,min(b+4,y2)),fill=color,width=2)
        if row["truth"]:
            detail=f'标注 {len(row["truth"])}    预测 {len(row["boxes"])}    最佳 IoU {row["best_iou"]:.3f}'
        else:
            detail="标注 0    预测 1    预测框与苹果标注匹配"
        draw.text((x+12,y+350),detail,font=font(18),fill=ink)
        draw.text((x+12,y+379),name,font=font(14),fill=subtle)
    draw.text((24,945),"图像与标注来自 VigiClaw；此图叠加本项目检测框，样本编号对应逐帧报告。",font=font(17),fill=subtle)
    destination.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(destination)


if __name__=="__main__":
    main()
