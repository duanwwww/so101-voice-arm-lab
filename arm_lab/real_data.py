"""Evaluate the frozen color detector and audit gripper traces on two pinned public datasets."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from hashlib import sha256
import json
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
import time

import cv2
import httpx
import numpy as np

from .vision import detect


VISION_REPO = "pengxianglu/vigiclaw-yolo-dataset"
VISION_REV = "e2e9acfbfdf827776c06673ee4103c970ac0d29b"
ARM_REPO = "aswinkumar99/LeRobot-SO101-Pick-Place"
ARM_REV = "6f1ceec7b23629d9fcdaf819a7d312f019172977"
ARM_TASK = "task1-single-sponge-no-distractors"


def asset(client, cache, repo, revision, name):
    path = cache / repo / revision / name
    if path.is_file():
        return path
    if client is None:
        raise FileNotFoundError(f"离线缓存缺少 {repo}/{name}，先运行一次联网评测。")
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}"
    for attempt in range(4):
        try:
            response = client.get(url)
            response.raise_for_status()
            break
        except httpx.HTTPError as exc:
            if isinstance(exc,httpx.HTTPStatusError) and exc.response.status_code not in (408,429,500,502,503,504):
                raise RuntimeError(f"读取 {repo}/{name} 返回 HTTP {exc.response.status_code}") from None
            if attempt==3:
                raise RuntimeError(f"下载 {repo}/{name} 失败。已缓存的文件可在重跑时复用。") from None
            time.sleep(2**attempt)
    with NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        temporary = Path(tmp.name)
        tmp.write(response.content)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def iou(a,b):
    x,y,w,h = a
    u,v,p,q = b
    overlap = max(0,min(x+w,u+p)-max(x,u))*max(0,min(y+h,v+q)-max(y,v))
    return overlap / (w*h+p*q-overlap)


def ratio(n,d):
    return round(n/d,4) if d else None


def median(values):
    return round(float(np.median(values)),3) if len(values) else None


def vision_score(client,cache):
    def get(name):
        return asset(client,cache,VISION_REPO,VISION_REV,"real/"+name)
    classes = get("classes.txt").read_text(encoding="utf-8").splitlines()
    red = classes.index("red cube")
    metadata = get("images/metadata.jsonl").read_bytes()
    rows = [json.loads(line) for line in metadata.splitlines() if line.strip()]
    if len(rows)!=100 or len({row["file_name"] for row in rows})!=len(rows):
        raise ValueError("Pinned real image manifest is incomplete")

    def score(row):
        name = row["file_name"]
        if not re.fullmatch(r"[\w-]+\.jpg",name):
            raise ValueError("Unexpected image filename")
        labels = row["objects"]
        targets = [box for category,box in zip(labels["categories"],labels["bbox"],strict=True) if category==red]
        if len(targets)!=1:
            raise ValueError("This fixed evaluation expects one labeled red cube per frame")
        truth = targets[0]
        data = get("images/"+name).read_bytes()
        frame = cv2.imdecode(np.frombuffer(data,dtype=np.uint8),cv2.IMREAD_COLOR)
        if frame is None or frame.shape[:2]!=(row["height"],row["width"]):
            raise ValueError(f"Cannot decode {name} at its annotated dimensions")
        # No crops, per-image ROI or calibration from the simulation enter the real-image detector.
        found = detect(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),{})["objects"]
        overlaps = [iou(obj["box"],truth) for obj in found]
        best = int(np.argmax(overlaps)) if found else None
        best_iou = overlaps[best] if found else 0
        hit = int(best_iou>=.5)
        center = np.array(truth[:2])+np.array(truth[2:])/2
        return {"file":name,"sha256":sha256(data).hexdigest(),"candidates":len(found),
                "tp":hit,"fp":len(found)-hit,"fn":1-hit,"best_iou":round(best_iou,4),
                "matched_center_error_px":round(float(np.linalg.norm(found[best]["pixel"]-center)),3) if hit else None}
    with ThreadPoolExecutor(max_workers=4) as pool:
        scores = list(pool.map(score,rows))
    tp,fp,fn = (sum(row[k] for row in scores) for k in ("tp","fp","fn"))
    counts = [row["candidates"] for row in scores]
    return {"dataset":VISION_REPO,"revision":VISION_REV,"subset":"real / train (all images)",
            "original_test_split":False,"iou_threshold":.5,"manifest_sha256":sha256(metadata).hexdigest(),
            "frames":len(scores),"labeled_red_cubes":tp+fn,"tp":tp,"fp":fp,"fn":fn,
            "precision":ratio(tp,tp+fp),"recall":ratio(tp,tp+fn),"f1":ratio(2*tp,2*tp+fp+fn),
            "zero_candidates":counts.count(0),"one_candidate":counts.count(1),
            "multiple_candidates":sum(n>1 for n in counts),"unique_and_correct":sum(row["candidates"]==1 and row["tp"]==1 for row in scores),
            "median_best_iou_all_frames":median([row["best_iou"] for row in scores]),
            "median_center_error_matched_px":median([row["matched_center_error_px"] for row in scores if row["tp"]]),
            "per_image":scores}


def first_cycle(values,lower,upper):
    """First sustained high, low, high pattern. Return start frames of all three runs."""
    states = np.where(values>=upper,1,np.where(values<=lower,-1,0))
    targets = (1,-1,1)
    starts, run = [], 0
    for frame,state in enumerate(states):
        run = run+1 if state==targets[len(starts)] else 0
        if run==3:
            starts.append(frame-2)
            run = 0
            if len(starts)==3:
                return starts
    return None


def gripper_score(client,cache):
    import pyarrow.parquet as pq
    splits = {}
    for split in ("fixed-locations","random-locations"):
        def get(name):
            return asset(client,cache,ARM_REPO,ARM_REV,f"{ARM_TASK}/{split}/{name}")
        info = json.loads(get("meta/info.json").read_text(encoding="utf-8"))
        path = get("data/chunk-000/file-000.parquet")
        cols = ["episode_index","frame_index","action","observation.state"]
        table = pq.read_table(path,columns=cols)
        if table.num_rows != info["total_frames"] or info["fps"]!=30:
            raise ValueError(f"Unexpected frame count or fps in {split}")
        arrays = {}
        for name in ("action","observation.state"):
            grip = info["features"][name]["names"].index("gripper.pos")
            arrays[name] = np.asarray(table[name].to_pylist(),dtype=float)[:,grip]
        ids = np.asarray(table["episode_index"]); frames = np.asarray(table["frame_index"])
        if len(np.unique(ids))!=info["total_episodes"] or not all(np.isfinite(x).all() for x in arrays.values()):
            raise ValueError(f"Missing episodes or invalid gripper values in {split}")
        episodes = {}
        for episode in np.unique(ids):
            idx = np.flatnonzero(ids==episode)
            idx = idx[np.argsort(frames[idx])]
            if not np.array_equal(frames[idx],np.arange(len(idx))):
                raise ValueError(f"Non-contiguous frame_index: {split}, episode {episode}")
            episodes[int(episode)] = (arrays["action"][idx],arrays["observation.state"][idx])
        splits[split] = (episodes,info,sha256(path.read_bytes()).hexdigest())
    fixed = splits["fixed-locations"][0]
    low, high = np.percentile(np.concatenate([a for a,_ in fixed.values()]), [5,95])
    if high<=low:
        raise ValueError("No measurable gripper range")
    lower,upper = low+.25*(high-low), low+.75*(high-low)
    report = {"dataset":ARM_REPO,"revision":ARM_REV,"task":ARM_TASK,"fps":30,
              "calibration":"5th/95th percentiles of commanded gripper channel on fixed locations; low/high at 25%/75% of that interval",
              "low_threshold":float(lower),"high_threshold":float(upper),"min_run_frames":3,"match_tolerance_frames":15,"splits":{}}
    for split,(episodes,info,digest) in splits.items():
        records = []
        for episode,(cmd,state) in episodes.items():
            command = first_cycle(cmd,lower,upper)
            observed = first_cycle(state,lower,upper)
            lag = [observed[k]-command[k] for k in (1,2)] if command and observed else None
            records.append({"episode":episode,"frames":len(cmd),"command_high_low_high":command,
                            "measured_high_low_high":observed,"low_high_lag_frames":lag,
                            "mean_absolute_command_state_difference":round(float(np.mean(np.abs(cmd-state))),4),
                            "both_edges_within_tolerance":bool(lag and max(map(abs,lag))<=15)})
        lags = [r["low_high_lag_frames"] for r in records if r["low_high_lag_frames"] is not None]
        report["splits"][split] = {"episodes":len(records),"frames":info["total_frames"],"source_sha256":digest,
                       "command_cycles":sum(r["command_high_low_high"] is not None for r in records),
                       "measured_cycles":sum(r["measured_high_low_high"] is not None for r in records),"both_cycles":len(lags),
                       "both_edges_within_tolerance":sum(r["both_edges_within_tolerance"] for r in records),
                       "mean_absolute_command_state_difference":round(float(np.mean(
                           np.concatenate([np.abs(c-s) for c,s in episodes.values()]))),4),
                       "episodes_ever_reaching_high_state":sum(bool(np.any(s>=upper)) for _,s in episodes.values()),
                       "episodes_ever_reaching_low_state":sum(bool(np.any(s<=lower)) for _,s in episodes.values()),
                       "median_low_lag_frames":median([p[0] for p in lags]),"median_high_lag_frames":median([p[1] for p in lags]),
                       "per_episode":records}
    return report


def evaluate_real(out,cache,offline=False):
    with (nullcontext(None) if offline else httpx.Client(follow_redirects=True,timeout=30)) as client:
        vision = vision_score(client,cache)
        traces = gripper_score(client,cache)
    data = {"model_calls":0,"hardware_connected":False,"vision":vision,"gripper_trace":traces}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"External RGB: {vision['frames']} frames; TP={vision['tp']}, FP={vision['fp']}, FN={vision['fn']}, unique correct={vision['unique_and_correct']}.")
    for split,row in traces["splits"].items():
        print(f"{split}: {row['both_edges_within_tolerance']}/{row['episodes']} episodes with aligned low/high transitions.")
    print(out)
    return data
