"""Offline chronological holdout experiment on published red-cube bounding boxes."""
import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

import cv2
import numpy as np

from arm_lab.real_data import VISION_REPO, VISION_REV, iou
from arm_lab.vision import detect


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BoxFilter:
    name: str
    min_area_fraction: float
    max_area_fraction: float
    min_short_long_ratio: float

    def accepts(self, box, pixels):
        w, h = box[2:]
        return (self.min_area_fraction <= w*h/pixels <= self.max_area_fraction
                and min(w,h)/max(w,h) >= self.min_short_long_ratio)


# Three prespecified candidates. Resolve ties by preferring the earlier, wider rule.
FILTERS = [
    BoxFilter("wide", .00025, .05, .40),
    BoxFilter("medium", .0008, .05, .60),
    BoxFilter("strict", .002, .04, .72),
]


def split_of(filename):
    minute = filename[6:19]
    if not "20260515_2232" <= minute <= "20260515_2249":
        raise ValueError(f"Unexpected capture minute in {filename}")
    if minute <= "20260515_2240":
        return "development"
    if minute == "20260515_2241":
        return "gap"
    return "holdout"


def observations(folder, rows, red):
    result = []
    for row in rows:
        name = row["file_name"]
        if Path(name).name != name or not name.endswith(".jpg"):
            raise ValueError(f"Unexpected filename {name}")
        blob = (folder/name).read_bytes()
        bgr = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pixels = rgb.shape[0]*rgb.shape[1]
        candidates = [o["box"] for o in detect(rgb, {})["objects"]]
        # Labels enter scoring only after all candidates have been produced.
        targets = [box for c,box in zip(row["objects"]["categories"],row["objects"]["bbox"],strict=True)
                   if c == red]
        if len(targets) != 1:
            raise ValueError(f"Expected one red label in {name}")
        result.append((name, sha256(blob).hexdigest(), candidates, pixels, targets[0]))
    return result


def score(observed, rule=None):
    totals = dict.fromkeys(("tp", "fp", "fn", "unique_correct", "unique_wrong", "empty", "multiple"), 0)
    per_image = []
    for name, digest, boxes, pixels, truth in observed:
        kept = [b for b in boxes if rule is None or rule.accepts(b, pixels)]
        matched = any(iou(box, truth) >= .5 for box in kept)
        count = len(kept)
        totals["tp"] += int(matched)
        totals["fp"] += count-int(matched)
        totals["fn"] += int(not matched)
        totals["unique_correct"] += int(count == 1 and matched)
        totals["unique_wrong"] += int(count == 1 and not matched)
        totals["empty"] += int(count == 0)
        totals["multiple"] += int(count > 1)
        per_image.append({"name":name, "sha256":digest, "candidates":count, "hit":matched})
    tp, fp, fn = (totals[key] for key in ("tp","fp","fn"))
    return {"images":len(observed), **totals,
            "precision":round(tp/(tp+fp), 4) if tp+fp else None,
            "recall":round(tp/(tp+fn), 4), "per_image":per_image}


def selection_order(result):
    return result["unique_correct"], -result["unique_wrong"], result["tp"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT/"outputs"/"public-data")
    parser.add_argument("--out", type=Path, default=ROOT/"outputs"/"vision_ablation.json")
    args = parser.parse_args()
    base = args.cache/VISION_REPO/VISION_REV/"real"
    red = (base/"classes.txt").read_text(encoding="utf-8").splitlines().index("red cube")
    manifest = (base/"images"/"metadata.jsonl").read_bytes()
    groups = {name:[] for name in ("development","gap","holdout")}
    for raw in manifest.splitlines():
        row = json.loads(raw)
        groups[split_of(row["file_name"])].append(row)
    if [len(group) for group in groups.values()] != [54,5,41]:
        raise ValueError("Unexpected split size for pinned manifest")
    dev = observations(base/"images", groups["development"], red)
    scores = [score(dev, rule) for rule in FILTERS]
    winner = max(range(len(FILTERS)), key=lambda idx:selection_order(scores[idx]))
    selected = FILTERS[winner]
    # Load the reserved subset after choosing and freezing the operating point.
    holdout = observations(base/"images", groups["holdout"], red)
    report = {
        "dataset":VISION_REPO, "revision":VISION_REV, "manifest_sha256":sha256(manifest).hexdigest(),
        "model_calls":0, "hardware_connected":False, "detector_default_changed":False,
        "method":"Fixed original HSV detector; standalone postfilter uses candidate box area divided "
                 "by image area and its shorter/longer side ratio, without cropping or workcell ROI. "
                 "Match one red label per image at IoU >= 0.5, all remaining boxes are FP.",
        "split":"2026-05-15 filename clock. Development 22:32-22:40 (54); exclude 22:41 (5); "
                "holdout 22:42-22:49 (41). Entire capture minutes stay together. "
                "Same capture period, not an independent device or session; prior full-set baseline known.",
        "unused_gap":[r["file_name"] for r in groups["gap"]],
        "selection":"On development, maximize unique_correct, minimize unique_wrong, maximize TP, "
                    "then prefer widest specified rule. Apply winner once on reserved images.",
        "prespecified_rules":[asdict(f) for f in FILTERS], "selected_rule":asdict(selected),
        "development_grid":{f.name:{k:v for k,v in metrics.items() if k != "per_image"}
                            for f,metrics in zip(FILTERS,scores,strict=True)},
        "development":{"before":score(dev), "after":scores[winner]},
        "holdout":{"before":score(holdout), "after":score(holdout, selected)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Frozen rule: {selected}.")
    for group in ("development","holdout"):
        for version, result in report[group].items():
            print(group, version, {k:v for k,v in result.items() if k != "per_image"})


if __name__ == "__main__":
    main()
