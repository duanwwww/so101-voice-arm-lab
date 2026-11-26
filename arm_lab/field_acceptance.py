"""Prepare and score independently measured hardware trials. No device or model calls."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median


FIELDS = ("id","setup","kind","instruction","target","axis","requested",
          "program_status","actual_station","dx_mm","dy_mm","dz_mm","wrist_deg","evidence")
STATUSES = {"success","already_done","unverified","clarify","rejected","stopped","model_error"}


def template(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for setup in ("正常摆位","目标位遮挡","取料位边缘偏放"):
        for station in ("A","B","C"):
            target = {"A":"C","B":"A","C":"B"}[station]
            rows.append(dict(setup=setup,kind="transfer",instruction=f"把 {station} 区木块放到 {target} 区",target=target))
    for axis, label in (("x","前"),("y","左"),("z","上")):
        rows.append(dict(setup="独立刻度测量",kind="move",instruction=f"向{label}移动一厘米",axis=axis,requested=10))
    for angle in (3,-3,5):
        direction = "下" if angle>0 else "上"
        rows.append(dict(setup="独立角度测量",kind="wrist",instruction=f"手腕向{direction}转{abs(angle)}度",requested=angle))
    # Protect an existing notebook containing field measurements.
    with path.open("x",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS)
        writer.writeheader()
        for i,row in enumerate(rows,1):
            writer.writerow(dict(id=i,**row))


def number(row, key):
    n=float(row[key])
    if not math.isfinite(n):
        raise ValueError(f"第 {row['id']} 行的 {key} 必须是有限数值")
    return n


def score(path):
    records = []
    errors = {"move":[], "wrist":[]}
    ids = set()
    with path.open(encoding="utf-8-sig",newline="") as f:
        reader=csv.DictReader(f)
        if reader.fieldnames != list(FIELDS):
            raise ValueError("列名不匹配，请从 template 生成记录表")
        for row in reader:
            key=row["id"]
            if not key or key in ids:
                raise ValueError("缺少回合编号或编号重复")
            ids.add(key)
            if not (row["program_status"] or row["evidence"]):
                if any(row[k] for k in ("actual_station","dx_mm","dy_mm","dz_mm","wrist_deg")):
                    raise ValueError(f"第 {key} 行已有测量，请补齐回执与记录依据")
                continue
            if row["program_status"] not in STATUSES or not row["evidence"].strip():
                raise ValueError(f"第 {key} 行缺少合法回执或记录依据")
            kind=row["kind"]
            if kind=="transfer":
                if row["target"] not in ("A","B","C") or row["actual_station"] not in ("A","B","C","outside","held","unknown"):
                    raise ValueError(f"第 {key} 行需要人工记录目标与实际落点")
                achieved = None if row["actual_station"]=="unknown" else row["actual_station"]==row["target"]
                error=None
            elif kind in errors:
                if kind=="move":
                    if row["axis"] not in ("x","y","z"):
                        raise ValueError(f"第 {key} 行缺少运动轴")
                    desired=[number(row,"requested") if a==row["axis"] else 0 for a in "xyz"]
                    error=math.dist(desired,[number(row,f"d{a}_mm") for a in "xyz"])
                else:
                    error=abs(number(row,"wrist_deg")-number(row,"requested"))
                errors[kind].append(error)
                achieved=error <= (5 if kind=="move" else 2)
            else:
                raise ValueError(f"第 {key} 行未知动作")
            records.append(dict(id=key,kind=kind,program_status=row["program_status"],achieved=achieved,
                                error=error,evidence=row["evidence"]))
    return dict(planned=len(ids),recorded=len(records),pending=len(ids)-len(records),
                physically_unknown=sum(r["achieved"] is None for r in records),
                false_success=sum(r["program_status"] in ("success","already_done") and
                                  r["achieved"] is False for r in records),
                unconfirmed_but_achieved=sum(r["program_status"]=="unverified" and r["achieved"] is True for r in records),
                error={k:dict(n=len(v),median=median(v) if v else None,max=max(v,default=None)) for k,v in errors.items()},
                records=records)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action",choices=("template","score"))
    cli.add_argument("csv",type=Path)
    cli.add_argument("--out",type=Path)
    args=cli.parse_args()
    if args.action=="template":
        template(args.csv)
    else:
        output=json.dumps(score(args.csv),indent=2,ensure_ascii=False)
        if args.out:
            args.out.parent.mkdir(exist_ok=True,parents=True)
            args.out.write_text(output,encoding="utf-8")
        else:
            print(output)


if __name__=="__main__":
    main()
