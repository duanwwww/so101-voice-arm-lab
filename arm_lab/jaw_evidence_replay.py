"""Offline audit of what real gripper position traces can support about contact."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .real_data import ARM_REPO, ARM_REV, ARM_TASK, asset


HASHES = {
    "fixed-locations": "7420379125baa954c01d62b730823536aaa577e766eba4b2907b640e223603b6",
    "random-locations": "a66c5322639e2aad56ea53fbc9efca4ce3ef98fc551822b82afab2e0f4deb82e",
}
WINDOW = 15  # samples at 30 Hz; observational thresholds in dataset-native units


def load_split(cache,split):
    prefix = f"{ARM_TASK}/{split}"
    info_path = asset(None,cache,ARM_REPO,ARM_REV,prefix+"/meta/info.json")
    data_path = asset(None,cache,ARM_REPO,ARM_REV,prefix+"/data/chunk-000/file-000.parquet")
    with data_path.open("rb") as src:
        if hashlib.file_digest(src,"sha256").hexdigest()!=HASHES[split]:
            raise ValueError(f"Source file differs from the pinned trace: {split}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    table = pq.read_table(data_path,columns=["episode_index","frame_index","action","observation.state"])
    if info["fps"] != 30 or table.num_rows != info["total_frames"]:
        raise ValueError("Unexpected frame count or sampling rate")
    channels = {}
    for field in ("action","observation.state"):
        names = info["features"][field]["names"]
        if not all(name.endswith(".pos") for name in names):
            raise ValueError("Unexpected telemetry schema; review contact evidence interpretation")
        channels[field] = np.array(table[field].to_pylist())[:,names.index("gripper.pos")]
    episode_ids = np.asarray(table["episode_index"])
    frames = np.asarray(table["frame_index"])
    episodes = {}
    for episode in np.unique(episode_ids):
        indices = np.flatnonzero(episode_ids == episode)
        indices = indices[np.argsort(frames[indices])]
        if not np.array_equal(frames[indices],np.arange(len(indices))):
            raise ValueError("Non-contiguous episode")
        action, observed = (channels[name][indices] for name in ("action","observation.state"))
        if not (np.isfinite(action).all() and np.isfinite(observed).all()):
            raise ValueError("Missing position samples")
        episodes[int(episode)] = (action,observed)
    return episodes,info["features"]["observation.state"]["names"]


def first_interval(cmd, obs, low):
    if len(obs)<WINDOW:
        return None
    commands = np.lib.stride_tricks.sliding_window_view(cmd,WINDOW)
    measured = np.lib.stride_tricks.sliding_window_view(obs,WINDOW)
    # This intentionally optimistic position-only cue is used to find ambiguous
    # examples. Neither its presence nor its absence certifies a grasp.
    candidates = np.flatnonzero((measured.max(axis=1)<=low) &
                                (np.ptp(measured,axis=1)<=0.5) &
                                (np.abs(commands-measured).max(axis=1)<=1.0))
    if not len(candidates):
        return None
    i = int(candidates[0])
    return {"first_frame":i,"last_frame":i+WINDOW-1,"minimum_position":float(measured[i].min()),
            "maximum_position":float(measured[i].max()),"max_command_error":float(np.abs(commands[i]-measured[i]).max())}


def replay(cache):
    sets = {name:load_split(cache,name) for name in HASHES}
    fixed = sets["fixed-locations"][0]
    low5,high95 = np.percentile(np.concatenate([cmd for cmd,_ in fixed.values()]),(5,95))
    low = float(low5+0.25*(high95-low5))
    result = {"dataset":ARM_REPO,"revision":ARM_REV,"task":ARM_TASK,"model_calls":0,"serial_writes":0,
              "fps":30,"window_samples":WINDOW,"max_observed_range":0.5,"max_abs_command_error":1.0,
              "low_position_threshold_fixed_split":low,"inspection":"recorded position channels only; footage not analysed",
              "missing_independent_fields":["left/right contact","object off table","actuator fault status","grip force"],
              "splits":{}}
    for split,(episodes,fields) in sets.items():
        rows = []
        for episode,(cmd,obs) in episodes.items():
            interval = first_interval(cmd,obs,low)
            rows.append({"episode":episode,"frames":len(obs),"low_stable_tracking_window":interval,
                         "grasp_evidence":"unverified"})
        result["splits"][split] = {"source_sha256":HASHES[split],"state_names":fields,
                                  "episodes":len(rows),"frames":sum(x["frames"] for x in rows),
                                  "episodes_with_low_stable_tracking":sum(x["low_stable_tracking_window"] is not None for x in rows),
                                  "window_already_at_first_frame":sum(x["low_stable_tracking_window"] is not None and
                                                                      x["low_stable_tracking_window"]["first_frame"]==0 for x in rows),
                                  "independently_verified_grasps":0,"position_only_unverified":len(rows),"rows":rows}
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--cache",type=Path,default=Path("outputs/public-data"))
    cli.add_argument("--out",type=Path,default=Path("results/safety/real_jaw_evidence.json"))
    args = cli.parse_args()
    result = replay(args.cache)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    for name,row in result["splits"].items():
        print(name, "low/stable/tracking",row["episodes_with_low_stable_tracking"],"/",row["episodes"],
              "independent grasp confirmations",row["independently_verified_grasps"])


if __name__ == "__main__":
    main()
