"""Read fixed/wrist cameras or recorded videos and save bounded, timestamped field evidence."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import cv2


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def save_report(folder, report):
    pending = folder / "capture.tmp"
    pending.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    pending.replace(folder / "capture.json")


def get_source(device, video, backend, start_s, stack):
    if device is not None:
        if device < 0:
            raise ValueError("Camera index must be non-negative")
        cap = cv2.VideoCapture(device, {"dshow":cv2.CAP_DSHOW, "msmf":cv2.CAP_MSMF}[backend])
        info = {"type":"device","device_index":device,"backend":backend}
    else:
        if not video.is_file():
            raise ValueError(f"Missing local video: {video}")
        cap = cv2.VideoCapture(str(video.resolve()))
        info = {"type":"video","file":video.name,"file_size":video.stat().st_size,"start_s":start_s}
    stack.callback(cap.release)
    if not cap.isOpened():
        raise ValueError(f"Cannot open specified source: {info}")
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if width*height > 3840*2160:
        raise ValueError("Choose a source at or below 4K")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps) if math.isfinite(fps) and fps > 0 else None
    info.update(reported_size=[int(width),int(height)], reported_fps=fps)
    if video is not None:
        if not fps:
            raise ValueError("Video needs a readable nominal FPS for time-based sampling")
        length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        info["reported_frames"] = int(length) if math.isfinite(length) and length>0 else None
    return cap, info


def timed_grab(cap, source, relative_s):
    row = {}
    if source["type"]=="video":
        frame_no = round((source["start_s"]+relative_s)*source["reported_fps"])
        row["requested_video_frame"] = frame_no
        if source["reported_frames"] is not None and frame_no >= source["reported_frames"]:
            return {**row,"failure":"past_video_end"}
        if not cap.set(cv2.CAP_PROP_POS_FRAMES,frame_no):
            return {**row,"failure":"seek_failed"}
    if not cap.grab():
        return {**row,"failure":"grab_failed"}
    row.update(grab_host_ns=time.perf_counter_ns(), grab_utc=utc_now())
    return row


def save_frame(cap, source_info, folder, name, row):
    ok, frame = cap.retrieve()
    row["retrieve_host_ns"] = time.perf_counter_ns()
    if not ok or frame is None:
        row["failure"] = "retrieve_failed"
        return
    h,w = frame.shape[:2]
    row["received_size"] = [w,h]
    if source_info["type"]=="video":
        ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        row["reported_video_time_ms"] = round(ms,3) if math.isfinite(ms) and ms>=0 else None
    if w*h > 3840*2160:
        raise ValueError("Decoded source exceeded 4K")
    scale = min(1,1280/w,720/h)
    if scale<1:
        frame = cv2.resize(frame, (max(1,round(w*scale)),max(1,round(h*scale))),
                           interpolation=cv2.INTER_AREA)
    ok, jpg = cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY,85])
    if not ok:
        raise OSError("JPEG encoding failed")
    path = folder / name
    path.write_bytes(jpg.tobytes())
    row.update(file=name,sha256=hashlib.sha256(jpg).hexdigest(),saved_size=list(frame.shape[:2][::-1]))


def capture(args):
    if not 1<=args.trial<=9999 or not 1<=args.samples<=30 or not 1<=args.fps<=10 or (args.samples-1)/args.fps>15:
        raise ValueError("Use trial 1–9999, 1–30 samples, 1–10 Hz and a sampling window no longer than 15 s")
    if args.fixed_device is not None and args.fixed_device==args.wrist_device:
        raise ValueError("The fixed and wrist cameras must have different device indices")
    if args.fixed_video and args.wrist_video and args.fixed_video.resolve()==args.wrist_video.resolve():
        raise ValueError("Use distinct recordings for fixed and wrist views")
    specifications = {"fixed":(args.fixed_device,args.fixed_video,args.start_s)}
    if args.wrist_device is not None or args.wrist_video is not None:
        start = args.wrist_start_s if args.wrist_start_s is not None else args.start_s
        specifications["wrist"] = (args.wrist_device,args.wrist_video,start)
    elif args.wrist_start_s is not None:
        raise ValueError("A wrist start time needs a wrist video")
    if args.wrist_start_s is not None and args.wrist_video is None:
        raise ValueError("--wrist-start-s only applies to a recorded wrist video")
    for device, video, start in specifications.values():
        if not math.isfinite(start) or start<0 or (start and device is not None):
            raise ValueError("Offsets must be non-negative finite times for a recorded video")
    folder = args.out / f"{args.trial:02d}" / args.phase
    folder.mkdir(parents=True,exist_ok=False)
    report = {"trial":args.trial,"phase":args.phase,"start_utc":utc_now(),"status":"in_progress",
              "requested_samples_per_view":args.samples,"sampling_hz":args.fps,"sources":{},"samples":[],
              "skew_clock":"host monotonic grab completion times; hardware exposure timestamps unavailable"}
    save_report(folder,report)
    try:
        with ExitStack() as stack:
            opened = {}
            for label,(device,video,start) in specifications.items():
                cap, source = get_source(device,video,args.backend,start,stack)
                opened[label] = cap
                report["sources"][label] = source
            live = any(s["type"]=="device" for s in report["sources"].values())
            started_ns = time.perf_counter_ns()
            for i in range(args.samples):
                target = i/args.fps
                if live:
                    delay = target - (time.perf_counter_ns()-started_ns)/1e9
                    if delay>0:
                        time.sleep(delay)
                sample = {"i":i,"requested_offset_s":round(target,6),"views":{}}
                if live:
                    lateness = max(0,(time.perf_counter_ns()-started_ns)/1e9-target)
                    sample["schedule_lateness_ms"] = round(lateness*1000,3)
                    if lateness >= 1/args.fps:
                        sample["views"] = {label:{"failure":"missed_sample_deadline"} for label in opened}
                        report["samples"].append(sample)
                        save_report(folder,report)
                        continue
                # Grab both feeds before decoding either JPEG. Driver buffering is unknown.
                for label, cap in opened.items():
                    sample["views"][label] = timed_grab(cap,report["sources"][label],target)
                for label,row in sample["views"].items():
                    if "failure" not in row:
                        save_frame(opened[label],report["sources"][label],folder,f"{label}-{i:03d}.jpg",row)
                views = list(sample["views"].values())
                if len(views)==2 and all("file" in view for view in views):
                    sample["grab_skew_ms"] = round(abs(views[1]["grab_host_ns"]-views[0]["grab_host_ns"])/1e6,3)
                report["samples"].append(sample)
                save_report(folder,report)
        stats = {}
        for label in specifications:
            seen = sum("file" in s["views"][label] for s in report["samples"])
            stats[label] = {"saved":seen,"missing_samples":args.samples-seen}
            if report["sources"][label]["type"]=="video":
                numbers = [s["views"][label]["requested_video_frame"] for s in report["samples"]
                           if "file" in s["views"][label]]
                stats[label]["reused_video_samples"] = len(numbers)-len(set(numbers))
        skews = [s["grab_skew_ms"] for s in report["samples"] if "grab_skew_ms" in s]
        report.update(status="complete" if all(not s["missing_samples"] for s in stats.values()) else "partial",
                      counts=stats, paired_samples=len(skews), maximum_grab_skew_ms=max(skews,default=None),
                      median_grab_skew_ms=statistics.median(skews) if skews else None)
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status="interrupted" if isinstance(exc,KeyboardInterrupt) else "error",error=str(exc))
        raise
    finally:
        report["finish_utc"] = utc_now()
        save_report(folder,report)
    return folder / "capture.json", report


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("fixed","wrist"):
        group = cli.add_mutually_exclusive_group(required=name=="fixed")
        group.add_argument(f"--{name}-device",type=int,help="Explicit local camera index")
        group.add_argument(f"--{name}-video",type=Path,help="Existing local recording")
    cli.add_argument("--trial",type=int,required=True,help="Field notebook trial ID")
    cli.add_argument("--phase",choices=("before","after","during"),required=True)
    cli.add_argument("--samples",type=int,default=3)
    cli.add_argument("--fps",type=float,default=3,help="Requested sample rate (1–10 per second)")
    cli.add_argument("--start-s",type=float,default=0,help="Evidence window start within recorded files")
    cli.add_argument("--wrist-start-s",type=float)
    cli.add_argument("--backend",choices=("dshow","msmf"),default="dshow",help="Windows USB camera backend")
    cli.add_argument("--out",type=Path,default=Path("outputs/field-camera"))
    args = cli.parse_args(argv)
    try:
        path,report = capture(args)
    except (ValueError,OSError,cv2.error) as exc:
        cli.exit(2,f"Camera evidence failed: {exc}\n")
    print(f"{report['status']} | {report['counts']} | {path}")
    return 0 if report["status"]=="complete" else 1


if __name__=="__main__":
    raise SystemExit(main())
