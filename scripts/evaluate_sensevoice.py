"""One approved local SenseVoice pass over the frozen 24 human and 4 synthetic clips."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import io
import json
import os
from pathlib import Path
import platform
import time
import wave

from arm_lab import asr_benchmark as bench
from scripts import compare_asr_passes as comparison


OUT = bench.EVIDENCE / "sensevoice" / "speech.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_row(sample, output, elapsed):
    reference = bench.cer_text(sample["reference"])
    return {**sample, "output":output, "inference_seconds":round(elapsed,6),
            "chars":len(reference), "char_edits":bench.edit_distance(reference,bench.cer_text(output)),
            "ref_anchors":bench.anchors(sample["reference"]), "hyp_anchors":bench.anchors(output)}


def summarize(report):
    inventory = json.loads((bench.EVIDENCE/"asr_inventory.json").read_bytes())
    previous = {beam:json.loads((bench.EVIDENCE/f"asr_beam{beam}.json").read_bytes()) for beam in (1,3)}
    existing = comparison.compare(inventory,previous)
    outputs = {row["id"]:row for row in report["rows"]}
    if set(outputs) != {row["id"] for row in inventory["clips"]}:
        raise ValueError("Expected every original human clip exactly once")
    pairs = existing["rows"]
    for pair in pairs:
        row = outputs[pair["id"]]
        if row["sha256"]!=pair["sha256"] or row["reference"]!=pair["reference"]:
            raise ValueError("Cannot compare different audio or references")
        pair["sensevoice"] = {key:row[key] for key in ("output","char_edits","inference_seconds")}
        pair["sensevoice"]["tokens"] = comparison.tokens(row["output"])
    groups = {}
    for group in ("development","evaluation","all"):
        subset = [row for row in pairs if group=="all" or row["group"]==group]
        groups[group] = {name:comparison.summarize(subset,name)
                         for name in ("beam1","beam3","sensevoice")}
    report["comparison"] = {"groups":groups, "rows":pairs,
                            "historical_whisper_runs":True, "new_whisper_model_calls":0,
                            "policy":existing["scoring_policy"],
                            "split_note":"All 24 clips come from AISHELL-1 train. Evaluation is the project's speaker-held-out subset, not the official AISHELL-1 test split.",
                            "timing_note":"SenseVoice wall time includes audio decode after one model load, no warm-up. Whisper timings are earlier runs on this computer, not a randomized concurrent speed comparison."}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--approve-local-asr",action="store_true")
    cli.add_argument("--download-model",action="store_true")
    cli.add_argument("--out",type=Path,default=OUT)
    cli.add_argument("--cache",type=Path,default=bench.DEFAULT_CACHE)
    cli.add_argument("--model-dir",type=Path)
    args = cli.parse_args()
    if not args.approve_local_asr:
        cli.error("Explicit approval is required for the 24 human and 4 synthetic clips")
    if args.out.exists():
        cli.error("Choose a new output path to preserve the previous run")
    from arm_lab.sensevoice import DEFAULT_MODEL_DIR, SenseVoiceRecognizer, prepare_model
    model_dir = args.model_dir or DEFAULT_MODEL_DIR
    samples = bench.prepare(args.cache,offline=True)
    inventory = json.loads((bench.EVIDENCE/"asr_inventory.json").read_bytes())
    comparison.validate_inventory(inventory)
    if [(s["id"],s["sha256"],s["reference"]) for s in samples] != [
        (s["id"],s["sha256"],s["reference"]) for s in inventory["clips"]
    ]:
        raise ValueError("Audio cache differs from the frozen inventory")
    weights = prepare_model(model_dir,allow_download=args.download_model)
    before = time.perf_counter()
    model = SenseVoiceRecognizer(model_dir,num_threads=4,use_itn=True)
    loaded = time.perf_counter()-before
    report = {"created_utc":datetime.now(timezone.utc).isoformat(), "status":"in_progress",
              "model":"SenseVoiceSmall", "device":"cpu", "compute_type":"int8", "num_threads":4,
              "language":"zh", "use_itn":True, "vad_filter":False, "weights":weights,
              "model_load_seconds":round(loaded,6), "local_model_calls":0, "cloud_calls":0,
              "dataset":bench.REPO,"revision":bench.REV,
              "cer_script_policy":"Windows zh-CN simplified" if os.name=="nt" else "original Unicode",
              "environment":{"python":platform.python_version(), "system":platform.system(),
                             **{name:version(name) for name in ("sherpa-onnx","av","numpy")}},
              "source_sha256":{name:digest(bench.EVIDENCE/name) for name in
                               ("asr_inventory.json","asr_beam1.json","asr_beam3.json","asr_keyword_review.json","prompt_trial_real.json")},
              "rows":[], "synthetic":[]}
    bench.save(args.out,report)
    previous = json.loads((bench.EVIDENCE/"prompt_trial_real.json").read_bytes())
    synthetic = [case for case in json.loads((bench.ROOT/"configs/prompt_trial.json").read_bytes())["cases"] if "wav" in case]
    inputs = [(sample,args.cache/sample["speaker"]/(sample["id"]+".wav"),"rows") for sample in samples]
    for case in synthetic:
        path = bench.EVIDENCE/"prompt_trial_audio"/case["wav"]
        blob = bench.read_verified(path,previous["signature"]["wave_sha256"][case["id"]])
        with wave.open(io.BytesIO(blob)) as audio:
            seconds = audio.getnframes()/audio.getframerate()
        inputs.append(({**case,"sha256":hashlib.sha256(blob).hexdigest(),"seconds":seconds,
                        "source":"Windows offline System.Speech / Microsoft Huihui Desktop"},path,"synthetic"))
    for sample,path,section in inputs:
        report["local_model_calls"] += 1
        report["current_clip"] = sample["id"]
        bench.save(args.out,report)
        before = time.perf_counter()
        try:
            output = model.transcribe(bench.read_verified(path,sample["sha256"]))
        except Exception as exc:
            report.update(status="inference_error",error_type=type(exc).__name__)
            bench.save(args.out,report)
            raise
        elapsed = time.perf_counter()-before
        row = score_row(sample,output,elapsed)
        if section=="synthetic":
            row["historical_whisper"] = previous["audio"][sample["id"]]
        report[section].append(row)
        bench.save(args.out,report)
        print(sample["id"],output,f"{elapsed:.3f}s",flush=True)
    summarize(report)
    report.pop("current_clip")
    report.update(status="complete")
    bench.save(args.out,report)
    for group,summary in report["comparison"]["groups"].items():
        result=summary["sensevoice"]
        print(group,f"CER={result['cer']:.2%}",f"sentence={result['normalized_sentence_exact']}/{result['clips']}",
              f"decode={result['inference_seconds']:.3f}s")


if __name__=="__main__":
    main()
