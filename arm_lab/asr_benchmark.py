"""Fixed real Mandarin recordings for a local-only speech benchmark."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tarfile
import tempfile
import time
import unicodedata
import wave

import httpx


ROOT = Path(__file__).resolve().parents[1]
REPO = "AISHELL/AISHELL-1"
REV = "bbe295d530192a4cd41644b711c9aecd087df653"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/data_aishell"
TRANSCRIPT = "aishell_transcript_v0.8.txt"
TRANSCRIPT_HASH = "b5f33b9e0b47548e20a5ea4e504297f80df41d559133924c4e5b7b544c15b5c4"
# Frozen from the publisher's transcript prior to ASR. Speaker groups are disjoint.
SPEAKERS = {
    "S0019": ("evaluation", "d503949c0b78d0637cdb8dac0a08c2680f3148f15baea3423bd88d2ce1656995", (308,329,462,489,493)),
    "S0039": ("development", "0c7b706c503f881803f5085e2049558b05639dd69b9a18a905ac68fe9b4de754", (176,231,457,458)),
    "S0045": ("evaluation", "8f40ffde68adb1d996b575247438a7dcd08c1c61d41bec5cab38859bd8e6be0d", (259,330,403,487)),
    "S0075": ("development", "e985259e6e6b88e7dbbe2c4d7b68ff93efb34df8b95ae81a78f6762ea5f0a4ad", (338,386,388,390,392)),
    "S0076": ("evaluation", "e386f910ae13d2021d52016a5106d41417638000d8a3c5ac9ae827c22c8b2f89", (340,344,348,349,353,355)),
}
DEFAULT_CACHE = ROOT / "outputs" / "public-data" / REPO / REV
EVIDENCE = ROOT / "results" / "safety"
DEFAULT_MODEL = ROOT / "outputs" / "models" / "faster-whisper-small"
MODEL_REPO = "Systran/faster-whisper-small"
MODEL_REV = "536b0662742c02347bc0e980a01041f333bce120"
MODEL_FILES = {
    "model.bin":"3e305921506d8872816023e4c273e75d2419fb89b24da97b4fe7bce14170d671",
    "config.json":"e5047537059bd8f182d9ca64c470201585015187",
    "tokenizer.json":"7818adb6de9fa3064d3ff81226fdd675be1f6344",
    "vocabulary.txt":"c9074644d9d1205686f16d411564729461324b75",
}


def read_verified(path, digest):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"Checksum mismatch: {path.name}")
    return data


def download(client, url, path, digest):
    for attempt in range(4):
        try:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                with path.open("wb") as handle:
                    for chunk in response.iter_bytes():
                        handle.write(chunk)
            read_verified(path, digest)
            return
        except (httpx.HTTPError, ValueError):
            if attempt == 3:
                raise RuntimeError(f"Failed to verify download: {path.name}") from None
            time.sleep(2**attempt)


def prepare(cache, offline):
    cache.mkdir(parents=True, exist_ok=True)
    text_path = cache / TRANSCRIPT
    with httpx.Client(follow_redirects=True, timeout=45) as client:
        if not text_path.exists() and not offline:
            with tempfile.TemporaryDirectory(dir=cache) as tmp:
                tmpfile = Path(tmp) / TRANSCRIPT
                download(client, f"{BASE}/transcript/{TRANSCRIPT}", tmpfile, TRANSCRIPT_HASH)
                tmpfile.replace(text_path)
        transcripts = {}
        for line in read_verified(text_path, TRANSCRIPT_HASH).decode().splitlines():
            key, words = line.split(maxsplit=1)
            transcripts[key] = words.replace(" ", "")

        def fetch_speaker(speaker):
            group, digest, numbers = SPEAKERS[speaker]
            ids = [f"BAC009{speaker}W{number:04d}" for number in numbers]
            folder = cache / speaker
            folder.mkdir(exist_ok=True)
            index = folder / "source.json"
            if not index.exists():
                if offline:
                    raise FileNotFoundError(f"Missing {speaker}; run prepare online once.")
                with tempfile.TemporaryDirectory(dir=folder) as tmp:
                    scratch = Path(tmp)
                    archive = scratch / f"{speaker}.tar.gz"
                    download(client, f"{BASE}/wav/{archive.name}", archive, digest)
                    source = {}
                    with tarfile.open(archive, "r:gz") as files:
                        for item in files:
                            key = Path(item.name).stem
                            if key not in ids or not item.isfile():
                                continue
                            data = files.extractfile(item).read()
                            (scratch / f"{key}.wav").write_bytes(data)
                            source[key] = {"archive_path":item.name, "sha256":hashlib.sha256(data).hexdigest()}
                    if set(source) != set(ids):
                        raise ValueError(f"Missing utterance in {archive.name}")
                    for key in ids:
                        (scratch / f"{key}.wav").replace(folder / f"{key}.wav")
                    temp_index = scratch / "source.json"
                    temp_index.write_text(json.dumps(source), encoding="utf-8")
                    temp_index.replace(index)
            source = json.loads(index.read_text(encoding="utf-8"))
            samples = []
            for key in ids:
                data = read_verified(folder / f"{key}.wav", source[key]["sha256"])
                with wave.open(io.BytesIO(data)) as recording:
                    if (recording.getnchannels(), recording.getsampwidth(), recording.getframerate()) != (1,2,16000):
                        raise ValueError(f"Unexpected PCM format: {key}")
                    seconds = recording.getnframes() / recording.getframerate()
                    if not 0 < seconds <= 15:
                        raise ValueError(f"Length outside approved scope: {key}")
                samples.append({"id":key, "speaker":speaker, "group":group, **source[key],
                                "seconds":seconds, "reference":transcripts[key]})
            return samples

        with ThreadPoolExecutor(max_workers=3) as pool:
            samples = [sample for batch in pool.map(fetch_speaker, SPEAKERS) for sample in batch]
    if len(samples) != 24 or sum(s["seconds"] for s in samples) > 360:
        raise ValueError("Invalid benchmark batch")
    return samples


# Functional tests of numeric spelling and competing direction/unit words use this
# frozen vocabulary. It is used only *after* decoding and is never sent to Whisper.
SEMANTIC_WORDS = {
    "direction": ("向左","向右","向前","向后","向上","向下","左侧","右侧","朝上","左转","右转"),
    "motion": ("移动","滑动","旋转","抓取","放下"),
    "negation_or_stop": ("不能","不要","停止"),
}
COUNT_WORDS = "零〇一二两三四五六七八九十百千"
QUANTITY = rf"(?:\d+(?:\.\d+)?|[{COUNT_WORDS}]+(?:点[零〇一二三四五六七八九]+)?)"
ANCHORS = re.compile(
    rf"(?P<number>{QUANTITY})(?P<unit>毫米|厘米|公分|mm|cm|度|°)|"
    + "|".join(word for words in SEMANTIC_WORDS.values() for word in words)
)


def simplified(text):
    """Use the installed Windows locale tables; record the script policy in reports."""
    if not text or os.name != "nt":
        return text
    import ctypes
    convert = ctypes.windll.kernel32.LCMapStringEx
    flags = 0x02000000  # LCMAP_SIMPLIFIED_CHINESE
    length = convert("zh-CN",flags,text,-1,None,0,None,None,0)
    if not length:
        raise ctypes.WinError()
    output = ctypes.create_unicode_buffer(length+1)
    if not convert("zh-CN",flags,text,-1,output,length,None,None,0):
        raise ctypes.WinError()
    return output.value


def standard_text(text):
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(simplified(text).split())


def cer_text(text):
    return "".join(char for char in standard_text(text) if char.isalnum())


def numeral(text):
    if text[0].isascii():
        return text
    digits = "零一二三四五六七八九"
    integer, _, decimals = text.replace("两","二").replace("〇","零").partition("点")
    if any(scale in integer for scale in "千百十"):
        value, previous = 0, 0
        for char in integer:
            if char in digits:
                previous = digits.index(char)
            else:
                value += (previous or 1) * {"十":10,"百":100,"千":1000}[char]
                previous = 0
        integer = str(value + previous)
    else:
        integer = "".join(str(digits.index(char)) for char in integer)
    if decimals:
        return integer + "." + "".join(str(digits.index(char)) for char in decimals)
    return integer


def anchors(text):
    result = []
    for match in ANCHORS.finditer(standard_text(text)):
        if match["number"]:
            unit = {"公分":"cm","厘米":"cm","毫米":"mm","度":"°"}.get(match["unit"],match["unit"])
            result.append("quantity:" + numeral(match["number"]) + unit)
        else:
            token = match[0]
            kind = next(kind for kind, words in SEMANTIC_WORDS.items() if token in words)
            result.append(kind + ":" + token)
    return result


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis)+1))
    for i, expected in enumerate(reference, start=1):
        row = [i]
        for j, observed in enumerate(hypothesis, start=1):
            row.append(min(row[-1]+1, previous[j]+1, previous[j-1]+(expected!=observed)))
        previous = row
    return previous[-1]


def group_score(rows):
    if not rows:
        return None
    errors = sum(row["char_edits"] for row in rows)
    chars = sum(row["chars"] for row in rows)
    groups = {}
    for kind in (*SEMANTIC_WORDS, "quantity"):
        positive = [r for r in rows if any(a.startswith(kind+":") for a in r["ref_anchors"])]
        exact = sum([a for a in r["ref_anchors"] if a.startswith(kind+":")] ==
                    [a for a in r["hyp_anchors"] if a.startswith(kind+":")] for r in positive)
        groups[kind] = {"correct":exact,"positive_clips":len(positive)}
    return {"clips":len(rows),"characters":chars,"char_edits":errors,"cer":round(errors/chars,4),
            "ordered_anchor_exact":sum(r["ref_anchors"]==r["hyp_anchors"] for r in rows),
            "anchor_groups":groups,
            "inference_seconds":round(sum(r["inference_seconds"] for r in rows),3)}


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w",encoding="utf-8",dir=path.parent,delete=False) as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)
        temporary = Path(handle.name)
    temporary.replace(path)


def inventory(samples):
    from .planner import local_intent
    rows = [{**row, "reference_anchors":anchors(row["reference"]),
             "rules_intent_on_reference":local_intent(row["reference"]).kind} for row in samples]
    return {"dataset":REPO,"revision":REV,"transcript_sha256":TRANSCRIPT_HASH,"license":"Apache-2.0",
            "model_calls":0,"seconds":round(sum(r["seconds"] for r in rows),3),
            "speaker_groups":{g:sum(r["group"]==g for r in rows) for g in ("development","evaluation")},
            "non_arm_references_proposing_motion":sum(r["rules_intent_on_reference"] not in ("ask","observe") for r in rows),
            "clips":rows}


def check_weight(path, expected):
    if not path.exists():
        return False
    with path.open("rb") as handle:
        if len(expected)==64:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        else:
            data = handle.read()
            actual = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    return actual==expected


def weights(folder, allow_download):
    folder.mkdir(parents=True,exist_ok=True)
    for name, expected in MODEL_FILES.items():
        path = folder / name
        if check_weight(path, expected):
            continue
        if path.exists():
            raise ValueError(f"Existing weight differs from the pinned model: {name}")
        if not allow_download:
            raise FileNotFoundError(f"Missing approved local weight: {name}")
        with tempfile.TemporaryDirectory(dir=folder) as tmp, httpx.Client(follow_redirects=True,timeout=60) as client:
            pending = Path(tmp) / name
            # Never write a signed URL from the model host to the result or terminal.
            try:
                with client.stream("GET",f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REV}/{name}") as response:
                    response.raise_for_status()
                    with pending.open("wb") as handle:
                        for chunk in response.iter_bytes():
                            handle.write(chunk)
            except httpx.HTTPError:
                raise RuntimeError(f"Model download interrupted: {name}") from None
            if not check_weight(pending, expected):
                raise ValueError(f"Model checksum mismatch: {name}")
            pending.replace(path)


def infer(samples, cache, model_dir, beam, out, allow_download):
    from faster_whisper import WhisperModel
    if out.exists():
        raise FileExistsError(f"Result already exists: {out}. Choose a new output for a new run.")
    weights(model_dir,allow_download)
    model = WhisperModel(str(model_dir),device="cpu",compute_type="int8",local_files_only=True)
    result = {"dataset":REPO,"revision":REV,"model":MODEL_REPO,"model_revision":MODEL_REV,
              "beam_size":beam,"compute_type":"int8","device":"cpu","language":"zh","vad_filter":False,
              "cer_script_policy":"Windows zh-CN simplified" if os.name=="nt" else "original Unicode",
              "status":"in_progress","model_calls":0,"rows":[]}
    save(out,result)
    for sample in samples:
        result["model_calls"] += 1
        save(out,result)
        before = time.perf_counter()
        try:
            segments, _ = model.transcribe(str(cache/sample["speaker"]/(sample["id"]+".wav")),
                                            language="zh",beam_size=beam,vad_filter=False)
            output = "".join(s.text for s in segments).strip()
        except Exception as exc:
            result.update(status="inference_error",failed_sample=sample["id"],error_type=type(exc).__name__)
            save(out,result)
            raise
        elapsed = time.perf_counter()-before
        reference = sample["reference"]
        result["rows"].append({**sample, "output":output,"inference_seconds":round(elapsed,3),
                               "chars":len(cer_text(reference)),
                               "char_edits":edit_distance(cer_text(reference),cer_text(output)),
                               "ref_anchors":anchors(reference),"hyp_anchors":anchors(output)})
        save(out,result)
    result.update(status="complete", scores={name:group_score([r for r in result["rows"] if r["group"]==name])
                                             for name in ("development","evaluation")})
    result["scores"]["all"] = group_score(result["rows"])
    save(out,result)
    return result


def main():
    cli = argparse.ArgumentParser(description="AISHELL-1 Mandarin public audio selection and local ASR benchmark")
    cli.add_argument("mode", choices=("prepare","run"))
    cli.add_argument("--cache",type=Path,default=DEFAULT_CACHE)
    cli.add_argument("--out",type=Path)
    cli.add_argument("--offline",action="store_true",help="Read cached data only")
    cli.add_argument("--beam",type=int,choices=(1,3),default=1)
    cli.add_argument("--model-dir",type=Path,default=DEFAULT_MODEL)
    cli.add_argument("--approve-local-asr",action="store_true",help="Explicitly authorize one local pass of the 24 selected recordings")
    cli.add_argument("--download-model",action="store_true",help="Also allow fetching the pinned Whisper Small weights")
    args = cli.parse_args()
    if args.mode=="run" and not args.approve_local_asr:
        cli.error("Running ASR requires explicit approval for these 24 recordings.")
    if args.offline and args.download_model:
        cli.error("--offline cannot be combined with --download-model")
    samples = prepare(args.cache,args.offline)
    if args.mode=="prepare":
        result = inventory(samples)
        save(args.out or EVIDENCE/"asr_inventory.json",result)
        print(f"Verified {len(samples)} clips, {result['seconds']:.2f} s; 0 model calls.")
    else:
        result = infer(samples,args.cache,args.model_dir,args.beam,
                       args.out or EVIDENCE/f"asr_beam{args.beam}.json",args.download_model)
        print(json.dumps(result["scores"],ensure_ascii=False))


if __name__=="__main__":
    main()
