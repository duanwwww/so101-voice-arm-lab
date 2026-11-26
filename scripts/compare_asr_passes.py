"""Compare two completed local ASR runs using the saved transcriptions only."""
import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median

from arm_lab import asr_benchmark as bench


BEAMS = (1, 3)
SPLITS = ("development", "evaluation", "all")
NUMBERS = re.compile(bench.QUANTITY)
UNITS = re.compile(r"毫米|厘米|公分|mm|cm|米")
UNIT_NAMES = {"毫米": "mm", "厘米": "cm", "公分": "cm", "米": "m"}
# Avoid reading the lexical adverb 一定 ('definitely') as the count one.
NON_COUNT_ONE = ("一定", "一起", "一直")


def numeric_sequence(text):
    # In “一百二十千克” the 千 belongs to the unit. Keep this boundary in both scripts.
    text = text.replace("千克", "kg")
    values = []
    for match in NUMBERS.finditer(text):
        if match.group() == "一":
            right = text.startswith(NON_COUNT_ONE, match.start())
            left = match.start() > 0 and text[match.start() - 1] == "统"
            if right or left:
                continue
        values.append(format(Decimal(bench.numeral(match.group())).normalize(), "f"))
    return values


def length_units(text):
    result = []
    for match in UNITS.finditer(text):
        unit = match.group()
        # A bare 米 is a unit after a number. 小米 is a brand in this corpus.
        if unit == "米" and not re.search(rf"(?:{bench.QUANTITY})$", text[:match.start()]):
            continue
        result.append(UNIT_NAMES.get(unit, unit))
    return result


def tokens(raw):
    text = bench.standard_text(raw)
    values = {
        "direction": [],
        "number": numeric_sequence(text),
        "length_unit": length_units(text),
        "negation_or_stop": [],
        "quantity_with_unit": [],
        "motion": [],
    }
    for anchor in bench.anchors(raw):
        kind, value = anchor.split(":", 1)
        values["quantity_with_unit" if kind == "quantity" else kind].append(value)
    return values


def validate_inventory(inventory):
    rows = inventory["clips"]
    groups = Counter(row["group"] for row in rows)
    if len({row["id"] for row in rows}) != 24 or groups != {"development": 9, "evaluation": 15}:
        raise ValueError("Expected the 24 frozen clips and the original 9/15 split")
    speakers = {group: {row["speaker"] for row in rows if row["group"] == group} for group in groups}
    if speakers["development"] & speakers["evaluation"]:
        raise ValueError("Speaker leakage across the development and evaluation groups")
    if any(not re.fullmatch("[a-f0-9]{64}", row["sha256"]) for row in rows):
        raise ValueError("Missing SHA-256 in the audio inventory")
    if inventory["dataset"] != bench.REPO or inventory["revision"] != bench.REV:
        raise ValueError("Unexpected source dataset or revision")


def validated_rows(inventory, report, beam):
    expected = {
        "status": "complete", "model_calls": 24, "beam_size": beam,
        "model": bench.MODEL_REPO, "model_revision": bench.MODEL_REV,
        "dataset": bench.REPO, "revision": bench.REV,
        "device": "cpu", "compute_type": "int8", "language": "zh", "vad_filter": False,
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Incomplete or incompatible beam={beam} report")
    if len(report["rows"]) != 24:
        raise ValueError(f"Expected exactly 24 decoding results for beam={beam}")
    by_id = {}
    input_fields = ("id", "sha256", "reference", "group", "speaker", "seconds")
    for source, row in zip(inventory["clips"], report["rows"], strict=True):
        if any(source[key] != row[key] for key in input_fields):
            raise ValueError(f"Different input or order in beam={beam}, expected {source['id']}")
        reference, output = bench.cer_text(row["reference"]), bench.cer_text(row["output"])
        edits = bench.edit_distance(reference, output)
        if (edits != row["char_edits"] or len(reference) != row["chars"]
                or bench.anchors(row["output"]) != row["hyp_anchors"]):
            raise ValueError(f"Stored scores differ on {row['id']}; check the report's script normalization")
        elapsed = row["inference_seconds"]
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError(f"Invalid inference time on {row['id']}")
        by_id[row["id"]] = {
            "output": row["output"], "char_edits": edits,
            "inference_seconds": elapsed, "tokens": tokens(row["output"]),
        }
    return by_id


def summarize(rows, key):
    total_chars = sum(row["chars"] for row in rows)
    edits = sum(row[key]["char_edits"] for row in rows)
    utterances = sum(row[key]["char_edits"] == 0 for row in rows)
    durations = [row[key]["inference_seconds"] for row in rows]
    decode_seconds = sum(durations)
    audio = sum(row["seconds"] for row in rows)
    slots = {}
    for kind in rows[0]["reference_tokens"]:
        positive = [row for row in rows if row["reference_tokens"][kind]]
        wrong = [row["id"] for row in positive if row["reference_tokens"][kind] != row[key]["tokens"][kind]]
        extra = [row["id"] for row in rows if not row["reference_tokens"][kind] and row[key]["tokens"][kind]]
        slots[kind] = {
            "correct": len(positive) - len(wrong), "reference_positive": len(positive),
            "incorrect_ids": wrong, "added_on_reference_negative_ids": extra,
        }
    return {
        "clips": len(rows), "reference_characters": total_chars, "character_edits": edits,
        "cer": round(edits / total_chars, 6),
        "character_accuracy_1_minus_cer": round(1 - edits / total_chars, 6),
        "normalized_sentence_exact": utterances,
        "normalized_sentence_accuracy": round(utterances / len(rows), 6),
        "audio_seconds": round(audio, 3), "inference_seconds": round(decode_seconds, 3),
        "mean_seconds_per_clip": round(decode_seconds / len(rows), 3),
        "median_seconds_per_clip": round(median(durations), 3),
        "rtf": round(decode_seconds / audio, 4), "slots": slots,
    }


def compare(inventory, reports):
    validate_inventory(inventory)
    if reports[1]["cer_script_policy"] != reports[3]["cer_script_policy"]:
        raise ValueError("Script-normalization policy differs between the two passes")
    indexed = {beam: validated_rows(inventory, reports[beam], beam) for beam in BEAMS}
    pairs = []
    for source in inventory["clips"]:
        row = {key: source[key] for key in ("id", "group", "speaker", "sha256", "reference", "seconds")}
        row.update(chars=len(bench.cer_text(row["reference"])), reference_tokens=tokens(row["reference"]))
        for beam in BEAMS:
            row[f"beam{beam}"] = indexed[beam][row["id"]]
        pairs.append(row)
    groups = {}
    for group in SPLITS:
        rows = [row for row in pairs if group == "all" or row["group"] == group]
        edit_reductions = [row["beam1"]["char_edits"] - row["beam3"]["char_edits"] for row in rows]
        groups[group] = {f"beam{beam}": summarize(rows, f"beam{beam}") for beam in BEAMS}
        groups[group]["paired"] = {
            "lower_cer_with_beam3": sum(n > 0 for n in edit_reductions),
            "equal_edits": edit_reductions.count(0),
            "higher_cer_with_beam3": sum(n < 0 for n in edit_reductions),
            "normalized_output_changed_ids": [
                row["id"] for row in rows
                if bench.cer_text(row["beam1"]["output"]) != bench.cer_text(row["beam3"]["output"])
            ],
        }
    return {"verified_model_calls_in_input_runs": sum(r["model_calls"] for r in reports.values()),
            "comparison_model_calls": 0, "cer_script_policy": reports[1]["cer_script_policy"],
            "scoring_policy": {
                "text":"CER and exact sentences follow the frozen benchmark, including numeral spelling differences.",
                "keyword":"Ordered exact tokens per reference-positive clip, extra tokens on reference-negative clips listed separately; lexical direction, negation/stop, motion and quantity pairs follow the frozen anchor list.",
                "number":"Separate supplementary whole-clip numeral sequence, including non-length counts and times; unify Chinese and Arabic spellings, treat 千克 as the unit kg and exclude lexical 一 in 一定, 一起, 一直, 统一. A numeral is still counted when its unit is missing.",
                "length_unit":"Separate supplementary whole-clip millimetre, centimetre and metre unit sequence; centimetre spellings are equivalent. A bare 米 requires a preceding numeral. These automatic views do not replace semantic review.",
                "time":"Sum of per-clip wall times after model load, with no new warmup or inference passes. RTF divides this sum by the clips' total audio duration."},
            "groups": groups, "rows": pairs}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--directory", type=Path, default=bench.EVIDENCE)
    cli.add_argument("--out", type=Path)
    args = cli.parse_args()
    paths = {key: args.directory / f"asr_{key}.json" for key in ("inventory", "beam1", "beam3")}
    raw = {key: path.read_bytes() for key, path in paths.items()}
    reports = {beam: json.loads(raw[f"beam{beam}"]) for beam in BEAMS}
    result = compare(json.loads(raw["inventory"]), reports)
    result["input_sha256"] = {paths[key].name: hashlib.sha256(blob).hexdigest() for key, blob in raw.items()}
    path = args.out or args.directory / "asr_comparison.json"
    if path.resolve() in {p.resolve() for p in paths.values()}:
        cli.error("Output path must differ from the source reports")
    bench.save(path, result)
    for beam in BEAMS:
        v = result["groups"]["all"][f"beam{beam}"]
        print(f"beam={beam}  CER={v['cer']:.2%}  exact={v['normalized_sentence_exact']}/{v['clips']}  decode={v['inference_seconds']}s")


if __name__ == "__main__":
    main()
