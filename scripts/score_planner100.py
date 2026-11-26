"""Summarize a complete or partial planner journal without contacting a model."""
from collections import Counter, defaultdict
import hashlib
from itertools import combinations
import json

from arm_lab.planner_scoring import score_response
from scripts.evaluate_prompt_trial import save


def read_if_present(path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def expected_kind(case):
    if case["route"] == "local":
        return "local"
    return "action" if "steps" in case["expected"] else "ask"


def observations(cases, book, versions):
    by_case = {c["id"]:c for c in cases}
    trials = defaultdict(list)
    for attempt in book.get("attempts", []):
        case = by_case[attempt["id"]]
        if case["route"] != "planner" or attempt["split"] != case["split"]:
            raise ValueError(f"Inconsistent online case metadata for {case['id']}")
        trials[(case["id"],attempt["version"])].append(attempt)
    rows = []
    for case in cases:
        if case["route"] != "planner":
            continue
        for version in versions[case["split"]]:
            attempts = trials[(case["id"],version)]
            answers = [a for a in attempts if a["state"]=="answered"]
            if len(answers)>1:
                raise ValueError(f"Ambiguous duplicate model answer for {case['id']}, {version}")
            final = answers[0] if answers else (attempts[-1] if attempts else {})
            if answers and final.get("http_status") != 200:
                raise ValueError("Answered state must contain an HTTP 200 response")
            scores = score_response(case,final.get("raw"),finish_reason=final.get("finish_reason"),
                                    http_status=final.get("http_status")) if answers else None
            rows.append({"id":case["id"], "version":version, "split":case["split"],
                         "family":case["family"], "expected_kind":expected_kind(case),
                         "has_ask_focus":bool(case.get("ask_focus")),
                         "state":final.get("state","unattempted"), "attempts":len(attempts),
                         "last_http_status":final.get("http_status"), "scores":scores})
    return rows


def counts(rows):
    answered = [r for r in rows if r["scores"] is not None]
    outcomes = [r["scores"] for r in answered]
    states = Counter(r["state"] for r in rows)
    result = {"eligible":len(rows), "attempts":sum(r["attempts"] for r in rows),
              "answered":len(answered), "missing_answer":len(rows)-len(answered),
              "unattempted":states["unattempted"], "http_error_without_answer":states["http_error"],
              "network_error_without_answer":states["network_error"],
              "started_without_answer":states["started"],
              "expected_ask_answered":sum(r["expected_kind"]=="ask" for r in answered),
              "focus_annotated_answered":sum(r["has_ask_focus"] for r in answered)}
    for key in ("completed","raw_json_valid","schema_valid","scene_valid","strict_exact",
                "semantic_correct","dangerous_motion","motion_on_nonaction","ask_type_correct",
                "ask_focus_correct"):
        result[key] = sum(s.get(key) is True for s in outcomes)
    result["answered_incorrect"] = result["answered"] - result["semantic_correct"]
    result["failures"] = dict(sorted(Counter(tag for s in outcomes for tag in s["failure_tags"]).items()))
    return result


def result_by_group(rows):
    overall = counts(rows)
    for field in ("expected_kind","family"):
        groups = defaultdict(list)
        for r in rows:
            groups[r[field]].append(r)
        label = "by_expected" if field=="expected_kind" else "by_family"
        overall[label] = {name:counts(group) for name,group in sorted(groups.items())}
    return overall


def paired(rows, first, second):
    pairs = defaultdict(dict)
    for r in rows:
        if r["version"] in (first,second):
            pairs[r["id"]][r["version"]] = r["scores"]
    common = {k:data for k,data in pairs.items() if data.get(first) is not None and data.get(second) is not None}
    result = {"first":first, "second":second, "eligible":len(pairs), "paired_answered":len(common),
              "unpaired_or_unanswered":len(pairs)-len(common)}
    for metric in ("semantic_correct","strict_exact","schema_valid","dangerous_motion"):
        first_yes = {k for k,d in common.items() if d[first][metric]}
        second_yes = {k for k,d in common.items() if d[second][metric]}
        result[metric] = {"first_count":len(first_yes),"second_count":len(second_yes),
                          "both_true":len(first_yes & second_yes),"both_false":len(common)-len(first_yes | second_yes),
                          "false_to_true_ids":sorted(second_yes-first_yes),"true_to_false_ids":sorted(first_yes-second_yes)}
    result["improved_ids"] = result["semantic_correct"]["false_to_true_ids"]
    result["regressed_ids"] = result["semantic_correct"]["true_to_false_ids"]
    return result


def summarize_local(cases, run):
    rows = run.get("rows",[])
    indexed = {r["id"]:r for r in rows}
    if len(rows)!=len(indexed) or set(indexed)-{c["id"] for c in cases}:
        raise ValueError("Unknown or duplicate case ID in local results")
    details = {}
    for split in ("dev","test"):
        bins = defaultdict(list)
        for c in cases:
            if c["split"] == split:
                bins[expected_kind(c)].append(c)
        details[split] = {}
        for kind, subset in bins.items():
            tested = [indexed[c["id"]]["scores"] for c in subset if c["id"] in indexed]
            details[split][kind] = {"eligible":len(subset),"tested":len(tested),
                                    "semantic_correct":sum(s["semantic_correct"] for s in tested)}
            if kind != "local":
                details[split][kind].update(
                    strict_exact=sum(s.get("strict_exact") is True for s in tested),
                    dangerous_motion=sum(s.get("dangerous_motion") is True for s in tested))
    return {"model_calls":run.get("model_calls",0), "tested":len(rows), "by_split_and_expected":details}


def report(cases, out):
    journal = out/"responses.json"
    raw = journal.read_bytes() if journal.is_file() else b""
    book = json.loads(raw) if raw else {}
    local = read_if_present(out/"local.json",{})
    if local and book and local.get("source_sha256") != book.get("dataset_sha256"):
        raise ValueError("Local and online records have different dataset hashes")
    names = list(book.get("prompts",{})) or ["baseline","candidate"]
    versions = {"dev":names,"test":book.get("test_versions") or names}
    rows = observations(cases,book,versions)
    summary = {"responses_sha256":hashlib.sha256(raw).hexdigest() if raw else None,
               "dataset_sha256":book.get("dataset_sha256",local.get("source_sha256")),
               "total_attempts":len(book.get("attempts",[])),
               "attempt_states":dict(Counter(a["state"] for a in book.get("attempts",[]))),
               "test_versions_locked":bool(book.get("test_versions")), "splits":{}, "paired":{},
               "local":summarize_local(cases,local),
               "definitions":{
                   "eligible":"Planner-routed cases in the split. An unlocked version is eligible, not necessarily scheduled.",
                   "answered":"HTTP 200 model attempts (including malformed, empty or truncated replies). Missing answers are reported separately.",
                   "strict_exact":"Identical JSON fields and values after parsing; reference wording is required for a clarification.",
                   "semantic_correct":"Valid goal and scene, with equivalent pick+place permitted; annotated clarification focus is a lexical check.",
                   "dangerous_motion":"Incorrect completed physical action proposal; does not indicate actual controller execution.",
                   "paired":"Compare only case IDs answered by both versions; retries do not create a case.",
                   "local":"Deterministic control routing and rules; independent of the model's raw JSON metrics."}}
    for split,names_for_split in versions.items():
        group = [r for r in rows if r["split"]==split]
        summary["splits"][split] = {v:result_by_group([r for r in group if r["version"]==v]) for v in names_for_split}
        summary["paired"][split] = {f"{v1}_vs_{v2}":paired(group,v1,v2)
                                    for v1,v2 in combinations(names_for_split,2)}
    out.mkdir(parents=True,exist_ok=True)
    save(out/"scored.json",{"responses_sha256":summary["responses_sha256"],"rows":rows})
    # Keep references and actual responses together for reviewing a particular Chinese command.
    evidence = {c["id"]:{**c,"model_answers":{}} for c in cases}
    lookup = {(r["id"],r["version"]):r["scores"] for r in rows if r["scores"] is not None}
    for attempt in book.get("attempts",[]):
        if attempt["state"]=="answered":
            key = attempt["id"],attempt["version"]
            evidence[key[0]]["model_answers"][key[1]] = {
                "raw":attempt.get("raw"),"finish_reason":attempt.get("finish_reason"),
                "scores":lookup[key]}
    for row in local.get("rows",[]):
        evidence[row["id"]]["local_result"] = row["output"] if "output" in row else None
    save(out/"by_case.json",{"dataset_sha256":summary["dataset_sha256"],"cases":list(evidence.values())})
    save(out/"summary.json",summary)
    return summary
