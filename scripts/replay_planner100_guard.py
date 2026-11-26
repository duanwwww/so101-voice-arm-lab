"""Replay previously captured model choices through the production plan validator, offline."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from unittest.mock import patch

from httpx import Client, MockTransport, Response

from arm_lab.model_access import ModelAccess, ModelFailure, campus_payload, planning_task
from arm_lab.planner import actions
from arm_lab.planner_scoring import FIELDS, has_motion, score_response
from scripts.evaluate_planner100 import DATA, OUT
from scripts.evaluate_prompt_trial import save


def intent_document(intent):
    if intent.kind == "ask":
        return {"action":"ask","question":intent.message}
    if intent.kind in ("pause","resume","ignore","status"):
        return {"local":intent.kind}
    steps = []
    for step in actions(intent):
        row = {"action":step.kind}
        if step.kind in ("transfer","pick"):
            row.update(object="red_1",**{"from":step.source})
        if step.kind in ("transfer","place"):
            row["to"] = step.target
        if step.kind == "move":
            row.update(axis=step.axis,mm=step.amount)
        if step.kind == "wrist":
            row.update(deg=step.amount)
        steps.append(row)
    return {"steps":steps}


def compatibility_view(case, raw, finish_reason, strict):
    """Ignore only fences and a missing steps envelope when diagnosing physical meaning.

    Preserve the original string inside the wrapper so duplicate-key checks and numeric
    precision are unchanged. This diagnostic does not alter the strict model score.
    """
    if not isinstance(raw,str) or finish_reason != "stop":
        return strict, []
    text = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*([\s\S]+?)\s*```",text,re.I)
    changes = []
    if fenced:
        text = fenced[1]
        changes.append("markdown_fence")
    try:
        obj = json.loads(text)
    except (ValueError,RecursionError):
        return strict,changes
    if isinstance(obj,dict) and isinstance(obj.get("action"),str) and obj["action"] in FIELDS and "steps" not in obj:
        text = '{"steps":[' + text + ']}'
        changes.append("bare_action")
    return (score_response(case,text,finish_reason=finish_reason) if changes else strict), changes


def replay_choice(case, record, instructions):
    raw, finish = record.get("raw"), record.get("finish_reason")
    strict = score_response(case,raw,finish_reason=finish,http_status=record["http_status"])
    parsed, adjustments = compatibility_view(case,raw,finish,strict)
    expected_payload = campus_payload(instructions, planning_task(case["command"],case["scene"]))
    def respond(request):
        if json.loads(request.content) != expected_payload:
            raise AssertionError("Replay request differs from the frozen prompt, text or station observations")
        return Response(200,json={"choices":[{"message":{"content":raw},"finish_reason":finish}]})
    transport = MockTransport(respond)
    accepted = None
    reason = None
    access = ModelAccess()
    with patch("arm_lab.model_access.PLAN_INSTRUCTIONS",instructions), \
            patch("arm_lab.model_access.campus_api_key",return_value="in-memory-placeholder"), \
            patch("arm_lab.model_access.httpx.Client",side_effect=lambda **kw:Client(transport=transport,**kw)):
        try:
            accepted = intent_document(access.plan(case["command"],case["scene"],approved_once=True))
        except ModelFailure as exc:
            reason = str(exc)
    after = None
    if accepted is not None and "local" not in accepted:
        after = score_response(case,json.dumps(accepted,ensure_ascii=False,allow_nan=False))
    actual_motion = has_motion(accepted)
    guard_goal_match = bool(after and after["semantic_correct"])
    admitted_wrong = actual_motion and not guard_goal_match
    proposed_wrong = parsed["dangerous_motion"]
    outcome = ("rejected" if accepted is None else "admitted_motion" if actual_motion
               else "asked" if accepted.get("action")=="ask" else "nonmoving_result")
    return {"id":case["id"],"split":case["split"],"family":case["family"],"version":record["version"],
            "guard_outcome":outcome,"accepted_plan":accepted,"reason":reason,
            "finish_reason":finish,"mock_http_responses_used":access.used["deepseek"],
            "raw_strict_schema":strict["schema_valid"],"raw_semantic_correct":strict["semantic_correct"],
            "raw_failure_tags":strict["failure_tags"],"format_adjustments_for_diagnosis":adjustments,
            "correct_meaning_in_bare_action":"bare_action" in adjustments and parsed["semantic_correct"],
            "wrong_motion_proposed":proposed_wrong,
            "wrong_motion_blocked_or_asked":proposed_wrong and not actual_motion,
            "wrong_motion_still_admitted":proposed_wrong and admitted_wrong,
            "wrong_motion_changed_to_correct_motion":proposed_wrong and actual_motion and guard_goal_match,
            "any_wrong_motion_admitted":admitted_wrong,"guard_goal_match":guard_goal_match}


def counts(rows):
    keys = ("raw_semantic_correct","guard_goal_match","wrong_motion_proposed","wrong_motion_blocked_or_asked",
            "wrong_motion_still_admitted","wrong_motion_changed_to_correct_motion","any_wrong_motion_admitted",
            "correct_meaning_in_bare_action")
    values = {key:sum(r[key] for r in rows) for key in keys}
    values["replayed_answers"] = len(rows)
    values["outcomes"] = dict(Counter(r["guard_outcome"] for r in rows))
    values["wrong_motion_admitted_ids"] = [r["id"] for r in rows if r["any_wrong_motion_admitted"]]
    values["bare_correct_and_admitted"] = sum(r["correct_meaning_in_bare_action"] and r["guard_goal_match"] for r in rows)
    return values


def replay(cases, book):
    by_id = {c["id"]:c for c in cases}
    rows = []
    seen = set()
    for record in book.get("attempts",[]):
        if record["state"] != "answered":
            continue
        case = by_id[record["id"]]
        name = record["version"]
        key = case["id"],name
        if case["route"] != "planner" or case["split"] != record["split"] or key in seen or record.get("http_status")!=200:
            raise ValueError(f"Inconsistent answer identity {key}")
        seen.add(key)
        rows.append(replay_choice(case,record,book["prompts"][name]))
    summary = {}
    for split in ("dev","test"):
        versions = book.get("test_versions") if split == "test" else None
        summary[split] = {v:counts([r for r in rows if r["version"]==v and r["split"]==split])
                          for v in (versions or book.get("prompts",{}))}
    return {"online_model_calls":0,"robot_or_simulator_steps":0,"source_attempts":len(book.get("attempts",[])),
            "mode":"MockTransport replay through ModelAccess.plan. No live motion or geometric path checks.",
            "counts_definition":"A physical proposal is wrong when it fails the gold meaning after stripping optional Markdown fences and wrapping a bare action. Typed or range errors are retained. Unfinished replies have no completed physical proposal. Schema validity and raw model correctness always refer to the original reply.",
            "splits":summary,"rows":rows}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--directory",type=Path,default=OUT)
    args = cli.parse_args()
    source = (args.directory/"responses.json").read_bytes()
    book = json.loads(source)
    dataset = DATA.read_bytes()
    if book["dataset_sha256"] != hashlib.sha256(dataset).hexdigest():
        raise ValueError("Recorded dataset checksum differs")
    result = replay(json.loads(dataset)["cases"],book)
    result["dataset_sha256"] = book["dataset_sha256"]
    result["responses_sha256"] = hashlib.sha256(source).hexdigest()
    path = args.directory/"guard_replay.json"
    save(path,result)
    print(f"Wrote {len(result['rows'])} replayed answers to {path}")


if __name__ == "__main__":
    main()
