"""Run the paired 100-command benchmark. Local mode and summaries never contact a model."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
from pathlib import Path
import random
import time

import httpx

from arm_lab.flow import interaction_intent
from arm_lab.model_access import DEEPSEEK_ENDPOINT, campus_api_key, campus_payload, planning_task
from arm_lab.planner import actions, check_steps, local_intent, offline_reply, source_slot
from scripts.evaluate_prompt_trial import exclusive, save


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "configs/planner100.json"
OUT = ROOT / "results/safety/planner100"
RUNS = OUT / "responses.json"
BASE = ROOT / "configs/prompt_trial.json"
MAX_ATTEMPTS = 250


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def dataset():
    cases = json.loads(DATA.read_text(encoding="utf-8"))["cases"]
    if len(cases) != 100 or len({c["id"] for c in cases}) != 100:
        raise ValueError("Expected 100 unique case IDs")
    if Counter(c["split"] for c in cases) != {"dev":40, "test":60}:
        raise ValueError("Unexpected split")
    return cases


def prompt(name):
    if name == "baseline":
        return json.loads(BASE.read_text(encoding="utf-8"))["prompts"]["baseline"]
    if name not in ("candidate","revised","finalist"):
        raise ValueError("Unknown prompt")
    return (ROOT / f"configs/planner100_{name}.txt").read_text(encoding="utf-8").strip()


def rules_result(case):
    from arm_lab.planner_scoring import score_local, score_response
    scene = case["scene"]
    text = case["command"]
    directive = interaction_intent(text)
    if directive:
        if case["route"] == "local":
            return {"local":directive.kind}, score_local(case,directive.kind)
        result = {"action":"ask","question":directive.message}
        scores = score_response(case,json.dumps(result,ensure_ascii=False)) if directive.kind=="ask" else score_local(case,directive.kind)
        return {"local":directive.kind,"message":directive.message}, scores
    if case["route"] == "local":
        return {"local":None}, score_local(case,None)
    intent = offline_reply(scene["dialogue"],text) if scene.get("dialogue") else local_intent(text)
    issue = check_steps(intent,scene) if intent.kind != "ask" else None
    if issue and issue[0]=="rejected":
        scores = score_response(case,None,http_status=None)
        scores.update(failure_tags=["local_rejection"],safe_refusal=True)
        return {"local":"rejected","message":issue[1]},scores
    if intent.kind == "ask" or issue:
        body = {"action":"ask","question":intent.message if not issue else issue[1]}
    else:
        slot, _ = source_slot(scene)
        held = bool(scene.get("holding"))
        steps = []
        for step in actions(intent):
            row = {"action":step.kind}
            if step.kind=="transfer" and held:
                row = {"action":"place"}
            elif step.kind in ("transfer","pick"):
                row.update(object="red_1", **{"from":step.source or slot})
            if step.kind in ("transfer","place"):
                row["to"] = step.target
                held, slot = False, step.target
            elif step.kind=="pick":
                held, slot = True, None
            elif step.kind=="move":
                row.update(axis=step.axis,mm=step.amount)
            elif step.kind=="wrist":
                row.update(deg=step.amount)
            steps.append(row)
        body = {"steps":steps}
    return body,score_response(case,json.dumps(body,ensure_ascii=False))


def local(cases):
    rows = []
    for case in cases:
        body, scores = rules_result(case)
        rows.append({"id":case["id"],"split":case["split"],"family":case["family"],"route":case["route"],
                     "output":body,"scores":scores})
    save(OUT/"local.json", {"source_sha256":hashlib.sha256(DATA.read_bytes()).hexdigest(),
                            "model_calls":0, "rows":rows})
    print(f"Scored {len(rows)} local cases.")


def select_challenger(cases):
    """Freeze a blinded comparison before the first held-out request."""
    from arm_lab.planner_scoring import score_response
    with exclusive(OUT/"run.lock"):
        run = json.loads(RUNS.read_text(encoding="utf-8"))
        decision = OUT/"selection.json"
        if decision.exists():
            return json.loads(decision.read_text(encoding="utf-8"))["challenger"]
        if run.get("test_versions"):
            raise ValueError("Cannot create a new selection after test has begun")
        versions = ("baseline","candidate","revised","finalist")
        intended = {v:(24 if v=="finalist" else 36) for v in versions}
        data = {c["id"]:c for c in cases if c["split"]=="dev"}
        answers = {v:{r["id"]:r for r in run["attempts"] if r["split"]=="dev"
                      and r["version"]==v and r["state"]=="answered"} for v in versions}
        if any(len(answers[v])!=intended[v] for v in versions):
            raise ValueError("Missing developer-panel responses")
        common = sorted(set.intersection(*(set(by_id) for by_id in answers.values())))
        if len(common)!=24:
            raise ValueError("The four versions did not see the same 24-case panel")
        scores = {}
        for v in versions:
            rows = [(data[k],score_response(data[k],answers[v][k]["raw"],finish_reason=answers[v][k]["finish_reason"]))
                    for k in common]
            scores[v] = {"correct_action_json":sum(s["semantic_correct"] for c,s in rows if "steps" in c["expected"]),
                         "correct_all":sum(s["semantic_correct"] for _,s in rows),
                         "motion_on_nonaction":sum(s["motion_on_nonaction"] for _,s in rows),
                         "prompt_characters":len(run["prompts"][v])}
        def rank(v):
            r=scores[v]
            return (r["correct_action_json"],r["correct_all"],-r["motion_on_nonaction"],-r["prompt_characters"])
        winner=max(versions[1:],key=rank)
        save(decision,{"frozen_utc":now(),"dataset_sha256":run["dataset_sha256"],"matched_dev_ids":common,
                       "action_count":18,"ask_count":6,"scores":scores,"challenger":winner,
                       "dev_best_including_baseline":max(versions,key=rank),"test_versions":["baseline",winner]})
        return winner


def delay_for(response, default):
    value = response.headers.get("Retry-After","").strip()
    if value.isdigit():
        return max(default,int(value))
    if value:
        try:
            return max(default,parsedate_to_datetime(value).timestamp()-time.time())
        except (ValueError, TypeError, OverflowError):
            pass
    return default


def run_online(cases, args):
    if not 0 < args.approved_attempts <= MAX_ATTEMPTS:
        raise ValueError("Supply the authorized cumulative attempt limit")
    key = campus_api_key()
    if not key:
        raise ValueError("Campus credential unavailable")
    signature = hashlib.sha256(DATA.read_bytes()).hexdigest()
    split = args.split
    variants = args.variants
    coverage = getattr(args,"coverage",False)
    selected = [c for c in cases if c["split"]==split and c["route"]=="planner"
                and interaction_intent(c["command"]) is None]
    if split=="dev" and "finalist" in variants and not coverage:
        if variants!=["finalist"]:
            raise ValueError("The supplementary candidate uses its fixed development panel")
        panel_rng = random.Random(27091)
        selected = (panel_rng.sample([c for c in selected if "steps" in c["expected"]],18)
                    + panel_rng.sample([c for c in selected if "question" in c["expected"]],6))
    random.Random(442 if split=="dev" else 817).shuffle(selected)
    planned = [(case,v) for i,case in enumerate(selected) for v in (variants if i%2==0 else variants[::-1])]
    with exclusive(OUT/"run.lock"), httpx.Client(timeout=httpx.Timeout(70,connect=15), follow_redirects=False) as client:
        book = json.loads(RUNS.read_text(encoding="utf-8")) if RUNS.exists() else {
            "dataset_sha256":signature,"prompts":{},"attempts":[],"temperature":0,"max_output_tokens":160,
            "model":"deepseek-chat","created_utc":now(),"next_request_after":0}
        if book["dataset_sha256"] != signature:
            raise ValueError("Dataset changed after first request")
        if coverage:
            choice = json.loads((OUT/"selection.json").read_text(encoding="utf-8"))
            if split!="dev" or variants!=[choice["challenger"]] or book.get("test_versions")!=choice["test_versions"]:
                raise ValueError("Coverage can only complete the already-frozen final prompt")
            required={(c["id"],v) for c in cases if c["route"]=="planner" and c["split"]=="test"
                      for v in book["test_versions"]}
            answered={(r["id"],r["version"]) for r in book["attempts"] if r["state"]=="answered"}
            if not required.issubset(answered):
                raise ValueError("Finish paired held-out responses before the optional coverage pass")
        elif book.get("test_versions") and (split=="dev" or book["test_versions"]!=variants):
            raise ValueError("Held-out evaluation is already locked")
        for v in variants:
            body = prompt(v)
            if v in book["prompts"] and book["prompts"][v]!=body:
                raise ValueError("A versioned prompt changed during evaluation")
            book["prompts"][v] = body
        if split=="test":
            chosen = OUT/"selection.json"
            if chosen.is_file() and json.loads(chosen.read_text(encoding="utf-8"))["test_versions"]!=variants:
                raise ValueError("Held-out variants differ from the frozen development selection")
            book["test_versions"] = variants
        save(RUNS,book)
        # Never automatically retry a request with an unknown outcome after a process exit.
        if any(r["state"]=="started" for r in book["attempts"]):
            raise ValueError("Interrupted request with unknown outcome; inspect journal")
        finished = {(r["id"],r["version"]) for r in book["attempts"] if r["state"]=="answered"}
        for case,v in planned:
            if (case["id"],v) in finished:
                continue
            payload = campus_payload(book["prompts"][v], planning_task(case["command"],case["scene"]))
            while len(book["attempts"]) < args.approved_attempts:
                wait = max(0,book["next_request_after"]-time.time())
                if wait:
                    time.sleep(wait)
                row = {"id":case["id"],"split":split,"version":v,"state":"started","utc":now()}
                if coverage:
                    row["phase"] = "postlock_coverage"
                book["attempts"].append(row)
                save(RUNS,book)
                started = time.monotonic()
                try:
                    response = client.post(DEEPSEEK_ENDPOINT,json=payload,headers={"Authorization":"Bearer "+key})
                    row["http_status"] = response.status_code
                    if response.status_code==200:
                        try:
                            answer = response.json()
                            choice = answer["choices"][0]
                            row.update(raw=choice["message"]["content"],finish_reason=choice.get("finish_reason"),
                                       usage={k:answer.get("usage",{}).get(k) for k in ("prompt_tokens","completion_tokens")})
                        except (ValueError,KeyError,IndexError,TypeError,AttributeError):
                            row.update(raw=None,finish_reason="malformed_response")
                        row["state"] = "answered"
                    else:
                        row["state"] = "http_error"
                except httpx.HTTPError as exc:
                    row.update(state="network_error",error_type=type(exc).__name__)
                row["seconds"] = round(time.monotonic()-started,3)
                book["next_request_after"] = time.time() + max(0,args.interval-row["seconds"])
                transient = row.get("http_status") in (429,502,503,504) or row["state"]=="network_error"
                if transient:
                    failures = 0
                    for last in reversed(book["attempts"]):
                        if last.get("http_status") not in (429,502,503,504) and last["state"]!="network_error":
                            break
                        failures += 1
                    default = min(120, 30 * 2**min(2,failures-1))
                    wait = default if row["state"]=="network_error" else delay_for(response,default)
                    book["next_request_after"] = time.time()+wait
                    row["retry_after_seconds"] = round(wait,2)
                    save(RUNS,book)
                    reason = row.get("http_status", row.get("error_type"))
                    print(f"{reason} on attempt {len(book['attempts'])}; backoff {wait:.0f}s",flush=True)
                    if failures>=3 or wait>300:
                        return "checkpoint_rate_limited" if reason==429 else "checkpoint_transient_error"
                    continue
                save(RUNS,book)
                if row["state"] != "answered":
                    return row["state"]
                print(f"{len(book['attempts'])} {split} {case['id']} {v} {row['finish_reason']}",flush=True)
                finished.add((case["id"],v))
                break
            else:
                return "budget_exhausted"
        return "complete"


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode",choices=("local","online","score","select"))
    cli.add_argument("--split",choices=("dev","test"),default="dev")
    cli.add_argument("--variants",nargs="+",choices=("baseline","candidate","revised","finalist"),default=["baseline","candidate"])
    cli.add_argument("--approved-attempts",type=int,default=0)
    cli.add_argument("--interval",type=float,default=6.2,help="Minimum seconds between start times")
    cli.add_argument("--coverage",action="store_true",help="After the blind test, complete missing dev rows without tuning")
    args=cli.parse_args()
    if args.interval<0 or len(set(args.variants))!=len(args.variants):
        cli.error("Invalid interval or duplicate variant")
    cases=dataset()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.mode=="local":
        local(cases)
    elif args.mode=="select":
        print(select_challenger(cases))
    elif args.mode=="score":
        from scripts.score_planner100 import report
        report(cases,OUT)
    else:
        status=run_online(cases,args)
        print(status,flush=True)
        if status!="complete":
            raise SystemExit(2)


if __name__=="__main__":
    main()
