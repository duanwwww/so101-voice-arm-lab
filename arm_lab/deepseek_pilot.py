"""Two fixed text-only requests, isolated from the general-purpose web console."""
import json
from pathlib import Path

from .model_access import DEEPSEEK_ENDPOINT, DEEPSEEK_MODEL, PLAN_MAX_TOKENS, ModelAccess, ModelDenied, ModelFailure, campus_api_key


def run_pilot(out, access=None):
    out = Path(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    report = {"provider":"上海交通大学 · 致远一号","endpoint":DEEPSEEK_ENDPOINT,
              "model":DEEPSEEK_MODEL,"limit":2,"max_output_tokens_per_call":PLAN_MAX_TOKENS,
              "inputs":"自建中文口令和 A/B 工位摘要","attempts":0,"cases":[]}

    def save(status):
        report["status"] = status
        out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
        return report

    try:
        key_present = bool(campus_api_key())
    except ModelFailure as exc:
        report["detail"] = str(exc)
        return save("configuration_error")
    if not key_present:
        return save("missing_credential")
    if access is None:
        access = ModelAccess()
        try:
            access.authorize("deepseek",2,f"向上海交大 {DEEPSEEK_MODEL}（{DEEPSEEK_ENDPOINT}）发送两条固定口令和 A/B 工位摘要。"
                             f"合计模型输入少于 2000 个汉字，每次最多输出 {PLAN_MAX_TOKENS} tokens，使用校方分配的配额。")
        except ModelDenied:
            return save("not_approved")
    if access.remaining["deepseek"] != 2 or access.used["deepseek"]:
        return save("not_approved")
    from .flow import execute
    from .simulator import Simulator
    sim = Simulator()
    try:
        for slot, text in [("A","把红色木块放到 C 区"),("B","请把红块送到C区")]:
            sim.reset(slot)
            try:
                receipt = execute(sim,text,planner=lambda command,scene:access.plan(command,scene,station_only=True))
                report["cases"].append({"initial":slot,"receipt":receipt})
                if receipt["status"] != "success" or receipt["planner"] != DEEPSEEK_MODEL:
                    break
            except (ModelFailure,ModelDenied) as exc:
                report["cases"].append({"initial":slot,"error":str(exc)})
                break
    finally:
        sim.close()
    report["attempts"] = access.used["deepseek"]
    ok = len(report["cases"])==2 and all(x.get("receipt",{}).get("status")=="success"
               and x["receipt"]["planner"]==DEEPSEEK_MODEL for x in report["cases"])
    return save("passed" if ok else "failed")
