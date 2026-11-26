"""One real local ASR request, a local clarification, and an explicit test-supplied answer."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch
from uuid import uuid4

import cv2
from fastapi.testclient import TestClient
import numpy as np

from arm_lab.flow import execute
from arm_lab.model_access import ModelAccess
from arm_lab.web import create_app
from scripts.evaluate_prompt_trial import exclusive, save
from scripts.evaluate_sensevoice_flow import OUT, ROOT, physical_snapshot


def run(out):
    if out.exists():
        raise ValueError("Preserve the existing result; choose another output for an authorized repeat.")
    wav = ROOT/"results/safety/prompt_trial_audio/transfer.wav"
    source = json.loads((ROOT/"results/safety/prompt_trial_real.json").read_bytes())
    blob = wav.read_bytes()
    if hashlib.sha256(blob).hexdigest()!=source["signature"]["wave_sha256"]["transfer"]:
        raise ValueError("Synthetic source differs from the earlier experiment")
    report = {"mode":"real_local_sensevoice_then_local_rules_with_explicit_clarification",
              "audio_source":"Existing Windows offline TTS recording, transfer.wav",
              "reference":"把红色木块放到右边的工位", "audio_sha256":hashlib.sha256(blob).hexdigest(),
              "clarification_answer":"B", "clarification_source":"supplied_reference_clarification",
              "clarification_note":"Test-supplied simulated human confirmation from the reference goal; not automatically inferred or counted as unedited ASR correctness.",
              "scene_resets":0, "cloud_calls":0, "turns":[], "status":"started"}
    out.parent.mkdir(parents=True,exist_ok=True)
    frames=out.with_suffix(".frames")
    frames.mkdir(exist_ok=True)
    persist=lambda:save(out,report)
    access=ModelAccess()
    access.asr_engine="sensevoice"
    access.remaining["asr"]=1
    persist()
    with exclusive(out.with_suffix(".lock")), ExitStack() as stack:
        app=create_app(access,layout="A",playback_speed=0)
        client=stack.enter_context(TestClient(app,base_url="http://127.0.0.1",client=("127.0.0.1",55555)))
        worker=stack.enter_context(ThreadPoolExecutor(1))
        report["before"]=physical_snapshot(app.state.sim)
        started=time.perf_counter()
        response=client.post("/api/transcribe",files={"file":(wav.name,blob,"audio/wav")})
        report["asr"]={"http_status":response.status_code,"response":response.json(),
                       "seconds_including_lazy_model_load":round(time.perf_counter()-started,3)}
        persist()
        if response.status_code!=200:
            report["status"]="asr_failed"
            persist()
            return report
        text=response.json()["text"]
        reply_to=None
        for index,command in enumerate((text,"B")):
            state=client.get("/api/state").json()
            command_id,token=str(uuid4()),str(uuid4())
            body={"id":command_id,"text":command,"revision":state["revision"],
                  "issued_at_ms":int(time.time()*1000),"watchdog_token":token,"use_deepseek":False}
            if reply_to:
                body["reply_to"]=reply_to
            row={"turn":index+1,"command":body,"frames":[],"before":physical_snapshot(app.state.sim),
                 "source":"unedited_new_asr" if index==0 else "test_supplied_human_confirmation",
                 "successful_monitor_renewals":0}
            report["turns"].append(row)
            def monitored_execute(sim,text,**kwargs):
                previous=kwargs.get("observer")
                def observe(phase,scene,frame):
                    if previous:
                        previous(phase,scene,frame)
                    if phase in ("before","after","lift") or phase.startswith("verify_"):
                        name=f"turn{index+1}-{phase}.jpg"
                        cv2.imwrite(str(frames/name),frame)
                        row["frames"].append({"phase":phase,"image":name,"scene":scene,
                                              "physics":physical_snapshot(sim)})
                kwargs["observer"]=observe
                return execute(sim,text,**kwargs)
            persist()
            with patch("arm_lab.web.execute",side_effect=monitored_execute):
                pending=worker.submit(client.post,"/api/commands",json=body)
                while not pending.done():
                    fresh=client.get("/api/state")
                    image=client.get("/camera.jpg")
                    if fresh.status_code==200 and image.status_code==200:
                        decoded=cv2.imdecode(np.frombuffer(image.content,np.uint8),cv2.IMREAD_COLOR)
                        if decoded is not None:
                            beat=client.post("/api/heartbeat",json={"id":command_id,"watchdog_token":token})
                            row["successful_monitor_renewals"]+=beat.status_code==200
                    time.sleep(.1)
                response=pending.result()
            row.update(http_status=response.status_code,receipt=response.json(),
                       after=physical_snapshot(app.state.sim),state_after=client.get("/api/state").json())
            persist()
            if index==0:
                if row["receipt"].get("status")!="clarify" or not row["receipt"].get("clarification"):
                    report["status"]="expected_clarification_missing"
                    break
                reply_to=row["receipt"]["id"]
        turns=report["turns"]
        first=turns[0]
        no_motion=(first["before"]["block"]["time_s"]==first["after"]["block"]["time_s"] and
                   np.allclose(first["before"]["joints"],first["after"]["joints"],atol=1e-10,rtol=0))
        verified=[]
        if len(turns)==2:
            verified=[frame["physics"]["block"] for frame in turns[1]["frames"]
                      if frame["phase"].startswith("verify_")]
        checks={"asked_before_motion":first["receipt"].get("status")=="clarify" and no_motion,
                "reply_linked_to_question":len(turns)==2 and turns[1]["command"].get("reply_to")==first["receipt"]["id"],
                "no_scene_reset":report["scene_resets"]==0,
                "successful_second_turn":len(turns)==2 and turns[1]["receipt"].get("status")=="success",
                "physical_B_on_desk_three_times":len(verified)==3 and all(
                    block["inside"]=="B" and "desk" in block["contacts"] for block in verified)}
        report.update(status="complete",checks=checks,passed=all(checks.values()),
                      raw_asr_first_goal_achieved=first["receipt"].get("status")=="success"
                          and first["after"]["block"]["inside"]=="B",
                      assisted_recovery_completed=all(checks.values()),
                      asr_calls=access.used["asr"],after=physical_snapshot(app.state.sim))
        persist()
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approve-local-asr",action="store_true")
    parser.add_argument("--out",type=Path,default=OUT/"recovery.json")
    args=parser.parse_args()
    if not args.approve_local_asr:
        parser.error("Approve one local ASR call for the existing synthetic transfer recording.")
    result=run(args.out)
    print(json.dumps({k:result.get(k) for k in ("status","asr_calls","cloud_calls","checks","passed")},ensure_ascii=False))
