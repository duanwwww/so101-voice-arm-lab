import time
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
import numpy as np

from arm_lab.model_access import ModelAccess, ModelDenied
from arm_lab.planner import decode_model_plan, local_intent, offline_reply
from arm_lab.vision import detect
from arm_lab.scene import STATIONS
from arm_lab.simulator import Simulator
from arm_lab.web import create_app


class ParsingTests(unittest.TestCase):
    def test_station_answer_replaces_unresolved_label_without_dropping_other_actions(self):
        question = "要把木块放到哪个工位？请回复 A、B 或 C。"
        for text, source in (("把红色木块放到右干的工位。",""),
                             ("从 A 区把木块搬到右面的未知工位。","A"),
                             ("把木块搬过去。","")):
            with self.subTest(text=text):
                intent = offline_reply({"command":text,"question":question},"B")
                self.assertEqual((intent.kind,intent.source,intent.target),("transfer",source,"B"))
        for text in ("把木块放到右干的工位，再向左移动五毫米。",
                     "把木块放到右边再抓工位。",
                     "不要把木块放到右干的工位。",
                     "把木块放到右边五厘米的工位。",
                     "给这个积木换个家。"):
            with self.subTest(text=text):
                self.assertEqual(offline_reply({"command":text,"question":question},"B").kind,"ask")

    def test_cloud_pilot_without_credential_makes_zero_requests(self):
        from arm_lab.deepseek_pilot import run_pilot
        with TemporaryDirectory() as folder, patch.dict("os.environ",{"SJTU_DEEPSEEK_API_KEY":"","DASHSCOPE_API_KEY":"other-provider"}, clear=True), \
                patch("arm_lab.model_access.httpx.Client",side_effect=AssertionError("network used")):
            result = run_pilot(Path(folder)/"pilot.json")
            self.assertEqual((result["status"], result["attempts"]),("missing_credential",0))

    def test_numeric_units_and_directions(self):
        for text, axis, value in [("向前移动五厘米","x",50),("再往右半厘米","y",-5),
                                  ("向下移十五毫米","z",-15),("向左１.５厘米","y",15)]:
            with self.subTest(text=text):
                intent = local_intent(text)
                self.assertEqual((intent.kind,intent.axis,intent.amount),("move",axis,value))
        self.assertEqual(local_intent("腕部向上转二点五度").amount,-2.5)

    def test_ambiguous_and_negated_commands(self):
        for text in ("向下几度","不要把红块放到C","", "向后移五"):
            with self.subTest(text=text):
                self.assertEqual(local_intent(text).kind,"ask")
        self.assertEqual(local_intent("向前五厘米再向左五厘米").kind,"sequence")
        self.assertEqual(local_intent("向前抓取木块").kind,"pick")

    def test_model_plan_is_grounded_and_has_only_known_actions(self):
        scene = {"objects":[{"slot":"B"}]}
        plan = {"action":"transfer","object":"red_1","from":"B","to":"C"}
        self.assertEqual(decode_model_plan(plan,scene).kind,"transfer")
        for bad in ({**plan,"from":"A"},{**plan,"to":"X"},{**plan,"axis":"x"},{"action":"execute_code"}):
            with self.assertRaises(ValueError):
                decode_model_plan(bad,scene)

    def test_model_denied_before_inference(self):
        access = ModelAccess()
        with patch("arm_lab.model_access.httpx.Client",side_effect=AssertionError("network used")):
            with self.assertRaises(ModelDenied):
                access.plan("把红块放到C",{"objects":[]})
            with self.assertRaises(ModelDenied):
                access.transcribe(b"fake-audio")
        self.assertEqual(sum(access.used.values()),0)

    def test_vision_reports_two_candidates_and_outside(self):
        frame=np.zeros((80,120,3),dtype=np.uint8)
        frame[15:25,15:25,0] = 245
        frame[40:50,80:90,0] = 245
        self.assertEqual([o["slot"] for o in detect(frame,{"A":(5,5,35,35)})["objects"]],["A",None])


class WebTests(unittest.TestCase):
    def test_projected_xy_for_all_stations_and_held_block(self):
        sim=Simulator()
        self.addCleanup(sim.close)
        for station,xy in STATIONS.items():
            sim.reset(station)
            scene,_=sim.camera()
            obj=scene["objects"][0]
            self.assertEqual(obj["slot"],station)
            np.testing.assert_allclose(obj["xy_mm"],np.array(xy)*1000,atol=2)
        sim.pick("C", lambda phase:None)
        held,_ = sim.camera()
        self.assertTrue(held["holding"])
        self.assertTrue(all("xy_mm" not in item for item in held["objects"]))

    def test_camera_command_dedup_stale_and_approval(self):
        access=ModelAccess()
        with TestClient(create_app(access,playback_speed=0)) as c:
            self.assertEqual(c.get("/").status_code,200)
            state=c.get("/api/state").json()
            self.assertEqual(state["quota"], {"asr":0,"deepseek":0,"tts":0})
            self.assertEqual(state["scene"]["objects"][0]["slot"],"A")
            self.assertTrue(c.get("/camera.jpg").content.startswith(b"\xff\xd8"))
            body={"id":str(uuid4()),"watchdog_token":str(uuid4()),"text":"向前移动五厘米","revision":state["revision"],"use_qwen":False,
                  "issued_at_ms":int(time.time()*1000)}
            first=c.post("/api/commands",json=body)
            self.assertEqual(first.json()["status"],"success",first.text)
            self.assertLess(first.json()["measurement"]["error_mm"],5)
            self.assertEqual(c.post("/api/commands",json=body).json(),first.json())
            self.assertEqual(c.post("/api/commands",json={**body,"text":"向左五厘米"}).status_code,409)
            self.assertEqual(c.post("/api/commands",json={**body,"id":str(uuid4())}).status_code,409)
            fresh={**body,"id":str(uuid4()),"revision":first.json()["revision"]}
            self.assertEqual(c.post("/api/commands",json={**fresh,"issued_at_ms":1}).status_code,409)
            self.assertEqual(c.post("/api/commands",json={**fresh,"use_qwen":True}).status_code,422)
            self.assertEqual(c.post("/api/commands",json={**fresh,"use_deepseek":True}).status_code,403)
            self.assertEqual(c.post("/api/transcribe",files={"file":("a.wav",b"sample","audio/wav")}).status_code,403)
            self.assertEqual(c.post("/api/speak").status_code,403)
            self.assertEqual(sum(access.used.values()),0)
            self.assertEqual(c.post("/api/reset",json={"layout":"B"}).status_code,200)
            self.assertEqual(c.get("/api/state").json()["scene"]["objects"][0]["slot"],"B")


if __name__=="__main__":
    unittest.main()
