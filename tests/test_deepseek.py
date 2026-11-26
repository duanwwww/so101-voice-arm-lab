"""SJTU request contract and simulated motion. HTTP responses are generated in memory."""
import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from httpx import Client, MockTransport, Response
from fastapi.testclient import TestClient
import numpy as np

from arm_lab.deepseek_pilot import run_pilot
from arm_lab.flow import execute
from arm_lab.model_access import ModelAccess, ModelFailure, campus_api_key
from arm_lab.simulator import Simulator
from arm_lab.web import create_app


def mock_campus(handler):
    return patch(
        "arm_lab.model_access.httpx.Client",
        side_effect=lambda **options: Client(transport=MockTransport(handler), **options),
    )


def completion(plan, finish="stop"):
    return Response(200, json={"choices": [{
        "finish_reason": finish, "message": {"content": json.dumps(plan)},
    }]})


class CampusPlannerTests(unittest.TestCase):
    def setUp(self):
        self.access = ModelAccess()
        self.access.remaining["deepseek"] = 2
        env = patch.dict("os.environ", {"SJTU_DEEPSEEK_API_KEY": "test-only-campus-key"})
        env.start()
        self.addCleanup(env.stop)

    def test_two_slot_contract_through_simulation(self):
        seen_slots = []
        input_lengths = []

        def reply(request):
            self.assertEqual(request.method, "POST")
            self.assertEqual(str(request.url), "https://models.sjtu.edu.cn/api/v1/chat/completions")
            self.assertEqual(request.headers["Authorization"], "Bearer test-only-campus-key")
            payload = json.loads(request.content)
            self.assertEqual(set(payload), {"model","temperature","max_tokens","stream","messages"})
            self.assertEqual((payload["model"], payload["max_tokens"], payload["stream"]), ("deepseek-chat",160,False))
            messages = payload["messages"]
            self.assertEqual([m["role"] for m in messages], ["system","user"])
            user = json.loads(messages[1]["content"])
            self.assertEqual(set(user), {"command","objects"})
            slot = ("A","B")[len(seen_slots)]
            self.assertEqual(user["objects"], [{"id":"red_1","slot":slot}])
            self.assertIn("红", user["command"])
            input_lengths.append(sum(len(m["content"]) for m in messages))
            seen_slots.append(slot)
            return completion({"action":"transfer","object":"red_1","from":slot,"to":"C"})

        with TemporaryDirectory() as folder, mock_campus(reply):
            report = run_pilot(Path(folder)/"pilot.json", self.access)
        self.assertEqual(seen_slots, ["A","B"])
        self.assertLess(sum(input_lengths), 2000)
        self.assertEqual((report["status"], report["attempts"]), ("passed", 2))
        for case in report["cases"]:
            receipt = case["receipt"]
            self.assertEqual(receipt["planner"], "deepseek-chat")
            self.assertEqual(receipt["plan"][0], "pick("+case["initial"]+")")
            self.assertEqual(receipt["after"]["objects"][0]["slot"], "C")

    def test_conflicting_plan_stops_and_explicit_offline_metric_stays_local(self):
        sim = Simulator("B")
        self.addCleanup(sim.close)
        initial = sim.data.qpos.copy()
        with mock_campus(lambda req: completion({"action":"transfer","object":"red_1","from":"A","to":"C"})) as client:
            with self.assertRaises(ModelFailure):
                execute(sim, "把红块放到C", planner=self.access.plan)
            np.testing.assert_array_equal(sim.data.qpos, initial)
            receipt = execute(sim, "向前移动五厘米")
            self.assertEqual((receipt["planner"], receipt["status"]), ("rules","success"))
            self.assertEqual(client.call_count, 1)

    def test_http_failures_and_truncation_stop_the_pilot(self):
        cases = [
            Response(307, headers={"Location":"https://example.invalid/elsewhere"}),
            Response(401), Response(429),
            completion({"action":"ask","question":"请确认。"}, finish="length"),
            Response(200, json={"choices":[{"finish_reason":"stop","message":{"content":"bad json"}}]}),
        ]
        for response in cases:
            with self.subTest(response=response), TemporaryDirectory() as folder:
                destinations = []
                def reply(request):
                    destinations.append(request.url.host)
                    return response
                with mock_campus(reply):
                    access = ModelAccess()
                    access.remaining["deepseek"] = 2
                    report = run_pilot(Path(folder)/"pilot.json", access)
                    self.assertEqual((report["attempts"], report["status"]), (1, "failed"))
                    self.assertEqual(destinations, ["models.sjtu.edu.cn"])
                    self.assertEqual(len(report["cases"]), 1)

    def test_preflight_does_not_spend_allowance(self):
        with patch.dict("os.environ", {"SJTU_DEEPSEEK_API_KEY":"   ", "DEEPSEEK_API_KEY":"other-key"}, clear=True), \
                mock_campus(lambda request: self.fail("unexpected campus request")):
            with self.assertRaises(ModelFailure):
                self.access.plan("把红块放到C", {"objects":[{"slot":"A"}]})
            self.assertEqual((self.access.remaining["deepseek"], self.access.used["deepseek"]), (2, 0))

    def test_existing_course_file_binds_key_to_campus(self):
        with TemporaryDirectory() as folder:
            cfg = Path(folder) / ".env.worker"
            with patch.dict("os.environ", {"SJTU_DEEPSEEK_ENV_FILE":str(cfg)}, clear=True), \
                    mock_campus(lambda request: self.fail("unexpected campus request")):
                cfg.write_text("SJTU_API_KEY='test-key'\nSJTU_API_BASE_URL=https://models.sjtu.edu.cn/api/v1/\n"
                               "SJTU_MODEL=deepseek-reasoner\n", encoding="utf-8-sig")
                self.assertEqual(campus_api_key(), "test-key")
                cfg.write_text("SJTU_API_KEY=test-key\nSJTU_API_BASE_URL=https://api.deepseek.com\n", encoding="utf-8")
                result = run_pilot(Path(folder)/"result.json", self.access)
                self.assertEqual((result["status"], result["attempts"]), ("configuration_error", 0))
                cfg.unlink()
                with self.assertRaises(ModelFailure):
                    campus_api_key()

    def test_voice_custom_command_feedback_and_consent(self):
        phrase = "让眼前的小家伙在左侧格子安家，随后把腕部轻轻抬高两度"
        requests = []
        def reply(request):
            payload = json.loads(request.content)
            task = json.loads(payload["messages"][1]["content"])
            requests.append(task)
            if "status" in task:
                self.assertEqual((task["status"],task["slots"]), ("success",["A"]))
                return completion({"speech":"木块已经到达左侧工位，手腕已抬高。"})
            self.assertEqual(task["command"],phrase)
            self.assertEqual(len(task["objects"]),1)
            obj=task["objects"][0]
            self.assertEqual((obj["id"],obj["slot"]),("red_1","B"))
            np.testing.assert_allclose(obj["xy_mm"],[215,-77],atol=2)
            return completion({"steps":[{"action":"transfer","from":"B","object":"red_1","to":"A"},
                                        {"action":"wrist","deg":-2}]})
        access = ModelAccess()
        origin = "http://127.0.0.1:8768"
        def permit(kind):
            return {"Origin":origin,"X-Arm-Approval":kind}
        with TestClient(create_app(access,"B",playback_speed=0),base_url=origin,client=("127.0.0.1",1234)) as c, \
                mock_campus(reply), patch.object(access,"transcribe",return_value=phrase) as asr, \
                patch.object(access,"speak",new_callable=AsyncMock,return_value=b"test-mp3") as tts:
            audio = {"file":("command.wav",b"synthetic-test-input","audio/wav")}
            transcript = c.post("/api/transcribe",headers=permit("asr"),files=audio).json()["text"]
            asr.assert_called_once_with(b"synthetic-test-input",approved_once=True)
            body = {"id":str(uuid4()),"text":transcript,"revision":0,"issued_at_ms":int(time.time()*1000),"use_deepseek":True}
            for headers in ({}, {"Origin":"https://example.invalid","X-Arm-Approval":"deepseek"}):
                self.assertEqual(c.post("/api/commands",json=body,headers=headers).status_code,403)
            self.assertEqual(requests, [])
            result = c.post("/api/commands",json=body,headers=permit("deepseek")).json()
            self.assertEqual(result["status"],"success",result)
            self.assertEqual(result["plan"],["pick(B)","place(A)","verify(A)","wrist_delta(-2 deg)"])
            self.assertEqual(result["after"]["objects"][0]["slot"],"A")
            self.assertEqual(c.post("/api/commands",json=body,headers=permit("deepseek")).json(),result)
            url = "?receipt_id="+result["id"]
            speech = c.post("/api/feedback"+url,headers=permit("deepseek")).json()["speech"]
            self.assertEqual(c.post("/api/speak"+url,headers=permit("tts")).content,b"test-mp3")
            tts.assert_awaited_once_with(result["message"]+" "+speech,approved_once=True)
            self.assertEqual((len(requests),access.used["deepseek"],access.remaining["deepseek"]),(2,2,0))

    def test_model_clarification_remembers_command_and_rechecks_scene(self):
        seen = []
        def reply(request):
            task = json.loads(json.loads(request.content)["messages"][1]["content"])
            seen.append(task)
            if "dialogue" not in task:
                return completion({"action":"ask","question":"送到哪个工位？"})
            return completion({"steps":[{"action":"transfer","object":"red_1","from":"C","to":"B"}]})
        with TestClient(create_app(self.access,"C",playback_speed=0)) as c, mock_campus(reply):
            def send(text, rev, prior=None):
                return c.post("/api/commands",json={"text":text,"revision":rev,"id":str(uuid4()),"watchdog_token":str(uuid4()),
                              "issued_at_ms":int(time.time()*1000),"use_deepseek":True,"reply_to":prior})
            first = send("给这个积木换个家",0).json()
            self.assertEqual((first["status"],first["plan"]),("clarify",[]))
            second = send("右手边的那个",first["revision"],first["id"]).json()
            self.assertEqual(second["status"],"success")
            self.assertEqual(seen[1]["dialogue"],{"command":"给这个积木换个家","question":"送到哪个工位？"})
            self.assertEqual(seen[1]["objects"][0]["slot"],"C")
            self.assertEqual(send("改去左侧",second["revision"],first["id"]).status_code,409)

    def test_unrelated_action_or_invalid_tail_never_moves_the_arm(self):
        sim = Simulator("B")
        self.addCleanup(sim.close)
        initial = sim.joints_now()
        for text, plan in [
                ("向左移动一厘米",{"steps":[{"action":"move","axis":"x","mm":10}]}),
                ("顺次做两个指定动作",{"steps":[{"action":"move","axis":"x","mm":10},{"action":"wrist","deg":55}]})]:
            with self.subTest(text=text), mock_campus(lambda req:completion(plan)):
                with self.assertRaises(ModelFailure):
                    execute(sim,text,planner=self.access.plan)
                np.testing.assert_array_equal(initial,sim.joints_now())
