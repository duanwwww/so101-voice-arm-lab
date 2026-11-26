"""The normal console uses a fixed planner without per-request approval."""
import time
import unittest
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from arm_lab.model_access import ModelAccess
from arm_lab.planner import Intent
from arm_lab.web import create_app


class SimpleConsoleTests(unittest.TestCase):
    def envelope(self, client, text, **extra):
        state = client.get("/api/state").json()
        return {"id":str(uuid4()), "watchdog_token":str(uuid4()), "text":text,
                "revision":state["revision"], "issued_at_ms":time.time_ns()//1_000_000, **extra}

    def test_default_reasoner_is_used_even_when_legacy_client_requests_rules(self):
        with patch.object(ModelAccess, "plan", return_value=Intent("move","y",2)) as planner, \
             TestClient(create_app(playback_speed=0), base_url="http://localhost",
                        client=("127.0.0.1",1234)) as client:
            state = client.get("/api/state").json()
            self.assertEqual(state["planner_model"], "deepseek-reasoner")
            self.assertNotIn("quota", state)
            self.assertNotIn("local_approval", state)
            body = self.envelope(client,"向左移动两毫米",use_deepseek=False)
            response = client.post("/api/commands",json=body)
            self.assertEqual(response.json()["status"], "success",response.text)
            self.assertEqual(response.json()["planner"],"deepseek-reasoner")
            self.assertTrue(planner.call_args.kwargs["approved_once"])
            self.assertEqual(client.post("/api/commands",json=body).json(),response.json())
            self.assertEqual(planner.call_count,1)

    def test_same_origin_remote_user_can_transcribe_and_plan_without_approval(self):
        access = ModelAccess(planner_model="deepseek-reasoner")
        origin = "http://arm.local:8769"
        with TestClient(create_app(access,playback_speed=0),base_url=origin,
                        client=("192.0.2.24",1234)) as client, \
             patch.object(access,"transcribe",return_value="向左移动两毫米") as asr, \
             patch.object(access,"plan",return_value=Intent("move","y",2)) as planner:
            headers={"Origin":origin}
            transcript=client.post("/api/transcribe",headers=headers,
                                   files={"file":("test.wav",b"audio-fixture","audio/wav")})
            self.assertEqual(transcript.status_code,200)
            asr.assert_called_once_with(b"audio-fixture",approved_once=True)
            result=client.post("/api/commands",headers=headers,
                               json=self.envelope(client,transcript.json()["text"]))
            self.assertEqual(result.json()["status"],"success",result.text)
            self.assertTrue(planner.call_args.kwargs["approved_once"])
            self.assertEqual(access.remaining,{"asr":0,"deepseek":0,"tts":0})

    def test_cross_origin_writes_never_call_a_model_or_change_scene(self):
        access=ModelAccess()
        with TestClient(create_app(access,playback_speed=0),base_url="http://localhost") as client, \
             patch.object(access,"plan") as planner, patch.object(access,"transcribe") as asr:
            before=client.get("/api/state").json()
            body=self.envelope(client,"向左移动两毫米")
            for headers in ({"Origin":"https://example.invalid"}, {"Origin":"null"},
                            {"Origin":"http://["}, {"Sec-Fetch-Site":"cross-site"}):
                with self.subTest(headers=headers):
                    self.assertEqual(client.post("/api/commands",headers=headers,json=body).status_code,403)
                    self.assertEqual(client.post("/api/transcribe",headers=headers,
                        files={"file":("a.wav",b"fixture","audio/wav")}).status_code,403)
                    self.assertEqual(client.post("/api/reset",headers=headers,json={"layout":"B"}).status_code,403)
            planner.assert_not_called()
            asr.assert_not_called()
            after=client.get("/api/state").json()
            self.assertEqual((before["revision"],before["layout"],before["sim_seconds"]),
                             (after["revision"],after["layout"],after["sim_seconds"]))

    def test_local_control_and_invalid_audio_do_not_call_models(self):
        access=ModelAccess()
        with TestClient(create_app(access,playback_speed=0)) as client, \
             patch.object(access,"plan") as planner, patch.object(access,"transcribe") as asr:
            for phrase,status in (("不要向前移动一厘米","ignored"),("向前移动五","clarify"),
                                  ("现在在做什么","observed")):
                response=client.post("/api/commands",json=self.envelope(client,phrase))
                self.assertEqual(response.json()["status"],status)
            self.assertEqual(client.post("/api/transcribe",
                files={"file":("empty.wav",b"","audio/wav")}).status_code,400)
            planner.assert_not_called()
            asr.assert_not_called()
            self.assertEqual(client.post("/api/speak").status_code,404)
            self.assertEqual(client.post("/api/feedback").status_code,404)

    def test_missing_key_reports_model_error_instead_of_running_local_rules(self):
        with patch.dict("os.environ",{},clear=True), \
             patch("arm_lab.model_access.httpx.Client.post",side_effect=AssertionError("unexpected cloud call")), \
             TestClient(create_app(playback_speed=0)) as client:
            before=client.get("/api/state").json()
            result=client.post("/api/commands",json=self.envelope(client,"向左移动两毫米")).json()
            self.assertEqual(result["status"],"model_error",result)
            self.assertEqual(result["planner"],"deepseek-reasoner")
            after=client.get("/api/state").json()
            self.assertEqual((before["tcp_mm"],before["sim_seconds"]),(after["tcp_mm"],after["sim_seconds"]))


if __name__=="__main__":
    unittest.main()
