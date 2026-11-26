from concurrent.futures import ThreadPoolExecutor
from threading import Event
import time
import unittest
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from arm_lab.model_access import ModelAccess
from arm_lab.planner import Intent
from arm_lab.web import create_app


class WatchdogTests(unittest.TestCase):
    def test_expired_link_discards_a_slow_plan_before_motion(self):
        entered, finish = Event(), Event()
        access = ModelAccess()
        access.remaining["deepseek"] = 1

        def delayed_plan(*args, **kwargs):
            entered.set()
            if not finish.wait(5):
                raise TimeoutError("Test planner was never released")
            return Intent("move", "x", 10)

        with TestClient(create_app(access, playback_speed=0, watchdog_seconds=.25)) as client, \
                ThreadPoolExecutor(1) as worker, patch.object(access, "plan", side_effect=delayed_plan):
            before = client.get("/api/state").json()
            body = dict(id=str(uuid4()), watchdog_token=str(uuid4()), text="向前移动一厘米",
                        revision=before["revision"], issued_at_ms=int(time.time()*1000), use_deepseek=True)
            future = worker.submit(client.post, "/api/commands", json=body)
            try:
                self.assertTrue(entered.wait(2))
                deadline = time.monotonic()+3
                while time.monotonic()<deadline:
                    state = client.get("/api/state").json()
                    if state["watchdog_expired"]:
                        break
                    time.sleep(.02)
                self.assertTrue(state["busy"] and state["stopped"] and state["watchdog_expired"])
            finally:
                finish.set()
            receipt = future.result(timeout=4).json()
            self.assertEqual((receipt["status"], receipt["plan"]), ("unverified", []))
            self.assertEqual(receipt["message"].count("心跳超时"), 1)
            self.assertEqual(client.get("/api/state").json()["sim_seconds"], before["sim_seconds"])

    def test_remote_command_without_a_distinct_token_never_starts(self):
        with TestClient(create_app(playback_speed=0)) as client:
            key = str(uuid4())
            body = dict(id=key, text="向前移动一厘米", revision=0, issued_at_ms=int(time.time()*1000))
            self.assertEqual(client.post("/api/commands", json=body).status_code,422)
            self.assertEqual(client.post("/api/commands", json=dict(body,watchdog_token=key)).status_code,422)
            s=client.get("/api/state").json()
            self.assertEqual(s["revision"],0)
            self.assertFalse(s["busy"] or s["watchdog_active"])
