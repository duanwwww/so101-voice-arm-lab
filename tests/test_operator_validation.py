from concurrent.futures import ThreadPoolExecutor
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from arm_lab.field_acceptance import FIELDS, score, template
from arm_lab.simulator import Simulator
from arm_lab.web import create_app
from arm_lab.wrist_view import confirm_drop, observe_wrist


class OperatorTests(unittest.TestCase):
    def test_inflight_stop_preempts_motion_until_reset(self):
        with TestClient(create_app(playback_speed=3),base_url="http://127.0.0.1",
                        client=("127.0.0.1",12345)) as client, ThreadPoolExecutor(1) as pool:
            def submit(text):
                revision=client.get("/api/state").json()["revision"]
                return client.post("/api/commands",json=dict(id=str(uuid4()),text=text,revision=revision,
                                                             issued_at_ms=int(time.time()*1000)))
            self.assertEqual(client.get("/camera/wrist.jpg").headers["content-type"],"image/jpeg")
            future=pool.submit(submit,"把木块放到 C 区")
            deadline=time.monotonic()+5
            while time.monotonic()<deadline:
                s=client.get("/api/state").json()
                if s["busy"] and s["phase"]=="moving":
                    break
                time.sleep(.01)
            else:
                self.fail("The command did not enter the live movement phase")
            self.assertTrue(client.post("/api/stop").json()["stopped"])
            receipt=future.result(timeout=8).json()
            self.assertEqual(receipt["status"],"unverified")
            self.assertNotIn("release",receipt["phases"])
            self.assertEqual(submit("向左移动一厘米").status_code,409)
            self.assertEqual(client.post("/api/reset",json={"layout":"A"}).status_code,200)
            self.assertFalse(client.get("/api/state").json()["stopped"])
            client.post("/api/stop")
            self.assertTrue(client.get("/api/state").json()["fault"])

    def test_wrist_fallback_requires_clear_independent_evidence(self):
        s=Simulator("A")
        self.addCleanup(s.close)
        for name in ("A","B","C"):
            s.reset(name)
            self.assertEqual([o["slot"] for o in observe_wrist(s)[0]],[name])
            self.assertEqual(confirm_drop(s,name,dict(holding=False,objects=[])),(True,"wrist"))
        with patch("arm_lab.wrist_view.observe_wrist",side_effect=AssertionError("unexpected fallback")):
            for scene in (dict(holding=True,objects=[]),dict(holding=False,objects=[dict(slot="B")]),
                          dict(holding=False,objects=[dict(slot="C"),dict(slot="C")])):
                self.assertEqual(confirm_drop(s,"C",scene),(False,"unconfirmed"))
        s.reset("outside")
        self.assertEqual(confirm_drop(s,"C",dict(holding=False,objects=[])),(False,"unconfirmed"))

    def test_manual_records_distinguish_unknown_false_success_and_numeric_error(self):
        with TemporaryDirectory() as folder:
            p=Path(folder)/"field.csv"
            template(p)
            self.assertEqual(score(p)["pending"],15)
            with self.assertRaises(FileExistsError):
                template(p)
            with p.open(encoding="utf-8-sig",newline="") as f:
                rows=list(csv.DictReader(f))
            rows[0].update(evidence="test fixture",actual_station="B",program_status="success")
            rows[1].update(evidence="test fixture",actual_station="unknown",program_status="unverified")
            rows[9].update(evidence="test fixture",dx_mm="10",dy_mm="3",dz_mm="4",program_status="success")
            with p.open("w",encoding="utf-8-sig",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            result=score(p)
            self.assertEqual((result["recorded"],result["physically_unknown"],result["false_success"]),(3,1,1))
            self.assertEqual(result["error"]["move"]["max"],5)
