from concurrent.futures import ThreadPoolExecutor
from threading import Event
import time
import unittest
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
import numpy as np

from arm_lab.flow import execute
from arm_lab.model_access import ModelAccess
from arm_lab.planner import Intent
from arm_lab.simulator import Simulator
from arm_lab.web import create_app


class LiveInteractionTests(unittest.TestCase):
    def test_followups_during_planning_do_not_replace_the_active_lease_or_start_an_old_plan(self):
        entered, finish = Event(), Event()
        access = ModelAccess()
        access.remaining["deepseek"] = 1

        def delayed(*args, **kwargs):
            entered.set()
            if not finish.wait(5):
                raise TimeoutError("Test planner was not released")
            return Intent("move","x",10)

        with TestClient(create_app(access, layout="B", playback_speed=0, allow_local_rules=True)) as client, ThreadPoolExecutor(1) as worker, \
                patch.object(access, "plan", side_effect=delayed) as model:
            def state():
                return client.get("/api/state").json()
            def envelope(text, **extra):
                return dict(id=str(uuid4()),text=text,revision=state()["revision"],
                            issued_at_ms=time.time_ns()//1_000_000, **extra)
            def send(text, **extra):
                return client.post("/api/commands",json=envelope(text,**extra))
            original = envelope("向前移动一厘米", watchdog_token=str(uuid4()), use_deepseek=True)
            before = state()
            future = worker.submit(client.post,"/api/commands",json=original)
            try:
                self.assertTrue(entered.wait(2))
                active = state()
                for text, status in [("不要向前移动","ignored"), ("不要暂停","ignored"), ("现在在做什么","observed")]:
                    r = send(text,use_deepseek=True).json()
                    self.assertEqual((r["status"],r["plan"]),(status,[]))
                    self.assertEqual(state()["revision"],active["revision"])
                    self.assertIsNone(state()["last"])
                    self.assertFalse(state()["paused"])
                self.assertEqual(send("向后移动五毫米").status_code,409)
                self.assertEqual(client.post("/api/heartbeat",json={"id":original["id"],
                                       "watchdog_token":original["watchdog_token"]}).status_code,200)
                pause = envelope("暂停当前动作",use_deepseek=True)
                r = client.post("/api/commands",json=pause).json()
                self.assertEqual(r["status"],"paused")
                self.assertEqual(client.post("/api/commands",json=pause).json(),r)
                self.assertEqual(send("恢复控制").status_code,409)
            finally:
                finish.set()
            ended = future.result(timeout=5).json()
            self.assertEqual((ended["status"],ended["plan"]),("paused",[]))
            self.assertEqual(state()["tcp_mm"],before["tcp_mm"])
            self.assertEqual(send("向前移动五毫米").status_code,409)
            resume = envelope("恢復控制",use_deepseek=True)
            resumed = client.post("/api/commands",json=resume).json()
            self.assertEqual(resumed["status"],"resumed")
            self.assertFalse(state()["paused"])
            self.assertEqual(state()["sim_seconds"],before["sim_seconds"])
            self.assertEqual(state()["layout"],"B")
            self.assertEqual(send("向前移动五毫米",watchdog_token=str(uuid4())).json()["status"],"success")
            self.assertEqual(send("暂停").json()["status"],"paused")
            self.assertEqual(client.post("/api/commands",json=resume).json(),resumed)
            self.assertTrue(state()["paused"])
            client.post("/api/stop")
            self.assertEqual(send("恢复控制").status_code,409)
            self.assertTrue(state()["stopped"])
            self.assertEqual(model.call_count,1)

    def test_pause_holding_keeps_physical_scene_and_accepts_new_destination(self):
        sim = Simulator("B")
        self.addCleanup(sim.close)
        self.assertEqual(execute(sim,"向前抓取木块")["status"],"success")
        where, when = sim.data.qpos.copy(), sim.data.time
        self.assertEqual(execute(sim,"暂停当前动作")["status"],"paused")
        self.assertTrue(sim.holding())
        self.assertEqual(sim.jaw_mode,"hold")
        self.assertEqual(execute(sim,"解除暂停")["status"],"resumed")
        np.testing.assert_array_equal(sim.data.qpos,where)
        self.assertEqual(sim.data.time,when)
        placed = execute(sim,"放到 A 区")
        self.assertEqual(placed["status"],"success")
        self.assertEqual(placed["after"]["objects"][0]["slot"],"A")

    def test_frozen_idle_hold_has_no_lease_and_emergency_stop_still_latches(self):
        app = create_app(layout="B",playback_speed=0,watchdog_seconds=.25, allow_local_rules=True)
        with TestClient(app,base_url="http://localhost",client=("127.0.0.1",1234)) as client:
            def state():
                return client.get("/api/state").json()
            def send(text):
                return client.post("/api/commands",json=dict(id=str(uuid4()),text=text,
                    revision=state()["revision"],issued_at_ms=time.time_ns()//1_000_000))
            self.assertEqual(send("向前抓取木块").json()["status"],"success")
            self.assertTrue(state()["scene"]["holding"])
            self.assertEqual(send("暂停").json()["status"],"paused")
            snapshot = state()
            time.sleep(.32)
            idle = state()
            self.assertTrue(idle["paused"])
            self.assertFalse(idle["watchdog_active"] or idle["watchdog_expired"] or idle["busy"])
            self.assertEqual((snapshot["sim_seconds"],snapshot["tcp_mm"]),
                             (idle["sim_seconds"],idle["tcp_mm"]))
            self.assertTrue(client.post("/api/stop").json()["stopped"])
            self.assertEqual(app.state.sim.stop_response,"bounded_hold")
            self.assertFalse(state()["paused"])
            self.assertEqual(send("恢复控制").status_code,409)

    def test_pause_inflight_motion_keeps_layout_and_position(self):
        with TestClient(create_app(layout="B",playback_speed=2, allow_local_rules=True),
                        base_url="http://localhost",client=("127.0.0.1",1234)) as client, ThreadPoolExecutor(1) as worker:
            def state():
                return client.get("/api/state").json()
            def send(text):
                return client.post("/api/commands",json=dict(id=str(uuid4()),text=text,
                    revision=state()["revision"],issued_at_ms=time.time_ns()//1_000_000)).json()
            before = state()
            future = worker.submit(send,"把木块放到 C 区")
            deadline = time.monotonic()+5
            while time.monotonic()<deadline:
                current = state()
                if current["busy"] and current["phase"]=="moving" and current["tcp_mm"] != before["tcp_mm"]:
                    break
                time.sleep(.02)
            else:
                self.fail("Did not see movement")
            self.assertEqual(send("暂停当前动作")["status"],"paused")
            result = future.result(6)
            self.assertEqual(result["status"],"paused")
            self.assertNotIn("release",result["phases"])
            stopped = state()
            self.assertEqual(send("恢复控制")["status"],"resumed")
            self.assertEqual(state()["tcp_mm"],stopped["tcp_mm"])
            self.assertEqual(state()["layout"],"B")
            self.assertEqual(state()["sim_seconds"],stopped["sim_seconds"])
            self.assertNotEqual(stopped["tcp_mm"],before["tcp_mm"])
