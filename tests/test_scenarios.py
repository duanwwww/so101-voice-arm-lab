import unittest

from fastapi.testclient import TestClient
import numpy as np

from arm_lab.real_data import first_cycle, iou
from arm_lab.scene import LAYOUTS
from arm_lab.simulator import Simulator
from arm_lab.web import create_app


class ScenarioTests(unittest.TestCase):
    def test_reset_restores_camera_and_props(self):
        sim=Simulator("A")
        self.addCleanup(sim.close)
        scene,frame=sim.camera()
        for layout,expected_count in (("two_red",2),("cover_source",0),("dim",1),
                                      ("reflective_band",2),("clutter",1)):
            with self.subTest(layout=layout):
                sim.reset(layout)
                self.assertEqual(len(sim.camera()[0]["objects"]),expected_count)
                sim.reset("A")
                observed,image=sim.camera()
                self.assertEqual(observed,scene)
                # A repeated OpenGL render can round a channel by one 8-bit level.
                # Check the entire scene as well as its exact detected objects above.
                np.testing.assert_allclose(image,frame,rtol=0,atol=1)

    def test_api_lists_and_selects_the_same_scenes(self):
        with TestClient(create_app(playback_speed=0, allow_local_rules=True)) as c:
            listing=c.get("/api/scenarios").json()
            self.assertEqual([r["id"] for r in listing],list(LAYOUTS))
            self.assertTrue(all(r["command"] for r in listing))
            self.assertEqual(c.post("/api/reset",json={"layout":"two_red"}).status_code,200)
            state=c.get("/api/state").json()
            self.assertEqual(state["layout"],"two_red")
            self.assertEqual(len(state["scene"]["objects"]),2)
            self.assertEqual(c.post("/api/reset",json={"layout":"missing"}).status_code,422)
            self.assertEqual(c.get("/api/state").json()["layout"],"two_red")


class DataMetricTests(unittest.TestCase):
    def test_iou_uses_xywh_and_counts_partial_overlap(self):
        box=[10,20,8,4]
        self.assertEqual(iou(box,box),1)
        self.assertEqual(iou(box,[18,20,8,4]),0)
        self.assertAlmostEqual(iou(box,[14,20,8,4]),1/3)

    def test_trace_cycle_needs_three_consecutive_samples_per_state(self):
        sequence=np.array([9,9,4,9,9,9,5,1,1,1,4,9,9,9])
        self.assertEqual(first_cycle(sequence,2,8),[3,7,11])
        self.assertIsNone(first_cycle(sequence[:-1],2,8))
