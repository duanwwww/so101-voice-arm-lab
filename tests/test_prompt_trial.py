import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from arm_lab.model_access import ModelFailure
from scripts import evaluate_prompt_trial as trial


class PromptTrialTests(unittest.TestCase):
    def test_dry_requires_no_audio_or_services(self):
        self.assertEqual(trial.run("dry", None, None, None)["calls"],0)

    def test_raw_reply_scored_before_controller(self):
        self.assertFalse(trial.expected_proposal("ask",{"action":"ask","question":None}))
        self.assertFalse(trial.expected_proposal("ask",trial.mocked_reply("move")))
        self.assertTrue(trial.expected_proposal("transfer",{"steps":[
            {"action":"pick","from":"A","object":"red_1"},{"action":"place","to":"B"}]}))
        wrong = trial.mocked_reply("move")
        wrong["steps"][0]["axis"] = "x"
        self.assertFalse(trial.expected_proposal("move",wrong))

    def test_failed_gateway_is_charged_once_and_stops(self):
        a = trial.TrialAccess()
        a.mock = False
        a.row, a.report, a.variant = {"goal":"transfer"},{"campus_attempts":0},"test"
        saved_counts = []
        a.persist = lambda:saved_counts.append(a.report["campus_attempts"])
        with patch("arm_lab.model_access.ModelAccess.chat_json",side_effect=ModelFailure("fixture offline")):
            with self.assertRaises(trial.EndpointStopped):
                a.chat_json("",{"command":"test"},"fixture",True)
        self.assertEqual(saved_counts, [1])
        a.report["campus_attempts"] = trial.MAX_CAMPUS
        with patch("arm_lab.model_access.ModelAccess.chat_json") as request:
            with self.assertRaises(trial.EndpointStopped):
                a.chat_json("",{},"fixture",True)
            request.assert_not_called()

    def test_fixture_runs_camera_and_physics_and_is_idempotent(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp)/"report.json"
            with patch("arm_lab.model_access.httpx.Client",side_effect=AssertionError("unexpected network")):
                report = trial.run("mock",out,None,None)
                self.assertEqual(json.loads(json.dumps(report)),trial.run("mock",out,None,None))
            self.assertEqual((report["status"],report["asr_attempts"],report["campus_attempts"]),("complete",0,0))
            self.assertEqual(report["mock_requests"],sum("planner_input" in r for r in report["trials"]))
            self.assertLessEqual(report["mock_requests"],trial.MAX_CAMPUS)
            self.assertTrue(all(row["scores"]["end_to_end_correct"] for row in report["trials"]))
            negative = next(r for r in report["trials"] if r["goal"]=="ignore")
            self.assertNotIn("planner_input",negative)
            normal = report["trials"][0]
            self.assertEqual(normal["physics"]["verify_3"]["inside"], "B")
            self.assertTrue((out.with_suffix(".frames")/normal["camera_frames"]["after"]).is_file())
            # A process that crashed after reserving a request must never silently resend it.
            report.update(status="in_progress")
            report["trials"][-1]["state"] = "started"
            out.write_text(json.dumps(report),encoding="utf-8")
            with patch("arm_lab.model_access.httpx.Client",side_effect=AssertionError):
                resumed = trial.run("mock",out,None,None)
            self.assertEqual(resumed["status"],"interrupted_unknown_outcome")

    def test_paired_denominator_includes_truncated_and_excludes_unanswered(self):
        rejected = {"raw_plan_correct":None,"end_to_end_correct":False,"movement_on_nonaction":False}
        accepted = {"raw_plan_correct":True,"end_to_end_correct":True,"movement_on_nonaction":False}
        rows = [
            {"case":"one","variant":"baseline","http_status":200,"raw_model_text":"{}","scores":accepted},
            {"case":"one","variant":"compact","http_status":200,"raw_model_text":'{',"scores":rejected},
            {"case":"two","variant":"baseline","http_status":429,"scores":rejected},
        ]
        report = {"trials":rows,"prompts":{"baseline":"a","compact":"b"},"asr_attempts":0,"campus_attempts":3,
                  "audio":{},"signature":{"mode":"real"},"status":"endpoint_stopped"}
        summary = trial.summarize(report)
        self.assertEqual(summary["same_answered_cases"],["one"])
        self.assertEqual(summary["paired"]["baseline"]["raw_model_goal_match"],1)
        self.assertEqual(summary["paired"]["compact"]["raw_model_goal_match"],0)

    def test_replay_only_uses_recorded_scene_and_reply(self):
        a = trial.ReplayAccess()
        a.report, a.row = {"responses_replayed":0}, {}
        task = {"command":"?", "objects":[]}
        a.recorded = {"planner_input":task, "http_status":200, "raw_model_text":"{", "finish_reason":"length"}
        with patch("arm_lab.model_access.httpx.Client",side_effect=AssertionError):
            with self.assertRaisesRegex(ModelFailure,"differs"):
                a.chat_json("",{"command":"different"},"unused",False)
            self.assertEqual(a.report["responses_replayed"],0)
            with self.assertRaisesRegex(ModelFailure,"incomplete"):
                a.chat_json("",task,"unused",False)
            self.assertEqual(a.row["finish_reason"],"length")
            a.recorded["raw_model_json"] = trial.mocked_reply("move")
            self.assertEqual(a.chat_json("",task,"unused",False),a.recorded["raw_model_json"])
