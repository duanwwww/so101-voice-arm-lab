import json
from unittest.mock import patch
import unittest

from scripts.replay_planner100_guard import replay


def case(id_, command, expected):
    return {"id":id_,"command":command,"scene":{"objects":[{"id":"red_1","slot":"A"}]},
            "split":"dev","route":"planner","family":"test","expected":expected}


def answer(case_, body, finish="stop"):
    return {"state":"answered","http_status":200,"id":case_["id"],"version":"baseline",
            "split":case_["split"],"raw":body if isinstance(body,str) else json.dumps(body),"finish_reason":finish}


class TestGuardReplay(unittest.TestCase):
    def test_wrong_proposals_blocked_or_admitted_and_bare_correct_separated(self):
        left = {"action":"move","axis":"y","mm":12}
        goal = case("a","向左移动十二毫米",{"steps":[left]})
        unclear = case("b","帮小家伙搬个家",{"action":"ask","question":"要搬到哪一站？"})
        same = {**goal,"id":"c"}
        stopped = {**goal,"id":"d"}
        wrong_transfer = {"steps":[{"action":"transfer","object":"red_1","from":"A","to":"B"}]}
        journal={"prompts":{"baseline":"Original prompt"},"attempts":[
            answer(goal,{"steps":[{**left,"axis":"x"}]}),answer(unclear,wrong_transfer),
            answer(same,left),answer(stopped,left,finish="length"),
            {"state":"http_error","http_status":429,"id":"e","version":"baseline","split":"dev"}]}
        with patch("socket.socket.connect",side_effect=AssertionError("real socket in offline replay")):
            run = replay([goal,unclear,same,stopped],journal)
        summary = run["splits"]["dev"]["baseline"]
        self.assertEqual((run["online_model_calls"],run["robot_or_simulator_steps"]), (0,0))
        self.assertEqual((run["source_attempts"],summary["replayed_answers"]), (5,4))
        self.assertEqual(summary["wrong_motion_proposed"],2)
        self.assertEqual(summary["wrong_motion_blocked_or_asked"],1)
        self.assertEqual(summary["wrong_motion_admitted_ids"],["b"])
        self.assertEqual(summary["bare_correct_and_admitted"],1)
        self.assertEqual(summary["raw_semantic_correct"],0)
        self.assertEqual(summary["guard_goal_match"],1)
        self.assertEqual(run["rows"][-1]["guard_outcome"],"rejected")

    def test_fence_or_ask_never_receives_unearned_raw_credit(self):
        left={"action":"move","axis":"y","mm":12}
        goal=case("a","向左移动十二毫米",{"steps":[left]})
        unclear=case("b","帮小家伙搬个家",{"action":"ask","question":"准备搬到哪里？"})
        bad='{"steps":[{"action":"move","axis":"y","mm":12,"mm":12}]}'
        duplicates={**goal,"id":"c"}
        rows=replay([goal,unclear,duplicates],{"prompts":{"baseline":"prompt"},"attempts":[
            answer(goal,"```json\n"+json.dumps({"steps":[left]})+"\n```"),answer(unclear,unclear["expected"]),
            answer(duplicates,bad)]})["rows"]
        self.assertFalse(rows[0]["raw_strict_schema"])
        self.assertTrue(rows[0]["guard_goal_match"])
        self.assertEqual(rows[0]["format_adjustments_for_diagnosis"],["markdown_fence"])
        self.assertTrue(rows[1]["raw_semantic_correct"])
        self.assertEqual(rows[1]["guard_outcome"],"asked")
        self.assertIn("duplicate_key",rows[2]["raw_failure_tags"])
        self.assertFalse(rows[2]["raw_semantic_correct"])


if __name__=="__main__":
    unittest.main()
