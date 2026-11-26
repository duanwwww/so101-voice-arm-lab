import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from arm_lab.planner_scoring import score_local, score_response


LEFT = {"action":"move", "axis":"y", "mm":12}
PICK = {"action":"pick", "object":"red_1", "from":"A"}
PLACE = {"action":"place", "to":"C"}
TRANSFER = {"action":"transfer", "object":"red_1", "from":"A", "to":"C"}
ASK = {"action":"ask", "question":"角度是多少度？"}


def sample(expected, **metadata):
    return {"route":"planner", "scene":{"objects":[{"id":"red_1", "slot":"A"}]},
            "expected":expected, **metadata}


def response(expected, proposed, **metadata):
    return score_response(sample(expected, **metadata), json.dumps(proposed, ensure_ascii=False))


class PlannerScoringTests(unittest.TestCase):
    def test_exact_and_equivalent_with_order_and_context(self):
        gold = {"steps":[TRANSFER, LEFT]}
        for reply, exact in [(gold,True), ({"steps":[PICK,PLACE,LEFT]},False)]:
            result = response(gold,reply)
            self.assertTrue(result["semantic_correct"],result)
            self.assertEqual(result["strict_exact"],exact)
            self.assertFalse(result["dangerous_motion"])
        wrong_order = response(gold,{"steps":[LEFT,TRANSFER]})
        self.assertIn("step_order",wrong_order["failure_tags"])
        self.assertFalse(wrong_order["semantic_correct"])
        wrong_place = response(gold,{"steps":[{**TRANSFER,"to":"B"},LEFT]})
        self.assertIn("wrong_to",wrong_place["failure_tags"])
        wrong_source = response(gold,{"steps":[{**TRANSFER,"from":"B"},LEFT]})
        self.assertFalse(wrong_source["scene_valid"])
        self.assertIn("wrong_from",wrong_source["failure_tags"])
        self.assertIn("empty_gripper",response(gold,{"steps":[PLACE]})["failure_tags"])
        self.assertTrue(response({"steps":[PLACE]}, {"steps":[PLACE]},
                                 scene={"objects":[],"holding":True})["semantic_correct"])

    def test_unrequested_steps_and_wrong_move_slots(self):
        gold = {"steps":[LEFT]}
        mutations = [([LEFT,LEFT],"extra_steps"), ([{**LEFT,"axis":"x"}],"wrong_axis"),
                     ([{**LEFT,"mm":-12}],"wrong_sign"), ([{**LEFT,"mm":1.2}],"wrong_mm"),
                     ([{"action":"wrist","deg":12}],"action_kind")]
        for steps, label in mutations:
            with self.subTest(steps=steps):
                got = response(gold,{"steps":steps})
                self.assertIn(label,got["failure_tags"])
                self.assertTrue(got["dangerous_motion"])
                self.assertFalse(got["semantic_correct"])
        self.assertIn("missing_steps",response({"steps":[LEFT,{"action":"observe"}]}, gold)["failure_tags"])
        flipped = response({"steps":[LEFT,{**LEFT,"mm":3}]},{"steps":[{**LEFT,"mm":3},LEFT]})
        self.assertIn("step_order",flipped["failure_tags"])

    def test_strict_format_rejects_ambiguous_or_invalid_json(self):
        good = '{"steps":[{"action":"move","axis":"y","mm":12}]}'
        gold = sample({"steps":[LEFT]})
        for bad in [good+" "+good,'```json\n'+good+'\n```',good.replace('"mm":12','"mm":7,"mm":12'),
                    good.replace("12","NaN"),good.replace("12","Infinity")]:
            result = score_response(gold,bad)
            self.assertFalse(result["raw_json_valid"],bad)
            self.assertFalse(result["semantic_correct"],bad)
        for invalid in [LEFT,{"steps":[]},{"steps":[LEFT]*5},{"steps":[LEFT],"extra":2},
                        {"steps":[{**LEFT,"mm":True}]},{"steps":[{**LEFT,"mm":"12"}]},
                        {"steps":[{**LEFT,"mm":51}]},{"steps":[{**LEFT,"mm":0}]},
                        {"steps":[{**PICK,"object":"red_2"}]}, {"action":"ask","question":" "}]:
            result = response(gold["expected"],invalid)
            self.assertTrue(result["raw_json_valid"])
            self.assertFalse(result["schema_valid"],invalid)
        for raw in [good.replace("12","12.00000000000000001"),good.replace("12","120e-1")]:
            got = score_response(gold,raw)
            self.assertEqual(got["strict_exact"],"120e-1" in raw)

    def test_ask_focus_is_distinct_from_deciding_to_ask(self):
        case = sample(ASK, ask_focus=[["角度","几度"]])
        for q, focus in [("要转几度？",True),("放到哪个工位？",False)]:
            result = score_response(case,json.dumps({**ASK,"question":q}))
            self.assertTrue(result["ask_type_correct"])
            self.assertEqual(result["ask_focus_correct"],focus)
            self.assertEqual(result["semantic_correct"],focus)
            self.assertFalse(result["strict_exact"])
        got = response(ASK,{"steps":[LEFT]})
        self.assertTrue(got["motion_on_nonaction"])
        self.assertFalse(response({"steps":[LEFT]},ASK)["dangerous_motion"])
        self.assertFalse(response(ASK,{"steps":[{"action":"observe"}]})["dangerous_motion"])

    def test_incomplete_calls_and_local_decisions(self):
        case = sample({"steps":[LEFT]})
        for kwargs in ({"http_status":429},{"http_status":None},{"finish_reason":"length"}):
            result = score_response(case,json.dumps(case["expected"]),**kwargs)
            self.assertFalse(result["completed"])
            self.assertFalse(result["strict_exact"])
            self.assertFalse(result["semantic_correct"])
        self.assertFalse(score_response(case,None)["completed"])
        local = sample({"local":"ignore"},route="local")
        self.assertTrue(score_local(local,"ignore")["local_correct"])
        self.assertFalse(score_local(local,"pause")["local_correct"])
        got = score_response(local,json.dumps({"steps":[LEFT]}))
        self.assertTrue(got["motion_on_nonaction"])
        self.assertIn("unexpected_model_call",got["failure_tags"])

    def test_frozen_100_gold_roundtrips(self):
        path = Path(__file__).resolve().parents[1]/"configs"/"planner100.json"
        dataset = json.loads(path.read_text(encoding="utf-8"))
        rows = dataset["cases"] if isinstance(dataset,dict) else dataset
        self.assertEqual(len(rows),100)
        self.assertEqual(len({row["id"] for row in rows}),100)
        for row in rows:
            with self.subTest(id=row["id"]):
                if row["route"]=="local":
                    score = score_local(row,row["expected"]["local"])
                else:
                    score = score_response(row,json.dumps(row["expected"],ensure_ascii=False))
                    self.assertTrue(score["strict_exact"],score)
                self.assertTrue(score["semantic_correct"],score)

    def test_partial_summary_keeps_gateway_failures_and_retries_out_of_accuracy(self):
        from scripts.score_planner100 import report
        a = sample({"steps":[LEFT]},id="D1",split="dev",family="move")
        b = sample(ASK,id="D2",split="dev",family="unknown",ask_focus=[["角度","几度"]])
        c = sample({"local":"ignore"},id="T1",split="test",family="control",route="local")
        d = sample({"steps":[PLACE]},id="T2",split="test",family="drop",scene={"objects":[],"holding":True})
        cases=[a,b,c,d]
        def answer(case,v,body):
            return {"id":case["id"],"version":v,"split":case["split"],"state":"answered","http_status":200,
                    "finish_reason":"stop","raw":json.dumps(body,ensure_ascii=False)}
        with TemporaryDirectory() as directory:
            out=Path(directory)
            empty=report(cases,out)
            self.assertEqual(empty["splits"]["dev"]["baseline"]["missing_answer"],2)
            self.assertEqual(empty["paired"]["dev"]["baseline_vs_candidate"]["paired_answered"],0)
            failure={"id":"D1","split":"dev","version":"baseline","state":"http_error","http_status":429}
            attempts=[failure,answer(a,"baseline",ASK),answer(a,"candidate",a["expected"]),
                      {**failure,"id":"D2"},answer(b,"candidate",ASK)]
            (out/"responses.json").write_text(json.dumps({"prompts":{"baseline":"original","candidate":"edited"},
                "dataset_sha256":"fixture","attempts":attempts}),encoding="utf-8")
            (out/"local.json").write_text(json.dumps({"source_sha256":"fixture","model_calls":0,"rows":[
                {"id":"T1","scores":score_local(c,"ignore")}]}),encoding="utf-8")
            result=report(cases,out)
            base=result["splits"]["dev"]["baseline"]
            self.assertEqual((base["attempts"],base["answered"],base["missing_answer"],base["answered_incorrect"]), (3,1,1,1))
            pair=result["paired"]["dev"]["baseline_vs_candidate"]
            self.assertEqual((pair["paired_answered"],pair["improved_ids"],pair["regressed_ids"]),(1,["D1"],[]))
            self.assertEqual(result["splits"]["dev"]["candidate"]["by_expected"]["ask"]["ask_focus_correct"],1)
            self.assertEqual(result["local"]["by_split_and_expected"]["test"]["local"]["semantic_correct"],1)


if __name__=="__main__":
    unittest.main()
