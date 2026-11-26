import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from scripts import evaluate_planner100 as bench


class Planner100RunnerTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.out = Path(tmp.name)
        self.log = self.out/"responses.json"
        out_patch = patch.multiple(bench, OUT=self.out, RUNS=self.log)
        out_patch.start()
        self.addCleanup(out_patch.stop)
        key = patch.object(bench,"campus_api_key",return_value="unit-fixture")
        key.start()
        self.addCleanup(key.stop)
        self.args = SimpleNamespace(split="dev",variants=["baseline","candidate"],approved_attempts=2,interval=0)
        self.inputs = [{"id":"one","split":"dev","command":"向左移动一毫米","scene":{"objects":[]},
                        "route":"planner","expected":{"SECRET":"do not send"}},
                       {"id":"stop","split":"dev","command":"暂停","scene":{"objects":[]},"route":"local"}]

    def with_fake(self, handler):
        client = httpx.Client(transport=httpx.MockTransport(handler))
        return patch.object(bench.httpx,"Client",return_value=client)

    def test_attempt_cap_idempotence_and_gold_isolation(self):
        def reply(request):
            body=json.loads(request.content)
            self.assertEqual(set(json.loads(body["messages"][1]["content"])),{"command","objects"})
            self.assertNotIn("SECRET",body["messages"][1]["content"])
            self.assertEqual(body["max_tokens"],160)
            return httpx.Response(200,json={"choices":[{"message":{"content":"{}"},"finish_reason":"stop"}]})
        self.args.approved_attempts=1
        with self.with_fake(reply):
            self.assertEqual(bench.run_online(self.inputs,self.args),"budget_exhausted")
        self.args.approved_attempts=2
        with self.with_fake(reply):
            self.assertEqual(bench.run_online(self.inputs,self.args),"complete")
        with self.with_fake(lambda req:self.fail("repeated paid call")):
            self.assertEqual(bench.run_online(self.inputs,self.args),"complete")
        rows=json.loads(self.log.read_text(encoding="utf-8"))["attempts"]
        self.assertEqual([r["id"] for r in rows],["one","one"])

    def test_rate_limit_is_counted_and_checkpointed(self):
        with self.with_fake(lambda req:httpx.Response(429,headers={"Retry-After":"301"})):
            self.assertEqual(bench.run_online(self.inputs,self.args),"checkpoint_rate_limited")
        book=json.loads(self.log.read_text(encoding="utf-8"))
        self.assertEqual(len(book["attempts"]),1)
        self.assertEqual(book["attempts"][0]["retry_after_seconds"],301)
        self.assertGreater(book["next_request_after"],0)

    def test_postlock_coverage_cannot_spend_before_heldout_is_complete(self):
        import hashlib
        book = {"dataset_sha256":hashlib.sha256(bench.DATA.read_bytes()).hexdigest(),
                "prompts":{"baseline":bench.prompt("baseline"),"finalist":bench.prompt("finalist")},
                "test_versions":["baseline","finalist"],"attempts":[],"next_request_after":0}
        self.log.write_text(json.dumps(book),encoding="utf-8")
        (self.out/"selection.json").write_text(json.dumps({"challenger":"finalist",
                  "test_versions":["baseline","finalist"]}),encoding="utf-8")
        self.args.coverage=True
        self.args.variants=["finalist"]
        cases=self.inputs + [{**self.inputs[0],"split":"test","id":"held"}]
        with self.with_fake(lambda req:self.fail("No request before held-out completion")):
            with self.assertRaisesRegex(ValueError,"Finish paired"):
                bench.run_online(cases,self.args)
