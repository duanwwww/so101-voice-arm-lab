"""Reasoning budget and final-content isolation; all HTTP calls are in memory."""
import json
import unittest
from unittest.mock import patch

import httpx

from arm_lab.model_access import ModelAccess, ModelFailure


class ReasonerConfigTests(unittest.TestCase):
    def test_model_specific_budget_and_timeout_keep_legacy_runs_reproducible(self):
        legacy = ModelAccess()
        reasoner = ModelAccess(planner_model="deepseek-reasoner")
        self.assertEqual((legacy.planner_model, legacy.planner_max_tokens, legacy.planner_timeout),
                         ("deepseek-chat", 160, 30))
        self.assertEqual((reasoner.planner_max_tokens, reasoner.planner_timeout), (4096, 120))
        with self.assertRaises(ValueError):
            ModelAccess(planner_model="unverified-provider")
        with self.assertRaises(ValueError):
            ModelAccess(max_tokens=0)

    def client(self, access, message, finish="stop"):
        real_client = httpx.Client
        def handle(request):
            body = json.loads(request.content)
            self.assertEqual((body["model"], body["max_tokens"]), ("deepseek-reasoner", 4096))
            return httpx.Response(200, json={"choices": [{"finish_reason": finish, "message": message}]})
        def factory(**options):
            self.assertEqual(options["timeout"], 120)
            self.assertFalse(options["follow_redirects"])
            return real_client(transport=httpx.MockTransport(handle), **options)
        return patch("arm_lab.model_access.httpx.Client", side_effect=factory)

    def test_only_final_content_can_become_an_action(self):
        access = ModelAccess(planner_model="deepseek-reasoner")
        message = {"reasoning_content": '{"steps":[{"action":"move","axis":"z","mm":50}]}',
                   "content": '{"steps":[{"action":"move","axis":"y","mm":12}]}'}
        with self.client(access, message), patch("arm_lab.model_access.campus_api_key", return_value="offline-test-key"):
            plan = access.plan("向左移动十二毫米", {"objects": []}, approved_once=True)
        from arm_lab.planner import actions
        action = actions(plan)[0]
        self.assertEqual((action.axis, action.amount), ("y", 12))
        self.assertEqual(access.used["deepseek"], 1)

    def test_reasoning_without_final_json_is_rejected(self):
        access = ModelAccess(planner_model="deepseek-reasoner")
        message = {"reasoning_content": '{"steps":[{"action":"home"}]}', "content": None}
        with self.client(access, message), self.assertRaises(ModelFailure):
            access.chat_json("test", {}, "offline-test-key", True)

    def test_truncated_reply_never_becomes_a_plan(self):
        access = ModelAccess(planner_model="deepseek-reasoner")
        with self.client(access, {"content": '{"steps":[{"action":"home"}]}'}, "length"), \
                self.assertRaises(ModelFailure):
            access.chat_json("test", {}, "offline-test-key", True)


if __name__ == "__main__":
    unittest.main()
