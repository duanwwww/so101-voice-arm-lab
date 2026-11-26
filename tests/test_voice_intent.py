import unittest
from unittest.mock import patch

from arm_lab.model_access import ModelAccess
from arm_lab.planner import Intent, check_command_match, classify_control_text, local_intent


class VoiceControlTests(unittest.TestCase):
    def test_specific_prohibitions_do_not_interrupt_another_task(self):
        for text in ("不要向前移动一厘米", "请先别把红块放到 C", "不能向后走",
                     "不要暂停", "請別抓木塊", "不要转腕，也不要移动"):
            with self.subTest(text=text):
                self.assertEqual(classify_control_text(text).kind, "ignore")

    def test_pause_is_immediate_only_when_explicit(self):
        for text in ("请暂停当前动作", "停止", "先别动", "不要继续执行当前任务",
                     "麻烦你停一下吧", "請先暫停", "先停，然后向后挪五毫米"):
            with self.subTest(text=text):
                self.assertEqual(classify_control_text(text).kind, "pause")
        for text in ("不要去A，改去B", "先回到原位然后别动手腕", "拿稳以后暂停"):
            with self.subTest(text=text):
                self.assertEqual(classify_control_text(text).kind, "ask")

    def test_polite_request_and_nonnegative_words_still_reach_planning(self):
        for text in ("请问能不能向前移动一厘米", "麻烦你可不可以向前移动一厘米"):
            with self.subTest(text=text):
                self.assertIsNone(classify_control_text(text))
                intent = local_intent(text)
                self.assertEqual((intent.kind, intent.axis, intent.amount), ("move", "x", 10))
        for text in ("识别桌上的红块", "分别观察A和B", "把别的木块放到这里"):
            with self.subTest(text=text):
                self.assertIsNone(classify_control_text(text))

    def test_control_does_not_request_credentials_or_consume_a_call(self):
        access = ModelAccess()
        with patch("arm_lab.model_access.campus_api_key") as key, patch.object(access, "chat_json") as chat:
            for text, kind in (("不要向前移动一厘米", "ignore"),
                               ("暂停当前操作", "pause"), ("不要去B，但是去C", "ask")):
                self.assertEqual(access.plan(text, {"objects": []}).kind, kind)
            key.assert_not_called()
            chat.assert_not_called()
        self.assertEqual(access.used["deepseek"], 0)

    def test_even_an_injected_motion_plan_cannot_overrule_a_prohibition(self):
        scene = {"objects": []}
        for text in ("不要向前移动一厘米", "暂停", "先向前走，然后不要动手腕"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                check_command_match(text, Intent("move", "x", 10), scene)
        check_command_match("能不能向前移动一厘米", Intent("move", "x", 10), scene)

    def test_recorded_whisper_traditional_text_still_binds_the_axis(self):
        text = "向左移動12毫米"
        with self.assertRaises(ValueError):
            check_command_match(text, Intent("move", "x", -12), {"objects":[]})
        check_command_match(text, Intent("move", "y", 12), {"objects":[]})

    def test_missing_distance_never_moves_and_a_unit_reply_uses_the_same_scene(self):
        import time
        from uuid import uuid4
        from fastapi.testclient import TestClient
        from arm_lab.web import create_app
        access = ModelAccess()
        with TestClient(create_app(access,layout="A",playback_speed=0, allow_local_rules=True)) as client, patch.object(access,"plan") as model:
            def send(text, cloud=False, prior=None):
                rev = client.get("/api/state").json()["revision"]
                response = client.post("/api/commands",json={"id":str(uuid4()),"watchdog_token":str(uuid4()),
                    "text":text,"revision":rev,"issued_at_ms":round(time.time()*1000),"use_deepseek":cloud,"reply_to":prior})
                self.assertEqual(response.status_code,200,response.text)
                return response.json()
            first = send("向前移動5", cloud=True)
            self.assertEqual((first["status"],first["plan"]),("clarify",[]))
            second = send("毫米", prior=first["id"])
            self.assertEqual(second["plan"],["move_delta(x,5 mm)"])
            self.assertEqual(second["status"],"success")
            model.assert_not_called()

    def test_free_wording_cannot_invent_a_relative_motion(self):
        with self.assertRaises(ValueError):
            check_command_match("给小方块换个家",Intent("move","x",50),{"objects":[]})
        with self.assertRaises(ValueError):
            check_command_match("请把小方块送过去，并把腕部轻轻抬高两度",
                                Intent("wrist","wrist_flex",2),{"objects":[]})


if __name__ == "__main__":
    unittest.main()
