"""Offline semantic regressions for accepted wording and malformed motion plans.

The seven positive cases and actual planner replies predate this validator change.
Counterexamples mutate their actions, not the frozen evaluation data. No API,
ASR or physics execution is needed to check the command/plan contract here.
"""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from arm_lab.planner import check_command_match, decode_model_plan


ROOT = Path(__file__).resolve().parents[1]
CASE_IDS = ("P013", "P017", "P058", "P059", "P066", "P075", "P081")
CASES = {c["id"]: c for c in json.loads((ROOT / "configs/planner100.json").read_text(encoding="utf-8"))["cases"]}
SCENE = {"objects": [{"id": "red_1", "slot": "A", "xy_mm": [215, 77]}]}


def move(axis, mm):
    return {"action": "move", "axis": axis, "mm": mm}


def wrist(deg):
    return {"action": "wrist", "deg": deg}


class GuardVocabularyTests(unittest.TestCase):
    def accepts(self, text, body, scene=None):
        scene = SCENE if scene is None else scene
        intent = decode_model_plan(body, scene)
        check_command_match(text, intent, scene)

    def rejects(self, text, body, scene=None):
        with self.assertRaises(ValueError, msg=f"Unexpected admission: {text!r} -> {body!r}"):
            self.accepts(text, body, scene)

    def test_seven_preexisting_correct_proposals_are_admitted(self):
        for case_id in CASE_IDS:
            c = CASES[case_id]
            with self.subTest(case=case_id):
                self.accepts(c["command"], c["expected"], c["scene"])

    def test_saved_reasoner_answers_pass_without_changing_transcript_or_json(self):
        report = json.loads((ROOT / "results/safety/reasoner/paired.json").read_text(encoding="utf-8"))
        replies = {r["case_id"]: r for r in report["attempts"]
                   if r["model"] == "deepseek-reasoner" and r["state"] == "answered"}
        for case_id in CASE_IDS:
            c = CASES[case_id]
            with self.subTest(case=case_id):
                self.accepts(c["command"], json.loads(replies[case_id]["raw"]), c["scene"])

    def test_each_motion_rejects_reversed_sign_wrong_axis_and_wrong_scale(self):
        for case_id in CASE_IDS:
            c = CASES[case_id]
            for index, step in enumerate(c["expected"]["steps"]):
                amount = "mm" if step["action"] == "move" else "deg"
                mutations = [("reverse_sign", amount, -step[amount]),
                             ("unit_factor_ten", amount, step[amount] * 10)]
                if amount == "mm":
                    mutations.append(("wrong_axis", "axis", {"x": "y", "y": "z", "z": "x"}[step["axis"]]))
                else:
                    # Correct angle magnitude must not authorize a translation.
                    changed = deepcopy(c["expected"])
                    changed["steps"][index] = move("z", step[amount])
                    with self.subTest(case=case_id, step=index, mutation="wrong_action"):
                        self.rejects(c["command"], changed, c["scene"])
                for name, field, value in mutations:
                    changed = deepcopy(c["expected"])
                    changed["steps"][index][field] = value
                    with self.subTest(case=case_id, step=index, mutation=name):
                        self.rejects(c["command"], changed, c["scene"])

    def test_motion_cannot_be_silently_removed_or_extended(self):
        for case_id in CASE_IDS:
            c = CASES[case_id]
            steps = c["expected"]["steps"]
            for index in range(len(steps)):
                omitted = deepcopy(steps)
                omitted.pop(index)
                with self.subTest(case=case_id, omitted=index):
                    self.rejects(c["command"], {"steps": omitted or [{"action": "observe"}]}, c["scene"])
            with self.subTest(case=case_id, mutation="extra_step"):
                self.rejects(c["command"], {"steps": [*deepcopy(steps), move("x", 1)]}, c["scene"])

    def test_each_reordering_of_four_step_case_is_rejected(self):
        c = CASES["P075"]
        for left in range(4):
            for right in range(left + 1, 4):
                changed = deepcopy(c["expected"])
                rows = changed["steps"]
                rows[left], rows[right] = rows[right], rows[left]
                with self.subTest(swapped=(left, right)):
                    self.rejects(c["command"], changed, c["scene"])

    def test_prohibition_is_not_removed_by_wording_normalization(self):
        for case_id in CASE_IDS:
            if case_id == "P081":
                continue
            c = CASES[case_id]
            with self.subTest(case=case_id):
                self.rejects("不要" + c["command"], c["expected"], c["scene"])

    def test_fractional_units_are_complete_quantities(self):
        for text, step in (
            ("手腕往上抬两度半", wrist(-2.5)),
            ("手腕往下转2度半", wrist(2.5)),
            ("向左移动两厘米半", move("y", 25)),
            ("向后移动半厘米", move("x", -5)),
        ):
            with self.subTest(text=text):
                self.accepts(text, {"steps": [step]})

    def test_position_prefix_and_vertical_motion_preserve_previous_acceptance(self):
        for text, step in (("从现在的位置向前挪一点二厘米。", move("x", 12)),
                           ("请竖直向下挪五厘米。", move("z", -50))):
            with self.subTest(text=text):
                self.accepts(text, {"steps": [step]})
                self.rejects(text, {"steps": [move(step["axis"], -step["mm"])]})

    def test_explicit_base_axes_keep_coordinate_frame_and_sign(self):
        for axis in ("x", "y", "z"):
            for word, sign in (("正", 1), ("负", -1)):
                text = f"朝底座的 {axis.upper()} {word}方向走 3 mm"
                with self.subTest(text=text):
                    self.accepts(text, {"steps": [move(axis, sign * 3)]})
                    self.rejects(text, {"steps": [move(axis, -sign * 3)]})

    def test_uncertain_spelling_decimal_and_colloquial_fraction_need_clarification(self):
        for text, step in (
            ("向左移动十二豪米", move("y", 12)),
            ("向左移动两点毫米", move("y", 2)),
            ("向左移动十几毫米", move("y", 10)),
            ("向左移动两厘米五", move("y", 20)),
            ("向左移动两厘米五", move("y", 25)),
            ("向左移动2.5厘米半", move("y", 30)),
            ("手腕往上抬两度半半", wrist(-2.5)),
            ("向下移动一厘", move("z", -10)),
        ):
            with self.subTest(text=text, step=step):
                self.rejects(text, {"steps": [step]})

    def test_missing_units_cannot_be_filled_by_the_model(self):
        for text, step in (
            ("机械臂往后退2", move("x", -2)),
            ("末端垂直下降1.5", move("z", -1.5)),
            ("让手腕朝下俯五点五", wrist(5.5)),
            ("朝底座的X正方向走16", move("x", 16)),
        ):
            with self.subTest(text=text):
                self.rejects(text, {"steps": [step]})

    def test_unit_answer_uses_only_valid_missing_unit_context(self):
        c = CASES["P081"]
        for answer in ("单位是厘米。", "单位为厘米", "厘米"):
            with self.subTest(answer=answer):
                self.accepts(answer, c["expected"], c["scene"])
        for scene in (SCENE, {**SCENE, "dialogue": {"command": "向上移动2毫米", "question": "请选择目标工位。"}}):
            with self.subTest(scene=scene):
                self.rejects("单位是厘米", c["expected"], scene)
        for answer in ("单位是厘米和毫米", "单位是厘米，不要移动", "单位是厘米还是毫米"):
            with self.subTest(answer=answer):
                self.rejects(answer, c["expected"], c["scene"])

    def test_non_base_coordinate_words_cannot_be_discarded(self):
        for text in ("沿相机坐标向左移动5毫米", "按照画面向左移动5毫米", "沿工具坐标向左移动5毫米",
                     "按相机坐标，向左移动5毫米", "以画面为参照，然后向左移动5毫米"):
            with self.subTest(text=text):
                self.rejects(text, {"steps": [move("y", 5)]})

    def test_free_form_text_cannot_lose_its_literal_motion(self):
        self.rejects("请慢一点向左移动5毫米", {"steps": [
            {"action": "transfer", "object": "red_1", "from": "A", "to": "B"}]})
        self.rejects("向左移动5毫米然后手腕向下转", {"steps": [move("y", 5)]})

    def test_existing_physical_limits_are_unchanged(self):
        for text, step in (("往后退六厘米", move("x", -60)),
                           ("手腕往下转十六度", wrist(16)),
                           ("向左移动零毫米", move("y", 0))):
            with self.subTest(text=text):
                self.rejects(text, {"steps": [step]})


if __name__ == "__main__":
    unittest.main()
