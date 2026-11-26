"""Offline boundary tests for live-pair provenance, budget and physical scoring."""
import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts import evaluate_sensevoice_pairs as pairs


class SenseVoicePairTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for patcher in (
            patch.object(pairs, "ROOT", self.root),
            patch("httpx.Client.post", side_effect=AssertionError("unexpected model request")),
            patch.object(pairs, "create_app", side_effect=AssertionError("unexpected simulator startup")),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        audio = b"fixture-only; not a recording or an ASR result"
        audio_dir = self.root / "results/safety/prompt_trial_audio"
        audio_dir.mkdir(parents=True)
        (audio_dir / "transfer.wav").write_bytes(audio)
        (self.root / "configs").mkdir()
        self.reference = "把红色木块放到右边的工位"
        (self.root / "configs/prompt_trial.json").write_text(json.dumps({"cases": [
            {"id": "transfer", "reference": self.reference, "wav": "transfer.wav"}
        ]}), encoding="utf-8")
        self.transcript = "把红色木块放到右干的工位。"
        self.source = self.root / "source.json"
        self.fixture = {
            "mode": "real_sensevoice_to_http_to_live_deepseek", "asr_engine": "sensevoice",
            "planner_prompt_sha256": pairs.PLAN_PROMPT_SHA256,
            "summary": {"total": 4, "cloud_calls": 2},
            "campus_attempts": [{"task": {"command": self.transcript}, "state": "answered"},
                                {"task": {"command": "向左移动12毫米。"}, "state": "answered"}],
            "rows": [{"id": "transfer", "reference": self.reference,
                      "audio_sha256": hashlib.sha256(audio).hexdigest(), "asr_http_status": 200,
                      "asr_response": {"text": self.transcript, "engine": "sensevoice"},
                      "command": {"text": self.transcript}}],
        }
        self.write_source(self.fixture)

    def write_source(self, data):
        self.source.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def test_dry_run_freezes_pairs_and_preserves_raw_transcript(self):
        result = pairs.run(self.source, self.root / "unused.json", 8, dry_run=True)
        self.assertEqual([row["id"] for row in result["trials"]],
                         ["A_noisy", "A_reference", "C_reference", "C_noisy"])
        self.assertEqual(result["source_sha256"], hashlib.sha256(self.source.read_bytes()).hexdigest())
        for row in result["trials"]:
            self.assertEqual(row["text"], self.transcript if row["variant"] == "noisy" else self.reference + "。")
            self.assertEqual(row["expected"]["steps"], [{"action": "transfer", "object": "red_1",
                                                         "from": row["layout"], "to": "B"}])
        self.assertFalse((self.root / "unused.json").exists())

    def test_budget_counts_prior_attempts_including_failures(self):
        for approval in (0, 5, 9):
            with self.subTest(approval=approval), self.assertRaises(ValueError):
                pairs.prepare(self.source, approval)
        failed = copy.deepcopy(self.fixture)
        failed["campus_attempts"].extend([{"task": {}, "state": "failed"}] * 2)
        failed["summary"]["cloud_calls"] = 4
        self.write_source(failed)
        with self.assertRaises(ValueError):
            pairs.prepare(self.source, 7)
        self.assertEqual(pairs.prepare(self.source, 8)["source_campus_attempts"], 4)

    def test_tampered_or_incomplete_source_is_rejected_before_requests(self):
        edits = {
            "mode": lambda d: d.update(mode="local_rules"),
            "prompt": lambda d: d.update(planner_prompt_sha256="wrong"),
            "incomplete": lambda d: d.update(summary={"total": 3, "cloud_calls": 2}),
            "attempts": lambda d: d["summary"].update(cloud_calls=1),
            "audio": lambda d: d["rows"][0].update(audio_sha256="wrong"),
            "edited_command": lambda d: d["rows"][0]["command"].update(text="把红块放到B"),
            "edited_reference": lambda d: d["rows"][0].update(reference="把红块放到A"),
            "other_asr": lambda d: d["rows"][0]["asr_response"].update(engine="whisper"),
        }
        for name, mutate in edits.items():
            with self.subTest(name=name):
                data = copy.deepcopy(self.fixture)
                mutate(data)
                self.write_source(data)
                with self.assertRaises(ValueError):
                    pairs.prepare(self.source, 8)

    def test_existing_output_cannot_be_overwritten_or_spend_more_calls(self):
        output = self.root / "retained.json"
        original = b'{"state":"partial"}'
        output.write_bytes(original)
        with self.assertRaises(ValueError):
            pairs.run(self.source, output, 8)
        self.assertEqual(output.read_bytes(), original)

    @staticmethod
    def physical_row():
        def block(z, contacts, inside):
            return {"xyz_m": [.2, 0, z], "contacts": contacts, "inside": inside}
        landed = block(.046, ["desk"], "B")
        return {"layout": "A", "receipt": {"status": "success"},
                "before": {"block": block(.046, ["desk"], "A")}, "after": {"block": landed},
                "frames": [{"phase": "lift", "physics": {"block": block(
                    .086, ["fixed_pad", "moving_pad"], None)}}] + [
                    {"phase": f"verify_{index}", "physics": {"block": copy.deepcopy(landed)}}
                    for index in (1, 2, 3)]}

    def test_success_requires_lift_and_three_released_landings(self):
        self.assertTrue(all(value is True for value in pairs.physical_checks(self.physical_row()).values()))
        row = self.physical_row()
        row["frames"][0]["physics"]["block"]["xyz_m"][2] = .05
        self.assertFalse(pairs.physical_checks(row)["single_lift_with_both_pads_and_over_20mm"])

    def test_finger_contact_or_wrong_target_cannot_count_as_released_landing(self):
        for field, bad_value in (("contacts", ["desk", "moving_pad"]), ("inside", "C")):
            with self.subTest(field=field):
                row = self.physical_row()
                row["frames"][2]["physics"]["block"][field] = bad_value
                self.assertFalse(pairs.physical_checks(row)["all_three_landings_inside_B_only_desk_contact"])
        row = self.physical_row()
        row["frames"][0]["physics"]["block"]["contacts"] = ["moving_pad"]
        self.assertFalse(pairs.physical_checks(row)["single_lift_with_both_pads_and_over_20mm"])

    def test_missing_physical_measurements_are_unknown_not_success(self):
        row = self.physical_row()
        row["frames"] = []
        scores = pairs.physical_checks(row)
        self.assertIsNone(scores["single_lift_with_both_pads_and_over_20mm"])
        self.assertIsNone(scores["all_three_landings_inside_B_only_desk_contact"])
        self.assertFalse(scores["verification_phases"])
        self.assertFalse(all(value is True for value in scores.values()))


if __name__ == "__main__":
    unittest.main()
