import copy
import json
import unittest

from scripts import compare_asr_passes as comparison


class KeywordTest(unittest.TestCase):
    def test_number_and_length_unit_are_independent(self):
        reference = comparison.tokens("向左移动七毫米，不要右转四次")
        hypothesis = comparison.tokens("向左移动7厘米，不要右转4次")
        self.assertEqual(reference["number"], ["7", "4"])
        self.assertEqual(reference["number"], hypothesis["number"])
        self.assertEqual(reference["direction"], hypothesis["direction"])
        self.assertEqual(reference["negation_or_stop"], hypothesis["negation_or_stop"])
        self.assertEqual(reference["length_unit"], ["mm"])
        self.assertEqual(hypothesis["length_unit"], ["cm"])
        self.assertNotEqual(reference["quantity_with_unit"], hypothesis["quantity_with_unit"])

    def test_missing_number_missing_unit_and_lexical_one(self):
        self.assertEqual(comparison.tokens("向右移七毫末")["number"], ["7"])
        self.assertEqual(comparison.tokens("向右移七毫末")["length_unit"], [])
        self.assertEqual(comparison.tokens("向右移厘米")["length_unit"], ["cm"])
        self.assertEqual(comparison.tokens("向右移厘米")["number"], [])
        noise = comparison.tokens("小米一定一起统一")
        self.assertEqual(noise["number"], [])
        self.assertEqual(noise["length_unit"], [])

    def test_kilogram_prefix_is_separate_from_its_number(self):
        self.assertEqual(comparison.tokens("一百九十厘米一百二十千克")["number"], ["190", "120"])
        self.assertEqual(comparison.tokens("190米120千克")["number"], ["190", "120"])


class ComparisonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bench = comparison.bench
        cls.inventory = json.loads((bench.EVIDENCE / "asr_inventory.json").read_bytes())
        rows = []
        for source in cls.inventory["clips"]:
            reference = source["reference"]
            rows.append(dict(source, output=reference, chars=len(bench.cer_text(reference)),
                             char_edits=0, hyp_anchors=bench.anchors(reference), inference_seconds=0.125))
        cls.reports = {
            beam: dict(status="complete", model_calls=24, beam_size=beam, model=bench.MODEL_REPO,
                       model_revision=bench.MODEL_REV, dataset=bench.REPO, revision=bench.REV,
                       device="cpu", compute_type="int8", language="zh", vad_filter=False,
                       cer_script_policy="test", rows=rows)
            for beam in comparison.BEAMS
        }

    def test_paired_splits_and_total_duration(self):
        result = comparison.compare(self.inventory, self.reports)
        self.assertEqual(result["verified_model_calls_in_input_runs"], 48)
        self.assertEqual(result["groups"]["development"]["beam1"]["clips"], 9)
        self.assertEqual(result["groups"]["evaluation"]["beam3"]["normalized_sentence_exact"], 15)
        all_clips = result["groups"]["all"]
        self.assertEqual(all_clips["beam1"]["inference_seconds"], 3.0)
        self.assertEqual(all_clips["beam1"]["normalized_sentence_accuracy"], 1.0)
        self.assertEqual(all_clips["paired"]["equal_edits"], 24)

    def test_incomplete_or_mismatched_audio_rejected(self):
        report = copy.deepcopy(self.reports[1])
        report["model_calls"] = 23
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            comparison.validated_rows(self.inventory, report, 1)
        report["model_calls"] = 24
        report["rows"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Different input"):
            comparison.validated_rows(self.inventory, report, 1)
