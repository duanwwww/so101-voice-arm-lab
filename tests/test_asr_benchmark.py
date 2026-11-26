import importlib.util
import io
import unittest
import wave
from unittest.mock import patch

from arm_lab import asr_benchmark as bench


class ASRBenchmarkTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("faster_whisper"), "voice extra required")
    def test_whisper_decoder_accepts_uploaded_pcm(self):
        from faster_whisper.audio import decode_audio

        audio = io.BytesIO()
        with wave.open(audio, "wb") as clip:
            clip.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            clip.writeframes(bytes(3200))
        audio.seek(0)
        self.assertEqual(len(decode_audio(audio)), 1600)

    def test_unit_and_direction_errors_remain_visible(self):
        ref = bench.anchors("向左滑动十三毫米，不要右转。")
        self.assertEqual(ref, ["direction:向左","motion:滑动","quantity:13mm",
                               "negation_or_stop:不要","direction:右转"])
        self.assertEqual(ref, bench.anchors("向左滑动13 mm 不要右转"))
        self.assertNotEqual(ref, bench.anchors("向右滑动13厘米，右转"))
        self.assertEqual(bench.anchors("三零厘米"), ["quantity:30cm"])
        self.assertEqual(bench.anchors("一百五十六毫米"), ["quantity:156mm"])
        self.assertEqual(bench.anchors("五点六毫米"), ["quantity:5.6mm"])

    def test_edit_distance_counts_insertions_deletions_substitutions(self):
        self.assertEqual(bench.edit_distance("左转", "右转"), 1)
        self.assertEqual(bench.edit_distance("左转", ""), 2)
        self.assertEqual(bench.edit_distance("右转", "向右转"), 1)
        self.assertEqual(bench.cer_text(" 五，公分！"),"五公分")

    def test_without_approval_no_data_or_models_touched(self):
        with patch("sys.argv", ["asr_benchmark","run"]), patch.object(bench, "prepare") as prepare:
            with self.assertRaises(SystemExit):
                bench.main()
            prepare.assert_not_called()

    def test_groups_are_fixed_by_speaker(self):
        groups = {g:sum(len(ids) for name,_,ids in bench.SPEAKERS.values() if g==name)
                  for g in ("development","evaluation")}
        self.assertEqual(groups,{"development":9,"evaluation":15})
