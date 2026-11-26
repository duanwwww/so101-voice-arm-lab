import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from arm_lab.field_cameras import main


class FieldCameraTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        for name,length in (("fixed",8),("wrist",3)):
            writer = cv2.VideoWriter(str(self.root/f"{name}.avi"),cv2.VideoWriter_fourcc(*"MJPG"),4,(80,60))
            self.assertTrue(writer.isOpened())
            for frame in range(length):
                source = np.zeros((60,80,3),dtype=np.uint8)
                source[:,:,0 if name=="fixed" else 2] = 50+frame*20
                writer.write(source)
            writer.release()
        self.args = ["--trial","1","--phase","before","--fps","4",
                     "--fixed-video",str(self.root/"fixed.avi"),"--out",str(self.root/"captures")]

    def report(self):
        return json.loads((self.root/"captures"/"01"/"before"/"capture.json").read_text(encoding="utf-8"))

    def test_file_pair_counts_unavailable_samples(self):
        code = main(self.args + ["--samples","4","--wrist-video",str(self.root/"wrist.avi")])
        report = self.report()
        self.assertEqual(code,1)
        self.assertEqual(report["counts"]["fixed"]["saved"],4)
        self.assertEqual(report["counts"]["wrist"]["missing_samples"],1)
        self.assertEqual(report["paired_samples"],3)
        self.assertEqual(report["samples"][-1]["views"]["wrist"]["failure"],"past_video_end")
        for sample in report["samples"][:3]:
            self.assertGreaterEqual(sample["grab_skew_ms"],0)
            row = sample["views"]["wrist"]
            self.assertEqual(row["received_size"],[80,60])
            path = self.root/"captures"/"01"/"before"/row["file"]
            self.assertEqual(row["sha256"],hashlib.sha256(path.read_bytes()).hexdigest())

    def test_fixed_only_and_offset_do_not_overwrite(self):
        self.assertEqual(main(self.args+["--start-s","0.5","--samples","2"]),0)
        report = self.report()
        self.assertEqual(list(report["sources"]),["fixed"])
        self.assertEqual(report["samples"][0]["views"]["fixed"]["requested_video_frame"],2)
        with self.assertRaises(SystemExit):
            main(self.args)
        self.assertEqual(report,self.report())

    def test_no_device_is_opened_without_explicit_source(self):
        with patch("cv2.VideoCapture") as op:
            with self.assertRaises(SystemExit):
                main(["--trial","1","--phase","before"])
            with self.assertRaises(SystemExit):
                main(self.args+["--samples","31"])
            op.assert_not_called()
