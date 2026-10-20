"""Exercise the video/export pipeline with a fake pose estimator.

This checks file handling without downloading a pretrained model.
"""

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from analysis import CountingSettings
from run_squat import process_video


class FakePoseModel:
    def __init__(self):
        self.frame = 0
        # Enough standing/bottom frames to pass the default smoothing and timing checks.
        self.angles = [165] * 7 + [135] * 5 + [88] * 9 + [130] * 5 + [165] * 8

    def __call__(self, _image):
        angle = math.radians(self.angles[min(self.frame, len(self.angles) - 1)])
        self.frame += 1
        points = np.zeros((1, 26, 2), dtype=float)
        scores = np.ones((1, 26), dtype=float)
        knee = np.array([190, 125], dtype=float)
        points[0, 14] = knee
        points[0, 16] = knee + [0, 55]
        points[0, 12] = knee + [55 * math.sin(angle), 55 * math.cos(angle)]
        return points, scores


class TestVideoProcessing(unittest.TestCase):
    def test_exports_with_stub_pose_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root / "test_squat.mp4"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"),
                                     12, (400, 240))
            self.assertTrue(writer.isOpened())
            try:
                for _ in range(34):
                    writer.write(np.zeros((240, 400, 3), dtype=np.uint8))
            finally:
                writer.release()

            with patch("run_squat.load_pose_model", return_value=FakePoseModel()):
                process_video(video, root / "results", "right", 0.45,
                              CountingSettings(), False)

            results = root / "results"
            for name in ("test_squat_annotated.mp4", "test_squat_frames.csv",
                         "test_squat_reps.csv", "test_squat_summary.json"):
                self.assertTrue((results / name).is_file(), name)
            summary = json.loads((results / "test_squat_summary.json").read_text())
            self.assertEqual(summary["frames_processed"], 34)
            self.assertEqual(summary["frames_with_valid_knee_angle"], 34)
            self.assertEqual(summary["completed_reps"], 1)
            self.assertIn("No real-video ground-truth accuracy", summary["evaluation_status"])


if __name__ == "__main__":
    unittest.main()
