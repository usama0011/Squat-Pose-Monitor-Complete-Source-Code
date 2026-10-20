import unittest

import numpy as np

from analysis import CountingSettings, SquatCounter, joint_angle, leg_measurement
from run_squat import single_person


class TestJointAngles(unittest.TestCase):
    def test_right_angle(self):
        self.assertAlmostEqual(joint_angle((0, 1), (0, 0), (1, 0)), 90)

    def test_straight_knee(self):
        self.assertAlmostEqual(joint_angle((-1, 0), (0, 0), (1, 0)), 180)

    def test_overlap_returns_none(self):
        self.assertIsNone(joint_angle((0, 0), (0, 0), (1, 0)))

    def test_nan_returns_none(self):
        self.assertIsNone(joint_angle((float("nan"), 0), (0, 0), (1, 0)))

    def test_landmark_confidence(self):
        points = np.zeros((26, 2))
        points[12], points[14], points[16] = (-1, 0), (0, 0), (1, 0)
        scores = np.ones(26)
        self.assertAlmostEqual(leg_measurement(points, scores), 180)
        scores[14] = 0.2
        self.assertIsNone(leg_measurement(points, scores))


class TestCounting(unittest.TestCase):
    def setUp(self):
        self.settings = CountingSettings(smoothing_window=1, confirm_seconds=0.1,
                                         min_rep_seconds=0.5)

    def run_angles(self, counter, angles, offset=0):
        for i, angle in enumerate(angles):
            counter.add(angle, offset + 0.12 * i)

    def test_full_rep(self):
        counter = SquatCounter(self.settings)
        self.run_angles(counter, [165, 165, 150, 135, 110, 95, 90, 90,
                                  110, 125, 145, 155, 165, 165])
        self.assertEqual(counter.count, 1)
        self.assertGreater(counter.repetitions[0]["range_of_motion_deg"], 55)

    def test_unfinished_rep(self):
        counter = SquatCounter(self.settings)
        self.run_angles(counter, [165, 165, 150, 120, 95, 90])
        self.assertEqual(counter.count, 0)

    def test_shallow_bend(self):
        counter = SquatCounter(self.settings)
        self.run_angles(counter, [165, 165, 150, 130, 120, 135, 155, 165])
        self.assertEqual(counter.count, 0)

    def test_missing_keypoints_reset_rep(self):
        counter = SquatCounter(self.settings)
        self.run_angles(counter, [165, 165, 140, 115, 95, 90])
        counter.missing(2.0)
        self.run_angles(counter, [110, 155, 165, 165], offset=2.1)
        self.assertEqual(counter.count, 0)
        self.assertGreaterEqual(counter.discarded, 1)

    def test_two_reps(self):
        counter = SquatCounter(self.settings)
        sequence = [165, 165, 150, 135, 110, 95, 90, 90,
                    110, 125, 145, 155, 165, 165]
        self.run_angles(counter, sequence + sequence[2:])
        self.assertEqual(counter.count, 2)

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            CountingSettings(bottom_angle=160, standing_angle=150)


class TestPersonSelection(unittest.TestCase):
    def test_one_person(self):
        pts = np.zeros((1, 26, 2))
        scores = np.ones((1, 26))
        p, s, status = single_person(pts, scores,
                                     {"hip": 12, "knee": 14, "ankle": 16}, 0.45)
        self.assertEqual(status, "tracked")

    def test_multiple_people_rejected(self):
        pts = np.zeros((2, 26, 2))
        scores = np.ones((2, 26))
        p, s, status = single_person(pts, scores,
                                     {"hip": 12, "knee": 14, "ankle": 16}, 0.45)
        self.assertIsNone(p)
        self.assertEqual(status, "multiple_people")

    def test_low_confidence_rejected(self):
        pts = np.zeros((1, 26, 2))
        scores = np.ones((1, 26))
        scores[0, 14] = 0.1
        _, _, status = single_person(pts, scores,
                                      {"hip": 12, "knee": 14, "ankle": 16}, 0.45)
        self.assertEqual(status, "low_confidence")


if __name__ == "__main__":
    unittest.main()
