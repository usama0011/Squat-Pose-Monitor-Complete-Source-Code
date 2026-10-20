"""Geometry and repetition counting for side-view squat videos.

This module does not depend on the pose model, so it can be tested with
coordinates and angle sequences before processing real videos.
"""

from collections import deque
from dataclasses import dataclass
from math import acos, degrees, isfinite
from statistics import median

import numpy as np


# Halpe26 landmarks (0-based indexing, RTMLib BodyWithFeet output)
LEGS = {
    "left": {"shoulder": 5, "hip": 11, "knee": 13, "ankle": 15},
    "right": {"shoulder": 6, "hip": 12, "knee": 14, "ankle": 16},
}


def joint_angle(a, vertex, b):
    """Return the interior 2D angle at `vertex`, in degrees.

    Invalid or overlapping points return None instead of producing NaNs.
    """
    points = [np.asarray(p, dtype=float)[:2] for p in (a, vertex, b)]
    if any(p.shape != (2,) or not np.all(np.isfinite(p)) for p in points):
        return None

    first = points[0] - points[1]
    second = points[2] - points[1]
    lengths = np.linalg.norm(first) * np.linalg.norm(second)
    if lengths < 1e-8:
        return None

    cosine = float(np.dot(first, second) / lengths)
    return degrees(acos(np.clip(cosine, -1.0, 1.0)))


def leg_measurement(points, confidence, side="right", min_confidence=0.45):
    """Measure the knee angle when all three landmarks are reliable."""
    ids = LEGS[side]
    required = (ids["hip"], ids["knee"], ids["ankle"])
    points = np.asarray(points)
    confidence = np.asarray(confidence)

    if points.ndim != 2 or confidence.ndim != 1:
        return None
    if points.shape[0] < 26 or confidence.shape[0] < 26:
        return None
    if any(not isfinite(float(confidence[i])) or confidence[i] < min_confidence
           for i in required):
        return None

    return joint_angle(points[required[0]], points[required[1]], points[required[2]])


@dataclass
class CountingSettings:
    # Initial values for side-view experiments, not medical form criteria.
    standing_angle: float = 150.0
    bottom_angle: float = 105.0
    confirm_seconds: float = 0.12
    min_rep_seconds: float = 0.65
    max_rep_seconds: float = 12.0
    max_missing_seconds: float = 0.4
    smoothing_window: int = 5

    def __post_init__(self):
        if not 0 < self.bottom_angle < self.standing_angle < 180:
            raise ValueError("Require 0 < bottom_angle < standing_angle < 180")
        if self.smoothing_window < 1:
            raise ValueError("Smoothing window must be at least 1")
        if min(self.confirm_seconds, self.min_rep_seconds,
               self.max_rep_seconds, self.max_missing_seconds) < 0:
            raise ValueError("Time settings cannot be negative")


class SquatCounter:
    """Count a complete standing -> bottom -> standing sequence.

    The counter only reports movement cycles, not whether a squat was done
    correctly. Failed tracking can interrupt and discard an unfinished rep.
    """

    def __init__(self, settings=None):
        self.settings = settings or CountingSettings()
        self.recent_angles = deque(maxlen=self.settings.smoothing_window)
        self.phase = "waiting"
        self.pending_phase = None
        self.pending_since = None
        self.start_time = None
        self.last_valid_time = None
        self.samples = []
        self.repetitions = []
        self.discarded = 0

    @property
    def count(self):
        return len(self.repetitions)

    def _confirmed(self, phase, timestamp):
        if self.pending_phase != phase:
            self.pending_phase = phase
            self.pending_since = timestamp
            return self.settings.confirm_seconds == 0
        return timestamp - self.pending_since >= self.settings.confirm_seconds

    def _clear_confirmation(self):
        self.pending_phase = None
        self.pending_since = None

    def _reset_cycle(self):
        if self.start_time is not None:
            self.discarded += 1
        self.phase = "waiting"
        self.start_time = None
        self.samples = []
        self._clear_confirmation()

    def missing(self, timestamp):
        """Mark a video frame in which no reliable knee angle is available."""
        self.recent_angles.clear()
        self._clear_confirmation()
        if (self.last_valid_time is not None
                and timestamp - self.last_valid_time > self.settings.max_missing_seconds):
            self._reset_cycle()

    def add(self, knee_angle, timestamp):
        """Consume a knee angle and return (smoothed_angle, rep_if_completed)."""
        if knee_angle is None or not isfinite(float(knee_angle)):
            self.missing(timestamp)
            return None, None

        self.last_valid_time = timestamp
        self.recent_angles.append(float(knee_angle))
        angle = float(median(self.recent_angles))
        result = None
        cfg = self.settings

        if self.phase == "waiting":
            if angle >= cfg.standing_angle:
                if self._confirmed("standing", timestamp):
                    self.phase = "standing"
                    self._clear_confirmation()
            else:
                self._clear_confirmation()

        elif self.phase == "standing":
            if angle < cfg.standing_angle:
                if self.start_time is None:
                    self.start_time = timestamp
                self.samples.append((timestamp, angle))
            elif self.start_time is not None:
                # Stood back up without getting low enough to count.
                self.start_time = None
                self.samples = []

            if angle <= cfg.bottom_angle:
                if self._confirmed("bottom", timestamp):
                    self.phase = "bottom"
                    self._clear_confirmation()
            else:
                self._clear_confirmation()

        elif self.phase == "bottom":
            self.samples.append((timestamp, angle))
            if timestamp - self.start_time > cfg.max_rep_seconds:
                self._reset_cycle()
            elif angle >= cfg.standing_angle:
                if self._confirmed("standing", timestamp):
                    duration = timestamp - self.start_time
                    if duration >= cfg.min_rep_seconds:
                        values = [sample[1] for sample in self.samples]
                        result = {
                            "rep": self.count + 1,
                            "start_s": round(self.start_time, 3),
                            "end_s": round(timestamp, 3),
                            "duration_s": round(duration, 3),
                            "min_knee_deg": round(min(values), 2),
                            "max_knee_deg": round(max(values), 2),
                            "range_of_motion_deg": round(max(values) - min(values), 2),
                        }
                        self.repetitions.append(result)
                    else:
                        self.discarded += 1
                    self.phase = "standing"
                    self.start_time = None
                    self.samples = []
                    self._clear_confirmation()
            else:
                self._clear_confirmation()

        return angle, result
