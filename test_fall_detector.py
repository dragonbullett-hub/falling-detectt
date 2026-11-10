import sys
import types
import unittest
from unittest.mock import patch


if "ultralytics" not in sys.modules:
    fake_ultralytics = types.ModuleType("ultralytics")
    fake_ultralytics.YOLO = object
    sys.modules["ultralytics"] = fake_ultralytics

import fall_detector


class FakeKeypoints:
    def __init__(self, people):
        self.xy = people


class FakeResult:
    def __init__(self, people):
        self.keypoints = FakeKeypoints(people)


class FakeModel:
    def __init__(self, frames):
        self.frames = iter(frames)

    def __call__(self, frame, conf, verbose):
        return [FakeResult(next(self.frames))]


def pose(center_y, width, height):
    min_x = 100.0
    max_x = min_x + width
    min_y = center_y - height / 2
    max_y = center_y + height / 2
    points = [
        (min_x, min_y),
        (max_x, min_y),
        (min_x, max_y),
        (max_x, max_y),
        ((min_x + max_x) / 2, center_y),
    ]
    return points + [points[-1]] * 12


class FallDetectorIntegrationTests(unittest.TestCase):
    def build_detector(self, samples):
        frames = [[pose(center_y, width, height)] for _, center_y, width, height in samples]
        model = FakeModel(frames)
        with patch.object(fall_detector, "YOLO", return_value=model):
            detector = fall_detector.FallDetector()
        detector.alert_count = 0
        detector.alert = lambda: setattr(detector, "alert_count", detector.alert_count + 1)
        return detector

    def run_samples(self, samples):
        detector = self.build_detector(samples)
        timestamps = [sample[0] for sample in samples]
        with (
            patch.object(fall_detector.time, "monotonic", side_effect=timestamps),
            patch.object(fall_detector.cv2, "rectangle", return_value=None),
            patch.object(fall_detector.cv2, "putText", return_value=None),
        ):
            for _ in samples:
                detector.detect(object())
        return detector

    def test_rapid_drop_then_horizontal_alerts_once(self):
        detector = self.run_samples([
            (0.00, 100, 50, 100),
            (0.20, 105, 50, 100),
            (0.40, 125, 55, 90),
            (0.55, 145, 100, 50),
            (0.80, 148, 100, 50),
            (1.20, 148, 100, 50),
            (1.60, 148, 100, 50),
        ])
        self.assertEqual(detector.alert_count, 1)
        self.assertTrue(detector.fall_detected)

    def test_slow_lie_down_does_not_alert_even_when_horizontal_persists(self):
        samples = [
            (0.00, 100, 50, 100),
            (0.50, 105, 50, 100),
            (1.00, 110, 50, 100),
            (1.50, 115, 55, 90),
        ]
        samples.extend(
            (2.00 + index * 0.10, 120, 100, 50)
            for index in range(20)
        )
        detector = self.run_samples(samples)
        self.assertEqual(detector.alert_count, 0)
        self.assertFalse(detector.fall_detected)


if __name__ == "__main__":
    unittest.main()
