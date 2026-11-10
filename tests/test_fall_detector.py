import http.client
import io
import json
import sys
import types
import unittest
import urllib.error
from unittest.mock import patch


if "ultralytics" not in sys.modules:
    fake_ultralytics = types.ModuleType("ultralytics")
    fake_ultralytics.YOLO = object
    sys.modules["ultralytics"] = fake_ultralytics

if "cv2" not in sys.modules:
    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.rectangle = lambda *a, **k: None
    fake_cv2.putText = lambda *a, **k: None
    fake_cv2.FONT_HERSHEY_SIMPLEX = 0
    sys.modules["cv2"] = fake_cv2

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

    def test_wide_upright_pose_is_not_labeled_horizontal(self):
        detector = self.build_detector([(0.00, 100, 100, 90)])

        with (
            patch.object(fall_detector.time, "monotonic", return_value=0.00),
            patch.object(fall_detector.cv2, "rectangle", return_value=None),
            patch.object(fall_detector.cv2, "putText", return_value=None) as put_text,
        ):
            detector.detect(object())

        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertNotIn("HORIZONTAL", labels)

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

    def test_alert_sends_telegram_message_when_configured(self):
        with (
            patch.object(fall_detector, "YOLO"),
            patch.object(fall_detector, "TELEGRAM_BOT_TOKEN", "123456:ABCDEF"),
            patch.object(fall_detector, "TELEGRAM_CHAT_ID", "987654321"),
            patch.object(fall_detector.urllib.request, "urlopen") as urlopen,
        ):
            detector = fall_detector.FallDetector()
            self.assertTrue(detector.alert())

        urlopen.assert_called_once()
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.telegram.org/bot123456:ABCDEF/sendMessage")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(
            json.loads(request.data.decode("utf-8")),
            {"chat_id": "987654321", "text": "⚠️ ตรวจพบการล้ม! กรุณาตรวจสอบทันที"},
        )
        self.assertEqual(request.get_header("Content-type"), "application/json")
        self.assertEqual(urlopen.call_args.kwargs, {"timeout": 5})

    def test_alert_respects_cooldown_between_telegram_sends(self):
        with (
            patch.object(fall_detector, "YOLO"),
            patch.object(fall_detector, "TELEGRAM_BOT_TOKEN", "123456:ABCDEF"),
            patch.object(fall_detector, "TELEGRAM_CHAT_ID", "987654321"),
            patch.object(fall_detector, "ALERT_COOLDOWN", 10),
            patch.object(fall_detector.time, "monotonic", side_effect=[10.0, 15.0, 21.0]),
            patch.object(fall_detector.urllib.request, "urlopen") as urlopen,
        ):
            detector = fall_detector.FallDetector()
            self.assertTrue(detector.alert())
            self.assertEqual(detector.last_alert_time, 10.0)
            self.assertFalse(detector.alert())
            self.assertEqual(detector.last_alert_time, 10.0)
            self.assertEqual(urlopen.call_count, 1)
            self.assertTrue(detector.alert())
            self.assertEqual(detector.last_alert_time, 21.0)
            self.assertEqual(urlopen.call_count, 2)

    def test_alert_skips_when_telegram_not_configured(self):
        for token, chat_id in [
            ("", "987654321"),
            ("123456:ABCDEF", ""),
            (" \t", "987654321"),
            ("123456:ABCDEF", " \t"),
        ]:
            with self.subTest(token_missing=not token.strip(), chat_missing=not chat_id.strip()):
                with (
                    patch.object(fall_detector, "YOLO"),
                    patch.object(fall_detector, "TELEGRAM_BOT_TOKEN", token),
                    patch.object(fall_detector, "TELEGRAM_CHAT_ID", chat_id),
                    patch.object(fall_detector.urllib.request, "urlopen") as urlopen,
                ):
                    detector = fall_detector.FallDetector()
                    self.assertFalse(detector.alert())
                    self.assertIsNone(detector.last_alert_time)
                    urlopen.assert_not_called()

    def test_alert_handles_network_error_without_raising(self):
        errors = [
            urllib.error.URLError("network down"),
            urllib.error.URLError("network down: 123456:ABCDEF"),
            urllib.error.HTTPError(
                "https://api.telegram.org/bot123456:ABCDEF/sendMessage",
                403,
                "forbidden",
                None,
                None,
            ),
            OSError("connection failed: 123456:ABCDEF"),
            ValueError("invalid request: 123456:ABCDEF"),
            http.client.InvalidURL("invalid URL: 123456:ABCDEF"),
            http.client.BadStatusLine("bad response: 123456:ABCDEF"),
        ]
        for error in errors:
            with self.subTest(error_type=type(error).__name__):
                with (
                    patch.object(fall_detector, "YOLO"),
                    patch.object(fall_detector, "TELEGRAM_BOT_TOKEN", "123456:ABCDEF"),
                    patch.object(fall_detector, "TELEGRAM_CHAT_ID", "987654321"),
                    patch.object(fall_detector, "ALERT_COOLDOWN", 10),
                    patch.object(fall_detector.time, "monotonic", side_effect=[10.0, 11.0]),
                    patch.object(fall_detector.urllib.request, "urlopen") as urlopen,
                    patch("sys.stdout", new_callable=io.StringIO) as stdout,
                ):
                    detector = fall_detector.FallDetector()
                    urlopen.side_effect = error
                    self.assertFalse(detector.alert())
                    self.assertIsNone(detector.last_alert_time)
                    self.assertNotIn("123456:ABCDEF", stdout.getvalue())
                    self.assertNotIn(str(error), stdout.getvalue())
                    urlopen.side_effect = None
                    self.assertTrue(detector.alert())
                    self.assertEqual(detector.last_alert_time, 11.0)
                    self.assertEqual(urlopen.call_count, 2)
                    if isinstance(error, urllib.error.HTTPError):
                        self.assertTrue(error.closed)


if __name__ == "__main__":
    unittest.main()
