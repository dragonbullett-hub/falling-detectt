import unittest

from fall_logic import FallTemporalClassifier


class FallTemporalClassifierTests(unittest.TestCase):
    def setUp(self):
        self.detector = FallTemporalClassifier(
            rapid_drop_ratio=0.35,
            rapid_speed_ratio=0.55,
            transition_window=1.0,
            horizontal_confirm_time=0.20,
            recovery_time=0.50,
        )

    def feed(self, samples):
        result = False
        for timestamp, center_y, body_height, horizontal in samples:
            result = self.detector.update(
                timestamp=timestamp,
                center_y=center_y,
                body_height=body_height,
                horizontal=horizontal,
            )
        return result

    def test_rapid_descent_to_horizontal_is_fall(self):
        detected = self.feed([
            (0.00, 0.30, 0.50, False),
            (0.20, 0.34, 0.49, False),
            (0.40, 0.44, 0.44, False),
            (0.55, 0.54, 0.34, True),
            (0.80, 0.56, 0.31, True),
        ])
        self.assertTrue(detected)

    def test_slowly_lying_down_is_not_fall(self):
        detected = self.feed([
            (0.00, 0.30, 0.50, False),
            (0.50, 0.34, 0.48, False),
            (1.00, 0.39, 0.44, False),
            (1.50, 0.44, 0.40, False),
            (2.00, 0.49, 0.35, True),
            (2.30, 0.51, 0.32, True),
        ])
        self.assertFalse(detected)

    def test_already_lying_is_not_fall(self):
        detected = self.feed([
            (0.00, 0.55, 0.31, True),
            (0.30, 0.55, 0.31, True),
            (0.60, 0.56, 0.30, True),
            (0.90, 0.55, 0.31, True),
        ])
        self.assertFalse(detected)

    def test_alert_is_latched_until_recovery(self):
        first = self.feed([
            (0.00, 0.30, 0.50, False),
            (0.30, 0.40, 0.46, False),
            (0.50, 0.55, 0.32, True),
            (0.75, 0.56, 0.31, True),
        ])
        still_lying = self.detector.update(
            timestamp=2.00,
            center_y=0.56,
            body_height=0.31,
            horizontal=True,
        )
        self.assertTrue(first)
        self.assertFalse(still_lying)

        self.detector.update(
            timestamp=2.20,
            center_y=0.35,
            body_height=0.49,
            horizontal=False,
        )
        recovered = self.detector.update(
            timestamp=2.80,
            center_y=0.34,
            body_height=0.50,
            horizontal=False,
        )
        self.assertFalse(recovered)

        second = self.feed([
            (3.00, 0.34, 0.50, False),
            (3.20, 0.42, 0.45, False),
            (3.40, 0.55, 0.32, True),
            (3.65, 0.56, 0.31, True),
        ])
        self.assertTrue(second)


if __name__ == "__main__":
    unittest.main()
