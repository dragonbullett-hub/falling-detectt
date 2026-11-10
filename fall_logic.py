from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class PoseObservation:
    timestamp: float
    center_y: float
    body_height: float
    horizontal: bool


class FallTemporalClassifier:
    def __init__(
        self,
        rapid_drop_ratio,
        rapid_speed_ratio,
        transition_window,
        horizontal_confirm_time,
        recovery_time,
    ):
        self.rapid_drop_ratio = rapid_drop_ratio
        self.rapid_speed_ratio = rapid_speed_ratio
        self.transition_window = transition_window
        self.horizontal_confirm_time = horizontal_confirm_time
        self.recovery_time = recovery_time

        self.history = deque()
        self.horizontal_since = None
        self.rapid_transition_seen = False
        self.recovery_since = None
        self.active = False

    def update(self, *, timestamp, center_y, body_height, horizontal):
        observation = PoseObservation(
            timestamp=timestamp,
            center_y=center_y,
            body_height=max(body_height, 1e-6),
            horizontal=horizontal,
        )
        self.history.append(observation)
        self._trim_history(timestamp)

        if self.active:
            self._update_recovery(timestamp, horizontal)
            return False

        if not horizontal:
            self.horizontal_since = None
            self.rapid_transition_seen = False
            return False

        if self.horizontal_since is None:
            self.horizontal_since = timestamp

        self.rapid_transition_seen = (
            self.rapid_transition_seen
            or self._has_rapid_descent(observation)
        )

        if (
            self.rapid_transition_seen
            and timestamp - self.horizontal_since >= self.horizontal_confirm_time
        ):
            self.active = True
            self.recovery_since = None
            return True

        return False

    def _trim_history(self, timestamp):
        keep_after = timestamp - self.transition_window
        while self.history and self.history[0].timestamp < keep_after:
            self.history.popleft()

    def _has_rapid_descent(self, current):
        upright_samples = [
            sample
            for sample in self.history
            if not sample.horizontal and sample.timestamp < current.timestamp
        ]
        if not upright_samples:
            return False

        baseline = upright_samples[0]
        elapsed = current.timestamp - baseline.timestamp
        if elapsed <= 0 or elapsed > self.transition_window:
            return False

        drop = current.center_y - baseline.center_y
        drop_ratio = drop / baseline.body_height
        speed_ratio = drop_ratio / elapsed

        return (
            drop_ratio >= self.rapid_drop_ratio
            and speed_ratio >= self.rapid_speed_ratio
        )

    def _update_recovery(self, timestamp, horizontal):
        if horizontal:
            self.recovery_since = None
            return

        if self.recovery_since is None:
            self.recovery_since = timestamp
            return

        if timestamp - self.recovery_since >= self.recovery_time:
            self.active = False
            self.horizontal_since = None
            self.rapid_transition_seen = False
            self.recovery_since = None
            self.history.clear()
