# ==========================================
# fall_detector.py
# YOLO Pose + Temporal Fall Detection
# ==========================================

import cv2
import time

from ultralytics import YOLO

from config import (
    CONFIDENCE,
    FALL_RATIO,
    FALL_DROP_RATIO,
    FALL_SPEED_RATIO,
    FALL_TRANSITION_WINDOW,
    FALL_HORIZONTAL_CONFIRM_TIME,
    FALL_RECOVERY_TIME,
)
from fall_logic import FallTemporalClassifier


class FallDetector:

    def __init__(self):

        print("กำลังโหลด YOLO Pose...")

        self.model = YOLO("yolo11n-pose.pt")

        print("โหลด YOLO Pose สำเร็จ")

        self.classifier = FallTemporalClassifier(
            rapid_drop_ratio=FALL_DROP_RATIO,
            rapid_speed_ratio=FALL_SPEED_RATIO,
            transition_window=FALL_TRANSITION_WINDOW,
            horizontal_confirm_time=FALL_HORIZONTAL_CONFIRM_TIME,
            recovery_time=FALL_RECOVERY_TIME,
        )

        self.fall_detected = False


    def detect(self, frame):

        results = self.model(
            frame,
            conf=CONFIDENCE,
            verbose=False
        )

        poses = []

        for result in results:

            if result.keypoints is None:
                continue

            keypoints = result.keypoints.xy

            for points in keypoints:

                x_points = []
                y_points = []

                for point in points:
                    x = float(point[0])
                    y = float(point[1])

                    if x > 0 and y > 0:
                        x_points.append(x)
                        y_points.append(y)

                if len(x_points) < 5:
                    continue

                min_x = min(x_points)
                max_x = max(x_points)
                min_y = min(y_points)
                max_y = max(y_points)

                width = max_x - min_x
                height = max_y - min_y

                if height <= 0:
                    continue

                ratio = width / height
                horizontal = ratio > FALL_RATIO
                center_y = (min_y + max_y) / 2
                area = width * height

                poses.append({
                    "min_x": min_x,
                    "max_x": max_x,
                    "min_y": min_y,
                    "max_y": max_y,
                    "height": height,
                    "center_y": center_y,
                    "horizontal": horizontal,
                    "area": area,
                })

        # PR 1 keeps detection single-subject: the largest visible pose is the
        # primary subject. Stable multi-person tracking remains out of scope.
        primary_pose = max(poses, key=lambda pose: pose["area"], default=None)

        event_detected = False

        if primary_pose is not None:
            event_detected = self.classifier.update(
                timestamp=time.monotonic(),
                center_y=primary_pose["center_y"],
                body_height=primary_pose["height"],
                horizontal=primary_pose["horizontal"],
            )
            self.fall_detected = self.classifier.active
        else:
            self.fall_detected = self.classifier.active

        for pose in poses:
            x1 = int(pose["min_x"])
            y1 = int(pose["min_y"])
            x2 = int(pose["max_x"])
            y2 = int(pose["max_y"])

            is_primary = pose is primary_pose
            suspicious = is_primary and pose["horizontal"]

            color = (0, 0, 255) if suspicious else (0, 255, 0)
            thickness = 3 if suspicious else 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            if suspicious:
                label = "FALL" if self.fall_detected else "HORIZONTAL"
                cv2.putText(
                    frame,
                    label,
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    3,
                )

        if event_detected:
            self.alert()

        return frame


    def alert(self):

        print()
        print("================================")
        print("⚠️  ตรวจพบการล้ม!")
        print("================================")
        print()