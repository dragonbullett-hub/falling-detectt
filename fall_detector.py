# ==========================================
# fall_detector.py
# YOLO Pose + Temporal Fall Detection
# ==========================================

import http.client
import json
import secrets
import time
import urllib.error
import urllib.request

import cv2

from ultralytics import YOLO

from config import (
    ALERT_COOLDOWN,
    CONFIDENCE,
    FALL_RATIO,
    FALL_DROP_RATIO,
    FALL_SPEED_RATIO,
    FALL_TRANSITION_WINDOW,
    FALL_HORIZONTAL_CONFIRM_TIME,
    FALL_RECOVERY_TIME,
    TELEGRAM_ALERT_MESSAGE,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    TELEGRAM_TIMEOUT,
)
from fall_logic import FallTemporalClassifier


def send_telegram_alert(
    message: str = TELEGRAM_ALERT_MESSAGE,
    bot_token: str | None = None,
    chat_id: str | None = None,
    timeout: float = TELEGRAM_TIMEOUT,
    frame=None,
) -> bool:
    token = (TELEGRAM_BOT_TOKEN if bot_token is None else bot_token) or ""
    target_chat_id = (TELEGRAM_CHAT_ID if chat_id is None else chat_id) or ""
    token = token.strip()
    target_chat_id = target_chat_id.strip()
    if not token or not target_chat_id:
        return False

    if frame is None:
        endpoint = "sendMessage"
        payload = json.dumps({"chat_id": target_chat_id, "text": message}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
    else:
        try:
            encoded, image = cv2.imencode(".jpg", frame)
        except Exception:
            encoded = False

        if not encoded:
            print("ไม่สามารถบันทึกภาพแจ้งเตือน Telegram ได้")
            return send_telegram_alert(
                message=message,
                bot_token=token,
                chat_id=target_chat_id,
                timeout=timeout,
            )

        endpoint = "sendPhoto"
        boundary = f"----fall-detection-{secrets.token_hex(16)}"
        payload = b"".join((
            (
                f"--{boundary}\r\n"
                'Content-Disposition: form-data; name="chat_id"\r\n\r\n'
                f"{target_chat_id}\r\n"
                f"--{boundary}\r\n"
                'Content-Disposition: form-data; name="caption"\r\n\r\n'
                f"{message}\r\n"
                f"--{boundary}\r\n"
                'Content-Disposition: form-data; name="photo"; filename="fall.jpg"\r\n'
                "Content-Type: image/jpeg\r\n\r\n"
            ).encode("utf-8"),
            image.tobytes(),
            f"\r\n--{boundary}--\r\n".encode("utf-8"),
        ))
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}

    url = f"https://api.telegram.org/bot{token}/{endpoint}"
    try:
        request = urllib.request.Request(
            url,
            data=payload,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout):
            return True
    except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException) as exc:
        if isinstance(exc, urllib.error.HTTPError):
            exc.close()
        print(f"ไม่สามารถส่งแจ้งเตือน Telegram ได้: {type(exc).__name__}")

        return False



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
        self.last_alert_time = None


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
        timestamp = time.monotonic()
        event_detected = False

        if primary_pose is not None:
            event_detected = self.classifier.update(
                timestamp=timestamp,
                center_y=primary_pose["center_y"],
                body_height=primary_pose["height"],
                horizontal=primary_pose["horizontal"],
            )
        else:
            self.classifier.update_missing(timestamp=timestamp)

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
            self.alert(frame)

        return frame

    def alert(self, frame=None) -> bool:
        print()
        print("================================")
        print("⚠️  ตรวจพบการล้ม!")
        print("================================")
        print()

        now = time.monotonic()
        if self.last_alert_time is not None and (now - self.last_alert_time) < ALERT_COOLDOWN:
            return False

        sent = send_telegram_alert(frame=frame)
        if sent:
            self.last_alert_time = now

        return sent