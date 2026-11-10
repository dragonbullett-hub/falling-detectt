# ==========================================
# config.py
# ตั้งค่าระบบตรวจจับคนล้ม
# ==========================================

import os
from pathlib import Path

import truststore


def _load_dotenv():
    dotenv_path = Path(__file__).with_name(".env")
    if not dotenv_path.is_file():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

truststore.inject_into_ssl()

# -----------------------------
# Camera
# -----------------------------

# เลือกแหล่งภาพ:
#   0, 1, 2 ...  = กล้อง USB / เว็บแคม (เลขลำดับอุปกรณ์ เริ่มจาก 0)
#   "http://..." = กล้องโทรศัพท์ผ่าน Wi-Fi (IP Webcam)
CAMERA_SOURCE = 0

# URL กล้องโทรศัพท์ (ใช้เมื่อต้องการสลับกลับไปใช้โทรศัพท์:
# เปลี่ยน CAMERA_SOURCE = PHONE_CAMERA_URL)
PHONE_CAMERA_URL = "http://192.168.1.102:8080/video"

# ความละเอียดที่ขอจากกล้อง USB (กล้องจะใช้ค่าใกล้เคียงที่รองรับ)
# ยิ่งต่ำยิ่งเร็ว 640x480 เหมาะกับ YOLO nano
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS = 60

# ข้ามเฟรมก่อนส่งเข้า YOLO: 0 = ประมวลผลทุกเฟรม, 1 = ทุก 2 เฟรม
INFERENCE_SKIP = 0

# ปิดหน้าต่าง OpenCV เมื่อรันบนเครื่องที่ไม่มี display
HEADLESS = os.getenv("HEADLESS", "").lower() in {"1", "true", "yes"}




# -----------------------------
# YOLO
# -----------------------------

# ความมั่นใจขั้นต่ำของ YOLO
CONFIDENCE = 0.5


# -----------------------------
# Fall Detection
# -----------------------------

# ต้องกว้างกว่าสูงอย่างชัดเจนเพื่อไม่ให้ท่ายืนที่กางแขนถูกตีความเป็นท่านอน
FALL_RATIO = 1.3

# ต้องลดระดับลงอย่างรวดเร็วอย่างน้อยกี่เท่าของความสูงร่างกาย
FALL_DROP_RATIO = 0.20

# ความเร็วการลดระดับขั้นต่ำ (body-heights / second)
FALL_SPEED_RATIO = 0.30

# การเปลี่ยนจากยืนเป็นแนวนอนต้องเกิดภายในกี่วินาที
FALL_TRANSITION_WINDOW = 1.0

# ต้องคงท่าแนวนอนกี่วินาทีหลัง rapid descent จึงยืนยันการล้ม
FALL_HORIZONTAL_CONFIRM_TIME = 0.15

# ต้องกลับมาไม่เป็นแนวนอนต่อเนื่องกี่วินาทีจึง reset event
FALL_RECOVERY_TIME = 0.50

# -----------------------------
# Alert
# -----------------------------

# เวลาระหว่างการแจ้งเตือนแต่ละครั้ง
# หน่วยเป็นวินาที
ALERT_COOLDOWN = 10

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_ALERT_MESSAGE = "⚠️ ตรวจพบการล้ม! กรุณาตรวจสอบทันที"
TELEGRAM_TIMEOUT = 5