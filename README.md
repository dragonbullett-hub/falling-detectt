# falling-detect

ตรวจจับการล้มของคนจากกล้องแบบเรียลไทม์ ด้วย YOLO Pose + temporal fall classification.

## วิธีติดตั้ง

### บน Raspberry Pi (Bookworm)

```bash
# 1. ลง OS ล่าสุดแล้ว sudo raspi-config เปิด Camera, SSH
sudo apt update
sudo apt install -y python3-pip python3-opencv python3-picamera2

# 2. ติดตั้ง dependencies
pip3 install ultralytics opencv-python-headless
```

### บน Desktop / Dev

```bash
pip3 install ultralytics opencv-python-headless
```

## การตั้งค่า

แก้ไข `config.py`:

| ค่าคงที่ | หน้าที่ | ค่าเริ่มต้น |
|----------|---------|------------|
| `PHONE_CAMERA_URL` | IP ของกล้อง (IP Webcam / Pi Cam stream) | `http://192.168.1.102:8080/video` |
| `FALL_RATIO` | width/height ขั้นต่ำที่จะถือว่าเป็นท่าผ่าชิง | `1.2` |
| `FALL_DROP_RATIO` | ต้องลดระดับลงอย่างน้อยกี่เท่าของความสูงร่างกาย | `0.35` |
| `FALL_SPEED_RATIO` | ความเร็วต่ำสุด (body-heights/วินาที) | `0.55` |
| `FALL_TRANSITION_WINDOW` | หน้าต่างเวลาย้อนหลังสำหรับ rapid descent (วินาที) | `1.0` |
| `FALL_HORIZONTAL_CONFIRM_TIME` | ต้องอยู่ horizontal นาน enough ก่อนยืนยัน (วินาที) | `0.20` |
| `FALL_RECOVERY_TIME` | ต้องกลับมายืนต่อเนื่องนานเท่าไรจึง reset event (วินาที) | `0.50` |
| `CONFIDENCE` | ความมั่นใจขั้นต่ำของ YOLO | `0.5` |

## วิธีใช้งาน

### วิธีที่ 1: ใช้ IP Webcam (โทรศัพท์ / Pi Cam เป็น HTTP stream)

```bash
# บนโทรศัพท์: ติดตั้ง "IP Webcam" จาก Play Store → Start Server
# จะได้ URL เช่น http://192.168.1.102:8080/video

# ใส่ IP ลง config.py แล้วรัน:
python3 main.py
```

### วิธีที่ 2: ใช้ Pi Camera โดยตรง (Raspberry Pi)

```bash
# เปิด Pi Camera:
sudo raspi-config  → Interface Options → Camera → Enable

# รัน:
python3 main.py
```

## รันบน Raspberry Pi — Tips

** bottlenecks:**

1. **YOLO inference ช้า**
   - ใช้ `yolo11n-pose.pt` (nano) แล้ว
   - ปรับลด `CONFIDENCE` เป็น `0.4` เพื่อให้ detect เร็วขึ้น
   - ใช้ ONNX Runtime: ดูหัวข้อ `Optimization` ด้านล่าง

2. **FPS ต่ำ** — เพิ่ม `INFERENCE_SKIP = 1` ใน `config.py` (skip เฟรม):
   ```python
   INFERENCE_SKIP = 1  # ประมวลผลทุก 2 เฟรม
   ```

3. **ไม่มี display** — รันแบบ headless:
   ```bash
   python3 main.py 2>&1 | tee fall.log
   ```

## Optimization

### ONNX Runtime (เร็วขึ้นบน ARM)

```python
# ใน fall_detector.py หรือ script แยก:
model = YOLO("yolo11n-pose.pt")
model.export(format="onnx", half=True)  # สร้าง yolo11n-pose.onnx

# รันด้วย onnxruntime:
# pip3 install onnxruntime
```

### TensorRT Lite (Pi 5 / Coral USB Accelerator)

```bash
# Coral USB Accelerator:
pip3 install tflite-runtime
# ใช้模型 quantized ที่ 8-bit
```

## โครงสร้างไฟล์

```
falling-detect/
├── config.py              # ค่าคงที่
├── fall_detector.py       # YOLO + pose extraction + render
├── fall_logic.py          # Temporal fall classifier (PoseObservation, FallTemporalClassifier)
├── camera.py              # รับภาพจาก IP camera
├── main.py                # entry point
├── tests/
│   ├── test_fall_logic.py      # unit tests: temporal classifier
│   └── test_fall_detector.py   # integration tests: detector + fake YOLO
└── yolo11n-pose.pt        # YOLO Pose model (nano)
```

## รัน Tests

```bash
cd /home/mean/tmp/falling-detect
python3 -m unittest tests.test_fall_logic tests.test_fall_detector -v
```

## หมายเหตุ

- ตอนนี้รองรับ **หนึ่งคน** เป็น primary subject (pose ที่มีพื้นที่มากที่สุด)
- multi-person tracking ยังไม่ครอบคลุม
- `fall_logic.py` เป็น state machine แยก อิสระจาก YOLO — สามารถ replace เป็น model อื่นได้
