# ==========================================
# camera.py
# รับภาพจากกล้อง USB (เว็บแคม) หรือกล้อง IP ผ่าน Wi-Fi
# ==========================================

import sys

import cv2

from config import CAMERA_SOURCE, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS


def _open_capture(source):
    """เปิดกล้องโดยเลือก backend ที่เหมาะกับระบบปฏิบัติการ"""

    if isinstance(source, int):
        if sys.platform.startswith("win"):
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        elif sys.platform.startswith("linux"):
            backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
        else:
            backends = [cv2.CAP_ANY]

        for backend in backends:
            cap = cv2.VideoCapture(source, backend)
            if cap.isOpened():
                return cap
            cap.release()

        return cv2.VideoCapture(source)

    # สตรีมผ่านเครือข่าย (URL)
    return cv2.VideoCapture(source)


class Camera:

    # จำนวนครั้งที่อ่านภาพพลาดติดกันก่อนจะถือว่ากล้องหลุด
    MAX_FAILED_READS = 30

    def __init__(self, source=CAMERA_SOURCE):

        self.source = source
        self.is_usb = isinstance(source, int)
        self.failed_reads = 0

        print("กำลังเชื่อมต่อกล้อง...")
        print("แหล่งภาพ:", f"USB index {source}" if self.is_usb else source)

        self.cap = _open_capture(source)

        if not self.cap.isOpened():
            print("ไม่สามารถเปิดกล้องได้")
            if self.is_usb:
                print("ตรวจสอบว่าเสียบกล้อง USB แล้ว, ไม่มีโปรแกรมอื่นใช้กล้องอยู่")
                print("ลองเปลี่ยน CAMERA_SOURCE ใน config.py เป็น 1 หรือ 2")
                if sys.platform.startswith("linux"):
                    print("บน Linux ลองรัน: ls /dev/video*")
            else:
                print("ตรวจสอบ IP, พอร์ต และการเชื่อมต่อ Wi-Fi")
            return

        if self.is_usb:
            # MJPG ช่วยให้ได้ FPS สูงบนกล้อง USB ส่วนใหญ่
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"เชื่อมต่อกล้องสำเร็จ! ความละเอียด {width}x{height}")

    def read(self):

        ret, frame = self.cap.read()

        if not ret or frame is None:
            self.failed_reads += 1

            # กล้อง USB บางตัวอาจพลาดบางเฟรมตอนเริ่มต้น จึงลองใหม่ก่อน
            if self.failed_reads < self.MAX_FAILED_READS:
                return self._empty_frame()

            return None

        self.failed_reads = 0
        return frame

    def _empty_frame(self):
        """คืนภาพว่างชั่วคราวเมื่ออ่านพลาด เพื่อไม่ให้โปรแกรมหยุดทันที"""
        import numpy as np

        return np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)

    def release(self):

        self.cap.release()
