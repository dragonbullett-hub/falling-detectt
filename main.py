# ==========================================
# main.py
# โปรแกรมหลัก
# ==========================================

import cv2

from camera import Camera
from fall_detector import FallDetector


def main():

    print()
    print("==========================================")
    print("       FALL DETECTION SYSTEM")
    print("==========================================")
    print("       OpenCV + YOLO Pose")
    print("       Phone Camera")
    print("==========================================")
    print()


   
    camera = Camera()


    # ตรวจสอบว่ากล้องเปิดได้หรือไม่
    if not camera.cap.isOpened():

        print("ไม่สามารถเปิดกล้องได้")

        return


    # --------------------------------------
    # สร้างระบบตรวจจับคนล้ม
    # --------------------------------------

    detector = FallDetector()


    print()
    print("ระบบพร้อมใช้งาน")
    print("กด Q เพื่อออก")
    print()


    # ======================================
    # Main Loop
    # ======================================

    while True:


        # อ่านภาพจากโทรศัพท์
        frame = camera.read()


        # ถ้าอ่านภาพไม่ได้
        if frame is None:

            print("ไม่สามารถรับภาพจากโทรศัพท์")

            break


        # ----------------------------------
        # ส่งภาพไปตรวจจับ
        # ----------------------------------

        frame = detector.detect(frame)


        # ----------------------------------
        # แสดงสถานะ
        # ----------------------------------

        if detector.fall_detected:

            cv2.putText(
                frame,
                "!!! FALL DETECTED !!!",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 0, 255),
                3
            )

        else:

            cv2.putText(
                frame,
                "STATUS: NORMAL",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )


        # ----------------------------------
        # แสดงหน้าต่าง
        # ----------------------------------

        cv2.imshow(
            "Fall Detection - Phone Camera",
            frame
        )


        # ----------------------------------
        # กด Q เพื่อออก
        # ----------------------------------

        key = cv2.waitKey(1) & 0xFF


        if key == ord("q"):

            break


    # ======================================
    # ปิดระบบ
    # ======================================

    camera.release()

    cv2.destroyAllWindows()

    print()
    print("ปิดระบบเรียบร้อย")
    print()


# ==========================================
# Start
# ==========================================

if __name__ == "__main__":

    main()