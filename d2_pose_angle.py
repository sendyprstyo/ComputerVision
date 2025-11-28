import cv2
import numpy as np
from cvzone.PoseModule import PoseDetector

# Buka webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka. Coba ganti index 0/1/2.")

# Buat PoseDetector
detector = PoseDetector(
    staticMode=False,
    modelComplexity=1,
    enableSegmentation=False,
    detectionCon=0.5,
    trackCon=0.5
)

while True:
    ok, img = cap.read()
    if not ok:
        break

    # Deteksi pose
    img = detector.findPose(img)

    # Ambil daftar landmark
    lmList, bboxInfo = detector.findPosition(img, draw=True, bboxWithHands=False)

    if lmList:
        # Titik bahu kiri = 11
        # Titik siku kiri = 13
        # Titik pergelangan kiri = 15

        # Hitung sudut siku kiri
        angle, img = detector.findAngle(
            lmList[11][0:2],   # shoulder
            lmList[13][0:2],   # elbow
            lmList[15][0:2],   # wrist
            img=img,
            color=(0, 0, 255),
            scale=10
        )

        # Tampilkan tulisan sudut
        cv2.putText(img, f"Angle: {angle:.1f}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Cek apakah mendekati 50 derajat
        isClose = detector.angleCheck(myAngle=angle, targetAngle=50, offset=10)

        cv2.putText(img, f"Match 50 deg? {isClose}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    cv2.imshow("Pose + Angle", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
