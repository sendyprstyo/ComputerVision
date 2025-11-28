import cv2
import numpy as np
from collections import deque
from cvzone.PoseModule import PoseDetector

# Mode default (bisa diganti tekan 'm')
MODE = "squat"

# Buka kamera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Fullscreen window
cv2.namedWindow("Activity Counter", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Activity Counter", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Pose detector
detector = PoseDetector(
    staticMode=False,
    modelComplexity=1,
    enableSegmentation=False,
    detectionCon=0.5,
    trackCon=0.5
)

# Threshold setting
KNEE_DOWN = 80
KNEE_UP = 160

DOWN_R = 0.85
UP_R = 1.00

SAMPLE_OK = 4

count = 0
state = "up"
debounce = deque(maxlen=6)

def ratio_pushup(lm):
    sh = np.array(lm[11][1:3])
    wr = np.array(lm[15][1:3])
    hp = np.array(lm[23][1:3])
    return np.linalg.norm(sh - wr) / (np.linalg.norm(sh - hp) + 1e-8)

while True:
    ok, img = cap.read()
    if not ok:
        break

    img = cv2.resize(img, (1280, 720))

    img = detector.findPose(img, draw=True)
    lmList, _ = detector.findPosition(img, draw=False)

    flag = None

    if lmList:
        if MODE == "squat":
            angL, img = detector.findAngle(
                lmList[23][0:2], lmList[25][0:2], lmList[27][0:2],
                img=img, color=(0,0,255), scale=10
            )

            angR, img = detector.findAngle(
                lmList[24][0:2], lmList[26][0:2], lmList[28][0:2],
                img=img, color=(0,255,0), scale=10
            )

            ang = (angL + angR) / 2

            if ang < KNEE_DOWN:
                flag = "down"
            elif ang > KNEE_UP:
                flag = "up"

            cv2.putText(img, f"Knee Angle: {ang:.1f}", (30,140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 3)

        else:
            r = ratio_pushup(lmList)

            if r < DOWN_R:
                flag = "down"
            elif r > UP_R:
                flag = "up"

            cv2.putText(img, f"Ratio: {r:.2f}", (30,140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 3)

        debounce.append(flag)

        if debounce.count("down") >= SAMPLE_OK and state == "up":
            state = "down"

        if debounce.count("up") >= SAMPLE_OK and state == "down":
            state = "up"
            count += 1

    cv2.putText(img, f"Mode: {MODE.upper()}", (30,60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)

    cv2.putText(img, f"Count: {count}", (30,100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

    cv2.putText(img, f"State: {state}", (30,180),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)

    cv2.imshow("Activity Counter", img)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q') or key == 27:
        break

    if key == ord('m'):
        MODE = "pushup" if MODE == "squat" else "squat"

cap.release()
cv2.destroyAllWindows()
