import cv2
import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector

# Landmark mata kiri (EAR)
L_TOP = 159
L_BOTTOM = 145
L_LEFT = 33
L_RIGHT = 133

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Buka webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # kualitas besar
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Fullscreen window
cv2.namedWindow("Face Mesh + EAR", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Face Mesh + EAR", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# FaceMesh
detector = FaceMeshDetector(
    staticMode=False,
    maxFaces=1,
    minDetectionCon=0.5,
    minTrackCon=0.5
)

# Variabel EAR
blink_count = 0
closed_frames = 0
EYE_AR_THRESHOLD = 0.20
CLOSED_FRAMES_THRESHOLD = 3
is_closed = False

while True:
    ok, img = cap.read()
    if not ok:
        break

    # Resize agar fullscreen tetap proporsional
    img = cv2.resize(img, (1280, 720))

    img, faces = detector.findFaceMesh(img, draw=True)

    if faces:
        face = faces[0]

        v = distance(face[L_TOP], face[L_BOTTOM])
        h = distance(face[L_LEFT], face[L_RIGHT])
        ear = v / (h + 1e-8)

        cv2.putText(img, f"EAR: {ear:.3f}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3)

        if ear < EYE_AR_THRESHOLD:
            closed_frames += 1
            if closed_frames >= CLOSED_FRAMES_THRESHOLD and not is_closed:
                blink_count += 1
                is_closed = True
        else:
            closed_frames = 0
            is_closed = False

        cv2.putText(img, f"Blinks: {blink_count}", (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

    cv2.imshow("Face Mesh + EAR", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:   # ESC atau q
        break

cap.release()
cv2.destroyAllWindows()
