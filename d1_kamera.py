import cv2
import time

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka. Coba ganti index kamera: 0/1/2")

frames = 0
t0 = time.time()

cv2.namedWindow("Preview")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frames += 1

    if time.time() - t0 >= 1.0:
        cv2.setWindowTitle("Preview", f"Preview (FPS ~ {frames})")
        frames = 0
        t0 = time.time()

    cv2.imshow("Preview", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
