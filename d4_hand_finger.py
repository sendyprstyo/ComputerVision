import cv2
from cvzone.HandTrackingModule import HandDetector

# Buka kamera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Fullscreen window
cv2.namedWindow("Hand Detection + Fingers", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Hand Detection + Fingers", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Detector tangan
detector = HandDetector(
    staticMode=False,
    maxHands=1,
    modelComplexity=1,
    detectionCon=0.5,
    minTrackCon=0.5
)

while True:
    ok, img = cap.read()
    if not ok:
        break

    # Resize agar sesuai fullscreen
    img = cv2.resize(img, (1280, 720))

    hands, img = detector.findHands(img, draw=True, flipType=True)

    if hands:
        hand = hands[0]
        
        # Hitung jari (dari cvzone)
        fingers = detector.fingersUp(hand)    # contoh: [1,1,0,0,1]
        count = sum(fingers)

        # Tampilkan hasil
        cv2.putText(img, f"Fingers: {count}", (40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 5)

        cv2.putText(img, f"{fingers}", (40, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,0), 3)

    cv2.imshow("Hand Detection + Fingers", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
