import cv2
from cvzone.HandTrackingModule import HandDetector

# Buka kamera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Fullscreen window
cv2.namedWindow("Dual Hand + Fingers", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Dual Hand + Fingers", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Detector tangan
detector = HandDetector(
    staticMode=False,
    maxHands=2,              # <-- BISA DETEKSI 2 TANGAN
    modelComplexity=1,
    detectionCon=0.5,
    minTrackCon=0.5
)

while True:
    ok, img = cap.read()
    if not ok:
        break

    img = cv2.resize(img, (1280, 720))

    hands, img = detector.findHands(img, draw=True, flipType=True)

    if hands:
        # Jika dua tangan terbaca
        if len(hands) == 2:
            hand1 = hands[0]
            hand2 = hands[1]

            fingers1 = detector.fingersUp(hand1)
            fingers2 = detector.fingersUp(hand2)

            count1 = sum(fingers1)
            count2 = sum(fingers2)

            # Tangan 1
            cv2.putText(img, f"Left Hand: {count1}", (40, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,0), 4)
            cv2.putText(img, f"{fingers1}", (40, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 3)

            # Tangan 2
            cv2.putText(img, f"Right Hand: {count2}", (40, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,0), 4)
            cv2.putText(img, f"{fingers2}", (40, 290),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 3)

        # Jika hanya 1 tangan
        elif len(hands) == 1:
            hand = hands[0]
            fingers = detector.fingersUp(hand)
            count = sum(fingers)

            cv2.putText(img, f"Hand: {count}", (40, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,0), 4)
            cv2.putText(img, f"{fingers}", (40, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 3)

    cv2.imshow("Dual Hand + Fingers", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
