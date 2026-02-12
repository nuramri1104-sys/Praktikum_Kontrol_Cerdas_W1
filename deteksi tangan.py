import sys
print("Interpreter:", sys.executable)

import cv2
import mediapipe as mp

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

# Inisialisasi mediapipe (tanpa drawing landmark)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7
)

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # HANYA cek ada tangan atau tidak
    if results.multi_hand_landmarks:
        status = "Tangan Terdeteksi"
        print(status)
        warna = (0, 255, 0)
    else:
        status = "Tidak Ada Tangan"
        print(status)
        warna = (0, 0, 255)

    cv2.putText(img, status, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                warna,
                2)

    cv2.imshow("Deteksi Tangan Tanpa Landmark", img)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
