import sys
print("Interpreter:", sys.executable)

import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:

        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):

            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            label = results.multi_handedness[idx].classification[0].label

            # 🔄 Balik kanan dan kiri
            if label == "Right":
                label = "Left"
            else:
                label = "Right"

            print("Tangan:", label)

            cv2.putText(img, f"Tangan: {label}", (10,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    else:
        print("Tidak ada tangan")

    cv2.imshow("Deteksi Kanan/Kiri", img)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
