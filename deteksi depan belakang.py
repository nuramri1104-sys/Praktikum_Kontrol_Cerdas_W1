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
    image = cv2.flip(img, 0)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:

        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):

            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            handedness = results.multi_handedness[idx].classification[0].label

            thumb_tip = hand_landmarks.landmark[4]
            pinky_tip = hand_landmarks.landmark[20]


            if handedness == "Right":

                if thumb_tip.x < pinky_tip.x:
                    kondisi = "Tangan Kiri - Telapak (Depan)"
                else:
                    kondisi = "Tangan Kiri - Punggung (Belakang)"

            else:  # Left

                if thumb_tip.x > pinky_tip.x:
                    kondisi = "Tangan Kanan - Telapak (Depan)"
                else:
                    kondisi = "Tangan Kanan - Punggung (Belakang)"

            print("Kondisi:", kondisi)

            cv2.putText(img,
                        kondisi,
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2)

    else:
        cv2.putText(img,
                    "Tidak Ada Tangan",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2)

    cv2.imshow("Deteksi 4 Kondisi Tangan", img)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
