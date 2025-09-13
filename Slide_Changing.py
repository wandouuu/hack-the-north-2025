import cv2
import mediapipe as mp
import pyautogui

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

prev_x = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Index fingertip = landmark 8
            h, w, c = frame.shape
            x = int(hand_landmarks.landmark[8].x * w)

            if prev_x is not None:
                dx = x - prev_x

                # Swipe right
                if dx > 50:
                    pyautogui.press('right')
                    print("Next Slide")
                    prev_x = None  # reset so it doesn’t spam

                # Swipe left
                elif dx < -50:
                    pyautogui.press('left')
                    print("Previous Slide")
                    prev_x = None

            prev_x = x

    cv2.imshow("Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
