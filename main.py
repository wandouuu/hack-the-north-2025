import cv2
import mediapipe as mp
import pyautogui
import time

# For FPS calculation
def fps_handle(start):
    end = time.time()
    total_time = end - start
    if total_time != 0:
        fps = 1 / total_time
        print(f"FPS: {round(fps, 2)}")
        cv2.putText(frame, f"FPS: {int(fps)}", (20,70), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2, 1)
    else:
        print(f"FPS: ???")
        cv2.putText(frame, f"FPS: ???", (20,70), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2, 1)


def main():
    cap = cv2.VideoCapture(0)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, model_complexity=0)
    mp_draw = mp.solutions.drawing_utils

    prev_indx = None
    prev_mndx = None

    while True:
        start = time.time()
        global frame
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                fps_handle(start)
                # Index fingertip = landmark 8
                h, w, c = frame.shape
                indx = int(hand_landmarks.landmark[8].x * w)
                indy = int(hand_landmarks.landmark[8].y * h)

                #middle fingertip = landmark 12
                mndx = int(hand_landmarks.landmark[12].x * w)
                mndy = int(hand_landmarks.landmark[12].y * h)

                print(f"Index Finger Tip Coordinates: ({indx}, {indy})")
                print(f"Middle Finger Tip Coordinates: ({mndx}, {mndy})")

                if prev_indx is not None and prev_mndx is not None:
                    d_indx = indx - prev_indx
                    d_mndx = mndx - prev_mndx

                    # Swipe right
                    if d_indx > 150 and d_mndx > 150:
                        pyautogui.press('right')
                        print("Next Slide")
                        prev_indx = None  # reset so it doesn’t spam

                    # Swipe left
                    elif d_indx < -150 and d_mndx < -150:
                        pyautogui.press('left')
                        print("Previous Slide")
                        prev_indx = None

                prev_indx = indx
                prev_mndx = mndx

        cv2.imshow("Gesture Control", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()