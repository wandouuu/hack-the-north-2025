import cv2
import mediapipe as mp
import time
import numpy as np


# Precompute baseline S0 during a 1s neutral stance
S0 = None; state = "IDLE"
on_since = off_since = None
t_ms = lambda: int(time.time() * 1000)

def update_state(lm, w, h):
    global state, S0, on_since, off_since

    # pixels
    lx, ly = int(lm[11].x*w), int(lm[11].y*h)  # L shoulder
    rx, ry = int(lm[12].x*w), int(lm[12].y*h)  # R shoulder
    wx, wy = int(lm[15].x*w), int(lm[15].y*h)  # L wrist (mirror for right)
    S = max(np.hypot(rx-lx, ry-ly), 1e-6)

    # normalized depth gate
    S_norm = abs(lm[12].x - lm[11].x) + 1e-6
    forward_ok = (lm[15].z < lm[11].z - 0.10 * S_norm)

    # normalized raise amount
    raise_amt = (ly - wy) / S   # >0 when wrist above shoulder

    # optional smoothing here on raise_amt & forward_ok (EMA/One-Euro)

    now = t_ms()
    TH_ON, TH_OFF = 0.80, 0.60
    DWELL_ON, DWELL_OFF = 120, 150
    
    if state == "IDLE":
        if forward_ok and raise_amt > TH_ON:
            on_since = on_since or now
            if now - on_since >= DWELL_ON:
                mp.emit_draw_start()
                state = "DRAWING"; on_since = None
        else:
            on_since = None

    elif state == "DRAWING":
        if (not forward_ok) or (raise_amt < TH_OFF):
            off_since = off_since or now
            if now - off_since >= DWELL_OFF:
                mp.emit_draw_stop()
                state = "IDLE"; off_since = None
        else:
            off_since = None

    return state
    


# live video feed tracking pose of person
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7, 
                            model_complexity=0)
    mp_draw = mp.solutions.drawing_utils

    while True:
        start = time.time()
        global frame
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            end = time.time()
            total_time = end - start

            h, w, c = frame.shape
            # get coordinates of left and right wrists and shoulders
            left_wrist_x, left_wrist_y = int(results.pose_landmarks.landmark[15].x * w), int(results.pose_landmarks.landmark[15].y * h)
            right_wrist_x, right_wrist_y = int(results.pose_landmarks.landmark[16].x * w), int(results.pose_landmarks.landmark[16].y * h)
            left_shoulder_x, left_shoulder_y = int(results.pose_landmarks.landmark[11].x * w), int(results.pose_landmarks.landmark[11].y * h)
            right_shoulder_x, right_shoulder_y = int(results.pose_landmarks.landmark[12].x * w), int(results.pose_landmarks.landmark[12].y * h)


            # detect when left or right hand is raised above shoulder level
            # Use shoulder width as scale for depth-invariant threshold
            S = max(np.hypot(right_shoulder_x - left_shoulder_x, right_shoulder_y - left_shoulder_y), 1e-6)
            raise_amt_left = (left_shoulder_y - left_wrist_y) / S
            raise_amt_right = (right_shoulder_y - right_wrist_y) / S
            THRESHOLD = 0.8  # Similar to update_state()

            if raise_amt_left > THRESHOLD:
                print("Left hand raised (depth-invariant)")
                cv2.waitKey(5555)
                # call gesture function here
            elif raise_amt_right > THRESHOLD:
                print("Right hand raised (depth-invariant)")
                cv2.waitKey(5555)
                # call gesture function here



        cv2.imshow("Pose Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()