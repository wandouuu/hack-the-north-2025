import cv2
import mediapipe as mp
import time

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
            if left_wrist_y < left_shoulder_y - 450: # CHANGE MARGIN VALUE HERE
                print(left_wrist_y, left_shoulder_y)
                print("Left hand raised")
                cv2.waitKey(5555)
                # call gesture function here
            elif right_wrist_y < right_shoulder_y - 450:
                print("Right hand raised")
                cv2.waitKey(5555)
                # call gesture function here


        cv2.imshow("Pose Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()