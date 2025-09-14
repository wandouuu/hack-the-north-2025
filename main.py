import cv2
import mediapipe as mp
import pyautogui
import time
import sys
import math
from collections import deque
from PyQt5 import QtCore, QtGui, QtWidgets
import keyboard
import numpy as np
from drawing import OverlayWindow

# ---------- Config ----------
PINCH_THRESHOLD = 30  # px, strict pinch
SMOOTHING_WINDOW = 5
MIN_POINT_DIST = 12
CAM_INDEX = 0
FLIP_FRAME = True
STROKE_WIDTH = 6
CURSOR_RADIUS = 10
CAM_WIDTH, CAM_HEIGHT = 640, 480  # set to your webcam resolution
# ----------------------------

screen_w, screen_h = pyautogui.size()

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
    app = QtWidgets.QApplication(sys.argv)
    overlay = OverlayWindow()
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
                #cv2.waitKey(5555)
                # call gesture function here
                overlay.set_cursor(x, y)
                overlay.start_stroke(x, y)
                overlay.add_point(x, y)
                overlay.finish_stroke()
            elif right_wrist_y < right_shoulder_y - 450:
                print("Right hand raised")
                #cv2.waitKey(5555)
                overlay.set_cursor(x, y)
                overlay.start_stroke(x, y)
                overlay.add_point(x, y)
                overlay.finish_stroke()


        cv2.imshow("Pose Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()