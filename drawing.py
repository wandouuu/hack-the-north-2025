
"""
Air-writing overlay with:
- Cursor always following pointer finger (green circle)
- Strict pinch detection for drawing
- Red strokes (crisp, colored)
- Smooth, controlled strokes
- Full-screen mapping
- Transparent, click-through overlay
- Hotkeys: C to clear, Q to quit
- Eraser mode (make a fist to erase strokes)
"""

import sys
import math
import time
from collections import deque

import cv2
import mediapipe as mp
import pyautogui
from PyQt5 import QtCore, QtGui, QtWidgets
import keyboard
import numpy as np

# ---------- Config ----------
PINCH_THRESHOLD = 20   # px, strict pinch
SMOOTHING_WINDOW = 5
MIN_POINT_DIST = 12
CAM_INDEX = 0
FLIP_FRAME = True
STROKE_WIDTH = 6
CURSOR_RADIUS = 10
ERASER_RADIUS = 50      # big eraser radius
CAM_WIDTH, CAM_HEIGHT = 640, 480
# ----------------------------

# ---------- Webcam + MediaPipe ----------
class CamWorker(QtCore.QThread):
    point_signal = QtCore.pyqtSignal(int, int, bool, bool)  # (ix, iy, is_pinched, is_fist)

    def __init__(self):
        super().__init__()
        self._running = True
        self.cap = cv2.VideoCapture(CAM_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7, model_complexity=0)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, 
                                         min_tracking_confidence=0.6, model_complexity=0)

        self.hand_raised = False
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

    def run(self):

        #swipe detection state
        prev_indx = None
        prev_mndx = None
        SWIPE_THRESH = 150
        FINGERTIP_DIST_THRESH = 80


        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            if FLIP_FRAME:
                frame = cv2.flip(frame, 1)
            cam_h, cam_w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = self.pose.process(rgb)
            hand_results = self.hands.process(rgb)

            # Get pose landmarks
            left_wrist_y = right_wrist_y = left_shoulder_y = right_shoulder_y = None
            left_shoulder_x = right_shoulder_x = None
            if pose_results.pose_landmarks:
                left_wrist_y = int(pose_results.pose_landmarks.landmark[15].y * cam_h)
                right_wrist_y = int(pose_results.pose_landmarks.landmark[16].y * cam_h)
                left_shoulder_x = int(pose_results.pose_landmarks.landmark[11].x * cam_w)
                left_shoulder_y = int(pose_results.pose_landmarks.landmark[11].y * cam_h)
                right_shoulder_x = int(pose_results.pose_landmarks.landmark[12].x * cam_w)
                right_shoulder_y = int(pose_results.pose_landmarks.landmark[12].y * cam_h)

            # Use shoulder width as scale for depth-invariant threshold
            hand_is_raised = False
            if left_shoulder_x is not None and right_shoulder_x is not None:
                S = max(np.hypot(right_shoulder_x - left_shoulder_x, right_shoulder_y - left_shoulder_y), 1e-6)
                raise_amt_left = (left_shoulder_y - left_wrist_y) / S if left_wrist_y is not None else 0
                raise_amt_right = (right_shoulder_y - right_wrist_y) / S if right_wrist_y is not None else 0
                THRESHOLD = 0.1
                if raise_amt_left > THRESHOLD or raise_amt_right > THRESHOLD:
                    hand_is_raised = True

            if hand_is_raised and hand_results.multi_hand_landmarks:
                lm = hand_results.multi_hand_landmarks[0]
                ix = int(lm.landmark[8].x * cam_w)
                iy = int(lm.landmark[8].y * cam_h)
                tx = int(lm.landmark[4].x * cam_w)
                ty = int(lm.landmark[4].y * cam_h)
                

                # swipe detection
                mndx = int(lm.landmark[12].x * cam_w)
                mndy = int(lm.landmark[12].y * cam_h)
                fingertip_dist = math.hypot(ix - mndx, iy - mndy)
                if prev_indx is not None and prev_mndx is not None:
                    d_indx = ix - prev_indx
                    d_mndx = mndx - prev_mndx
                    # Only allow swipe if fingertips are close
                    if fingertip_dist < FINGERTIP_DIST_THRESH:
                        # Swipe right
                        if d_indx > SWIPE_THRESH and d_mndx > SWIPE_THRESH:
                            pyautogui.press('right')
                            print('Next Slide')
                            if hasattr(self, 'overlay'):
                                self.overlay.clear()
                            prev_indx = None  # reset so it doesn’t spam
                            prev_mndx = None
                        # Swipe left
                        elif d_indx < -SWIPE_THRESH and d_mndx < -SWIPE_THRESH:
                            pyautogui.press('left')
                            print('Previous Slide')
                            if hasattr(self, 'overlay'):
                                self.overlay.clear()
                            prev_indx = None
                            prev_mndx = None
                        else:
                            prev_indx = ix
                            prev_mndx = mndx
                    else:
                        prev_indx = ix
                        prev_mndx = mndx
                else:
                    prev_indx = ix
                    prev_mndx = mndx

                pinch_dist = math.hypot(ix - tx, iy - ty)
                is_pinched = pinch_dist < PINCH_THRESHOLD
                
                is_fist = True
                fid_lms = [4, 8, 12, 16, 20]
                for i in range(len(fid_lms)):
                    fx = int(lm.landmark[fid_lms[i]].x * cam_w)
                    fy = int(lm.landmark[fid_lms[i]].y * cam_h)
                    if i == 4:
                        next_x = int(lm.landmark[fid_lms[0]].x * cam_w)
                        next_y = int(lm.landmark[fid_lms[0]].y * cam_h)
                    else:
                        next_x = int(lm.landmark[fid_lms[i+1]].x * cam_w)
                        next_y = int(lm.landmark[fid_lms[i+1]].y * cam_h)
                    dist = math.hypot(fx - next_x, fy - next_y)
                    if dist >= 40:
                        is_fist = False
                        break
                self.overlay.on_point(ix, iy, is_pinched, is_fist)
            else:
                prev_indx = None
                prev_mndx = None
                self.overlay.on_point(-1, -1, False, False)

            #cv2.imshow("Air-writing", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.stop()






    def stop(self):
        self._running = False
        try:
            self.cap.release()
        except:
            pass
        self.quit()
        self.wait()

    @QtCore.pyqtSlot(int, int, bool)
    def set_hand_raised(self, sx, sy, is_raised):
        self.hand_raised = is_raised
