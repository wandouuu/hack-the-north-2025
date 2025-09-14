import cv2
import mediapipe as mp
import time
import numpy as np
from PyQt5 import QtCore
import pyautogui

# PoseWorker for detecting hand raise and emitting hand coordinates (sx, sy) along with the raise status
class PoseWorker(QtCore.QObject):
    hand_raise_signal = QtCore.pyqtSignal(int, int, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = True
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7, model_complexity=0)
        self.mp_draw = mp.solutions.drawing_utils
        self.screen_w, self.screen_h = pyautogui.size()

    @QtCore.pyqtSlot(np.ndarray)
    def process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        sx, sy = 0, 0

        if results.pose_landmarks:
            self.mp_draw.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            h, w, c = frame.shape
            left_wrist_x, left_wrist_y = int(results.pose_landmarks.landmark[15].x * w), int(results.pose_landmarks.landmark[15].y * h)
            right_wrist_x, right_wrist_y = int(results.pose_landmarks.landmark[16].x * w), int(results.pose_landmarks.landmark[16].y * h)
            left_shoulder_x, left_shoulder_y = int(results.pose_landmarks.landmark[11].x * w), int(results.pose_landmarks.landmark[11].y * h)
            right_shoulder_x, right_shoulder_y = int(results.pose_landmarks.landmark[12].x * w), int(results.pose_landmarks.landmark[12].y * h)

            # detect when left or right hand is raised above shoulder level
            if left_wrist_y < left_shoulder_y - 450:
                sx = int(left_wrist_x / w * self.screen_w)
                sy = int(left_wrist_y / h * self.screen_h)
                is_raised = True
            elif right_wrist_y < right_shoulder_y - 450:
                sx = int(right_wrist_x / w * self.screen_w)
                sy = int(right_wrist_y / h * self.screen_h)
                is_raised = True

        self.hand_raise_signal.emit(sx, sy, is_raised)
