
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

screen_w, screen_h = pyautogui.size()


# PoseWorker using main.py's main() logic
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
                # --- Swipe detection ---
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
                palm_x = int(lm.landmark[0].x * cam_w)
                palm_y = int(lm.landmark[0].y * cam_h)
                fingertip_ids = [8, 12, 16, 20]
                '''
                is_fist = False
                for fid in fingertip_ids:
                    fx = int(lm.landmark[fid].x * cam_w)
                    fy = int(lm.landmark[fid].y * cam_h)
                    dist = math.hypot(fx - palm_x, fy - palm_y)
                    if dist > 80:
                        is_fist = False
                        break
                '''
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

# ---------- Overlay ----------
class OverlayWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.FramelessWindowHint |
            QtCore.Qt.Tool
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.setGeometry(0, 0, screen_w, screen_h)

        self.strokes = []
        self.current_stroke = None
        self.is_drawing = False
        self.cursor_pos = None
        self.eraser_mode = False
        

        # Temporary image for drawing
        self.temp_img = np.zeros((screen_h, screen_w, 4), dtype=np.uint8)

        self.pen_color = (255, 0, 0, 200)  # Red RGBA
        self.pen_width = STROKE_WIDTH

        # For smoothing cursor movement
        self.smooth_queue = deque(maxlen=SMOOTHING_WINDOW)

        self.show()

    @QtCore.pyqtSlot(int, int, bool, bool)
    def on_point(self, ix, iy, is_pinched, is_fist):
        if ix == -1 or iy == -1:
            self.cursor_pos = None
            if self.is_drawing:
                self.finish_stroke()
            return

        # Map webcam coordinates to full screen
        sx = int(ix / CAM_WIDTH * screen_w)
        sy = int(iy / CAM_HEIGHT * screen_h)

        # Smooth cursor position
        self.smooth_queue.append((sx, sy))
        avg_x = int(sum(p[0] for p in self.smooth_queue) / len(self.smooth_queue))
        avg_y = int(sum(p[1] for p in self.smooth_queue) / len(self.smooth_queue))
        self.cursor_pos = QtCore.QPointF(avg_x, avg_y)

        # Switch eraser mode based on fist
        self.eraser_mode = is_fist

        if self.eraser_mode:
            self.erase_at_point(self.cursor_pos)
        else:
            # Append to stroke only if pinched
            if is_pinched:
                if not self.is_drawing:
                    self.start_stroke(self.cursor_pos)
                else:
                    last = self.current_stroke[-1] if self.current_stroke else None
                    if last is None or (abs(last.x() - self.cursor_pos.x()) + abs(last.y() - self.cursor_pos.y())) >= MIN_POINT_DIST:
                        self.append_point(self.cursor_pos)
            else:
                if self.is_drawing:
                    self.finish_stroke()

        self.update()

    def erase_at_point(self, pt):
        # Erase any stroke points within ERASER_RADIUS
        new_strokes = []
        for stroke in self.strokes:
            keep_stroke = []
            for p in stroke:
                dist = math.hypot(p.x() - pt.x(), p.y() - pt.y())
                if dist > ERASER_RADIUS:
                    keep_stroke.append(p)
            if len(keep_stroke) > 1:
                new_strokes.append(keep_stroke)
        self.strokes = new_strokes

    def start_stroke(self, pt):
        self.is_drawing = True
        self.current_stroke = [pt]
        self.update()

    def append_point(self, pt):
        self.current_stroke.append(pt)
        self.update()

    def finish_stroke(self):
        if self.current_stroke and len(self.current_stroke) > 1:
            self.strokes.append(self.current_stroke)
        self.current_stroke = None
        self.is_drawing = False
        self.update()

    def clear(self):
        self.strokes = []
        self.current_stroke = None
        self.is_drawing = False
        self.temp_img.fill(0)
        self.update()

    def paintEvent(self, event):
        self.temp_img.fill(0)

        # Draw strokes
        for stroke in self.strokes:
            if len(stroke) < 2:
                continue
            for i in range(1, len(stroke)):
                pt1 = (int(stroke[i-1].x()), int(stroke[i-1].y()))
                pt2 = (int(stroke[i].x()), int(stroke[i].y()))
                cv2.line(self.temp_img, pt1, pt2, self.pen_color, self.pen_width, lineType=cv2.LINE_AA)

        # Draw current stroke
        if self.current_stroke and len(self.current_stroke) >= 2:
            for i in range(1, len(self.current_stroke)):
                pt1 = (int(self.current_stroke[i-1].x()), int(self.current_stroke[i-1].y()))
                pt2 = (int(self.current_stroke[i].x()), int(self.current_stroke[i].y()))
                cv2.line(self.temp_img, pt1, pt2, self.pen_color, self.pen_width, lineType=cv2.LINE_AA)

        qt_img = QtGui.QImage(self.temp_img.data, self.temp_img.shape[1], self.temp_img.shape[0],
                              QtGui.QImage.Format_RGBA8888)
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, qt_img)

        # Draw cursor
        if self.cursor_pos:
            if self.eraser_mode:
                # Blue circle for eraser
                cursor_pen = QtGui.QPen(QtGui.QColor(0, 0, 255, 200))
                cursor_pen.setWidth(3)
                painter.setPen(cursor_pen)
                painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 255, 80)))
                painter.drawEllipse(self.cursor_pos, ERASER_RADIUS, ERASER_RADIUS)
            else:
                # Green circle for normal cursor
                cursor_pen = QtGui.QPen(QtGui.QColor(0, 255, 0, 180))
                cursor_pen.setWidth(2)
                painter.setPen(cursor_pen)
                painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 255, 0, 120)))
                painter.drawEllipse(self.cursor_pos, CURSOR_RADIUS, CURSOR_RADIUS)


# ---------- Main ----------

def main():
    app = QtWidgets.QApplication(sys.argv)
    overlay = OverlayWindow()
    cam_worker = CamWorker()
    cam_worker.overlay = overlay  # Pass overlay to CamWorker
    cam_worker.start()            # Start the thread (calls run() in a new thread)
    sys.exit(app.exec_())
    
   


if __name__ == "__main__":
    main()
