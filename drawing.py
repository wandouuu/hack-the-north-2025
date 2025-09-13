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


# ---------- Webcam + MediaPipe ----------
class CamWorker(QtCore.QThread):
    point_signal = QtCore.pyqtSignal(int, int, bool, bool)  # (ix, iy, is_pinched, is_fist)

    def __init__(self):
        super().__init__()
        self._running = True
        self.cap = cv2.VideoCapture(CAM_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

    def run(self):
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            if FLIP_FRAME:
                frame = cv2.flip(frame, 1)

            cam_h, cam_w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0]

                # Index finger tip
                ix = int(lm.landmark[8].x * cam_w)
                iy = int(lm.landmark[8].y * cam_h)
                # Thumb tip
                tx = int(lm.landmark[4].x * cam_w)
                ty = int(lm.landmark[4].y * cam_h)

                # Pinch detection
                pinch_dist = math.hypot(ix - tx, iy - ty)
                is_pinched = pinch_dist < PINCH_THRESHOLD

                # Fist detection: check if all fingertips are close to palm
                # Fingertips: 8, 12, 16, 20 | Palm center approx landmark 0
                palm_x = int(lm.landmark[0].x * cam_w)
                palm_y = int(lm.landmark[0].y * cam_h)
                fingertip_ids = [8, 12, 16, 20]
                is_fist = True
                for fid in fingertip_ids:
                    fx = int(lm.landmark[fid].x * cam_w)
                    fy = int(lm.landmark[fid].y * cam_h)
                    dist = math.hypot(fx - palm_x, fy - palm_y)
                    if dist > 80:   # threshold, tune if needed
                        is_fist = False
                        break

                self.point_signal.emit(ix, iy, is_pinched, is_fist)
            else:
                self.point_signal.emit(-1, -1, False, False)

            time.sleep(0.01)

    def stop(self):
        self._running = False
        try:
            self.cap.release()
        except:
            pass
        self.quit()
        self.wait()


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
    cam_worker.point_signal.connect(overlay.on_point)
    cam_worker.start()

    keyboard.add_hotkey('ctrl+c', overlay.clear)
    keyboard.add_hotkey('ctrl+q', lambda: (cam_worker.stop(), QtWidgets.QApplication.quit()))

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
