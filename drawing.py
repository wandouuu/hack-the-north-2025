"""
Air-writing overlay with:
- Cursor always following pointer finger (green circle)
- Strict pinch detection for drawing
- Red strokes (crisp, colored)
- Smooth, controlled strokes
- Full-screen mapping
- Transparent, click-through overlay
- Hotkeys: C to clear, Q to quit
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
PINCH_THRESHOLD = 30  # px, strict pinch
SMOOTHING_WINDOW = 5
MIN_POINT_DIST = 5
CAM_INDEX = 0
FLIP_FRAME = True
STROKE_WIDTH = 6
CURSOR_RADIUS = 10
CAM_WIDTH, CAM_HEIGHT = 640, 480  # set to your webcam resolution
# ----------------------------

screen_w, screen_h = pyautogui.size()


# ---------- Webcam + MediaPipe ----------
class CamWorker(QtCore.QThread):
    point_signal = QtCore.pyqtSignal(int, int, bool)

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

                pinch_dist = math.hypot(ix - tx, iy - ty)
                is_pinched = pinch_dist < PINCH_THRESHOLD

                self.point_signal.emit(ix, iy, is_pinched)
            else:
                self.point_signal.emit(-1, -1, False)

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

        # Temporary image for drawing
        self.temp_img = np.zeros((screen_h, screen_w, 4), dtype=np.uint8)

        self.pen_color = (255, 0, 0, 200)  # Red RGBA
        self.pen_width = STROKE_WIDTH

        # For smoothing cursor movement
        self.smooth_queue = deque(maxlen=SMOOTHING_WINDOW)

        self.show()

    @QtCore.pyqtSlot(int, int, bool)
    def on_point(self, ix, iy, is_pinched):
        if ix == -1 or iy == -1:
            self.cursor_pos = None
            if self.is_drawing:
                self.finish_stroke()
            return

        # Map webcam coordinates to full screen
        sx = int(ix / CAM_WIDTH * screen_w)
        sy = int(iy / CAM_HEIGHT * screen_h)

        # Smooth cursor position (always update!)
        self.smooth_queue.append((sx, sy))
        avg_x = int(sum(p[0] for p in self.smooth_queue) / len(self.smooth_queue))
        avg_y = int(sum(p[1] for p in self.smooth_queue) / len(self.smooth_queue))
        self.cursor_pos = QtCore.QPointF(avg_x, avg_y)

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

        # Always repaint so cursor is drawn even when not pinching
        self.update()

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
        # Clear temp image
        self.temp_img.fill(0)

        # Draw all strokes
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

        # Convert temp image to QImage directly to preserve red color
        qt_img = QtGui.QImage(self.temp_img.data, self.temp_img.shape[1], self.temp_img.shape[0],
                              QtGui.QImage.Format_RGBA8888)
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, qt_img)

        # Draw cursor (green circle) always
        if self.cursor_pos:
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

    # Hotkeys
    keyboard.add_hotkey('c', overlay.clear)
    keyboard.add_hotkey('q', lambda: (cam_worker.stop(), QtWidgets.QApplication.quit()))

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
