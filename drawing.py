"""
overlay_air_annotator.py

Transparent air-writing overlay using MediaPipe + OpenCV + PyQt5.
Features:
- Pinch (thumb+index) to start drawing.
- Release/separate fingers to finish stroke (spaces between letters).
- Hand must be above shoulder to draw.
- Click-through transparent overlay works over fullscreen apps like PowerPoint.
- Keyboard shortcuts:
    C -> clear overlay
    Q -> quit application
"""

import sys
import math
import time
from collections import deque

import cv2
import mediapipe as mp
import pyautogui
from PyQt5 import QtCore, QtGui, QtWidgets

# ---------- Config ----------
PINCH_THRESHOLD = 40           # px, distance thumb <-> index for pinch
PINCH_DEBOUNCE_FRAMES = 3     # frames for stable pinch/no-pinch
SMOOTHING_WINDOW = 4
MIN_POINT_DIST = 2             # px, for downsampling stroke points
CAM_INDEX = 0
FLIP_FRAME = True              # mirror camera for natural movement
# ----------------------------

screen_w, screen_h = pyautogui.size()

# ---------- Webcam + MediaPipe worker ----------
class CamWorker(QtCore.QThread):
    point_signal = QtCore.pyqtSignal(int, int, bool)  # screen_x, screen_y, is_pinched

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = True
        self.cap = cv2.VideoCapture(CAM_INDEX)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1,
                                         min_detection_confidence=0.6,
                                         min_tracking_confidence=0.6)
        self.pinch_history = deque(maxlen=PINCH_DEBOUNCE_FRAMES)
        self.smooth_queue = deque(maxlen=SMOOTHING_WINDOW)

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
                ix = int(lm.landmark[8].x * cam_w)  # index tip
                iy = int(lm.landmark[8].y * cam_h)
                tx = int(lm.landmark[4].x * cam_w)  # thumb tip
                ty = int(lm.landmark[4].y * cam_h)

                pinch_dist = math.hypot(ix - tx, iy - ty)
                self.pinch_history.append(pinch_dist)
                is_pinched = False
                if len(self.pinch_history) == self.pinch_history.maxlen:
                    med = sorted(self.pinch_history)[len(self.pinch_history)//2]
                    is_pinched = med < PINCH_THRESHOLD

                # Only draw if hand above shoulder (upper half)
                if iy < cam_h // 2:
                    # smoothing
                    self.smooth_queue.append((ix, iy))
                    avg_x = int(sum(p[0] for p in self.smooth_queue) / len(self.smooth_queue))
                    avg_y = int(sum(p[1] for p in self.smooth_queue) / len(self.smooth_queue))

                    # map to screen coordinates
                    sx = int(avg_x / cam_w * screen_w)
                    sy = int(avg_y / cam_h * screen_h)

                    self.point_signal.emit(sx, sy, is_pinched)
                else:
                    # hand too low: stop drawing
                    self.point_signal.emit(-1, -1, False)
            else:
                # no hand detected
                self.pinch_history.clear()
                self.smooth_queue.clear()
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

# ---------- Transparent Overlay ----------
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

        self.strokes = []          # list of completed strokes
        self.current_stroke = None
        self.is_drawing = False

        self.pen_color = QtGui.QColor(255, 0, 0, 200)
        self.pen_width = 6

        self.show()

        # --- Keyboard shortcuts ---
        clear_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("C"), self)
        clear_shortcut.activated.connect(self.clear)

        quit_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Q"), self)
        quit_shortcut.activated.connect(QtWidgets.QApplication.quit)

    @QtCore.pyqtSlot(int, int, bool)
    def on_point(self, sx, sy, is_pinched):
        if sx == -1 and sy == -1:
            # no valid hand: finish current stroke
            if self.is_drawing:
                self.finish_stroke()
            return

        pt = QtCore.QPointF(sx, sy)

        if is_pinched:
            if not self.is_drawing:
                self.start_stroke(pt)
            else:
                self.append_point(pt)
        else:
            if self.is_drawing:
                self.finish_stroke()

    def start_stroke(self, pt):
        self.is_drawing = True
        self.current_stroke = [pt]
        self.update()

    def append_point(self, pt):
        if self.current_stroke:
            last = self.current_stroke[-1]
            if (abs(last.x() - pt.x()) + abs(last.y() - pt.y())) < MIN_POINT_DIST:
                return
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
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        pen = QtGui.QPen(self.pen_color, self.pen_width, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
        painter.setPen(pen)

        # draw committed strokes
        for stroke in self.strokes:
            path = QtGui.QPainterPath()
            path.moveTo(stroke[0])
            for pt in stroke[1:]:
                path.lineTo(pt)
            painter.drawPath(path)

        # draw current stroke
        if self.current_stroke:
            path = QtGui.QPainterPath()
            path.moveTo(self.current_stroke[0])
            for pt in self.current_stroke[1:]:
                path.lineTo(pt)
            painter.drawPath(path)

# ---------- Main ----------
def main():
    app = QtWidgets.QApplication(sys.argv)

    overlay = OverlayWindow()
    cam_worker = CamWorker()
    cam_worker.point_signal.connect(overlay.on_point)
    cam_worker.start()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
