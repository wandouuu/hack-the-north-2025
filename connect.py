import math
from collections import deque

import cv2
import mediapipe as mp
import pyautogui
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np

screen_w, screen_h = pyautogui.size()

PINCH_THRESHOLD = 20   # px, strict pinch
SMOOTHING_WINDOW = 5
MIN_POINT_DIST = 12
CAM_INDEX = 0
FLIP_FRAME = True
STROKE_WIDTH = 6
CURSOR_RADIUS = 10
ERASER_RADIUS = 50 
CAM_WIDTH, CAM_HEIGHT = 640, 480

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
