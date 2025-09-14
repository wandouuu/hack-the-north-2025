
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
from collections import deque

import cv2
import mediapipe as mp
import pyautogui
from PyQt5 import QtCore, QtGui, QtWidgets
import keyboard
import numpy as np
from connect import OverlayWindow
from drawing import CamWorker

PINCH_THRESHOLD = 20   # px, strict pinch
SMOOTHING_WINDOW = 5
MIN_POINT_DIST = 12
CAM_INDEX = 0
FLIP_FRAME = True
STROKE_WIDTH = 6
CURSOR_RADIUS = 10
ERASER_RADIUS = 50 
CAM_WIDTH, CAM_HEIGHT = 640, 480

screen_w, screen_h = pyautogui.size()

def main():
    app = QtWidgets.QApplication(sys.argv)
    overlay = OverlayWindow()
    cam_worker = CamWorker()
    cam_worker.overlay = overlay  # Pass overlay to CamWorker
    cam_worker.start()            # Start the thread (calls run() in a new thread)
    sys.exit(app.exec_())
    
   


if __name__ == "__main__":
    main()
