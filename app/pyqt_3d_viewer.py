from PyQt6.QtWidgets import QMainWindow, QApplication
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import Qt
from OpenGL.GL import *
import sys
import numpy as np

class Pose3DViewer(QOpenGLWidget):
    def __init__(self, keypoints_3d_list):
        super().__init__()
        self.keypoints_3d_list = keypoints_3d_list
        self.rot_x = 0
        self.rot_y = 0
        self.last_pos = None
        self.setMinimumSize(600, 600)

    def initializeGL(self):
        glClearColor(1, 1, 1, 1)
        glEnable(GL_DEPTH_TEST)
        glPointSize(8)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0, 0, -5)
        glRotatef(self.rot_x, 1, 0, 0)
        glRotatef(self.rot_y, 0, 1, 0)

        glColor3f(1, 0, 0)

        for keypoints_3d in self.keypoints_3d_list:
            # 以骨架線連接
            skeleton = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (1, 5), (5, 6), (6, 7),
                (1, 8), (8, 9), (9, 10),
                # 根據你用的骨架定義增減
            ]

            for i, j in skeleton:
                glBegin(GL_LINES)
                glVertex3f(*keypoints_3d[i])
                glVertex3f(*keypoints_3d[j])
                glEnd()

            glBegin(GL_POINTS)
            for kp in keypoints_3d:
                glVertex3f(*kp)
            glEnd()

    def mousePressEvent(self, event):
        self.last_pos = event.position()

    def mouseMoveEvent(self, event):
        if self.last_pos is None:
            return
        dx = event.position().x() - self.last_pos.x()
        dy = event.position().y() - self.last_pos.y()
        self.rot_x += dy
        self.rot_y += dx
        self.update()
        self.last_pos = event.position()

def show_3d_pose_window(keypoints_3d_list):
    app = QApplication([])
    window = QMainWindow()
    widget = Pose3DViewer(keypoints_3d_list)
    window.setCentralWidget(widget)
    window.setWindowTitle("3D Pose Viewer")
    window.show()
    app.exec()
