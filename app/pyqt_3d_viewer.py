# pyqt_3d_viewer.py
from PyQt6.QtWidgets import QMainWindow, QSizePolicy, QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

class Pose3DViewer(QMainWindow):
    def __init__(self, keypoints_3d: np.ndarray):
        super().__init__()
        self.setWindowTitle("3D Pose Viewer")

        self.keypoints_3d = keypoints_3d

        # 創建 matplotlib 圖形與畫布
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.updateGeometry()

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.canvas)
        self.setCentralWidget(central_widget)

        self.ax = self.fig.add_subplot(111, projection='3d')
        self.draw_skeleton()

    def draw_skeleton(self):
        self.ax.clear()

        # 畫關節點
        self.ax.scatter(self.keypoints_3d[:, 0], self.keypoints_3d[:, 1], self.keypoints_3d[:, 2], c='r', s=30)

        # 骨架連接（COCO示意）
        skeleton = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (1, 5), (5, 6), (6, 7),
            (1, 8), (8, 9), (9, 10),
            (8, 12), (12, 13), (13, 14),
            (8, 11), (11, 12)
        ]
        for i, j in skeleton:
            x = [self.keypoints_3d[i, 0], self.keypoints_3d[j, 0]]
            y = [self.keypoints_3d[i, 1], self.keypoints_3d[j, 1]]
            z = [self.keypoints_3d[i, 2], self.keypoints_3d[j, 2]]
            self.ax.plot(x, y, z, c='b', linewidth=2)

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('3D Pose Estimation')
        self.canvas.draw()
