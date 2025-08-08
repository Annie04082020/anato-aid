import sys
import numpy as np
from PyQt6 import QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl

# 假資料：10 個關節點 (x, y, z)
num_joints = 10
keypoints_3d = np.random.rand(num_joints, 3) * 100

# 關節連線索引（假設骨架拓撲）
bones = [
    (0, 1), (1, 2), (2, 3),
    (1, 4), (4, 5),
    (0, 6), (6, 7),
    (0, 8), (8, 9)
]

class SkeletonViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Skeleton Viewer (PyQtGraph)")

        # 建立 OpenGL 視圖
        self.view = gl.GLViewWidget()
        self.view.opts['distance'] = 200
        self.setCentralWidget(self.view)

        # 畫點
        self.scatter = gl.GLScatterPlotItem(pos=keypoints_3d, color=(1, 0, 0, 1), size=5)
        self.view.addItem(self.scatter)

        # 畫線
        for bone in bones:
            pts = np.array([keypoints_3d[bone[0]], keypoints_3d[bone[1]]])
            plt = gl.GLLinePlotItem(pos=pts, color=(0, 1, 0, 1), width=2, antialias=True)
            self.view.addItem(plt)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    viewer = SkeletonViewer()
    viewer.show()
    sys.exit(app.exec())
