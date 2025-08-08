import os
import subprocess
import sys

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog,
    QVBoxLayout, QWidget, QSlider, QHBoxLayout
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt

python_path = sys.executable  # 當前 Python 執行路徑

MMPose_DIR = "./mmpose"
CHECKPOINT_PATH = "./hrnet_w32_coco_256x192-c78dce93_20200708.pth"
# CONFIG_PATH = "./mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py"
CONFIG_PATH = "./mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_gridmask-8xb64-210e_coco-256x192.py"
# CONFIG_PATH = "./mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-384x288.py"

class PoseEstimationUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Anato-Aid: Pose Estimator")
        self.image_path = None

        self.radius = 6       # 節點大小
        self.thickness = 4    # 骨架粗細

        layout = QVBoxLayout()

        self.label = QLabel("No image selected.")
        layout.addWidget(self.label)

        self.img_preview = QLabel()
        layout.addWidget(self.img_preview)

        btn_select = QPushButton("Select Image")
        btn_select.clicked.connect(self.select_image)
        layout.addWidget(btn_select)

        # 節點大小調整滑桿 + 顯示數字
        radius_layout = QHBoxLayout()
        radius_label = QLabel("Node Radius:")
        self.radius_slider = QSlider(Qt.Orientation.Horizontal)
        self.radius_slider.setMinimum(1)
        self.radius_slider.setMaximum(20)
        self.radius_slider.setValue(self.radius)
        self.radius_slider.valueChanged.connect(self.update_radius)
        self.radius_value_label = QLabel(str(self.radius))  # 新增顯示數字 Label
        radius_layout.addWidget(radius_label)
        radius_layout.addWidget(self.radius_slider)
        radius_layout.addWidget(self.radius_value_label)  # 加入數字顯示
        layout.addLayout(radius_layout)

        # 骨架粗細調整滑桿 + 顯示數字
        thickness_layout = QHBoxLayout()
        thickness_label = QLabel("Line Thickness:")
        self.thickness_slider = QSlider(Qt.Orientation.Horizontal)
        self.thickness_slider.setMinimum(1)
        self.thickness_slider.setMaximum(20)
        self.thickness_slider.setValue(self.thickness)
        self.thickness_slider.valueChanged.connect(self.update_thickness)
        self.thickness_value_label = QLabel(str(self.thickness))  # 新增顯示數字 Label
        thickness_layout.addWidget(thickness_label)
        thickness_layout.addWidget(self.thickness_slider)
        thickness_layout.addWidget(self.thickness_value_label)  # 加入數字顯示
        layout.addLayout(thickness_layout)

        btn_run = QPushButton("Run Pose Estimation")
        btn_run.clicked.connect(self.run_pose_estimation)
        layout.addWidget(btn_run)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def select_image(self):
        file_dialog = QFileDialog()
        img_path, _ = file_dialog.getOpenFileName(self, "Select Image", "./images")
        if img_path:
            self.image_path = img_path
            self.label.setText(f"Selected: {os.path.basename(img_path)}")
            # self.img_preview.setPixmap(QPixmap(img_path).scaled(256, 256))
            max_size = 800
            self.img_preview.setPixmap(QPixmap(img_path).scaled(max_size, max_size, Qt.AspectRatioMode.KeepAspectRatio))


    def update_radius(self, value):
        self.radius = value
        self.radius_value_label.setText(str(value))  # 同步更新數字顯示

    def update_thickness(self, value):
        self.thickness = value
        self.thickness_value_label.setText(str(value))  # 同步更新數字顯示

    def run_pose_estimation(self):
        if not self.image_path:
            self.label.setText("No image selected!")
            return

        output_dir = "./vis_results"
        os.makedirs(output_dir, exist_ok=True)

        command = [
            python_path,
            "mmpose/demo/topdown_demo_with_mmdet.py",
            "./mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py",  # det_config
            "./checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",   # det_checkpoint
            CONFIG_PATH,       # pose_config
            CHECKPOINT_PATH,   # pose_checkpoint
            "--input", self.image_path,
            "--output-root", output_dir,
            "--radius", str(self.radius),
            "--thickness", str(self.thickness),
            "--device", "cpu"
        ]

        try:
            subprocess.run(command, check=True)
            result_img = os.path.join(output_dir, os.path.basename(self.image_path))
            if os.path.exists(result_img):
                max_size = 800
                self.img_preview.setPixmap(QPixmap(result_img).scaled(max_size, max_size, Qt.AspectRatioMode.KeepAspectRatio))
                self.label.setText("Pose estimation complete!")
            else:
                self.label.setText("Failed: result image not found.")
        except subprocess.CalledProcessError as e:
            self.label.setText(f"Error: {e}")

if __name__ == "__main__":
    app = QApplication([])
    window = PoseEstimationUI()
    window.show()
    app.exec()
