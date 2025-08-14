# import 新增
# from pose_engine import run_2d_pose  # 新增
from pose_2d_engine import run_2d_pose
from pose3d_viewer import run_3d_pose, Pose3DViewer, show_pose3d_window, get_3d_keypoints  # 新增
from pyqt_3d_viewer import Pose3DViewer

import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog,
    QVBoxLayout, QWidget, QSlider, QHBoxLayout
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt

# 在 import 下面加回來
class PoseEstimationUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Anato-Aid: Pose Estimator")
        self.image_path = None
        self.radius = 6
        self.thickness = 4

        layout = QVBoxLayout()

        self.label = QLabel("No image selected.")
        layout.addWidget(self.label)

        self.img_preview = QLabel()
        layout.addWidget(self.img_preview)

        btn_select = QPushButton("Select Image")
        btn_select.clicked.connect(self.select_image)
        layout.addWidget(btn_select)

        radius_layout = QHBoxLayout()
        radius_label = QLabel("Node Radius:")
        self.radius_slider = QSlider(Qt.Orientation.Horizontal)
        self.radius_slider.setMinimum(1)
        self.radius_slider.setMaximum(20)
        self.radius_slider.setValue(self.radius)
        self.radius_slider.valueChanged.connect(self.update_radius)
        self.radius_value_label = QLabel(str(self.radius))
        radius_layout.addWidget(radius_label)
        radius_layout.addWidget(self.radius_slider)
        radius_layout.addWidget(self.radius_value_label)
        layout.addLayout(radius_layout)

        thickness_layout = QHBoxLayout()
        thickness_label = QLabel("Line Thickness:")
        self.thickness_slider = QSlider(Qt.Orientation.Horizontal)
        self.thickness_slider.setMinimum(1)
        self.thickness_slider.setMaximum(20)
        self.thickness_slider.setValue(self.thickness)
        self.thickness_slider.valueChanged.connect(self.update_thickness)
        self.thickness_value_label = QLabel(str(self.thickness))
        thickness_layout.addWidget(thickness_label)
        thickness_layout.addWidget(self.thickness_slider)
        thickness_layout.addWidget(self.thickness_value_label)
        layout.addLayout(thickness_layout)

        btn_run = QPushButton("Run Pose Estimation")
        btn_run.clicked.connect(self.run_pose_estimation)
        layout.addWidget(btn_run)
        
        # btn_run3d = QPushButton("Run 3D Pose Estimation")
        # btn_run3d.clicked.connect(self.run_pose_estimation_3d)
        # layout.addWidget(btn_run3d)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def select_image(self):
        file_dialog = QFileDialog()
        img_path, _ = file_dialog.getOpenFileName(self, "Select Image", "./images")
        if img_path:
            self.image_path = img_path
            self.label.setText(f"Selected: {os.path.basename(img_path)}")
            max_size = 800
            self.img_preview.setPixmap(QPixmap(img_path).scaled(max_size, max_size, Qt.AspectRatioMode.KeepAspectRatio))

    def update_radius(self, value):
        self.radius = value
        self.radius_value_label.setText(str(value))

    def update_thickness(self, value):
        self.thickness = value
        self.thickness_value_label.setText(str(value))

    def run_pose_estimation(self):  # 這裡用你目前寫的版本
        if not self.image_path:
            self.label.setText("No image selected!")
            return

        output_dir = "./../vis_results"
        os.makedirs(output_dir, exist_ok=True)

        success = run_2d_pose(
            img_path=self.image_path,
            output_dir=output_dir,
            rad=self.radius,
            thick=self.thickness,
            # return_data=True
        )

        if success:
            result_img = os.path.join(output_dir, os.path.basename(self.image_path))
            max_size = 800
            self.img_preview.setPixmap(QPixmap(result_img).scaled(max_size, max_size, Qt.AspectRatioMode.KeepAspectRatio))
            self.label.setText("Pose estimation complete!")
        else:
            self.label.setText("Failed to run pose estimation.")
    
    # def run_pose_estimation_3d(self):
    #     if not self.image_path:
    #         self.label.setText("No image selected!")
    #         return

    #     self.label.setText("Running 2D pose estimation...")
    #     try:
    #         det_results, pose2d_results = run_2d_pose(self.image_path, device='cpu')
    #     except Exception as e:
    #         self.label.setText(f"2D pose estimation error: {e}")
    #         return

    #     self.label.setText("Running 3D pose estimation...")
    #     try:
    #         keypoints_3d_list = run_3d_pose(pose2d_results, device='cpu')
    #     except Exception as e:
    #         self.label.setText(f"3D pose estimation error: {e}")
    #         return

        # 跳出新視窗顯示3D骨架（可旋轉）
        # show_3d_pose_window(keypoints_3d_list)
        # self.label.setText("3D pose estimation complete!")

