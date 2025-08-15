from pose_2d_engine import run_2d_pose
from pose_3d_engine import run_3d_pose
from pose3d_viewer import Pose3DViewer, show_pose3d_window, get_3d_keypoints
from pyqt_3d_viewer import Pose3DViewer

import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog,
    QVBoxLayout, QWidget, QSlider, QHBoxLayout, QGroupBox
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt


class PoseEstimationUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Anato-Aid: Pose Estimator")
        self.image_path = None
        self.radius = 6
        self.thickness = 4

        # 主容器 (水平布局)
        main_layout = QHBoxLayout()

        # ===== 左邊：圖片區 =====
        left_layout = QVBoxLayout()
        self.label = QLabel("No image selected.")
        left_layout.addWidget(self.label)

        self.img_preview = QLabel()
        left_layout.addWidget(self.img_preview)

        # ===== 右邊：控制區 =====
        right_layout = QVBoxLayout()

        # --- 圖片操作區 ---
        image_group = QGroupBox("Image Controls")
        img_group_layout = QVBoxLayout()

        btn_select = QPushButton("Select Image")
        btn_select.clicked.connect(self.select_image)
        img_group_layout.addWidget(btn_select)

        image_group.setLayout(img_group_layout)
        right_layout.addWidget(image_group)

        # --- 參數調整區 ---
        params_group = QGroupBox("Drawing Parameters")
        params_layout = QVBoxLayout()

        # Radius 調整
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
        params_layout.addLayout(radius_layout)

        # Thickness 調整
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
        params_layout.addLayout(thickness_layout)

        params_group.setLayout(params_layout)
        right_layout.addWidget(params_group)

        # --- 偵測操作區 ---
        actions_group = QGroupBox("Pose Estimation Actions")
        actions_layout = QVBoxLayout()

        btn_run = QPushButton("Run Pose Estimation")
        btn_run.clicked.connect(self.run_pose_estimation)
        actions_layout.addWidget(btn_run)

        btn_run3d = QPushButton("Run 3D Pose Estimation")
        btn_run3d.clicked.connect(self.run_pose_estimation_3d)
        actions_layout.addWidget(btn_run3d)

        actions_group.setLayout(actions_layout)
        right_layout.addWidget(actions_group)

        right_layout.addStretch()  # 把剩餘空間推到底

        # ===== 組合左右 =====
        main_layout.addLayout(left_layout, stretch=3)
        main_layout.addLayout(right_layout, stretch=1)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def select_image(self):
        file_dialog = QFileDialog()
        img_path, _ = file_dialog.getOpenFileName(self, "Select Image", "./images")
        if img_path:
            self.image_path = img_path
            self.label.setText(f"Selected: {os.path.basename(img_path)}")
            max_size = 580
            self.img_preview.setPixmap(QPixmap(img_path).scaled(max_size, max_size, Qt.AspectRatioMode.KeepAspectRatio))

    def update_radius(self, value):
        self.radius = value
        self.radius_value_label.setText(str(value))

    def update_thickness(self, value):
        self.thickness = value
        self.thickness_value_label.setText(str(value))

    def run_pose_estimation(self):
        if not self.image_path:
            self.label.setText("No image selected!")
            return

        output_dir = "./../2d_results"
        os.makedirs(output_dir, exist_ok=True)

        success = run_2d_pose(
            img_path=self.image_path,
            rad=self.radius,
            thick=self.thickness,
        )

        if success:
            result_img = os.path.join(output_dir, os.path.basename(self.image_path))
            max_size = 580
            self.img_preview.setPixmap(QPixmap(result_img).scaled(max_size, max_size, Qt.AspectRatioMode.KeepAspectRatio))
            self.label.setText("Pose estimation complete!")
        else:
            self.label.setText("Failed to run pose estimation.")

    def run_pose_estimation_3d(self):
        if not self.image_path:
            self.label.setText("No image selected!")
            return

        output_dir = "./../3d_results"
        os.makedirs(output_dir, exist_ok=True)

        success = run_3d_pose(
            img_path=self.image_path,
            rad=self.radius,
            thick=self.thickness,
        )

        if success:
            result_img = os.path.join(output_dir, os.path.basename(self.image_path))
            max_size = 580
            self.img_preview.setPixmap(QPixmap(result_img).scaled(max_size, max_size, Qt.AspectRatioMode.KeepAspectRatio))
            self.label.setText("3D Pose estimation complete!")
        else:
            self.label.setText("Failed to run 3D pose estimation.")
