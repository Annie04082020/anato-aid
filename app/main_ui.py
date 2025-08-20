from pose_2d_engine import run_2d_pose
from pose_3d_engine import run_3d_pose
from pose3d_viewer import Pose3DViewer, show_pose3d_window, get_3d_keypoints
from pyqt_3d_viewer import Pose3DViewer

import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog,
    QVBoxLayout, QWidget, QSlider, QHBoxLayout, QGroupBox, QComboBox
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
        
        btn_select_dir = QPushButton("Select Directory")
        btn_select_dir.clicked.connect(self.select_directory)
        img_group_layout.addWidget(btn_select_dir)

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

        # Run Mode 選擇
        mode_layout = QHBoxLayout()
        mode_label = QLabel("Run Mode:")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Single Image", "Batch (Directory)"])
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo)
        params_layout.addLayout(mode_layout)
        
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
        if self.mode_combo.currentText() == "Batch (Directory)":
            self.run_batch(mode="2d")
        else:
            self.run_single(mode="2d")

    def run_pose_estimation_3d(self):
        if self.mode_combo.currentText() == "Batch (Directory)":
            self.run_batch(mode="3d")
        else:
            self.run_single(mode="3d")

            
    def select_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory", "./images")
        if dir_path:
            self.image_path = dir_path
            self.label.setText(f"Selected directory: {os.path.basename(dir_path)}")
            self.img_preview.clear()  # 清除圖片預覽
            
    def run_batch(self, mode="2d"):
        if not self.image_path or not os.path.isdir(self.image_path):
            self.label.setText("No directory selected for batch mode!")
            return

        if mode == "2d":
            output_dir = "./../2d_results"
            runner = run_2d_pose
        else:
            output_dir = "./../3d_results"
            runner = run_3d_pose

        os.makedirs(output_dir, exist_ok=True)

        valid_ext = [".png", ".jpg", ".jpeg", ".bmp"]
        img_files = [f for f in os.listdir(self.image_path) if os.path.splitext(f)[1].lower() in valid_ext]

        if not img_files:
            self.label.setText("No images found in directory!")
            return

        last_result = None
        for img_name in img_files:
            img_path = os.path.join(self.image_path, img_name)
            success = runner(img_path=img_path, rad=self.radius, thick=self.thickness)
            if success:
                last_result = os.path.join(output_dir, img_name)

        if last_result:
            max_size = 580
            self.img_preview.setPixmap(QPixmap(last_result).scaled(max_size, max_size, Qt.AspectRatioMode.KeepAspectRatio))
            self.label.setText(f"Processed {len(img_files)} images. Last result shown.")
        else:
            self.label.setText("Batch processing failed.")