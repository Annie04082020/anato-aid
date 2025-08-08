import os
import subprocess
import sys

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog,
    QVBoxLayout, QWidget
)
from PyQt6.QtGui import QPixmap

python_path = sys.executable  # 抓當前執行這支 UI 的 Python 路徑

MMPose_DIR = "./mmpose"
CHECKPOINT_PATH = "./hrnet_w32_coco_256x192-c78dce93_20200708.pth"
CONFIG_PATH = f"{MMPose_DIR}/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192.py"

class PoseEstimationUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Anato-Aid: Pose Estimator")
        self.image_path = None
        self.running = False  # 防止連續點擊

        layout = QVBoxLayout()

        self.label = QLabel("No image selected.")
        layout.addWidget(self.label)

        self.img_preview = QLabel()
        layout.addWidget(self.img_preview)

        btn_select = QPushButton("Select Image")
        btn_select.clicked.connect(self.select_image)
        layout.addWidget(btn_select)

        self.btn_run = QPushButton("Run Pose Estimation")
        self.btn_run.clicked.connect(self.run_pose_estimation)
        layout.addWidget(self.btn_run)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def select_image(self):
        file_dialog = QFileDialog()
        img_path, _ = file_dialog.getOpenFileName(self, "Select Image", "./images")
        if img_path:
            self.image_path = img_path
            self.label.setText(f"Selected: {os.path.basename(img_path)}")
            self.img_preview.setPixmap(QPixmap(img_path).scaled(256, 256))

    def run_pose_estimation(self):
        if not self.image_path:
            self.label.setText("⚠️ No image selected!")
            return
        if self.running:
            self.label.setText("⏳ Pose estimation already running...")
            return

        self.running = True
        self.label.setText("Running pose estimation...")
        self.btn_run.setEnabled(False)

        output_dir = "./vis_results"
        os.makedirs(output_dir, exist_ok=True)

        command = [
            python_path, 
            "mmpose/demo/topdown_demo_with_mmdet.py",
            CONFIG_PATH,
            CHECKPOINT_PATH,
            self.image_path,
            "--det-config",
            "./mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py",
            "--det-checkpoint",
            "./checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
            "--out-img-root",
            output_dir
        ]

        try:
            subprocess.run(command, check=True)

            result_img = os.path.join(output_dir, os.path.basename(self.image_path))
            if os.path.exists(result_img):
                self.img_preview.setPixmap(QPixmap(result_img).scaled(256, 256))
                self.label.setText("✅ Pose estimation complete!")
            else:
                self.label.setText("❌ Failed: result image not found.")
        except subprocess.CalledProcessError as e:
            self.label.setText(f"❌ Error during pose estimation:\n{e}")
        finally:
            self.running = False
            self.btn_run.setEnabled(True)

if __name__ == "__main__":
    app = QApplication([])
    window = PoseEstimationUI()
    window.show()
    app.exec()
