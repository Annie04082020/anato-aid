# pose3d_viewer.py
import os
import numpy as np
from PyQt6 import QtWidgets
import pyqtgraph.opengl as gl

from mmpose.apis import init_model
from mmpose.apis.inference_3d import (
    extract_pose_sequence,
    inference_pose_lifter_model
)
from mmpose.structures import merge_data_samples

# 模型設定
LIFTER_CONFIG = './../mmpose/configs/body_3d_keypoint/lift_pose3d/h36m/simple_baseline3d_h36m.py'
LIFTER_CHECKPOINT = './../checkpoints/simple3Dbaseline_h36m-f0ad73a4_20210419.pth'

# 自定義骨架連接線（依 Human3.6M 關節順序調整）
SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10),
]

class Pose3DViewer(QtWidgets.QMainWindow):
    def __init__(self, keypoints_3d):
        super().__init__()
        self.setWindowTitle("3D Pose Viewer")
        self.resize(800, 600)

        # 3D 視圖
        self.view = gl.GLViewWidget()
        self.view.opts['distance'] = 3
        self.setCentralWidget(self.view)

        # 畫點
        self.scatter = gl.GLScatterPlotItem(pos=keypoints_3d, color=(1, 0, 0, 1), size=10)
        self.view.addItem(self.scatter)

        # 畫骨架線
        for i, j in SKELETON:
            if i < len(keypoints_3d) and j < len(keypoints_3d):
                pts = np.array([keypoints_3d[i], keypoints_3d[j]])
                line = gl.GLLinePlotItem(pos=pts, color=(0, 1, 0, 1), width=2, antialias=True)
                self.view.addItem(line)

def run_3d_pose(image_path, detection_result, pose2d_results):
    """執行 3D pose estimation 並回傳 numpy keypoints"""
    pose_lifter = init_model(LIFTER_CONFIG, LIFTER_CHECKPOINT, device='cpu')

    pose_seq = extract_pose_sequence(pose2d_results, frame_idx=0, causal=False, seq_len=1)
    merged_result = merge_data_samples(pose_seq)
    pose_lifted_results = inference_pose_lifter_model(pose_lifter, merged_result)

    # 取第一個人的骨架
    keypoints_3d = pose_lifted_results[0].pred_instances.keypoints3d[0].cpu().numpy()
    return keypoints_3d

def show_pose3d_window(image_path, det_results, pose2d_results):
    """整合流程：推論 + 顯示視窗"""
    keypoints_3d = run_3d_pose(image_path, det_results, pose2d_results)

    app = QtWidgets.QApplication([])
    viewer = Pose3DViewer(keypoints_3d)
    viewer.show()
    app.exec()
    
def get_3d_keypoints(pose2d_results):
    from mmpose.apis import init_model
    from mmpose.apis.inference_3d import extract_pose_sequence, inference_pose_lifter_model
    from mmpose.structures import merge_data_samples

    # 先初始化 model（路徑根據你實際位置調整）
    lifter = init_model(LIFTER_CONFIG, LIFTER_CHECKPOINT, device='cpu')

    pose_seq = extract_pose_sequence(pose2d_results, frame_idx=0, causal=False, seq_len=1)
    merged = merge_data_samples(pose_seq)
    pose_lifted_results = inference_pose_lifter_model(lifter, merged)

    # 取第一個人的3D keypoints numpy
    for res in pose_lifted_results:
        kpt3d = res.pred_instances.keypoints3d[0].cpu().numpy()
        return kpt3d

    return None
