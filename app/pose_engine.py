# pose_engine.py
import os
import subprocess
import sys
from mmpose.apis import inference_topdown, init_model as init_pose_model
from mmdet.apis import inference_detector, init_detector
# 設定參數與路徑（你可以改用 config.py 儲存）
python_path = sys.executable
MMPose_DIR = "./../mmpose"
CHECKPOINT_PATH = "./../checkpoints/hrnet_w32_coco_256x192-c78dce93_20200708.pth"
CONFIG_PATH = "./../mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_gridmask-8xb64-210e_coco-256x192.py"
DET_CONFIG_PATH = "./../mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py"
DET_CHECKPOINT_PATH = "./../checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"

# 初始化模型 (全域變數避免重複載入)
det_model = None
pose_model = None

def run_2d_pose(image_path, output_dir, radius=6, thickness=4, device="cpu", return_data=False):
    global det_model, pose_model
    command = [
        python_path,
        os.path.join(MMPose_DIR, "demo/topdown_demo_with_mmdet.py"),
        DET_CONFIG_PATH,
        DET_CHECKPOINT_PATH,
        CONFIG_PATH,
        CHECKPOINT_PATH,
        "--input", image_path,
        "--output-root", output_dir,
        "--radius", str(radius),
        "--thickness", str(thickness),
        "--device", device
    ]
    
    # # 初始化偵測模型
    # if det_model is None:
    #     det_model = init_detector(
    #         './../mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py',
    #         './../checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth',
    #         device='cpu'
    #     )

    # # 初始化姿態模型
    # if pose_model is None:
    #     pose_model = init_pose_model(
    #         './../mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py',
    #         './../checkpoints/hrnet_w32_coco_256x192-c78dce93_20200708.pth',
    #         device='cpu'
    #     )
        
    # # 1. 物件偵測
    # det_results = inference_detector(det_model, image_path)

    # # 2. 姿態估計 (針對人類類別的偵測框)
    # pose2d_results = inference_topdown(pose_model, image_path, det_results, bbox_thr=0.3)

    # # 3. 繪圖輸出
    # vis_path = os.path.join(output_dir, os.path.basename(image_path))
    # pose_model.vis_pose_result(
    #     image_path, pose2d_results, kpt_score_thr=0.3,
    #     radius=radius, thickness=thickness, show=False, out_file=vis_path
    # )

    # if return_data:
    #     return det_results, pose2d_results  # 這裡回傳 tuple
    # else:
        # return vis_path  # 原本的行為
    try:
        subprocess.run(command, check=True)
        result_img = os.path.join(output_dir, os.path.basename(image_path))
        return os.path.exists(result_img)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Pose estimation failed: {e}")
        return False
        
    