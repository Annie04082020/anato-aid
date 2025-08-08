import os
import subprocess
import sys
from mmpose.apis import init_model
# from mmpose.apis import (inference_topdown_pose_model, init_pose_model)
from mmdet.apis import inference_detector, init_detector
import cv2
import numpy as np

# 設定參數與路徑（你可以改用 config.py 儲存）
python_path = sys.executable
MMPose_DIR = "./../mmpose"
CHECKPOINT_PATH = "./../hrnet_w32_coco_256x192-c78dce93_20200708.pth"
CONFIG_PATH = "./../mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_gridmask-8xb64-210e_coco-256x192.py"
DET_CONFIG_PATH = "./../mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py"
DET_CHECKPOINT_PATH = "./../checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"

def run_2d_pose(image_path, output_dir, radius=6, thickness=4, device='cpu', return_data=False):
    # init detector
    det_model = init_detector(DET_CONFIG_PATH, DET_CHECKPOINT_PATH, device=device)
    # init pose model
    pose_model = init_pose_model(CONFIG_PATH, CHECKPOINT_PATH, device=device)
    # read image
    image = cv2.imread(image_path)
    # detect human bbox
    det_results = inference_detector(det_model, image)
    # pick only person class bbox
    person_results = [bbox for bbox in det_results[0] if bbox[4] > 0.5]
    # pose inference
    pose_results, _ = inference_topdown_pose_model(pose_model, image, person_results)
    
    # visualize and save image
    vis_img = vis_pose_result(pose_model, image, pose_results, radius=radius, thickness=thickness)
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(save_path, vis_img)
    
    if return_data:
        return det_results, pose_results
    else:
        return True
    
def vis_pose_result(pose_model, img, pose_results, radius=6, thickness=4):
    img = img.copy()
    for person in pose_results:
        keypoints = person['keypoints']
        for x, y, conf in keypoints:
            if conf > 0.3:
                cv2.circle(img, (int(x), int(y)), radius, (0, 255, 0), -1)
        # 這邊可以加連線
        # skeleton = pose_model.cfg.skeleton
        # for joint_start, joint_end in skeleton:
        #     cv2.line(img, (int(keypoints[joint_start][0]), int(keypoints[joint_start][1])),
        #                     (int(keypoints[joint_end][0]), int(keypoints[joint_end][1])), (255,0,0), thickness)
    return img