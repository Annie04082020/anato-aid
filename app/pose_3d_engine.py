from mmpose.apis import MMPoseInferencer

img_path = './../images/test3.jpg'   # 将img_path替换给你自己的路径

# 使用模型别名创建推理器
inferencer = MMPoseInferencer(pose3d="human3d")


def run_3d_pose(img_path,output_dir="./../3d_results",rad=6,thick=4,device="cpu"):
# MMPoseInferencer采用了惰性推断方法，在给定输入时创建一个预测生成器
    result_generator = inferencer(
        img_path, 
        show=False,
        radius=rad,          # Keypoint radius
        thickness=thick,       # Line thickness
        kpt_thr=0.35,       # Keypoint confidence threshold
        draw_bbox=True,    # Draw bounding boxes
        skeleton_style='mmpose',  # 'mmpose' or 'openpose'
        vis_out_dir='./../3d_results',
        pred_out_dir='./../3d_predictions'
    )
    result = next(result_generator)
    return result

# run_3d_pose(img_path,output_dir="./../3d_results",rad=6,thick=4,device="cpu")