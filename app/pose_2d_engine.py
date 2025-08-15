from mmpose.apis import MMPoseInferencer

# img_path = './../images/test3.jpg'   # 将img_path替换给你自己的路径

# 使用模型别名创建推理器
inferencer = MMPoseInferencer('human')

inferencer2 = MMPoseInferencer(  
 pose2d="./../rtmpose-m_8xb256-420e_coco-256x192.py",
 pose2d_weights="./../checkpoints/rtmpose-m_8xb256-420e_humanart-256x192-8430627b_20230611.pth"   
)

def run_2d_pose(img_path,output_dir='./../2d_results',rad=6,thick=4,device="cpu"):
# MMPoseInferencer采用了惰性推断方法，在给定输入时创建一个预测生成器
    result_generator = inferencer2(
        img_path, 
        show=False,
        radius=rad,          # Keypoint radius
        thickness=thick,       # Line thickness
        kpt_thr=0.4,       # Keypoint confidence threshold
        draw_bbox=True,    # Draw bounding boxes
        skeleton_style='mmpose',  # 'mmpose' or 'openpose'
        vis_out_dir='./../2d_results',
        pred_out_dir='./../2d_predictions'
    )
    result = next(result_generator)
    return result
