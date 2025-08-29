from mmpose.apis import init_model  
  
# 初始化 3D pose lifter 模型  
pose_lifter = init_model(  
    'configs/body_3d_keypoint/video_pose_lift/h36m/video-pose-lift_tcn-243frm-supv-cpn-ft_8xb128-200e_h36m.py',  
    'https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth',  
    device='cuda:0'  
)