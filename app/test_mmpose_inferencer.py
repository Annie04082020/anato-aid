import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 雖然沒用到直接調用，但 matplotlib 3d 需要
from mmpose import MMPoseInferencer

def visualize_3d_skeleton(keypoints_3d):
    """
    keypoints_3d: numpy array, shape (num_keypoints, 3)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 畫出關節點（紅色點）
    ax.scatter(keypoints_3d[:, 0], keypoints_3d[:, 1], keypoints_3d[:, 2], c='r', s=30)

    # 骨架連線，這裡用 COCO 的前半段連接示意
    skeleton = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (1, 5), (5, 6), (6, 7),
        (1, 8), (8, 9), (9, 10),
        (8, 12), (12, 13), (13, 14),
        (8, 11), (11, 12)
    ]
    for i, j in skeleton:
        x = [keypoints_3d[i, 0], keypoints_3d[j, 0]]
        y = [keypoints_3d[i, 1], keypoints_3d[j, 1]]
        z = [keypoints_3d[i, 2], keypoints_3d[j, 2]]
        ax.plot(x, y, z, c='b', linewidth=2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Pose Estimation (Interactable)')
    plt.show()

def run_3d_pose_infer(image_path):
    # 載入3D姿態推論器，會自動下載並快取權重
    inferencer = MMPoseInferencer('lift_simple_baseline3d_h36m', device='cpu')  # 或改成 'cuda' 加速

    # 推論，回傳結果包含2D和3D keypoints
    results = inferencer(image_path)

    # 結果結構會依模型不同有所差異，這裡示範拿第一個人物的3D關節點
    if not results or len(results) == 0:
        print("No pose detected.")
        return

    # 從結果取出3D keypoints (shape: num_instances x num_keypoints x 3)
    keypoints_3d = None
    for res in results:
        if 'keypoints_3d' in res:
            keypoints_3d = res['keypoints_3d']
            break
        # 部分版本可能用 pred_instances.keypoints3d (torch.Tensor)，可依實際格式調整

    if keypoints_3d is None:
        print("3D keypoints not found in results.")
        return

    # 如果是Tensor，轉成numpy
    if hasattr(keypoints_3d, 'cpu'):
        keypoints_3d = keypoints_3d.cpu().numpy()

    visualize_3d_skeleton(keypoints_3d[0])  # 取第一個人的3D關節

if __name__ == '__main__':
    image_path = 'your_image.jpg'  # <-- 這裡換成你的圖片路徑
    run_3d_pose_infer(image_path)
