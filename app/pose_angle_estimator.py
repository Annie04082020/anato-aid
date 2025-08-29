import numpy as np

# 假設輸入為 MMPose 2D 偵測的結果
# keypoints: numpy array of shape (17, 3) -> (x, y, score)
# 格式參考 COCO skeleton: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md
# 主要用到: Nose(0), L/R eye(1,2), L/R ear(3,4), L/R shoulder(5,6), L/R hip(11,12)

def angle_from_vector(v):
    """計算向量的 atan2 角度 (以水平軸為基準)。"""
    return np.degrees(np.arctan2(v[1], v[0]))

def compute_pose_angles(keypoints):
    result = {
        'yaw_bin': None,
        'pitch_bin': None,
        'roll_bin': None,
        'confidence': 0.0,
        'theta_img': None
    }

    if keypoints.shape[0] < 17:
        return result

    # 提取關鍵點
    nose = keypoints[0, :2]
    L_shoulder, R_shoulder = keypoints[5, :2], keypoints[6, :2]
    L_eye, R_eye = keypoints[1, :2], keypoints[2, :2]

    score = np.mean(keypoints[:, 2])

    # 肩線
    shoulder_vec = R_shoulder - L_shoulder
    if np.linalg.norm(shoulder_vec) > 1e-3:
        theta_img = angle_from_vector(shoulder_vec)
        result['theta_img'] = theta_img

        # roll: 肩線 vs 水平
        if abs(theta_img) < 15:
            result['roll_bin'] = '水平'
        elif theta_img > 0:
            result['roll_bin'] = '右傾'
        else:
            result['roll_bin'] = '左傾'

    # yaw 判斷：鼻子相對肩中心
    shoulder_mid = (L_shoulder + R_shoulder) / 2
    if np.linalg.norm(nose - shoulder_mid) > 1e-3:
        nose_rel = nose[0] - shoulder_mid[0]
        if nose_rel > 20:
            result['yaw_bin'] = '左側面'
        elif nose_rel < -20:
            result['yaw_bin'] = '右側面'
        else:
            result['yaw_bin'] = '正面/背面'

    # pitch 判斷：鼻子高度 vs 眼睛高度
    eye_mid_y = (L_eye[1] + R_eye[1]) / 2
    if not np.isnan(eye_mid_y):
        nose_eye_diff = nose[1] - eye_mid_y
        if nose_eye_diff < -10:
            result['pitch_bin'] = '仰頭'
        elif nose_eye_diff > 10:
            result['pitch_bin'] = '低頭'
        else:
            result['pitch_bin'] = '平視'

    # confidence: 平均關鍵點分數
    result['confidence'] = float(score)
    return result


if __name__ == "__main__":
    # 假設有一組模擬 keypoints (17,3)
    dummy_keypoints = np.zeros((17, 3))
    dummy_keypoints[0] = [50, 30, 0.9]   # nose
    dummy_keypoints[5] = [40, 50, 0.9]   # L shoulder
    dummy_keypoints[6] = [60, 50, 0.9]   # R shoulder
    dummy_keypoints[1] = [48, 28, 0.9]   # L eye
    dummy_keypoints[2] = [52, 28, 0.9]   # R eye

    angles = compute_pose_angles(dummy_keypoints)
    print(angles)
