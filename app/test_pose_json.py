# test_pose_json.py
import json
import matplotlib.pyplot as plt
import cv2

# 你要改成自己的路徑
json_path = './../results/pose_2d_results.json'
image_path = './../images/test3.jpg'

# 骨架關節間連線，跟 COCO keypoint skeleton 一樣 (你可以修改)
# skeleton = [
#     (0, 1), (1, 2), (2, 3), (3, 4),       # 右手臂
#     (0, 5), (5, 6), (6, 7), (7, 8),       # 左手臂
#     (0, 9), (9, 10), (10, 11),             # 右腿
#     (0, 12), (12, 13), (13, 14)            # 左腿
# ]

COCO_SKELETON = [(6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (5, 11), (11, 12), (5, 6),
    (0, 5), (0, 1)
]

def draw_pose(image, keypoints_with_scores, skeleton=COCO_SKELETON, threshold=0.3):
    img = image.copy()

    for (x, y, conf) in keypoints_with_scores:
        if conf > threshold:
            cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)

    for joint_start, joint_end in skeleton:
        if (keypoints_with_scores[joint_start][2] > threshold and
            keypoints_with_scores[joint_end][2] > threshold):
            start_point = (int(keypoints_with_scores[joint_start][0]), int(keypoints_with_scores[joint_start][1]))
            end_point = (int(keypoints_with_scores[joint_end][0]), int(keypoints_with_scores[joint_end][1]))
            cv2.line(img, start_point, end_point, (255, 0, 0), 2)

    return img

def main():
    with open(json_path, 'r', encoding='utf-8') as f:
        pose_data = json.load(f)

    # 你的 JSON 結構是 pose_data -> list[list[dict]]
    # 所以先拿第一層list，再拿第一個人 dict
    person_data = pose_data[0][0]

    keypoints = person_data['keypoints']
    scores = person_data['keypoint_scores']

    keypoints_with_scores = [(x, y, conf) for (x, y), conf in zip(keypoints, scores)]

    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot load image from {image_path}")
        return

    image_with_pose = draw_pose(image, keypoints_with_scores)

    plt.imshow(cv2.cvtColor(image_with_pose, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('2D Pose Visualization from JSON')
    plt.show()


if __name__ == "__main__":
    main()
