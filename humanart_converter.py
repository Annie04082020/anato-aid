import json  
import numpy as np  
  
class HumanArtToMMPoseConverter:  
    def __init__(self):  
        # COCO 17 關鍵點順序 (MMPose 標準)  
        self.coco_keypoints = [  
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',  
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',  
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',  
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'  
        ]  
          
        # Human-Art 21 關鍵點到 COCO 17 關鍵點的映射  
        # 根據 Human-Art 論文，前 17 個點對應 COCO 格式  
        self.humanart_to_coco_mapping = list(range(17))  
      
    def convert_annotation(self, humanart_annotation):  
        """  
        將 Human-Art 標註轉換為 MMPose 3D lifting 格式  
          
        Args:  
            humanart_annotation: Human-Art 格式的標註  
              
        Returns:  
            mmpose_annotation: MMPose 格式的標註  
        """  
        mmpose_annotation = {}  
          
        # 複製基本資訊  
        mmpose_annotation['image_id'] = humanart_annotation['image_id']  
        mmpose_annotation['id'] = humanart_annotation['id']  
        mmpose_annotation['bbox'] = humanart_annotation['bbox']  
        mmpose_annotation['area'] = humanart_annotation['area']  
        mmpose_annotation['category_id'] = humanart_annotation['category_id']  
        mmpose_annotation['iscrowd'] = humanart_annotation['iscrowd']  
          
        # 使用 COCO 17 關鍵點格式  
        if 'keypoints' in humanart_annotation:  
            # 直接使用已有的 COCO 格式關鍵點  
            mmpose_annotation['keypoints'] = humanart_annotation['keypoints']  
            mmpose_annotation['num_keypoints'] = humanart_annotation['num_keypoints']  
        elif 'keypoints_21' in humanart_annotation:  
            # 從 21 關鍵點提取前 17 個 COCO 關鍵點  
            keypoints_21 = np.array(humanart_annotation['keypoints_21']).reshape(-1, 3)  
            coco_keypoints = keypoints_21[:17].flatten().tolist()  
            mmpose_annotation['keypoints'] = coco_keypoints  
            mmpose_annotation['num_keypoints'] = np.sum(keypoints_21[:17, 2] > 0)  
          
        return mmpose_annotation  
      
    def convert_dataset(self, humanart_json_path, output_json_path):  
        """  
        轉換整個 Human-Art 資料集為 MMPose 格式  
          
        Args:  
            humanart_json_path: Human-Art JSON 檔案路徑  
            output_json_path: 輸出的 MMPose 格式 JSON 檔案路徑  
        """  
        with open(humanart_json_path, 'r') as f:  
            humanart_data = json.load(f)  
          
        mmpose_data = {  
            'info': humanart_data['info'],  
            'images': humanart_data['images'],  
            'categories': humanart_data['categories'],  
            'annotations': []  
        }  
          
        # 轉換所有標註  
        for annotation in humanart_data['annotations']:  
            converted_annotation = self.convert_annotation(annotation)  
            mmpose_data['annotations'].append(converted_annotation)  
          
        # 儲存轉換後的資料  
        with open(output_json_path, 'w') as f:  
            json.dump(mmpose_data, f, indent=2)  
          
        print(f"轉換完成！輸出檔案：{output_json_path}")  
        print(f"轉換了 {len(mmpose_data['annotations'])} 個標註")  
  
# 使用範例  
def main():  
    converter = HumanArtToMMPoseConverter()  
      
    # 轉換訓練資料  
    converter.convert_dataset(  
        'data/HumanArt/annotations/training_humanart.json',  
        'data/HumanArt/annotations/training_humanart_mmpose.json'  
    )  
      
    # 轉換驗證資料  
    converter.convert_dataset(  
        'data/HumanArt/annotations/validation_humanart.json',  
        'data/HumanArt/annotations/validation_humanart_mmpose.json'  
    )  
  
if __name__ == "__main__":  
    main()