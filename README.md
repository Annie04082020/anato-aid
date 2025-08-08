# AnatoAid

**AnatoAid** 是一個用於分析繪圖中人體結構合理性的 AI 工具。它能自動從手繪或合成圖像中擷取骨架資訊、偵測可能不合理的比例或關節角度，並提供自然語言的修正建議與視覺化標註。

## 功能模組

- 🎯 姿勢偵測：使用 MMPose 擷取人體 keypoints
- 🧠 結構分析：自訂比例與活動範圍規則檢查異常點
- 💬 自然語言建議：搭配 PoseScript 等模型產生人類可讀建議
- 🖼️ 視覺化：標註異常關節、生成修正建議圖（ControlNet）

## 專案目標

作為創作者的虛擬「姿勢教練」，幫助繪師在創作過程中快速發現結構問題，降低修圖成本，提升理解與實作能力。

---

🚧 _開發中，模組會持續補上。_

# Setup Instructions for MMPose + MMDetection Project

## 1. Create and activate virtual environment (Python 3.8)

```bash
conda create -n mmpose python=3.8 -y
conda activate mmpose
```

## 2. Install PyTorch and torchvision (CUDA 11.8)

```bash
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

## 3. Install MMCV and MMEngine (compatible versions)

```bash
pip install mmengine==0.10.1
pip install mmcv==2.0.1
```

## 4. Install MIM (OpenMMLab Installer)

```bash
pip install openmim
```

## 5. Install MMDetection and MMPose

```bash
mim install mmdet==3.1.0
mim install mmpose==1.2.0
```

## 6. (Optional) Install other dependencies

If your project uses other OpenMMLab tools or COCO-style dataset tools:

```bash
pip install opencv-python-headless
pip install scipy matplotlib pandas
```

## 7. (Optional) Save dependencies to requirements.txt

```bash
pip freeze > requirements.txt
```
