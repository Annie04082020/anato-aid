您想要在 MMPose 中進行 finetune 和訓練模型。讓我為您詳細說明需要準備的東西以及如何編寫 train.py。

## 需要準備的東西

### 1. 資料集準備

首先需要準備您的資料集。MMPose 支援多種資料集格式，最常見的是 COCO 格式 [1](#13-0) 。

您需要：

- 圖像檔案
- 標註檔案（JSON 格式）
- 資料集元資訊配置檔案

### 2. 配置檔案設定

配置檔案是 MMPose 訓練的核心，包含五個主要部分 [2](#13-1) ：

- **General**：基本配置（Hook、Logger 等）
- **Data**：資料集、資料載入器、資料增強
- **Training**：優化器、學習率調度、訓練輪數
- **Model**：模型結構、模組、損失函數
- **Evaluation**：評估指標

### 3. 資料處理管線配置

需要設定訓練和驗證的資料處理管線 [3](#13-2) ：

```python
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
```

## train.py 的編寫方式

### 1. 使用內建的訓練腳本

MMPose 提供了現成的訓練腳本 [4](#13-3) ：

```bash
python tools/train.py ${CONFIG_FILE} [ARGS]
```

### 2. 訓練命令範例

基本訓練命令：

```bash
python tools/train.py configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py
```

### 3. 常用訓練參數 [5](#13-4)

- `--work-dir`：指定訓練日誌和 checkpoint 存放目錄
- `--resume`：恢復訓練
- `--amp`：開啟混合精度訓練
- `--cfg-options`：臨時覆蓋配置檔案中的設定

### 4. 多卡訓練

如果您有多張 GPU，可以使用分散式訓練 [6](#13-5) ：

```bash
bash ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [PY_ARGS]
```

## Finetune 的特殊設定

### 1. 載入預訓練權重

在配置檔案中設定 `load_from` 參數來載入預訓練模型：

```python
load_from = 'path/to/pretrained_model.pth'
```

### 2. 凍結部分參數

如果您想凍結某些層的參數進行 finetune [7](#13-6) ：

```python
optim_wrapper = dict(
    optimizer=dict(...),
    paramwise_cfg=dict(
        custom_keys={
            'backbone.layer0': dict(lr_mult=0, decay_mult=0),
            'backbone.layer1': dict(lr_mult=0, decay_mult=0),
        }))
```

## 自定義資料集配置

如果使用自定義資料集，需要建立資料集元資訊配置檔案 [8](#13-7) ：

```python
train_dataloader = dict(
    dataset=dict(
        type='CocoDataset',  # 或您的自定義資料集類別
        data_root='your_data_root',
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/'),
        metainfo=dict(from_file='configs/_base_/datasets/custom.py'),
        ...),
)
```

## Notes

MMPose 的訓練系統高度模組化且基於配置檔案驅動。您主要需要準備好資料集、配置檔案，然後使用內建的 `tools/train.py` 腳本即可開始訓練。對於 finetune，重點是正確設定預訓練權重載入和可能的參數凍結策略。

Wiki pages you might want to explore:

- [Architecture (open-mmlab/mmpose)](/wiki/open-mmlab/mmpose#2)

---

根據 Human-Art 專案的架構，你需要使用 **MMPose 框架**來撰寫訓練腳本，因為 Human-Art 已經整合到 MMPose 中。 [1](#3-0)

## 訓練方式說明

Human-Art 專案**不維護獨立的訓練代碼庫**，而是完全依賴 MMPose 框架進行訓練。 [2](#3-1)

## 具體實作步驟

### 1. 使用 MMPose 框架

你需要直接使用 MMPose 的訓練腳本，因為 Human-Art 與 MSCOCO 格式相容，可以直接使用 MMPose 的 dataloader。 [2](#3-1)

### 2. 可用的配置檔案

專案提供了多種預訓練模型的配置檔案：

**ViTPose 系列：**

- `td-hm_ViTPose-small_8xb64-210e_humanart-256x192.py`
- `td-hm_ViTPose-base_8xb64-210e_humanart-256x192.py`

**HRNet 系列：**

- `td-hm_hrnet-w32_8xb64-210e_humanart-256x192.py`
- `td-hm_hrnet-w48_8xb32-210e_humanart-256x192.py`

**RTMPose 系列：**

- `rtmpose-t_8xb256-420e_humanart-256x192.py`
- `rtmpose-s_8xb256-420e_humanart-256x192.py` [3](#3-2)

### 3. 資料集格式

確保你的資料集按照以下結構組織：

````
data/
├── HumanArt/
│   ├── annotations/
│   │   ├── training_humanart_coco.json
│   │   └── validation_humanart_coco.json
│   └── images/
│       ├── real_human/
│       ├── 2D_virtual_human/
│       └── 3D_virtual_human/
``` [4](#3-3)

## 建議做法

1. **安裝 MMPose**：按照 MMPose 官方文件安裝框架
2. **下載 Human-Art 資料集**：使用提供的表單申請下載
3. **選擇配置檔案**：根據你的需求選擇適合的模型配置
4. **使用 MMPose 訓練腳本**：直接使用 MMPose 的 `tools/train.py`

**Notes**

由於 MMPose 框架經常更新，Human-Art 專案不維護獨立的訓練代碼。你需要參考 MMPose 的官方文件來了解具體的訓練命令和參數設置。所有的模型配置檔案都已經針對 Human-Art + COCO 聯合訓練進行了優化。

Wiki pages you might want to explore:
- [Overview (IDEA-Research/HumanArt)](/wiki/IDEA-Research/HumanArt#1)
````
