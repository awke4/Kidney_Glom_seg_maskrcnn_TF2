# 🔬 Kidney Glomerulus Segmentation using Mask R-CNN (TensorFlow 2.x)


本專案提供一套完整的腎絲球（glomerulus）語意分割教學流程，使用**改良後的 Mask R-CNN（TensorFlow 2.x 版本）**。

專案包含以下核心功能：

* ✔ Dataset 準備
* ✔ 訓練（COCO → Heads → Fine-tune All）
* ✔ 批次推論（Batch Inference）
* ✔ 評估（Dice / F1 / Precision / Recall / Confusion Matrix）
* ✔ 可在 Google Colab 一鍵執行的 Notebook

---

## 📂 專案結構
```markdown

Kidney_Glomerulus_seg_maskrcnn/
│
├── mrcnn/                                  \ Mask R-CNN 核心框架
|   └── model.py
|   └──...
├── samples/
│   └── kidney_glom/
│       ├── kidney_glom.py                  \ 訓練主程式
│       ├── batch_infer_kidney_glom.py      \ 批次推論
│       └── evaluate_kidney_glom.py         \ 評估指標
│
├── Kidney_Glomerulus_seg_maskrcnn.ipynb    \ Colab Notebook
│
├── requirements_no_tf.txt                  \ 不含 TensorFlow 的依賴(目前暫時不使用該requirements)
│
|── data_demo/                              \ 小型 dataset for demo
|   ├── images/
|   └── annotations/
|
|── datasets/                               \ 完整資料集 (28 份)
|   └── kidney_glom/
|       ├── images/
|       └── annotations/
|
└──results/
    └──weight/mask_rcnn_kidney_glom_0030.h5 \提供的訓練完權重
    └──datasets/ ...png                     \使用mask_rcnn_kidney_glom_0030.h5推論的結果
    └──confusion_matrix.png                 \產生的混淆矩陣
```

---

## 📦 Dataset 格式

專案採用以下結構：

```

dataset/
    kidney_glom/
        images/
            A.png
            B.png
            ...
        annotations/
            A.geojson
            B.geojson
            ...

````

* **資料要求：** 每張影像需有對應的 **GeoJSON polygon**（多邊形）。
* **支援：** 支援一張圖有多個腎絲球。
* **擴增：** 新資料可新建資料夾存放，只需要遵照images + annotations的架構即可
---

## 🧪 訓練（Training）

* **coco資料集：** 自行下載後放在主目錄Kidney_Glom_seg_maskrcnn_TF2下，[下載連結](https://github.com/matterport/Mask_RCNN/releases)

訓練流程包含 **Phase 1: Train Heads** 和 **Phase 2: Fine-tune all layers**。模型會儲存在 `logs_kidney_tf2/` 下，這裡提供兩種範例:

### 1. 使用 COCO 初始化權重 + 完整資料集

```bash
python samples/kidney_glom/kidney_glom.py train \
    --dataset dataset/kidney_glom \
    --weights coco \
    --logs logs_kidney_tf2
````

### 2\. 使用隨機初始化權重 + 小型資料集

```bash
python samples/kidney_glom/kidney_glom.py train \
    --dataset data_demo \
    --weights random \
    --logs logs_kidney_tf2
```
-----

## 🖼 批次推論所有資料（Batch Inference）

```bash
python samples/kidney_glom/batch_infer_kidney_glom.py \
    --dataset dataset/kidney_glom \
    --weights logs_kidney_tf2/.../mask_rcnn_kidney_glom_0030.h5
```

  * **輸出位置：** 推論結果會存到 `dataset/kidney_glom/results/`

-----

## 📊 模型評估（Evaluation）

支援 Dice、Precision、Recall、F1、Confusion Matrix。

```bash
python samples/kidney_glom/evaluate_kidney_glom.py \
    --dataset dataset/kidney_glom \
    --weights logs_kidney_tf2/.../mask_rcnn_kidney_glom_0030.h5 \
    --iou 0.5
```

  * **混淆矩陣輸出：** `dataset/kidney_glom/confusion_matrix.png`

-----

## 🚀 Google Colab Notebook

Notebook 提供一鍵執行所有流程，包括：

  * TensorFlow / 套件安裝
  * Repo clone
  * Dataset 載入 / 解壓
  * Debug → 確認 dataset 正確讀取
  * Training
  * Batch Inference
  * Evaluation（Dice / F1 / CM）
  * 自動畫出推論結果與混淆矩陣

-----

## 🧪 Demo Dataset (`data_demo/`)

提供 2 張圖片 + 2 個對應 GeoJSON，用於快速試跑流程。

**範例結構：**

```
data_demo/
    images/
        demo1.png
        demo2.png
    annotations/
        demo1.geojson
        demo2.geojson
```

-----

## 🧠 系統需求

| 環境 | 說明 |
| :--- | :--- |
| Google Colab | GPU (T4 / L4 / A100) |
| TensorFlow | 使用 Colab 內建（TF 2.16+ / 2.18） |
| Python | Colab 內建（3.12） |
| Keras2 | (在 Colab 需手動安裝) |


