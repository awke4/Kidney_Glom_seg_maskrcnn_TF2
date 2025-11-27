# samples/kidney_glom/evaluate_kidney_glom.py
#
# 評估 Mask R-CNN 腎絲球 segmentation (TP / FP / FN + F1 + Dice + Confusion Matrix)
# TF2.14 版，無 sklearn / 無 seaborn

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import skimage.io

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)

from mrcnn import model as modellib
from kidney_glom import KidneyInferenceConfig, KidneyDataset

IOU_THRESHOLD = 0.5


def compute_iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return inter / union


def plot_confusion_matrix(cm, save_path):
    classes = ["glomerulus", "background"]

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(cm, interpolation='nearest', cmap="Blues")

    cbar = ax.figure.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))

    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    ax.set_xlabel("True", fontsize=12)
    ax.set_ylabel("Predicted", fontsize=12)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.title("Confusion Matrix", fontsize=14)
    fig.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate glomeruli segmentation (TP/FP/FN + F1 + Dice)."
    )
    parser.add_argument("--dataset",
                        required=True,
                        help="資料集根目錄（內含 images/ 與 annotations/）")
    parser.add_argument("--weights",
                        required=True,
                        help="訓練好的 .h5 權重檔路徑")
    parser.add_argument("--iou",
                        required=False,
                        type=float,
                        default=IOU_THRESHOLD,
                        help=f"IoU 閥值（預設 {IOU_THRESHOLD}）")
    args = parser.parse_args()

    dataset_dir = args.dataset
    weights_path = args.weights
    iou_thr = args.iou

    print("Using dataset:", dataset_dir)
    print("Using weights:", weights_path)
    print("IoU threshold:", iou_thr)

    if not os.path.exists(weights_path):
        print("[ERROR] 找不到權重檔:", weights_path)
        return

    # inference model
    config = KidneyInferenceConfig()
    model = modellib.MaskRCNN(mode="inference", config=config,
                              model_dir=os.path.join(ROOT_DIR, "logs"))
    model.load_weights(weights_path, by_name=True)

    # dataset
    dataset = KidneyDataset()
    dataset.load_kidney(dataset_dir, "eval")
    dataset.prepare()

    print("Total images:", dataset.num_images)

    TP = FP = FN = 0
    dice_list = []  # 每一顆成功匹配 (TP) glomerulus 的 Dice

    for i in range(dataset.num_images):
        info = dataset.image_info[i]
        img_path = info["path"]
        print(f"[{i+1}/{dataset.num_images}] Evaluating {os.path.basename(img_path)}")

        image = skimage.io.imread(img_path)
        if image.ndim == 2:
            image = np.stack((image,) * 3, axis=-1)
        elif image.shape[-1] == 4:
            image = image[..., :3]

        gt_mask, _ = dataset.load_mask(i)
        gt_count = gt_mask.shape[-1]

        r = model.detect([image], verbose=0)[0]
        pred_mask = r["masks"]
        pred_count = pred_mask.shape[-1]

        matched_pred = set()

        # 找 TP / FN
        for g in range(gt_count):
            best_iou = 0.0
            best_idx = -1

            for p in range(pred_count):
                if p in matched_pred:
                    continue
                iou = compute_iou(gt_mask[..., g], pred_mask[..., p])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = p

            if best_iou >= iou_thr:
                TP += 1
                matched_pred.add(best_idx)

                # 計算這一對 (GT, Pred) 的 Dice
                g_mask = gt_mask[..., g]
                p_mask = pred_mask[..., best_idx]

                intersection = np.logical_and(g_mask, p_mask).sum()
                gt_area = g_mask.sum()
                pred_area = p_mask.sum()
                denom = gt_area + pred_area

                if denom > 0:
                    dice = 2.0 * intersection / denom
                    dice_list.append(dice)
            else:
                FN += 1

        # 找 FP
        for p in range(pred_count):
            if p not in matched_pred:
                FP += 1

    # Dice 指標
    if len(dice_list) > 0:
        mean_dice = float(np.mean(dice_list))
        median_dice = float(np.median(dice_list))
    else:
        mean_dice = 0.0
        median_dice = 0.0

    # Precision / Recall / F1
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    print("\n=== Evaluation Result ===")
    print(f"TP = {TP}")
    print(f"FP = {FP}")
    print(f"FN = {FN}")
    print(f"Precision = {precision:.4f}")
    print(f"Recall    = {recall:.4f}")
    print(f"F1-score  = {f1:.4f}")
    print(f"Dice (mean over matched glomeruli)   = {mean_dice:.4f}")
    print(f"Dice (median over matched glomeruli) = {median_dice:.4f}")
    print(f"#Matched glomeruli (for Dice)        = {len(dice_list)}")

    cm = np.array([[TP, FP],
                   [FN, 0]])  # TN 無意義，填 0 即可

    out_path = os.path.join(dataset_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, out_path)
    print("\nSaved confusion matrix to:", out_path)


if __name__ == "__main__":
    main()
