# samples/kidney_glom/batch_infer_kidney_glom.py
#
# 對 dataset_dir/images 底下所有 PNG 做腎絲球推論，
# 並把結果存到 dataset_dir/results/

import os
import sys
import argparse
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
from skimage.measure import find_contours

# ---- 路徑設定 ---- #
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)

from mrcnn import model as modellib, visualize
from kidney_glom import KidneyInferenceConfig


def show_instances_outline(image, boxes, masks, class_ids, class_names, scores=None, ax=None):
    """
    只畫 mask 的輪廓，不畫填滿、不畫 bbox 方框。
    保留你原本的邏輯。
    """
    if ax is None:
        _, ax = plt.subplots(1, figsize=(10, 10))

    ax.imshow(image)
    N = masks.shape[-1]
    colors = visualize.random_colors(N)

    for i in range(N):
        color = colors[i]
        m = masks[:, :, i]

        # padding 避免邊界問題
        padded = np.zeros((m.shape[0] + 2, m.shape[1] + 2), dtype=np.uint8)
        padded[1:-1, 1:-1] = m.astype(np.uint8)

        # 找輪廓
        contours = find_contours(padded, 0.5)
        for verts in contours:
            verts = np.fliplr(verts) - 1  # (row, col) -> (x, y)
            ax.plot(verts[:, 0], verts[:, 1], linewidth=2, color=color)

        # 畫文字 label（放在 bbox 左上角附近）
        if scores is not None:
            class_id = class_ids[i]
            label = class_names[class_id]
            score = scores[i]
            y1, x1, y2, x2 = boxes[i]
            caption = f"{label} {score:.3f}"
            ax.text(x1, y1 - 2, caption,
                    color="w", size=11, backgroundcolor="none")

    ax.axis("off")
    return ax


def main():
    parser = argparse.ArgumentParser(
        description="Batch inference for glomeruli using Mask R-CNN (TF2)."
    )
    parser.add_argument("--dataset",
                        required=True,
                        help="資料集根目錄（內含 images/ 與 annotations/）")
    parser.add_argument("--weights",
                        required=True,
                        help="訓練好的 .h5 權重檔路徑")
    parser.add_argument("--results_subdir",
                        required=False,
                        default="results",
                        help="結果輸出子資料夾名稱（預設: results）")
    args = parser.parse_args()

    dataset_dir = args.dataset
    images_dir = os.path.join(dataset_dir, "images")
    result_dir = os.path.join(dataset_dir, args.results_subdir)

    print("ROOT_DIR   :", ROOT_DIR)
    print("DATASET_DIR:", dataset_dir)
    print("IMAGES_DIR :", images_dir)
    print("RESULT_DIR :", result_dir)
    print("WEIGHTS    :", args.weights)

    if not os.path.exists(args.weights):
        print("[ERROR] 找不到權重檔:", args.weights)
        return

    if not os.path.isdir(images_dir):
        print("[ERROR] 找不到 images 資料夾:", images_dir)
        return

    os.makedirs(result_dir, exist_ok=True)

    # 建立 inference model
    config = KidneyInferenceConfig()
    model = modellib.MaskRCNN(mode="inference",
                              config=config,
                              model_dir=os.path.join(ROOT_DIR, "logs"))
    model.load_weights(args.weights, by_name=True)

    # class names
    class_names = ["BG", "glomerulus"]

    # 找出所有 PNG
    filenames = [f for f in os.listdir(images_dir)
                 if f.lower().endswith(".png")]
    filenames.sort()

    if not filenames:
        print("[WARN] 在資料夾裡找不到任何 PNG：", images_dir)
        return

    print("Found {} images.".format(len(filenames)))

    for idx, fname in enumerate(filenames, 1):
        image_path = os.path.join(images_dir, fname)
        print("\n[{}/{}] Processing {}".format(idx, len(filenames), image_path))

        image = skimage.io.imread(image_path)

        # 確保是 RGB 三通道
        if image.ndim == 2:
            image = np.stack((image,) * 3, axis=-1)
        elif image.shape[-1] == 4:
            image = image[..., :3]

        # 做推論
        results = model.detect([image], verbose=0)
        r = results[0]

        # 顯示並存圖到 RESULT_DIR
        fig, ax = plt.subplots(1, figsize=(10, 10))
        show_instances_outline(
            image, r["rois"], r["masks"], r["class_ids"],
            class_names, r["scores"], ax=ax
        )

        base = os.path.splitext(fname)[0]
        out_path = os.path.join(result_dir, base + "_maskrcnn_result.png")
        plt.savefig(out_path, bbox_inches="tight", dpi=200)
        plt.close(fig)

        print("Saved:", out_path)


if __name__ == "__main__":
    main()
