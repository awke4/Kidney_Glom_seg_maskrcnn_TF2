# samples/kidney_glom/kidney_glom.py
#
# 腎絲球 Mask R-CNN (TF2.14 友善版)
# 建議搭配 z-mahmud22/Mask-RCNN_TF2.14.0 使用

import os
import sys
import json
import argparse
import numpy as np
import skimage.io
import skimage.draw

# ----------------- 路徑設定 ----------------- #
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)

from mrcnn.config import Config
from mrcnn import model as modellib, utils

# 預設 logs 與 dataset 路徑（之後用 argparse 覆蓋）
DEFAULT_MODEL_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_DIR = os.path.join(ROOT_DIR, "datasets", "kidney_glom")
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")


# ----------------- Config ----------------- #
class KidneyConfig(Config):
    """腎絲球 segmentation 的基本設定"""
    NAME = "kidney_glom"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1 + 1  # background + glomerulus

    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 1024

    STEPS_PER_EPOCH = 10
    VALIDATION_STEPS = 5

    # DETECTION_MIN_CONFIDENCE = 0.7
    USE_MINI_MASK = False   # 避免 mini-mask / resize 的麻煩


class KidneyInferenceConfig(KidneyConfig):
    """推論用 Config"""
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.94


# ----------------- GeoJSON 工具 ----------------- #
def load_geojson_polygons(geojson_path):
    """讀取單一 GeoJSON，輸出 polygon list: [{"x":[...], "y":[...]}]"""
    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    polygons = []
    for feat in data.get("features", []):
        geom = feat.get("geometry", {})
        if geom.get("type") != "Polygon":
            continue
        rings = geom.get("coordinates", [])
        if not rings:
            continue
        coords = rings[0]
        xs = [pt[0] for pt in coords]
        ys = [pt[1] for pt in coords]
        polygons.append({"x": xs, "y": ys})

    return polygons


# ----------------- Dataset ----------------- #
class KidneyDataset(utils.Dataset):

    def load_kidney(self, dataset_dir, subset="train"):
        """
        載入腎絲球資料集。現在 subset 只是 tag，train/val/eval 都讀同一包。
        結構預期：
          dataset_dir/
            images/*.png
            annotations/*.geojson
        """
        self.add_class("kidney", 1, "glomerulus")

        images_dir = os.path.join(dataset_dir, "images")
        ann_dir    = os.path.join(dataset_dir, "annotations")

        image_files = [
            f for f in os.listdir(images_dir)
            if f.lower().endswith(".png")
        ]
        image_files.sort()

        for filename in image_files:
            image_path = os.path.join(images_dir, filename)
            base, _ = os.path.splitext(filename)
            geojson_path = os.path.join(ann_dir, base + ".geojson")

            if not os.path.exists(geojson_path):
                print("[WARN] 找不到對應 GeoJSON:", geojson_path)
                polygons = []
            else:
                polygons = load_geojson_polygons(geojson_path)

            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "kidney",
                image_id=base,
                path=image_path,
                width=width,
                height=height,
                polygons=polygons
            )

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        if info["source"] != "kidney":
            return super(KidneyDataset, self).load_mask(image_id)

        height = info["height"]
        width  = info["width"]
        polygons = info["polygons"]

        if not polygons:
            mask = np.zeros((height, width, 0), dtype=np.bool_)
            class_ids = np.array([], dtype=np.int32)
            return mask, class_ids

        mask = np.zeros((height, width, len(polygons)), dtype=np.bool_)

        for i, p in enumerate(polygons):
            rr, cc = skimage.draw.polygon(p["y"], p["x"])
            rr = np.clip(rr, 0, height - 1)
            cc = np.clip(cc, 0, width  - 1)
            mask[rr, cc, i] = True

        class_ids = np.array([1] * len(polygons), dtype=np.int32)
        return mask, class_ids

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "kidney":
            return info["path"]
        else:
            return super(KidneyDataset, self).image_reference(image_id)


# ----------------- 訓練流程 ----------------- #
def train(model, config, dataset_dir):
    """
    用你的資料集訓練 (train = val = 全部資料，小作弊)
    """
    dataset_train = KidneyDataset()
    dataset_train.load_kidney(dataset_dir, "train")
    dataset_train.prepare()

    dataset_val = KidneyDataset()
    dataset_val.load_kidney(dataset_dir, "val")
    dataset_val.prepare()

    print("Train images:", dataset_train.num_images)
    print("Val images  :", dataset_val.num_images)

    # Phase 1: 只訓練 heads
    print("\n=== Phase 1: Train heads ===")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                layers="heads")

    # Phase 2: 微調所有層
    print("\n=== Phase 2: Fine-tune all layers ===")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10.0,
                epochs=30,
                layers="all")


# ----------------- 主程式 ----------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Mask R-CNN (TF2) to detect glomeruli."
    )
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' 或 'debug'")
    parser.add_argument("--weights",
                        required=False,
                        default="coco",
                        help="coco / last / 自訂 .h5 路徑")
    parser.add_argument("--dataset",
                        required=False,
                        default=DEFAULT_DATASET_DIR,
                        help="資料集根目錄（內含 images/ 和 annotations/）")
    parser.add_argument("--logs",
                        required=False,
                        default=DEFAULT_MODEL_DIR,
                        help="log / checkpoints 輸出目錄")
    args = parser.parse_args()

    print("ROOT_DIR   :", ROOT_DIR)
    print("DATASET_DIR:", args.dataset)
    print("MODEL_DIR  :", args.logs)
    print("Command    :", args.command)
    print("Weights    :", args.weights)

    if args.command == "debug":
        config = KidneyConfig()
        config.display()
        dataset_train = KidneyDataset()
        dataset_train.load_kidney(args.dataset, "train")
        dataset_train.prepare()
        print("Train images:", dataset_train.num_images)
        for i in range(dataset_train.num_images):
            info = dataset_train.image_info[i]
            print(f"  {i}: id={info['id']}, file={os.path.basename(info['path'])}, polys={len(info['polygons'])}")
        sys.exit(0)

    assert args.command == "train", "目前只支援 train 或 debug"

    config = KidneyConfig()
    config.display()

    os.makedirs(args.logs, exist_ok=True)

    model = modellib.MaskRCNN(mode="training",
                              config=config,
                              model_dir=args.logs)

    # 載入權重
    if args.weights.lower() == "coco":
        if not os.path.exists(COCO_WEIGHTS_PATH):
            print("找不到 COCO 權重檔，請先下載 mask_rcnn_coco.h5 到：")
            print(" ", COCO_WEIGHTS_PATH)
            sys.exit(1)
        print("Loading weights from COCO:", COCO_WEIGHTS_PATH)
        model.load_weights(COCO_WEIGHTS_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif args.weights.lower() == "last":
        last_path = model.find_last()
        print("Loading last trained weights:", last_path)
        model.load_weights(last_path, by_name=True)
    elif args.weights.lower() == "random":
        # ⭐ 重要：完全不載任何權重，保持隨機初始化
        print("Training from random initialization (no pre-trained weights).")
    else:
        print("Loading custom weights:", args.weights)
        model.load_weights(args.weights, by_name=True)

    train(model, config, args.dataset)
