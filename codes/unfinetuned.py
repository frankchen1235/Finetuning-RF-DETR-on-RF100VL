import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import json
import numpy as np
from pathlib import Path
from PIL import Image
from rfdetr import RFDETRMedium
import torch



DATASET_DIR = "/scratch1/chenyinu/rf100_merged_final"
SPLIT = "test"                        
IOU_THR = 0.5                         # area-based matching threshold
CONF_THR = 0.01                       # keep low to count all predictions
SAVE_JSON = "/home1/chenyinu/rfdetr_unfinetuned_eval.json"

ann_path = f"{DATASET_DIR}/{SPLIT}/_annotations.coco.json"
img_dir = f"{DATASET_DIR}/{SPLIT}/"


# Load un-finetuned RF-DETR


print("Loading un-finetuned (pretrained COCO) RF-DETR...")
model = RFDETRMedium(resolution=640)
model.optimize_for_inference()  


# Load COCO annotations


with open(ann_path, "r") as f:
    coco = json.load(f)

images = {img["id"]: img for img in coco["images"]}

gt_by_image = {}
for ann in coco["annotations"]:
    img_id = ann["image_id"]
    gt_by_image.setdefault(img_id, []).append(ann)


# Helper functions


def area(bb):
    x1, y1, x2, y2 = bb
    return max(0, x2 - x1) * max(0, y2 - y1)

def iou_area(pred, gt):
    x1 = max(pred[0], gt[0])
    y1 = max(pred[1], gt[1])
    x2 = min(pred[2], gt[2])
    y2 = min(pred[3], gt[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    denom = area(pred) + area(gt) - inter + 1e-6
    return inter / denom

# Run evaluation


tp_total = 0
fp_total = 0
fn_total = 0

img_ids = sorted(images.keys())

for idx, img_id in enumerate(img_ids):

    img_info = images[img_id]
    img_path = os.path.join(img_dir, img_info["file_name"])

    if not os.path.exists(img_path):
        print("Missing:", img_path)
        continue

    # Load image
    image = Image.open(img_path).convert("RGB")

    # Run RF-DETR inference
    det = model.predict(image, threshold=CONF_THR)

    pred_boxes = det.xyxy
    pred_scores = det.confidence

    # Sort descending by confidence
    if len(pred_boxes) > 0:
        order = np.argsort(-pred_scores)
        pred_boxes = pred_boxes[order]
    else:
        pred_boxes = np.empty((0, 4))

    # Ground-truth
    gts = []
    for ann in gt_by_image.get(img_id, []):
        x, y, w, h = ann["bbox"]
        gts.append([x, y, x + w, y + h])

    gts = np.array(gts, dtype=float)

    # Evaluate matches
    if len(gts) == 0:
        tp = 0
        fp = len(pred_boxes)
        fn = 0
    else:
        used = np.zeros(len(gts), dtype=bool)
        tp = 0
        fp = 0

        for p in pred_boxes:
            ious = np.array([iou_area(p, g) for g in gts])
            best_idx = np.argmax(ious)
            best_iou = ious[best_idx]

            if best_iou >= IOU_THR and not used[best_idx]:
                tp += 1
                used[best_idx] = True
            else:
                fp += 1

        fn = np.sum(~used)

    tp_total += tp
    fp_total += fp
    fn_total += fn

    if (idx + 1) % 50 == 0:
        print(f"[{idx+1}/{len(img_ids)}]  TP:{tp_total}  FP:{fp_total}  FN:{fn_total}")


# Final Metrics


precision = tp_total / (tp_total + fp_total + 1e-6)
recall = tp_total / (tp_total + fn_total + 1e-6)
f1 = 2 * precision * recall / (precision + recall + 1e-6)

results = {
    "iou_thr": IOU_THR,
    "conf_thr": CONF_THR,
    "tp": int(tp_total),
    "fp": int(fp_total),
    "fn": int(fn_total),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
}

with open(SAVE_JSON, "w") as f:
    json.dump(results, f, indent=2)

print("\nFinal Un-Finetuned Metrics")
print(json.dumps(results, indent=2))
