import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import supervision as sv
from rfdetr import RFDETRMedium
from PIL import Image


print("CUDA available:", torch.cuda.is_available())


model = RFDETRMedium(
    pretrain_weights="/home1/chenyinu/final_output/checkpoint_best_total.pth"
)

model.optimize_for_inference()

# Load dataset
ds = sv.DetectionDataset.from_coco(
    images_directory_path="/scratch1/chenyinu/rf100_merged_final/test",
    annotations_path="/scratch1/chenyinu/rf100_merged_final/test/_annotations.coco.json",
)

# Inference on first image
path, image, annotations = ds[0]
image = Image.open(path).convert("RGB")

detections = model.predict(image, threshold=0.5)

# Visualization 
annotation_image = image.copy()
detection_image = image.copy()

bbox_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotations_labels = [
    f"{ds.classes[class_id]}" for class_id in annotations.class_id
]

detections_labels = [
    f"{ds.classes[class_id]} {confidence:.2f}"
    for class_id, confidence in zip(detections.class_id, detections.confidence)
]

annotation_image = bbox_annotator.annotate(annotation_image, annotations)
annotation_image = label_annotator.annotate(annotation_image, annotations, annotations_labels)

detection_image = bbox_annotator.annotate(detection_image, detections)
detection_image = label_annotator.annotate(detection_image, detections, detections_labels)

fig=sv.plot_images_grid(
    images=[annotation_image, detection_image],
    grid_size=(1, 2),
    titles=["Annotation", "Detection"]
)
plt.savefig("/home1/chenyinu/output_visualization.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved output_visualization.png")

