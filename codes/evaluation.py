import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import supervision as sv
from tqdm import tqdm
from supervision.metrics import MeanAveragePrecision
from rfdetr import RFDETRMedium
from PIL import Image


model = RFDETRMedium(pretrain_weights="/home1/chenyinu/final_output/checkpoint_best_total.pth")
model.optimize_for_inference()

ds = sv.DetectionDataset.from_coco(
    images_directory_path="/scratch1/chenyinu/rf100_merged_final/test",
    annotations_path=f"/scratch1/chenyinu/rf100_merged_final/test/_annotations.coco.json",
)

targets = []
predictions = []

for path, image, annotations in tqdm(ds):
    image = Image.open(path)
    detections = model.predict(image, threshold=0)

    targets.append(annotations)
    predictions.append(detections)

map_metric = MeanAveragePrecision()
map_result = map_metric.update(predictions, targets).compute()
print(map_result)

