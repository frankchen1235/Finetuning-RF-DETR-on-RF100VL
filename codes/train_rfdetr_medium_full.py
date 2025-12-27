
import os
from rfdetr import RFDETRMedium
import torch

# FORCE SINGLE GPU MODE
# HARD DISABLE DISTRIBUTED

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["LOCAL_RANK"] = "0"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"

dataset_dir = "/scratch1/chenyinu/rf100_merged_final"  
output_dir  = "/home1/chenyinu/final_output" 




# Model: RF-DETR Medium (best modern model)

model = RFDETRMedium()


# TRAIN

model.train(
    dataset_dir=dataset_dir,
    epochs=10,                 
    batch_size=2,              # RF-DETR Medium is large, 2 fits on A100
    grad_accum_steps=8,        # simulate batch 8
    lr=5e-5,                   # excellent LR for finetuning on large dataset
    num_workers=2,
    amp=True,
    output_dir=output_dir,
    eval_every=1,              
    save_best=True            
)

print("Training completed. Output saved to:", output_dir)
