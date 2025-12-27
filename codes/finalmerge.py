import os, json, shutil
from glob import glob
from tqdm import tqdm

base_dir = "/home1/chenyinu/rf100"         
output_dir = "/scratch1/chenyinu/rf100_merged_full"

splits = ["train", "valid", "test"]


# Create clean output structure
print("Creating fresh merge directory:", output_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for s in splits:
    split_path = os.path.join(output_dir, s)
    os.makedirs(split_path, exist_ok=True)

    # Remove any previous annotation file
    ann_path = os.path.join(split_path, "_annotations.coco.json")
    if os.path.exists(ann_path):
        os.remove(ann_path)

# Containers for incremental IDs
global_img_id = 1
global_ann_id = 1


global_categories = {}


print("🔍 Scanning RF100 datasets...")
all_datasets = sorted([
    d for d in os.listdir(base_dir)
    if os.path.isdir(os.path.join(base_dir, d))
])

for dataset in tqdm(all_datasets):
    ds_path = os.path.join(base_dir, dataset)

  
    if not os.path.isdir(ds_path):
        continue

    for split in splits:
        src_split = os.path.join(ds_path, split)
        ann_path = os.path.join(src_split, "_annotations.coco.json")

        if not os.path.exists(ann_path):
            continue  

        try:
            with open(ann_path, "r") as f:
                data = json.load(f)
        except:
            print(f" Skipping corrupted JSON: {ann_path}")
            continue

        images = data.get("images", [])
        anns = data.get("annotations", [])
        cats = data.get("categories", [])

        
        if len(images) == 0 or len(anns) == 0:
            continue

  
        for c in cats:
            global_categories[c["id"]] = c["name"]

        # Reassign IDs for merged dataset 

    
        old_to_new_imgid = {}

        for im in images:
            new_img_id = global_img_id
            old_to_new_imgid[im["id"]] = new_img_id

            im["id"] = new_img_id
            im["file_name"] = os.path.basename(im["file_name"])

            # copy image into new structure
            src_img = os.path.join(src_split, im["file_name"])
            dst_img = os.path.join(output_dir, split, im["file_name"])

            if os.path.exists(src_img):
                shutil.copy2(src_img, dst_img)

            global_img_id += 1

        # Fix annotation IDs
        for a in anns:
            a["id"] = global_ann_id
            a["image_id"] = old_to_new_imgid.get(a["image_id"], a["image_id"])
            global_ann_id += 1

        # Append to merged JSON 
        out_j = os.path.join(output_dir, split, "_annotations.coco.json")
        if os.path.exists(out_j):
            with open(out_j, "r") as f:
                merged = json.load(f)
        else:
            merged = {
                "info": {},
                "licenses": [],
                "categories": [],
                "images": [],
                "annotations": []
            }

        merged["images"].extend(images)
        merged["annotations"].extend(anns)

        with open(out_j, "w") as f:
            json.dump(merged, f)


# Insert Unified Category List (sorted by name)

print(" Finalizing category list...")

category_items = [{"id": cid, "name": name}
                  for cid, name in sorted(global_categories.items(), key=lambda x: x[1].lower())]

for split in splits:
    jpath = os.path.join(output_dir, split, "_annotations.coco.json")
    with open(jpath, "r") as f:
        d = json.load(f)
    d["categories"] = category_items
    with open(jpath, "w") as f:
        json.dump(d, f)

print("\n MERGE COMPLETE!")
print("Output folder:", output_dir)

