import os
import json

MERGED_DIR = "/scratch1/chenyinu/rf100_merged_final"
splits = ["train", "valid", "test"]

def fix_split(split):
    json_path = os.path.join(MERGED_DIR, split, "_annotations.coco.json")
    print(f"Fixing: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    categories = data["categories"]
    old_id_to_new = {}

    # Fix categories 
    for new_id, cat in enumerate(categories):
        old_id = cat["id"]
        old_id_to_new[old_id] = new_id
        cat["id"] = new_id  

        # keep supercategory
        if "supercategory" not in cat:
            cat["supercategory"] = "none"

    # Fix annotations 
    for ann in data["annotations"]:
        old_cat = ann["category_id"]
        if old_cat not in old_id_to_new:
            print("ERROR: annotation with invalid cat id:", old_cat)
            continue
        ann["category_id"] = old_id_to_new[old_cat]

    
    with open(json_path, "w") as f:
        json.dump(data, f)

    print(f"Fixed split: {split}")


for split in splits:
    fix_split(split)

print("\n🎉 All splits fixed! Category IDs are now 0-553.")
