import json
import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
import shutil
import sys

# --- CONFIGURATION ---
CRACK_DATA_DIR = r"C:\Users\srisu\Downloads\cracks\valid"
DRYWALL_DATA_DIR = r"C:\Users\srisu\Downloads\Drywall-Join-Detect\valid"

OUTPUT_DIR = "./processed_dataset_valid"
IMG_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "images")
MASK_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "masks")

def create_masks_from_coco(json_path, source_img_dir, target_img_dir, target_mask_dir, prompt, mode="polygon"):
    """
    Parses COCO JSON and creates binary masks with Robust Error Handling.
    """
    # 1. ERROR HANDLING: JSON Loading
    try:
        with open(json_path, 'r') as f:
            coco = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"\n[CRITICAL ERROR] Could not load JSON at {json_path}")
        print(f"Details: {e}")
        return

    # Basic check for required keys
    if 'images' not in coco or 'annotations' not in coco:
        print(f"\n[CRITICAL ERROR] JSON format invalid. Missing 'images' or 'annotations' keys.")
        return

    # Create ID to Filename map
    images = {img['id']: img for img in coco['images']}
    
    # Group annotations by image_id
    annotations = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations:
            annotations[img_id] = []
        annotations[img_id].append(ann)

    print(f"Processing {prompt} dataset...")
    
    success_count = 0
    error_count = 0
    
    for img_id, img_data in tqdm(images.items()):
        try:
            filename = img_data['file_name']
            
            # 2. ERROR HANDLING: Source Image Check
            src_path = os.path.join(source_img_dir, filename)
            if not os.path.exists(src_path):
                basename = os.path.basename(filename)
                src_path_alt = os.path.join(source_img_dir, basename)
                if os.path.exists(src_path_alt):
                    src_path = src_path_alt
                else:
                    error_count += 1
                    continue 

            h, w = img_data['height'], img_data['width']
            mask = np.zeros((h, w), dtype=np.uint8)

            # Draw annotations
            if img_id in annotations:
                for ann in annotations[img_id]:
                    if mode == "polygon":
                        if 'segmentation' in ann and ann['segmentation']:
                            for seg in ann['segmentation']:
                                try:
                                    poly = np.array(seg).reshape((-1, 2)).astype(np.int32)
                                    cv2.fillPoly(mask, [poly], 255)
                                except ValueError:
                                    print(f"[Warning] Skipped malformed polygon in {filename}")
                                    continue
                    
                    elif mode == "bbox":
                        try:
                            x, y, wb, hb = map(int, ann['bbox'])
                            cv2.rectangle(mask, (x, y), (x + wb, y + hb), 255, -1)
                        except (ValueError, TypeError):
                            print(f"[Warning] Skipped malformed bbox in {filename}")
                            continue

            # Save Files
            prefix = prompt.replace(" ", "_")
            new_filename = f"{prefix}_{filename}"
            new_filename = os.path.basename(new_filename) 
            
            shutil.copy(src_path, os.path.join(target_img_dir, new_filename))
            cv2.imwrite(os.path.join(target_mask_dir, new_filename), mask)
            
            success_count += 1

        except Exception as e:
            print(f"\n[Error] Failed processing file {filename}: {e}")
            error_count += 1
            continue

    print(f"Finished {prompt}. Success: {success_count}, Errors/Skipped: {error_count}")

def main():
    # 1. Setup Directories
    Path(IMG_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(MASK_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # 2. Process Cracks
    crack_json = os.path.join(CRACK_DATA_DIR, "_annotations.coco.json")
    if os.path.exists(crack_json):
        create_masks_from_coco(
            json_path=crack_json,
            source_img_dir=CRACK_DATA_DIR,
            target_img_dir=IMG_OUTPUT_DIR,
            target_mask_dir=MASK_OUTPUT_DIR,
            prompt="segment crack",
            mode="polygon"
        )
    else:
        print(f"Warning: Crack JSON not found at {crack_json}")

    # 3. Process Drywall
    drywall_json = os.path.join(DRYWALL_DATA_DIR, "_annotations.coco.json")
    if os.path.exists(drywall_json):
        create_masks_from_coco(
            json_path=drywall_json,
            source_img_dir=DRYWALL_DATA_DIR,
            target_img_dir=IMG_OUTPUT_DIR,
            target_mask_dir=MASK_OUTPUT_DIR,
            prompt="segment taping area",
            mode="bbox"
        )
    else:
        print(f"Warning: Drywall JSON not found at {drywall_json}")

    print("\n--- Preprocessing Summary ---")
    print(f"Check {OUTPUT_DIR} for results.")

if __name__ == "__main__":
    main()