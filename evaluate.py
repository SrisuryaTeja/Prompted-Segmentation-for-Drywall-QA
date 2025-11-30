import torch
import os
import numpy as np
import cv2
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from tqdm import tqdm

# --- CONFIGURATION ---
MODEL_PATH = "./fine_tuned_clipseg_epoch_10" 
VALID_DIR = "./processed_dataset_valid/images"
MASK_DIR = "./processed_dataset_valid/masks"

def calculate_metrics(pred_mask, gt_mask):
    # Flatten arrays
    pred = pred_mask.flatten()
    gt = gt_mask.flatten()
    
    # Intersection
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    
    # IoU
    iou = 1.0 if union == 0 else intersection / union

    # Dice
    dice_denom = pred.sum() + gt.sum()
    dice = 1.0 if dice_denom == 0 else (2. * intersection) / dice_denom
        
    return iou, dice

def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluator using device: {device}")

    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained(MODEL_PATH).to(device)
    model.eval()

    files = [f for f in os.listdir(VALID_DIR) if f.endswith(('.jpg', '.png'))]
    
    metrics = {
        "crack": {"iou": [], "dice": []},
        "drywall": {"iou": [], "dice": []}
    }

    print(f"Evaluating {len(files)} images...")

    for filename in tqdm(files):
        # 1. Determine prompt
        if "segment_crack" in filename:
            prompt = "segment crack"
            key = "crack"
        elif "segment_taping_area" in filename:
            prompt = "segment taping area"
            key = "drywall"
        else:
            continue

        # 2. Load Data
        img_path = os.path.join(VALID_DIR, filename)
        mask_path = os.path.join(MASK_DIR, filename)
        
        image = Image.open(img_path).convert("RGB")
        gt_mask = Image.open(mask_path).convert("L")
        
    
        gt_binary = np.array(gt_mask) > 128 

        # 3. Inference
        inputs = processor(
            text=[prompt], 
            images=[image], 
            padding="max_length", 
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            preds_352 = torch.sigmoid(outputs.logits[0]).cpu().numpy()
    
        # Get original dimensions from the loaded PIL image
        w, h = image.size 
        
        # Resize the 352x352 prediction to (W, H)
        preds_original = cv2.resize(preds_352, (w, h))
        
        # Threshold
        pred_mask = (preds_original > 0.5)

        # 4. Calculate Metrics (Now comparing at full resolution)
        iou, dice = calculate_metrics(pred_mask, gt_binary)
        metrics[key]["iou"].append(iou)
        metrics[key]["dice"].append(dice)

    print("\n--- FINAL GRADE REPORT ---")
    all_iou = []
    all_dice = []
    
    for task in ["crack", "drywall"]:
        if len(metrics[task]["iou"]) > 0:
            mIoU = np.mean(metrics[task]["iou"])
            mDice = np.mean(metrics[task]["dice"])
            print(f"[{task.upper()}] mIoU: {mIoU:.4f} | Dice: {mDice:.4f}")
            all_iou.extend(metrics[task]["iou"])
            all_dice.extend(metrics[task]["dice"])
        else:
            print(f"[{task.upper()}] No samples found.")
    
    if all_iou:
        print(f"\n[OVERALL] mIoU: {np.mean(all_iou):.4f} | Dice: {np.mean(all_dice):.4f}")

if __name__ == "__main__":
    evaluate()