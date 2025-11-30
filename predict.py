import torch
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image
import cv2
import numpy as np
import os

# --- CONFIGURATION ---
MODEL_PATH = "./fine_tuned_clipseg_epoch_10" 
TEST_IMAGE = r"D:\srisurya\Ml_projects\10x\segmentation-text-conditioned\processed_dataset\images\segment_taping_area_IMG_20220627_111721-jpg_1500x2000_jpg.rf.fb07305cb3b59a0eb7ddd73c169ddd53.jpg"
PROMPT = "segment taping area"

def predict():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load Model
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained(MODEL_PATH).to(device)

    # Load Image
    original_image = Image.open(TEST_IMAGE).convert("RGB")
    
    # Process
    inputs = processor(
        text=[PROMPT], 
        images=[original_image], 
        padding="max_length", 
        return_tensors="pt"
    ).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        # Output shape is usually (1, 352, 352)
        preds = outputs.logits
    
    prob_map = torch.sigmoid(preds[0]).cpu().numpy()

    w, h = original_image.size
    prob_map = cv2.resize(prob_map, (w, h)) 
    
    # Threshold
    binary_mask = (prob_map > 0.5).astype(np.uint8) * 255

    # NOTE: You might need to adjust this split depending on your exact filename structure
    base_name = os.path.basename(TEST_IMAGE).split('.')[0]
    clean_prompt = PROMPT.replace(" ", "_")
    save_name = f"{base_name}__{clean_prompt}.png"
    
    cv2.imwrite(save_name, binary_mask)
    print(f"Saved submission file: {save_name}")

if __name__ == "__main__":
    predict()