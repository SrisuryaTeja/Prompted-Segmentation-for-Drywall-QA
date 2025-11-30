import os
import time
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image, UnidentifiedImageError
import numpy as np
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm 

# --- CONFIGURATION ---
DATA_DIR = "./processed_dataset"
IMG_DIR = os.path.join(DATA_DIR, "images")
MASK_DIR = os.path.join(DATA_DIR, "masks")
MODEL_NAME = "CIDAS/clipseg-rd64-refined"

# HARDWARE SETTINGS
BATCH_SIZE = 4          
EPOCHS = 10
LEARNING_RATE = 2e-5
NUM_WORKERS = 0      


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred_logits, target):
        pred = torch.sigmoid(pred_logits)
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice

class DefectDataset(Dataset):
    def __init__(self, img_dir, mask_dir, processor, balance_weights=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.processor = processor
        
        # PROMPTS
        self.prompts_crack = [
            "segment crack", "segment wall crack",
            "segment surface crack", "segment line crack"
        ]
        self.prompts_drywall = [
            "segment taping area", "segment joint tape", 
            "segment drywall seam", "segment wall joint",
            "segment taping compound"
        ]

        
        try:
            all_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        except Exception as e:
            raise RuntimeError(f"Could not read directory {img_dir}: {e}")

        self.crack_files = [f for f in all_files if "segment_crack" in f]
        self.drywall_files = [f for f in all_files if "segment_taping_area" in f]
        
        print(f"Original Distribution: {len(self.crack_files)} Cracks, {len(self.drywall_files)} Drywall")

        # BALANCING
        if balance_weights and len(self.drywall_files) > 0:
            target_count = len(self.crack_files)
            repeat_factor = target_count // len(self.drywall_files)
            remainder = target_count % len(self.drywall_files)
            
            balanced_drywall = (self.drywall_files * repeat_factor) + self.drywall_files[:remainder]
            self.filenames = self.crack_files + balanced_drywall
            random.shuffle(self.filenames)
            print(f"Balanced Distribution: {len(self.crack_files)} Cracks, {len(balanced_drywall)} Drywall")
        else:
            self.filenames = all_files

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        
        # Random Prompt
        if filename.startswith("segment_crack"): prompt = random.choice(self.prompts_crack)
        elif filename.startswith("segment_taping_area"): prompt = random.choice(self.prompts_drywall)
        else: prompt = "segment defect"

        img_path = os.path.join(self.img_dir, filename)
        mask_path = os.path.join(self.mask_dir, filename)
        
        try:
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")

            inputs = self.processor(
                text=[prompt], 
                images=[image], 
                padding="max_length", 
                truncation=True, 
                return_tensors="pt"
            )
            
            mask = mask.resize((352, 352), resample=Image.NEAREST)
            mask = np.array(mask) / 255.0
            mask = (mask > 0.5).astype(float)
            
            return inputs, torch.tensor(mask).float()
        except Exception as e:
            print(f"[Warning] Skipped {filename}: {e}")
            return self.__getitem__(random.randint(0, len(self.filenames)-1))


def main():
  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        processor = CLIPSegProcessor.from_pretrained(MODEL_NAME)
        model = CLIPSegForImageSegmentation.from_pretrained(MODEL_NAME).to(device)
    except Exception as e:
        raise RuntimeError(f"Model load failed: {e}")

    dataset = DefectDataset(IMG_DIR, MASK_DIR, processor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    print(f"Training on {len(dataset)} samples.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    use_amp = (device == "cuda")
    scaler = GradScaler(enabled=use_amp)

    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = DiceLoss()

    model.train()
    total_start_time = time.time()

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        epoch_loss = 0
        
        loop = tqdm(dataloader, leave=True)
        
        for batch_idx, (inputs, gt_masks) in enumerate(loop):
            try:
                input_ids = inputs["input_ids"].squeeze(1).to(device)
                pixel_values = inputs["pixel_values"].squeeze(1).to(device)
                attention_mask = inputs["attention_mask"].squeeze(1).to(device)
                gt_masks = gt_masks.to(device)

                optimizer.zero_grad()

                with autocast(enabled=use_amp):
                    outputs = model(
                        input_ids=input_ids, 
                        pixel_values=pixel_values, 
                        attention_mask=attention_mask
                    )
                    preds = outputs.logits
                    loss = criterion_bce(preds, gt_masks) + criterion_dice(preds, gt_masks)

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                epoch_loss += loss.item()

                loop.set_description(f"Epoch {epoch+1}")
                loop.set_postfix(loss=loss.item())

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n[FATAL] OOM Error. Please restart with BATCH_SIZE={BATCH_SIZE//2}")
                    break
                else:
                    print(f"\n[Error] Batch failed: {e}")
                    continue

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")

        save_path = f"./fine_tuned_clipseg_epoch_{epoch+1}"
        model.save_pretrained(save_path)
        processor.save_pretrained(save_path)
        print(f"Checkpoint saved to {save_path}")

    print(f"Total Time: {(time.time() - total_start_time)/60:.2f} min")


    model.save_pretrained("./fine_tuned_clipseg_final")
    processor.save_pretrained("./fine_tuned_clipseg_final")

if __name__ == "__main__":
    main()