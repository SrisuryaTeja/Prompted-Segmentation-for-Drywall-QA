# Text-Conditioned Defect Segmentation using CLIPSeg

This repository contains a PyTorch implementation for segmenting industrial defects (**Cracks**) and construction details (**Drywall Joints**) using natural language prompts. 

Unlike traditional segmentation models (like U-Net) that require training on fixed classes, this project fine-tunes **CLIPSeg**, a multimodal model that generates segmentation masks based on arbitrary text queries.

## 📊 Performance

The model was fine-tuned on a custom dataset and evaluated on unseen validation data.

| Defect Type | Prompt | Average IoU per task | Dice Score |
| :--- | :--- | :--- | :--- |
| **Cracks** | *"segment crack"* | 0.492 | **0.638** |
| **Drywall** | *"segment taping area"* | 0.634 | **0.763** |
| **Overall** | — | **0.563** | **0.701** |

## 🏗️ Technical Highlights

* **Architecture:** `CIDAS/clipseg-rd64-refined` (Pre-trained on PhraseCut/Visual Genome).
* **Optimization:** Implemented **Mixed Precision (FP16)** and **Gradient Scaling** to enable training on consumer hardware (e.g., Laptops with 4GB-6GB VRAM).
* **Loss Function:** Hybrid **BCE + Dice Loss** to handle severe class imbalance (thin cracks vs. large backgrounds).
* **Robustness:** Includes text augmentation (randomizing prompts) and robust data preprocessing pipelines.

## 📂 Project Structure

```text
├── dataset_root/              # (Not included in repo) Raw Roboflow Data
│   ├── cracks/train/
│   └── drywall/train/
├── processed_dataset/         # Generated Binary Masks for Training
├── fine_tuned_clipseg_final/  # Saved Model Weights
│
├── preprocess_masks.py        # Script 1: Converts JSON Annotations -> Binary Masks
├── train.py                   # Script 2: Main Fine-Tuning Loop
├── predict.py                 # Script 3: Single Image Inference
├── evaluate.py                # Script 4: Validation Metrics (IoU/Dice)
├── requirements.txt           # Python Dependencies
└── README.md