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
```
## Getting Started
1. Installation    

Clone the repository and install the required dependencies.
```Bash

git clone [https://github.com/SrisuryaTeja/Prompted-Segmentation-for-Drywall-QA.git](https://github.com/SrisuryaTeja/Prompted-Segmentation-for-Drywall-QA.git)
cd Prompted-Segmentation-for-Drywall-QA

pip install -r requirements.txt
```



2. Data Setup
   
Since raw data is not uploaded to GitHub, you need to place your Roboflow exports in the dataset_root folder.

- Create a folder named dataset_root.

- Inside it, create folders for cracks and drywall.

- Place your COCO JSON exports inside the train and valid subfolders.

- Your path should look like this: ./dataset_root/cracks/train/_annotations.coco.json

## Usage Pipeline (Run in Order)
Follow these steps to reproduce the training and evaluation results.

Step 1: Preprocessing   
- Convert the raw COCO JSON annotations (Polygons/Boxes) into unified binary masks for the model.

```Bash

python preprocess_masks.py
```
- Output: Creates a processed_dataset/ folder containing images and corresponding black-and-white masks.

  

Step 2: Training (Fine-Tuning)   
- Start the training loop. This script uses Mixed Precision (AMP) to fit on Laptop GPUs.

```Bash

python train.py
```
Configuration: Default is BATCH_SIZE=4 and EPOCHS=10.

Output: Saves model checkpoints to ./fine_tuned_clipseg_epoch_X and the final model to ./fine_tuned_clipseg_final.



Step 3: Evaluation    
- Calculate mIoU and Dice scores on the validation set to verify performance.

```Bash

python evaluate.py
```
Output: Prints the "Final Grade Report" with metrics for both Cracks and Drywall.

Step 4: Inference    
- Run the model on a single test image to visualize the result.

Open predict.py and update the TEST_IMAGE path.

Run the script:

```Bash

python predict.py
```
Output: Generates prediction_mask.png.
