# Mask R-CNN Nanoparticle (Dot) Detection

This repository contains a Mask R-CNN–based pipeline for detecting and segmenting small “dot” objects (e.g., nanoparticles) in SEM images. The approach builds on PyTorch’s built-in **Mask R-CNN** implementation (`torchvision.models.detection.maskrcnn_resnet50_fpn_v2`) and showcases:

- **Data loading & augmentation** via custom transforms
- **Training** with a split of 80% training data and 20% validation data
- **Inference** and metric evaluation (mAP) on the validation set

---

## Achieved Metrics

Below are the AP metrics from the best current model:

Per-Threshold AP values: AP_0.50: 0.9985 AP_0.55: 0.9985 AP_0.60: 0.9985 AP_0.65: 0.9695 AP_0.70: 0.8500 AP_0.75: 0.6251 AP_0.80: 0.3572 AP_0.85: 0.1250 AP_0.90: 0.0052 AP_0.95: 0.0003

Summary Metrics: mAP@[0.5:0.95]: 0.5928

---

## Repository Structure

- **`train.py`**  
  - Script to train Mask R-CNN on custom “dot” dataset.
  - Splits data into 80% training, 20% validation.
  - Saves the best model checkpoint to `checkpoints/<timestamp>/best_model.pth`.

- **`predict.py`**  
  - Script to evaluate or predict on the validation set using a trained checkpoint or default pretrained weights. 
  - Prints per-threshold AP (IoU from 0.50 to 0.95).
  - Optionally saves annotated predictions and a CSV of detailed results.

- **`data/`**  
  - Contains helper modules: `data_analysis.py`, `dataset.py` for reading annotations, images, and building PyTorch datasets.

- **`checkpoints/`**  
  - Folder where training checkpoints are automatically stored during training.

---

## How to Train

1. **Prepare Your Dataset**  
   - Place `.jpg` images and corresponding `.json` annotation files in a single directory under `Dataset/`.  
   - Each JSON file should define polygon “dot” regions corresponding to the same-named image.

2. **Install Dependencies**  

Ensure PyTorch, Torchvision, tqdm, and other required libraries are installed.

Run train.py

python train.py \
    --data-path /path/to/dataset \
    --batch-size 2 \
    --epochs 30 \
    --learning-rate 5e-4 \
    --train-size 3072

Run predict.py

python predict.py \
  --data-path /path/to/dataset \
  --checkpoint-path checkpoints/<timestamp>/best_model.pth \
  --output-dir results/ \
  --evaluate-all
--data-path: same folder containing images + JSONs.

--checkpoint-path: the .pth file saved after training.

--output-dir: where results are saved.

--evaluate-all: do a full mAP evaluation on the 20% validation split.
