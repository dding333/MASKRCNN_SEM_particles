# Import Python Standard Library dependencies
import datetime
from functools import partial
from glob import glob
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
from typing import Any, Dict, Optional
# Add this import with the other Python Standard Library imports at the top
import sys
# Import utility functions
from cjm_psl_utils.core import download_file, file_extract, get_source_code
from cjm_pil_utils.core import resize_img, get_img_files, stack_imgs
from cjm_pytorch_utils.core import pil_to_tensor, tensor_to_pil, get_torch_device, set_seed, denorm_img_tensor, move_data_to_device
from cjm_pandas_utils.core import markdown_to_pandas, convert_to_numeric, convert_to_string
from cjm_torchvision_tfms.core import ResizeMax, PadSquare, CustomRandomIoUCrop

# Import the distinctipy module
from distinctipy import distinctipy

# Import matplotlib for creating plots
import matplotlib.pyplot as plt

# Import numpy
import numpy as np

# Import the pandas package
import pandas as pd
import argparse

# Import PIL for image manipulation
from PIL import Image, ImageDraw

# Import PyTorch dependencies
import torch
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchtnt.utils import get_module_summary
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.tv_tensors import BoundingBoxes, Mask
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import torchvision.transforms.v2  as transforms
from torchvision.transforms.v2 import functional as TF

# Import Mask R-CNN
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# Import tqdm for progress bar
from tqdm.auto import tqdm
from data.data_analysis import process_annotations, analyze_classes, setup_visualization, create_polygon_mask
from data.dataset import StudentIDDataset  # Import your dataset class


def create_model(num_classes, device, dtype=torch.float32):
    # Initialize Mask R-CNN with pretrained weights
    model = maskrcnn_resnet50_fpn_v2(weights='DEFAULT')
    
    # Modify box predictor
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)
    
    # Modify mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    dim_reduced = model.roi_heads.mask_predictor.conv5_mask.out_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, dim_reduced, num_classes
    )
    
    # Add model metadata
    model.to(device=device, dtype=dtype)
    model.device = device
    model.name = 'maskrcnn_resnet50_fpn_v2'
    
    return model

def get_transforms(train_sz):
    # Data augmentation transforms
    iou_crop = CustomRandomIoUCrop(
        min_scale=0.3, max_scale=1.0,
        min_aspect_ratio=0.5, max_aspect_ratio=2.0,
        sampler_options=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        trials=400, jitter_factor=0.25
    )
    
    data_aug_tfms = transforms.Compose([
        iou_crop,
        transforms.ColorJitter(
            brightness=(0.875, 1.125),
            contrast=(0.5, 1.5),
            saturation=(0.5, 1.5),
            hue=(-0.05, 0.05),
        ),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomEqualize(p=0.5),
        transforms.RandomPosterize(bits=3, p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
    ])

    # Resize/pad transforms
    resize_pad_tfm = transforms.Compose([
        ResizeMax(max_sz=train_sz),
        PadSquare(shift=True, fill=0),
        transforms.Resize([train_sz]*2, antialias=True)
    ])

    # Final processing transforms
    final_tfms = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.SanitizeBoundingBoxes(),
    ])

    return {
        'train': transforms.Compose([data_aug_tfms, resize_pad_tfm, final_tfms]),
        'valid': transforms.Compose([resize_pad_tfm, final_tfms])
    }
    
def run_epoch(model, loader, optimizer, scheduler, device, scaler, epoch, is_train=True):
    # Always keep model in train mode to get loss dictionaries
    model.train()
    epoch_loss = 0
    progress = tqdm(loader, desc=f"{'Train' if is_train else 'Valid'} Epoch {epoch}")

    with torch.set_grad_enabled(is_train):
        for batch_idx, (images, targets) in enumerate(progress):
            images = torch.stack(images).to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Different context managers for train/valid
            if is_train:
                context = torch.cuda.amp.autocast(enabled=scaler is not None)
            else:
                context = torch.no_grad()

            with context:
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            if is_train:
                optimizer.zero_grad()
                if scaler:
                    scaler.scale(losses).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    losses.backward()
                    optimizer.step()
                scheduler.step()

            epoch_loss += losses.item()
            progress.set_postfix(loss=losses.item(), avg_loss=epoch_loss/(batch_idx+1))
            
    return epoch_loss / len(loader)

#def train_model(model, train_loader, valid_loader, num_epochs, device):
    #optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR)
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=BASE_LR, total_steps=num_epochs*len(train_loader))
    #scaler = torch.cuda.amp.GradScaler() if 'cuda' in str(device) else None
    #best_loss = float('inf')

    # Create checkpoint directory
    #timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #checkpoint_dir = Path(f"checkpoints/{timestamp}")
    #checkpoint_dir.mkdir(parents=True, exist_ok=True)

    #for epoch in range(num_epochs):
        #train_loss = run_epoch(model, train_loader, optimizer, scheduler, device, scaler, epoch, is_train=True)
        #valid_loss = run_epoch(model, valid_loader, None, None, device, None, epoch, is_train=False)

        # Save best model
        #if valid_loss < best_loss:
            #best_loss = valid_loss
            #torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                #'optimizer_state_dict': optimizer.state_dict(),
                #'scheduler_state_dict': scheduler.state_dict(),
                #'train_loss': train_loss,
                #'valid_loss': valid_loss,}, checkpoint_dir / "best_model.pth")

        #print(f"Epoch {epoch+1}/{num_epochs} | "f"Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")

    #return model
    
def train_model(model, train_loader, valid_loader, num_epochs, device, args):
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=BASE_LR, total_steps=num_epochs*len(train_loader))
    scaler = torch.cuda.amp.GradScaler() if 'cuda' in str(device) else None
    best_loss = float('inf')

    # Create checkpoint directory
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_dir = Path(f"checkpoints/{timestamp}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create training log file
    log_file = checkpoint_dir / "training_log.txt"
    
    # Save command and configuration
    with open(log_file, 'w') as f:
        f.write(f"Training command: {' '.join(['python'] + sys.argv)}\n")
        f.write(f"Training started at: {timestamp}\n")
        f.write("\nConfiguration:\n")
        f.write(f"Data path: {args.data_path}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
        f.write(f"Train size: {args.train_size}\n")
        f.write(f"Device: {device}\n")
        f.write("\nTraining Log:\n")

    for epoch in range(num_epochs):
        train_loss = run_epoch(model, train_loader, optimizer, scheduler, 
                             device, scaler, epoch, is_train=True)
        valid_loss = run_epoch(model, valid_loader, None, None,
                             device, None, epoch, is_train=False)

        # Save best model
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
            }, checkpoint_dir / "best_model.pth")

        # Log epoch results
        epoch_log = (f"Epoch {epoch+1}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}\n")
        print(epoch_log, end='')
        with open(log_file, 'a') as f:
            f.write(epoch_log)

    # Save final log entry
    final_log = f"\nTraining completed. Best validation loss: {best_loss:.4f}\n"
    print(final_log)
    with open(log_file, 'a') as f:
        f.write(final_log)

    return model

def visualize_sample(dataset, int_colors, class_names, font_path, num_samples=2):
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 10))
    for idx in range(num_samples):
        image, target = dataset[np.random.randint(len(dataset))]
        image = (image * 255).byte().cpu()
        
        # Draw masks
        annotated = torchvision.utils.draw_segmentation_masks(
            image, target['masks'], alpha=0.3, 
            colors=[int_colors[label] for label in target['labels']]
        )
        
        # Draw boxes
        annotated = torchvision.utils.draw_bounding_boxes(
            annotated, target['boxes'],
            labels=[class_names[label] for label in target['labels']],
            colors=[int_colors[label] for label in target['labels']],
            font=font_path,
            font_size=20
        )
        
        axes[idx].imshow(annotated.permute(1, 2, 0).numpy())
        axes[idx].axis('off')
    plt.show()

def main(args):
    # Data preparation
    dataset_path = Path(args.data_path)
    img_dict = {f.stem: f for f in dataset_path.glob("*.jpg")}
    annotation_df = process_annotations(list(dataset_path.glob("*.json")), img_dict)
    class_names = analyze_classes(annotation_df)
    int_colors, font_path = setup_visualization(class_names)
    
    # Train/valid split
    keys = list(img_dict.keys())
    train_keys = keys[:int(0.8*len(keys))]
    valid_keys = keys[int(0.8*len(keys)):]
    
    # Create datasets
    transforms = get_transforms(TRAIN_SIZE)
    train_dataset = StudentIDDataset(train_keys, annotation_df, img_dict, 
                                   {c:i for i,c in enumerate(class_names)}, 
                                   transforms['train'])
    valid_dataset = StudentIDDataset(valid_keys, annotation_df, img_dict,
                                   {c:i for i,c in enumerate(class_names)}, 
                                   transforms['valid'])

    # Create data loaders
    loader_params = {
        'batch_size': BATCH_SIZE,
        'num_workers': NUM_WORKERS,
        'collate_fn': lambda x: tuple(zip(*x)),
        'pin_memory': True
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_params)
    valid_loader = DataLoader(valid_dataset, **loader_params)

    # Initialize model
    model = create_model(num_classes=len(class_names), device=DEVICE,dtype=torch.float32)
    model.to(DEVICE)

    # Train model
    trained_model = train_model(model, train_loader, valid_loader, EPOCHS, DEVICE, args)
    
    # Visualize results
    visualize_sample(valid_dataset, int_colors, class_names, font_path)

#if __name__ == "__main__":

    # Configuration
    #TRAIN_SIZE = 3072
    #BATCH_SIZE = 2  
    #BASE_LR = 5e-4
    #EPOCHS = 30
    #NUM_WORKERS = 4
    #DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #parser = argparse.ArgumentParser(description='Train Mask R-CNN for student ID detection')
    #parser.add_argument('--data-path', type=str, required=True, help='Path to dataset directory containing images and annotations')
    #args = parser.parse_args()
    
    #main(args.data_path)
    
if __name__ == "__main__":
    # Configuration
    parser = argparse.ArgumentParser(description='Train Mask R-CNN for student ID detection')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to dataset directory containing images and annotations')
    parser.add_argument('--batch-size', type=int, default=2,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=5e-4,
                       help='Initial learning rate')
    parser.add_argument('--train-size', type=int, default=3072,
                       help='Size to resize training images')
    
    args = parser.parse_args()
    
    # Set global constants from args
    TRAIN_SIZE = args.train_size
    BATCH_SIZE = args.batch_size
    BASE_LR = args.learning_rate
    EPOCHS = args.epochs
    NUM_WORKERS = 4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    main(args)