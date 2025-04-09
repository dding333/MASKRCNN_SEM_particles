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
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import box_iou


def calculate_metrics_per_threshold(predictions, targets):
    iou_thresholds = np.linspace(0.5, 0.95, 10)
    detailed_results = {}

    # For each IoU threshold, compute AP individually
    for iou in iou_thresholds:
        metric = MeanAveragePrecision(iou_thresholds=[iou], class_metrics=True)
        metric.update(predictions, targets)
        partial_res = metric.compute()
        # 'map' here is the AP at this single IoU threshold
        detailed_results[f"AP_{iou:.2f}"] = float(partial_res["map"])

    # Also compute standard "COCO style" average from 0.5 to 0.95
    coco_metric = MeanAveragePrecision(iou_thresholds=iou_thresholds.tolist(), class_metrics=True)
    coco_metric.update(predictions, targets)
    coco_res = coco_metric.compute()

    # Add entries for the old style
    detailed_results["map"] = float(coco_res["map"])       # identical to AP_0.5:0.95
    detailed_results["AP_0.50"] = float(coco_res["map_50"])
    detailed_results["AP_0.75"] = float(coco_res["map_75"])
    detailed_results["AP_0.5:0.95"] = float(coco_res["map"])

    return detailed_results

def predict_and_visualize(model, test_img, file_id, annotation_df, class_names, int_colors, font_path, device, threshold=0.5):
    # Preprocess image
    input_img = resize_img(test_img, target_sz=4096, divisor=1)
    min_img_scale = min(test_img.size) / min(input_img.size)

    # Convert to tensor
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True)
    ])
    input_tensor = transform(input_img).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        model_output = model(input_tensor)

    # Move outputs to CPU and filter predictions
    model_output = move_data_to_device(model_output, 'cpu')
    scores_mask = model_output[0]['scores'] > threshold

    # Scale bounding boxes and masks
    pred_bboxes = BoundingBoxes(model_output[0]['boxes'][scores_mask] * min_img_scale,
                              format='xyxy', canvas_size=test_img.size[::-1])

    # Convert model's class IDs to indices for class names
    pred_labels = [class_names[int(label)] for label in model_output[0]['labels'][scores_mask]]
    if len(pred_labels) == 0:
        print(f"No predictions above threshold {threshold} for image {file_id}")
    pred_scores = model_output[0]['scores'][scores_mask]

    # Process masks
    pred_masks = torch.nn.functional.interpolate(
        model_output[0]['masks'][scores_mask],
        size=test_img.size[::-1],
        mode='bilinear',
        align_corners=False
    )
    if pred_masks.shape[0] > 0:
        pred_masks = torch.cat([
            Mask(torch.where(mask >= threshold, 1, 0), dtype=torch.bool)
            for mask in pred_masks
        ])
    else:
        pred_masks = torch.zeros((0, 1, *test_img.size[::-1]), dtype=torch.bool)

    # Get ground truth data
    target_shapes = annotation_df.loc[file_id]['shapes']
    target_xy_coords = [[tuple(p) for p in shape['points']] for shape in target_shapes]
    target_mask_imgs = [create_polygon_mask(test_img.size, xy) for xy in target_xy_coords]

    # Convert mask images to tensors
    target_masks = Mask(torch.cat([
        Mask(transforms.PILToTensor()(mask_img).to(torch.bool)) for mask_img in target_mask_imgs
    ]))
    target_labels = [shape['label'] for shape in target_shapes]
    target_bboxes = BoundingBoxes(
        data=torchvision.ops.masks_to_boxes(target_masks),
        format='xyxy',
        canvas_size=test_img.size[::-1]
    )

    # Create visualizations
    img_tensor = transforms.PILToTensor()(test_img).to(torch.uint8)

    # Target visualization
    target_colors = [int_colors[class_names.index(label)] for label in target_labels]

    # Corrected mask generation
    mask_tensors = [transforms.PILToTensor()(mask_img).squeeze(0).to(torch.bool) for mask_img in target_mask_imgs]
    target_masks = Mask(torch.stack(mask_tensors))

    annotated_target = torchvision.utils.draw_segmentation_masks(
        img_tensor, target_masks, alpha=0.3, colors=target_colors
    )

    # Prediction visualization
    pred_colors = [int_colors[class_names.index(label)] for label in pred_labels]
    if pred_masks.shape[0] > 0:
        annotated_pred = torchvision.utils.draw_segmentation_masks(
            img_tensor, pred_masks, alpha=0.3, colors=pred_colors
        )
    else:
        annotated_pred = img_tensor.clone()

    annotated_pred = torchvision.utils.draw_bounding_boxes(
        annotated_pred,
        pred_bboxes,
        labels=[f"{label}\n{prob*100:.2f}%" for label, prob in zip(pred_labels, pred_scores)],
        colors=pred_colors,
        font=font_path,
        font_size=20
    )

    # Combine images
    combined_img = stack_imgs([tensor_to_pil(annotated_target), tensor_to_pil(annotated_pred)])

    # Create results dataframe
    results_df = pd.Series({
        "Target BBoxes:": [
            f"{label}:{bbox}" for label, bbox in zip(target_labels, np.round(target_bboxes.numpy(), 3))
        ],
        "Predicted BBoxes:": [
            f"{label}:{bbox}" for label, bbox in zip(pred_labels, np.round(pred_bboxes.numpy(), 3))
        ],
        "Confidence Scores:": [
            f"{label}: {prob*100:.2f}%" for label, prob in zip(pred_labels, pred_scores)
        ]
    }).to_frame()

    return combined_img, results_df


def evaluate_model(model, dataset, device, num_images=None, threshold=0.5):
    """
    Evaluate model on entire dataset or subset
    """
    model.eval()
    targets = []
    predictions = []

    indices = list(range(len(dataset)))
    if num_images and num_images < len(dataset):
        indices = random.sample(indices, num_images)

    for idx in tqdm(indices, desc="Evaluating"):
        image, target = dataset[idx]
        with torch.no_grad():
            output = model([image.to(device)])[0]

        # Remove threshold filtering for mAP calculation
        pred_boxes = output['boxes'].cpu()
        pred_scores = output['scores'].cpu()
        pred_labels = output['labels'].cpu()

        # Store predictions
        predictions.append({
            'boxes': pred_boxes,
            'scores': pred_scores,
            'labels': pred_labels
        })

        # Store targets
        targets.append({
            'boxes': target['boxes'],
            'labels': target['labels']
        })

    return calculate_metrics_per_threshold(predictions, targets)


def get_transforms(train_sz):
    # Simplified augmentation
    data_aug_tfms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomPhotometricDistort(p=0.2),
        transforms.RandomZoomOut(p=0.3)
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


def load_model(use_pretrained, checkpoint_path, num_classes, device):
    if use_pretrained:
        print("Warning: Pretrained COCO model may not align with your custom classes")
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

        model.eval()
        model.to(device=device, dtype=torch.float32)
        model.device = device
        model.name = 'maskrcnn_resnet50_fpn_v2'
        return model
    #else:
        #print("Loading fine-tuned model with custom architecture")
        #checkpoint = torch.load(checkpoint_path, map_location=device)

        #model = maskrcnn_resnet50_fpn_v2(weights=None)
        #model.rpn.anchor_generator.sizes = ((16, 32, 64),)
        #model.roi_heads.detections_per_img = 200

        # Modify box predictor
        #in_features_box = model.roi_heads.box_predictor.cls_score.in_features
        #model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

        # Modify mask predictor
        #in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        #model.roi_heads.mask_predictor = MaskRCNNPredictor(
            #in_features_mask,
            #model.roi_heads.mask_predictor.conv5_mask.out_channels,
            #num_classes
        #)

        #model.load_state_dict(checkpoint['model_state_dict'])
        #model.to(device)
        #model.eval()
        #return model
    else:
        print("Loading fine-tuned model with default architecture (matching training)")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Use weights=None here if you're loading your own fine-tuned checkpoint
        # (because the checkpoint itself contains all the trained parameters).
        model = maskrcnn_resnet50_fpn_v2(weights=None)

        # DO NOT override anchor sizes or detections_per_img here, so it stays at default:
        # model.rpn.anchor_generator.sizes = ...
        # model.roi_heads.detections_per_img = ...
        # (Remove these lines entirely)

        # Modify box predictor to match training setup
        in_features_box = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

        # Modify mask predictor to match training setup
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        dim_reduced = model.roi_heads.mask_predictor.conv5_mask.out_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, dim_reduced, num_classes)

        # Load your fine-tuned weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model



def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(args.data_path)
    img_dict = {f.stem: f for f in dataset_path.glob("*.jpg")}
    annotation_df = process_annotations(list(dataset_path.glob("*.json")), img_dict)
    class_names = analyze_classes(annotation_df)
    print(f"Number of classes: {len(class_names)}")
    print(f"Class names: {class_names}")
    int_colors, font_path = setup_visualization(class_names)

    model = load_model(args.use_pretrained, args.checkpoint_path, len(class_names), device)

    all_keys = list(img_dict.keys())
    split_idx = int(0.8 * len(all_keys))
    train_keys = all_keys[:split_idx]
    val_keys = all_keys[split_idx:]

    test_dataset = StudentIDDataset(
        val_keys,
        annotation_df,
        img_dict,
        {c: i for i, c in enumerate(class_names)},
        get_transforms(4096)['valid']
    )

    if args.evaluate_all:
        # Full evaluation
        metrics = evaluate_model(model, test_dataset, device, args.num_images, args.threshold)

        # Save to CSV
        metrics_df = pd.DataFrame({k: [v] for k, v in metrics.items()})
        metrics_df.to_csv(output_dir/"detailed_metrics.csv", index=False)

        # Print all per-threshold APs
        print("\nPer-Threshold AP values:")
        iou_thresholds = np.linspace(0.5, 0.95, 10)
        for iou in iou_thresholds:
            key = f"AP_{iou:.2f}"
            print(f"  {key}: {metrics[key]:.4f}")

        # Print summary metrics
        print("\nSummary Metrics:")
        print(f"mAP@[0.5:0.95]: {metrics['map']:.4f}")
        print(f"mAP@0.50: {metrics['AP_0.50']:.4f}")
        print(f"mAP@0.75: {metrics['AP_0.75']:.4f}")

    else:
        # Single image mode
        selected_keys = random.sample(val_keys, min(args.num_images, len(val_keys)))
        for file_id in selected_keys:
            test_img = Image.open(img_dict[file_id]).convert('RGB')
            result_img, results_df = predict_and_visualize(
                model, test_img, file_id, annotation_df,
                class_names, int_colors, font_path, device, args.threshold
            )
            result_img.save(output_dir/f"prediction_{file_id}.jpg")
            results_df.to_csv(output_dir/f"results_{file_id}.csv")

        print(f"Saved {len(selected_keys)} results to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mask R-CNN Detection and Evaluation')
    parser.add_argument('--data-path', required=True, help='Dataset directory path')
    parser.add_argument('--checkpoint-path', help='Path to fine-tuned model checkpoint')
    parser.add_argument('--use-pretrained', action='store_true', help='Use pretrained COCO weights')
    parser.add_argument('--output-dir', default='output', help='Output directory')
    parser.add_argument('--num-images', type=int, default=100, help='Number of images to process')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--evaluate-all', action='store_true', help='Run full evaluation on test set')

    args = parser.parse_args()

    if not args.use_pretrained and not args.checkpoint_path:
        raise ValueError("Must provide either --checkpoint-path or --use-pretrained")

    main(args)
