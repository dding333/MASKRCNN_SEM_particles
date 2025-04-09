# test_augmentation.py (Jupyter Notebook Exact Replica)
import torch
import torchvision
from PIL import Image
from functools import partial
from torchvision.tv_tensors import Mask, BoundingBoxes
from torchvision.transforms.v2 import functional as TF
import torchvision.transforms.v2 as transforms
from pathlib import Path
import matplotlib.pyplot as plt
from data.data_analysis import process_annotations, analyze_classes, setup_visualization, create_polygon_mask
from cjm_torchvision_tfms.core import CustomRandomIoUCrop, ResizeMax, PadSquare

def test_augmentations():
    # Suppress matplotlib debug output
    plt.set_loglevel("WARNING")
    
    # Initialize configuration (identical to notebook)
    train_sz = 4096
    # dataset_path = Path("Datasets/pytorch-for-information-extraction/code/datasets/detection/student-id")
    dataset_path = Path("/afs/crc.nd.edu/user/d/dding3/mask_rcnn_torch/3/Datasets/pytorch-for-information-extraction/code/datasets/detection/dots/jsons/output_json5/")
    
    # Load data and annotations (same as notebook)
    img_dict = {f.stem: f for f in dataset_path.glob("*.jpg")}
    annotation_paths = list(dataset_path.glob("*.json"))
    annotation_df = process_annotations(annotation_paths, img_dict)
    class_names = analyze_classes(annotation_df)
    print("class_names")
    print(class_names)
    int_colors, font_path = setup_visualization(class_names)

    # Select sample image (index 56 as in notebook)
    file_id = list(img_dict.keys())[56]
    sample_img = Image.open(img_dict[file_id]).convert("RGB")

    # Generate masks and boxes (identical to notebook cells)
    annotation = annotation_df.loc[file_id]
    labels = [shape['label'] for shape in annotation['shapes']]
    shape_points = [shape['points'] for shape in annotation['shapes']]
    xy_coords = [[tuple(p) for p in points] for points in shape_points]
    
    mask_imgs = [create_polygon_mask(sample_img.size, xy) for xy in xy_coords]
    masks = torch.cat([Mask(transforms.PILToTensor()(mask_img), dtype=torch.bool) for mask_img in mask_imgs])
    bboxes = BoundingBoxes(torchvision.ops.masks_to_boxes(masks), format='xyxy', canvas_size=sample_img.size[::-1])

    # Initialize transforms (same parameters as notebook)
    iou_crop = CustomRandomIoUCrop(
        min_scale=0.3, 
        max_scale=1.0, 
        min_aspect_ratio=0.5, 
        max_aspect_ratio=2.0, 
        sampler_options=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        trials=400, 
        jitter_factor=0.25
    )
    resize_max = ResizeMax(max_sz=train_sz)
    pad_square = PadSquare(shift=True, fill=0)
    final_resize = transforms.Resize([train_sz]*2, antialias=True)

    # Apply transforms step-by-step as in notebook
    targets = {
        'masks': Mask(masks),
        'boxes': bboxes,
        'labels': torch.tensor([class_names.index(label) for label in labels], dtype=torch.int64)
    }

    # 1. Crop
    cropped_img, cropped_targets = iou_crop(sample_img, targets)
    
    # 2. Resize
    resized_img, resized_targets = resize_max(cropped_img, cropped_targets)
    
    # 3. Pad
    padded_img, padded_targets = pad_square(resized_img, resized_targets)
    
    # 4. Final resize
    final_img, final_targets = final_resize(padded_img, padded_targets)
    
    # 5. Sanitize
    sanitized_img, sanitized_targets = transforms.SanitizeBoundingBoxes()(final_img, final_targets)

    # Visualize results matching notebook output
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image
    axs[0,0].imshow(sample_img)
    axs[0,0].set_title(f"Original\nSize: {sample_img.size}")
    
    # Intermediate steps
    steps = [
        (cropped_img, cropped_targets, "Cropped"),
        (resized_img, resized_targets, "Resized"),
        (padded_img, padded_targets, "Padded"),
        (final_img, final_targets, "Final Size"),
        (sanitized_img, sanitized_targets, "Sanitized")
    ]
    
    for idx, (img, tgt, name) in enumerate(steps, start=1):
        ax = axs[idx//3, idx%3]
        img_tensor = TF.to_image(img) if isinstance(img, Image.Image) else img
        # Generate colors based on each mask's class label
        colors_seg = [int_colors[label] for label in tgt['labels'].tolist()]
        colors_boxes = colors_seg  # Same colors for bounding boxes
        
        # Draw segmentation masks with class-specific colors
        annotated = torchvision.utils.draw_segmentation_masks(
            img_tensor, tgt['masks'], alpha=0.3, colors=colors_seg
        )
        # Draw bounding boxes with corresponding colors
        annotated = torchvision.utils.draw_bounding_boxes(
            annotated, tgt['boxes'], 
            labels=[class_names[i] for i in tgt['labels'].tolist()],
            colors=colors_boxes,
            width=2,
            font=font_path,
            font_size=25
        )
        # ACTUALLY DISPLAY THE ANNOTATED IMAGE
        ax.imshow(TF.to_pil_image(annotated))
        ax.set_title(f"{name}\nSize: {img.size if isinstance(img, Image.Image) else img.shape[-2:]}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("augmentation_steps.png", bbox_inches='tight')
    print("Augmentation test complete! Saved: augmentation_steps.png")

if __name__ == "__main__":
    test_augmentations()