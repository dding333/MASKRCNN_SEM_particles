# visualize_annotations.py
import torch
import torchvision
from PIL import Image, ImageDraw
from torchvision.tv_tensors import Mask, BoundingBoxes
from torchvision.transforms.v2 import functional as TF
import torchvision.transforms.v2  as transforms
from functools import partial
from pathlib import Path
import matplotlib.pyplot as plt

def create_polygon_mask(image_size, vertices):
    """
    Create a grayscale image with a white polygonal area on a black background.
    
    Args:
        image_size (tuple): (width, height) dimensions of the image
        vertices (list): List of (x,y) tuples defining polygon vertices
        
    Returns:
        PIL.Image: Binary mask image
    """
    mask_img = Image.new('L', image_size, 0)
    ImageDraw.Draw(mask_img).polygon(vertices, fill=255)
    return mask_img

def annotate_image(file_id, sample_img, annotation_df, class_names, int_colors, font_path):
    """
    Annotate image with masks and bounding boxes
    
    Args:
        file_id (str): Image filename stem (e.g. "10067")
        sample_img (PIL.Image): Input image
        annotation_df (DataFrame): Annotations dataframe
        class_names (list): List of class names
        int_colors (list): List of RGB color tuples
        font_path (str): Path to font file
        
    Returns:
        PIL.Image: Annotated image
    """
    # Create drawing function with fixed parameters
    draw_bboxes = partial(torchvision.utils.draw_bounding_boxes,
                         fill=False, 
                         width=2,
                         font=font_path,
                         font_size=25)

    # Process annotations - use passed file_id directly
    annotation = annotation_df.loc[file_id]  # Changed line
    
    # Generate masks
    labels = [shape['label'] for shape in annotation['shapes']]
    shape_points = [shape['points'] for shape in annotation['shapes']]
    xy_coords = [[tuple(p) for p in points] for points in shape_points]
    
    mask_imgs = [create_polygon_mask(sample_img.size, xy) for xy in xy_coords]
    masks = torch.cat([Mask(transforms.PILToTensor()(mask_img), dtype=torch.bool) for mask_img in mask_imgs])
    
    # Generate bounding boxes
    bboxes = torchvision.ops.masks_to_boxes(masks)
    
    # Convert image to tensor
    img_tensor = TF.to_image(sample_img)
    
    # Draw annotations
    colors = [int_colors[class_names.index(label)] for label in labels]
    annotated_tensor = torchvision.utils.draw_segmentation_masks(
        img_tensor, masks, alpha=0.3, colors=colors
    )
    
    annotated_tensor = draw_bboxes(
        annotated_tensor, 
        boxes=BoundingBoxes(bboxes, format="xyxy", canvas_size=sample_img.size[::-1]),
        labels=labels,
        colors=colors
    )
    
    return TF.to_pil_image(annotated_tensor)

def main():
    # Load dependencies from previous steps
    from data.data_analysis import process_annotations, setup_visualization
    
    # Path setup
    dataset_path = Path("/afs/crc.nd.edu/user/d/dding3/mask_rcnn_torch/3/Datasets/pytorch-for-information-extraction/code/datasets/detection/dots/jsons/output_json5/")
    img_dict = {f.stem: f for f in dataset_path.glob("*.jpg")}
    annotation_paths = list(dataset_path.glob("*.json"))
    
    # Process data
    annotation_df = process_annotations(annotation_paths, img_dict)
    class_names = ['background', 'dot']
    int_colors, font_path = setup_visualization(class_names)
    
    # Select sample image (using index 56 as in notebook)
    file_id = list(img_dict.keys())[46]
    sample_img = Image.open(img_dict[file_id]).convert("RGB")
    
    # Annotate image (pass file_id explicitly)
    annotated_img = annotate_image(file_id, sample_img, annotation_df, 
                                  class_names, int_colors, font_path)
    
    # Save and show result
    annotated_img.save("annotated_sample.png")
    plt.imshow(annotated_img)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()