# data/data_analysis.py
import json
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from distinctipy import distinctipy
from pathlib import Path
import requests
from PIL import Image, ImageDraw

def process_annotations(annotation_paths, img_dict):
    """Process JSON annotations into DataFrame"""
    print("\nProcessing annotations...")
    cls_dataframes = (pd.read_json(f, orient='index').transpose() 
                      for f in tqdm(annotation_paths))
    annotation_df = pd.concat(cls_dataframes, ignore_index=False)
    annotation_df['index'] = annotation_df['imagePath'].str.replace('.jpg', '')
    annotation_df = annotation_df.set_index('index').loc[list(img_dict.keys())]
    return annotation_df

def analyze_classes(annotation_df):
    """Analyze class distribution"""
    print("\nAnalyzing class distribution...")
    shapes_df = annotation_df['shapes'].explode().apply(pd.Series)
    class_names = shapes_df['label'].unique().tolist()
    
    # Add background class
    class_names = ['background'] + class_names
    
    # Plot distribution
    class_counts = shapes_df['label'].value_counts()
    plt.figure(figsize=(8,4))
    class_counts.plot(kind='bar')
    plt.title('Class Distribution')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.savefig('class_distribution.png')  # Save to root directory
    plt.close()
    
    return class_names

def setup_visualization(class_names):
    """Download font and generate colors"""
    # Download font to root directory
    font_url = "https://fonts.gstatic.com/s/roboto/v30/KFOlCnqEu92Fr1MmEU9vAw.ttf"
    font_path = Path("../KFOlCnqEu92Fr1MmEU9vAw.ttf")  # Save to root
    
    if not font_path.exists():
        print("\nDownloading font...")
        response = requests.get(font_url)
        font_path.write_bytes(response.content)
    
    # Generate colors
    colors = distinctipy.get_colors(len(class_names))
    int_colors = [tuple(int(channel * 255) for channel in color) for color in colors]
    
    # Save color swatch to root
    plt.figure(figsize=(len(class_names), 2))
    for i, (color, name) in enumerate(zip(colors, class_names)):
        plt.fill_between([i, i+1], 0, 1, color=color)
        plt.text(i + 0.5, 0.5, name, ha='center', va='center')
    plt.xlim(0, len(class_names))
    plt.axis('off')
    plt.savefig("color_swatch.png", bbox_inches='tight')  # Root directory
    plt.close()
    
    return int_colors, font_path

def create_polygon_mask(image_size, vertices):
    """
    Create binary mask from polygon vertices.
    (No path changes needed here as it's pure image processing)
    """
    mask = Image.new('L', image_size, 0)
    ImageDraw.Draw(mask).polygon(vertices, fill=255)
    return mask

if __name__ == "__main__":
    # Path setup - navigate from data/ to root/Datasets
    dataset_path = Path("../Datasets/pytorch-for-information-extraction/code/datasets/detection/student-id")
    
    # Get annotation files
    annotation_paths = list(dataset_path.glob('*.json'))
    img_dict = {f.stem: f for f in dataset_path.glob('*.jpg')}
    
    # Process annotations
    annotation_df = process_annotations(annotation_paths, img_dict)
    
    # Class analysis
    class_names = analyze_classes(annotation_df)
    print(f"\nClass names: {class_names}")
    
    # Visualization setup
    int_colors, font_path = setup_visualization(class_names)
    print("\nVisualization assets ready:")
    print(f"- Font: {font_path}")
    print("- Color swatch saved at ../color_swatch.png")