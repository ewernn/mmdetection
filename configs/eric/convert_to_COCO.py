import json
import pandas as pd
import os
from PIL import Image
from sklearn.model_selection import train_test_split

def split_data(df, test_size=0.2):
    # Split dataset into training and temp (for further splitting into validation and test)
    train_df, temp_df = train_test_split(df, test_size=test_size*2, random_state=42)
    # Split temp into validation and test
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    return train_df, val_df, test_df

def create_coco_format_subset(df, images_dir, output_json_path, subset_name):
    images = []
    annotations = []
    categories = [{'id': 1, 'name': 'vertebrae'}] # ERIC ADDED
    
    annotation_id = 1  # Start annotation IDs at 1
    for idx, row in df.iterrows():
        image_filename = row['Image']
        image_path = os.path.join(images_dir, image_filename)
        with Image.open(image_path) as img:
            width, height = img.size
        
        images.append({
            'id': idx + 1,
            'width': width,
            'height': height,
            'file_name': image_filename,
        })
        
        for bbox_idx in range(1, len(row), 2):
            if pd.isna(row[f'x{bbox_idx//2 + 1}']):
                break
            x_min = max(0, row[f'x{bbox_idx//2 + 1}'] * width)
            y_min = max(0, row[f'y{bbox_idx//2 + 1}'] * height)
            if bbox_idx+2 < len(row) and not pd.isna(row[f'x{bbox_idx//2 + 2}']):
                x_max = min(width, row[f'x{bbox_idx//2 + 2}'] * width)
                y_max = min(height, row[f'y{bbox_idx//2 + 2}'] * height)
                if x_min >= width or y_min >= height or x_max <= 0 or y_max <= 0 or (x_max - x_min) <= 0 or (y_max - y_min) <= 0:
                    continue
                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                annotations.append({
                    'id': annotation_id,
                    'image_id': idx + 1,
                    'category_id': 1,  # ERIC CHANGED
                    'bbox': bbox,
                    'area': (x_max - x_min) * (y_max - y_min),
                    'iscrowd': 0,
                    'segmentation': [],
                })
                annotation_id += 1
    
    coco_format_json = {
        'images': images,
        'annotations': annotations,
        'categories': categories,
    }
    
    subset_output_path = os.path.join(output_json_path, f'{subset_name}_Data_coco_format.json')
    with open(subset_output_path, 'w') as f:
        json.dump(coco_format_json, f, indent=4)

def create_coco_format(data_csv_path, images_dir, output_json_path):
    # Load the dataset
    df = pd.read_csv(data_csv_path)
    
    # Split the DataFrame into train, val, and test
    train_df, val_df, test_df = split_data(df, test_size=0.2)
    
    # Generate COCO format JSON files for each subset
    for subset_name, subset_df in zip(['train', 'val', 'test'], [train_df, val_df, test_df]):
        create_coco_format_subset(subset_df, images_dir, output_json_path, subset_name)

# Usage
data_csv_path = '/Users/ewern/Desktop/code/MetronMind/stacked_hourglass_point_localization/data/EqNeck/EqNeckData/Data.csv'  # Update with your actual path
images_dir = '/Users/ewern/Desktop/code/MetronMind/stacked_hourglass_point_localization/data/EqNeck/EqNeckData'               # Update with your actual path to images directory
output_json_path = '/Users/ewern/Desktop/code/MetronMind/stacked_hourglass_point_localization/data/EqNeck'  # Update with your desired output path

create_coco_format(data_csv_path, images_dir, output_json_path)
