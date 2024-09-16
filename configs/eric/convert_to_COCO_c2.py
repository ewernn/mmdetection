import os
import json
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

def create_coco_format(data_csv_path, images_dir, output_json_path, dataset_type, include_all_images=True):
    df = pd.read_csv(data_csv_path)
    # Convert x1, y1, x2, y2 to numeric, coercing errors to NaN
    for col in ['x1', 'y1', 'x2', 'y2']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # Clamping x1, y1, x2, y2 to be within [0.0, 1.0]
    df[['x1', 'y1', 'x2', 'y2']] = df[['x1', 'y1', 'x2', 'y2']].clip(lower=0.0, upper=1.0)
    df['has_bbox'] = df[['x1', 'y1', 'x2', 'y2']].notna().all(axis=1)
    
    if not include_all_images:
        df = df[df['has_bbox']]
    
    images = []
    annotations = []
    categories = [
        {'id': 1, 'name': 'c2_vertebrae'}
    ]
    annotation_id = 1
    
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
        
        if row['has_bbox']:
            coords = []
            for i in range(1, 3):
                x_key = f'x{i}'
                y_key = f'y{i}'
                if x_key in row and y_key in row and pd.notna(row[x_key]) and pd.notna(row[y_key]):
                    x = float(row[x_key]) * width
                    y = float(row[y_key]) * height
                    coords.append((x, y))
            
            valid_boxes = []
            for i in range(0, len(coords), 2):
                x, y = coords[i]
                x2, y2 = coords[i + 1]
                bbox_width = abs(x2 - x)
                bbox_height = abs(y2 - y)
                valid_boxes.append([min(x, x2), min(y, y2), bbox_width, bbox_height])
            
            valid_boxes.sort(key=lambda box: box[0])  # Sort by x-coordinate
            
            for i, bbox in enumerate(valid_boxes):
                category_id = 1 if i == 0 else 2  # 1 for right_kidney, 2 for left_kidney
                annotations.append({
                    'id': annotation_id,
                    'image_id': idx + 1,
                    'category_id': category_id,
                    'bbox': bbox,
                    'area': bbox[2] * bbox[3],
                    'iscrowd': 0,
                    'segmentation': [],
                })
                annotation_id += 1
    
    coco_format_json = {
        'images': images,
        'annotations': annotations,
        'categories': categories,
    }
    
    subset_name = f'{dataset_type}_{"all_images" if include_all_images else "only_with_bbox"}'
    subset_output_path = os.path.join(output_json_path, f'{subset_name}_Data_coco_format.json')
    
    os.makedirs(output_json_path, exist_ok=True)
    
    with open(subset_output_path, 'w') as f:
        json.dump(coco_format_json, f, indent=4)

    print(f"Created COCO format for {subset_name}: {subset_output_path}")

    print(f"Created COCO format for {subset_name}: {subset_output_path}")

# Usage
data_paths = [
    ('Test', '/Users/ewern/Desktop/code/MetronMind/c2/data/Test/Data_updated.csv', '/Users/ewern/Desktop/code/MetronMind/c2/data/Test/'),
    ('Train', '/Users/ewern/Desktop/code/MetronMind/c2/data/Train/Data_updated.csv', '/Users/ewern/Desktop/code/MetronMind/c2/data/Train/')
]
output_json_path = '/Users/ewern/Desktop/code/MetronMind/c2/data/c2_coco_format'

for dataset_type, data_csv_path, images_dir in data_paths:
    create_coco_format(data_csv_path, images_dir, output_json_path, dataset_type, include_all_images=True)
    create_coco_format(data_csv_path, images_dir, output_json_path, dataset_type, include_all_images=False)