import json
import pandas as pd
import os
from PIL import Image
from sklearn.model_selection import train_test_split

def categorize_data(df):
    def count_kidneys(row):
        coords = [row[f'x{i}'] for i in range(1, 5) if pd.notna(row[f'x{i}'])]
        return len(coords) // 2  # Each kidney has 2 coordinates

    df['kidney_count'] = df.apply(count_kidneys, axis=1)
    
    zero_one_two = df[df['kidney_count'].isin([0, 1, 2])]
    one_two = df[df['kidney_count'].isin([1, 2])]
    two_only = df[df['kidney_count'] == 2]
    
    return zero_one_two, one_two, two_only

def split_data(df, test_size=0.2):
    train_df, temp_df = train_test_split(df, test_size=test_size*2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    return train_df, val_df, test_df

def create_coco_format(data_csv_path, images_dir, output_json_path):
    df = pd.read_csv(data_csv_path)
    
    zero_one_two, one_two, two_only = categorize_data(df)
    
    datasets = {
        'zero_one_two': zero_one_two,
        'one_two': one_two,
        'two_only': two_only
    }
    
    total_abs_count = 0
    for dataset_name, dataset in datasets.items():
        train_df, val_df, test_df = split_data(dataset, test_size=0.2)
        
        for subset_name, subset_df in zip(['train', 'val', 'test'], [train_df, val_df, test_df]):
            create_coco_format_subset(subset_df, images_dir, output_json_path, f'{dataset_name}_{subset_name}')
    
    print(f"Total negative values changed to positive across all subsets: {total_abs_count}")

def create_coco_format_subset(df, images_dir, output_json_path, subset_name):
    images = []
    annotations = []
    categories = [
        {'id': 1, 'name': 'right_kidney'},
        {'id': 2, 'name': 'left_kidney'}
    ]
    
    annotation_id = 1
    valid_image_count = 0
    
    for idx, row in df.iterrows():
        image_filename = row['Image']
        image_path = os.path.join(images_dir, image_filename)
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            continue
        
        coords = []
        for i in range(1, 5):
            x, y = row.get(f'x{i}'), row.get(f'y{i}')
            if pd.notna(x) and pd.notna(y):
                coords.append((float(x) * width, float(y) * height))
        
        valid_boxes = []
        for i in range(0, len(coords), 2):
            if i + 1 < len(coords):
                x, y = coords[i]
                x2, y2 = coords[i + 1]
                bbox_width = abs(x2 - x)
                bbox_height = abs(y2 - y)
                bbox_area = bbox_width * bbox_height
                bbox_y = min(y, y2)
                
                if bbox_area <= 20474 and 20 <= bbox_y <= 875:
                    valid_boxes.append([min(x, x2), bbox_y, bbox_width, bbox_height])
        
        if len(valid_boxes) == 2:
            valid_image_count += 1
            images.append({
                'id': valid_image_count,
                'width': width,
                'height': height,
                'file_name': image_filename,
            })
            
            # Sort boxes by x-coordinate
            valid_boxes.sort(key=lambda box: box[0])
            
            for i, bbox in enumerate(valid_boxes):
                category_id = 1 if i == 0 else 2  # 1 for right_kidney, 2 for left_kidney
                
                annotations.append({
                    'id': annotation_id,
                    'image_id': valid_image_count,
                    'category_id': category_id,
                    'bbox': bbox,
                    'area': bbox[2] * bbox[3],
                    'iscrowd': 0,
                    'segmentation': [],
                })
                annotation_id += 1
    
    if images:  # Only create JSON if there are valid images
        coco_format_json = {
            'images': images,
            'annotations': annotations,
            'categories': categories,
        }
        
        subset_output_path = os.path.join(output_json_path, f'{subset_name}_Data_coco_format.json')
        with open(subset_output_path, 'w') as f:
            json.dump(coco_format_json, f, indent=4)

        print(f"Created COCO format subset: {subset_name}")
        print(f"Total images: {len(images)}")
        print(f"Total annotations: {len(annotations)}")
    else:
        print(f"No valid images found for subset: {subset_name}")

# Usage
data_csv_path = '/Users/ewern/Desktop/code/MetronMind/data/cat-dataset/Updated_Data_only2-sep14.csv'
images_dir = '/Users/ewern/Desktop/code/MetronMind/data/cat-dataset'
output_json_path = '/Users/ewern/Desktop/code/MetronMind/data/cat-dataset/coco_output'  # Changed to a directory path

# Create the output directory if it doesn't exist
os.makedirs(output_json_path, exist_ok=True)

create_coco_format(data_csv_path, images_dir, output_json_path)