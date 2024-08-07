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
    categories = [{'id': 1, 'name': 'vertebrae'}]
    
    annotation_id = 1
    abs_counter = 0  # Initialize the counter
    
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
        
        coords = []
        for i in range(1, 5):
            x, y = row.get(f'x{i}'), row.get(f'y{i}')
            if pd.notna(x) and pd.notna(y):
                coords.append((float(x) * width, float(y) * height))
        
        for i in range(0, len(coords), 2):
            x, y = coords[i]
            if i + 1 < len(coords):
                x2, y2 = coords[i + 1]
                width_diff = x2 - x
                height_diff = y2 - y
                bbox_width = abs(width_diff)
                bbox_height = abs(height_diff)
                
                # Increment counter only if abs() changed a negative value to positive
                if width_diff < 0:
                    abs_counter += 1
                if height_diff < 0:
                    abs_counter += 1
            else:
                bbox_width = bbox_height = 10
            
            bbox = [min(x, x2), min(y, y2), bbox_width, bbox_height]
            annotations.append({
                'id': annotation_id,
                'image_id': idx + 1,
                'category_id': 1,
                'bbox': bbox,
                'area': bbox_width * bbox_height,
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

    print(f"abs() function changed {abs_counter} negative values to positive for subset: {subset_name}")

# Usage
data_csv_path = '/Users/ewern/Desktop/code/MetronMind/data/cat-dataset/Data.csv'  # Update with your actual path
images_dir = '/Users/ewern/Desktop/code/MetronMind/data/cat-dataset'               # Update with your actual path to images directory
output_json_path = '/Users/ewern/Desktop/code/MetronMind/data/cat-dataset-json'  # Update with your desired output path

create_coco_format(data_csv_path, images_dir, output_json_path)
