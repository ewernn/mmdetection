import json
import pandas as pd
import os
import shutil
from PIL import Image
from sklearn.model_selection import train_test_split
import random

# Define paths
# named up to 7042 images
FIRST_DATASET_PATH = '/Users/ewern/Desktop/code/MetronMind/data/cat_kidney_dataset_csv_filtered'
FIRST_DATASET_COCO = {
    'train': '/Users/ewern/Desktop/code/MetronMind/data/cat_kidney_dataset_csv_filtered/COCO_2/train_Data_coco_format-labelme.json',
    'val': '/Users/ewern/Desktop/code/MetronMind/data/cat_kidney_dataset_csv_filtered/COCO_2/val_Data_coco_format-labelme.json',
    'test': '/Users/ewern/Desktop/code/MetronMind/data/cat_kidney_dataset_csv_filtered/COCO_2/test_Data_coco_format-labelme.json'
}
# Second dataset
SECOND_DATASET_PATH = '/Users/ewern/Desktop/code/MetronMind/data/DataCatSep26'
SECOND_DATASET_CSV = '/Users/ewern/Desktop/code/MetronMind/data/DataCatSep26/Data_only2_fixedSep26.csv'
OUTPUT_PATH = '/Users/ewern/Desktop/code/MetronMind/data/cat-data-combined-oct20'

def load_first_dataset():
    dataset = {'images': [], 'annotations': [], 'categories': []}
    for split, json_file in FIRST_DATASET_COCO.items():
        with open(json_file, 'r') as f:
            data = json.load(f)
        dataset['images'].extend(data['images'])
        dataset['annotations'].extend(data['annotations'])
        if not dataset['categories']:
            dataset['categories'] = data['categories']
    return dataset

def load_second_dataset(start_id):
    df = pd.read_csv(SECOND_DATASET_CSV)
    dataset = {'images': [], 'annotations': [], 'categories': []}
    annotation_id = start_id * 2  # Assuming each image has 2 annotations (left and right kidney)

    for idx, row in df.iterrows():
        # Check if all 8 points are present
        if pd.notna(row['x1']) and pd.notna(row['y1']) and pd.notna(row['x2']) and pd.notna(row['y2']) and \
           pd.notna(row['x3']) and pd.notna(row['y3']) and pd.notna(row['x4']) and pd.notna(row['y4']):
            
            image_id = start_id + idx
            new_name = f'Im{image_id}.tif'
            
            dataset['images'].append({
                'id': image_id,
                'file_name': new_name,
                'width': 1920,  # Assuming all images have the same dimensions
                'height': 1080  # Update these if the dimensions are different
            })
            
            # Create annotations for both kidneys
            for i in range(2):
                x1, y1 = row[f'x{2*i+1}'], row[f'y{2*i+1}']
                x2, y2 = row[f'x{2*i+2}'], row[f'y{2*i+2}']
                bbox = [min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)]
                dataset['annotations'].append({
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': i + 1,  # 1 for right kidney, 2 for left kidney
                    'bbox': bbox,
                    'area': bbox[2] * bbox[3],
                    'iscrowd': 0
                })
                annotation_id += 1

    dataset['categories'] = [
        {'id': 1, 'name': 'right_kidney'},
        {'id': 2, 'name': 'left_kidney'}
    ]
    
    print(f"Loaded {len(dataset['images'])} images with both kidneys from the second dataset.")
    return dataset

def combine_datasets(first_dataset, second_dataset):
    combined = {
        'images': first_dataset['images'] + second_dataset['images'],
        'annotations': first_dataset['annotations'] + second_dataset['annotations'],
        'categories': first_dataset['categories']
    }
    return combined

def create_new_split(combined_dataset):
    all_image_ids = [img['id'] for img in combined_dataset['images']]
    train_ids, test_val_ids = train_test_split(all_image_ids, test_size=0.2, random_state=42)
    val_ids, test_ids = train_test_split(test_val_ids, test_size=0.5, random_state=42)
    
    splits = {
        'train': {'images': [], 'annotations': []},
        'val': {'images': [], 'annotations': []},
        'test': {'images': [], 'annotations': []}
    }
    
    for img in combined_dataset['images']:
        if img['id'] in train_ids:
            splits['train']['images'].append(img)
        elif img['id'] in val_ids:
            splits['val']['images'].append(img)
        else:
            splits['test']['images'].append(img)
    
    for ann in combined_dataset['annotations']:
        if ann['image_id'] in train_ids:
            splits['train']['annotations'].append(ann)
        elif ann['image_id'] in val_ids:
            splits['val']['annotations'].append(ann)
        else:
            splits['test']['annotations'].append(ann)
    
    for split in splits.values():
        split['categories'] = combined_dataset['categories']
    
    return splits

def copy_images(dataset, src_path, dst_path, is_second_dataset=False):
    os.makedirs(dst_path, exist_ok=True)
    for img in dataset['images']:
        if is_second_dataset:
            # For the second dataset, we need to use the original filename to find the source file
            original_number = int(img['file_name'].split('.')[0][2:]) - 7543
            original_name = f"Im{original_number}.tif"
            src_file = os.path.join(src_path, original_name)
        else:
            src_file = os.path.join(src_path, img['file_name'])
        
        dst_file = os.path.join(dst_path, img['file_name'])
        try:
            if os.path.exists(src_file):
                shutil.copy2(src_file, dst_file)
                print(f"Copied: {src_file} -> {dst_file}")
            else:
                print(f"Warning: Source image not found, skipping: {src_file}")
        except Exception as e:
            print(f"Error copying file {src_file}: {str(e)}")

def create_coco_json(split_data, output_path):
    with open(output_path, 'w') as f:
        json.dump(split_data, f, indent=2)

def validate_output(output_path):
    for split in ['train', 'val', 'test']:
        json_path = os.path.join(output_path, f'{split}.json')
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        print(f"{split} split:")
        print(f"  Images: {len(data['images'])}")
        print(f"  Annotations: {len(data['annotations'])}")
        
        # Check if all images exist
        missing_images = []
        for img in data['images']:
            img_path = os.path.join(output_path, img['file_name'])
            if not os.path.exists(img_path):
                missing_images.append(img['file_name'])
        if missing_images:
            print(f"  Warning: {len(missing_images)} images not found:")
            for img_name in missing_images[:10]:  # Print first 10 missing images
                print(f"    - {img_name}")
            if len(missing_images) > 10:
                print(f"    ... and {len(missing_images) - 10} more")
        else:
            print("  All images found successfully")
    
    print("Validation complete.")

def main():
    print("Loading first dataset...")
    first_dataset = load_first_dataset()
    
    print("Loading second dataset...")
    start_id = 7544  # This is the ID after Im7543.tif
    second_dataset = load_second_dataset(start_id)
    
    print("Combining datasets...")
    combined_dataset = combine_datasets(first_dataset, second_dataset)
    
    print("Creating new splits...")
    new_splits = create_new_split(combined_dataset)
    
    print("Copying images...")
    copy_images(first_dataset, FIRST_DATASET_PATH, OUTPUT_PATH)
    copy_images(second_dataset, SECOND_DATASET_PATH, OUTPUT_PATH, is_second_dataset=True)
    
    print("Creating COCO JSON files...")
    for split_name, split_data in new_splits.items():
        create_coco_json(split_data, os.path.join(OUTPUT_PATH, f'{split_name}.json'))
    
    print("Validating output...")
    validate_output(OUTPUT_PATH)
    
    print("Process completed successfully.")

if __name__ == "__main__":
    main()
