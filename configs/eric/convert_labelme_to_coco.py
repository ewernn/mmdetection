import json
import os
from PIL import Image
import numpy as np
from shapely.geometry import Polygon
from sklearn.model_selection import train_test_split
import random

def convert_labelme_to_coco(labelme_dir, output_dir):
    all_data = []
    category_map = {}

    # First pass: collect all data and create category map
    for filename in os.listdir(labelme_dir):
        if filename.endswith('.json'):
            with open(os.path.join(labelme_dir, filename), 'r') as f:
                labelme_data = json.load(f)
            
            image_filename = labelme_data['imagePath']
            image_path = os.path.join(labelme_dir, image_filename)
            
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue
            
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
            except Exception as e:
                print(f"Error opening image {image_path}: {e}")
                continue
            
            annotations = []
            for shape in labelme_data['shapes']:
                label = shape['label']
                if label not in category_map:
                    category_id = len(category_map) + 1
                    category_map[label] = category_id
                else:
                    category_id = category_map[label]
                
                points = shape['points']
                polygon = Polygon(points)
                x, y, max_x, max_y = polygon.bounds
                bbox_width = max_x - x
                bbox_height = max_y - y
                
                annotations.append({
                    "category_id": category_id,
                    "bbox": [x, y, bbox_width, bbox_height],
                    "area": bbox_width * bbox_height,
                    "segmentation": [np.array(points).flatten().tolist()],
                    "iscrowd": 0
                })
            
            all_data.append({
                "file_name": image_filename,
                "width": width,
                "height": height,
                "annotations": annotations
            })

    # Split data into train, val, and test sets
    train_data, temp_data = train_test_split(all_data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # Create COCO format for each split
    for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        create_coco_format_subset(split_data, category_map, os.path.join(output_dir, f"{split_name}_coco_annotations.json"))

def create_coco_format_subset(data, category_map, output_file):
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    for category, category_id in category_map.items():
        coco_format["categories"].append({
            "id": category_id,
            "name": category
        })

    annotation_id = 1
    for image_id, image_data in enumerate(data, start=1):
        coco_format["images"].append({
            "id": image_id,
            "file_name": image_data["file_name"],
            "width": image_data["width"],
            "height": image_data["height"]
        })

        for ann in image_data["annotations"]:
            ann["id"] = annotation_id
            ann["image_id"] = image_id
            coco_format["annotations"].append(ann)
            annotation_id += 1

    with open(output_file, 'w') as f:
        json.dump(coco_format, f, indent=2)

    print(f"Created COCO format subset: {output_file}")
    print(f"Total images: {len(coco_format['images'])}")
    print(f"Total annotations: {len(coco_format['annotations'])}")

# Usage
labelme_dir = '/Users/ewern/Desktop/code/img_segmentation/cat_kidney_dataset_csv_filtered'
output_dir = '/Users/ewern/Desktop/code/MetronMind/data/cat-dataset/labelme_to_coco'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

convert_labelme_to_coco(labelme_dir, output_dir)