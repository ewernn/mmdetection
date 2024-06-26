import json
import os
from PIL import Image, ImageDraw
import numpy as np
from collections import defaultdict

def load_coco_annotations(json_file):
    with open(json_file, 'r') as f:
        coco_data = json.load(f)
    return coco_data

def draw_annotations(image, annotations, category_id_to_name):
    print(f"Number of annotations: {len(annotations)}")
    print(annotations)
    draw = ImageDraw.Draw(image)
    colors = ["red", "blue"]  # One color for each kidney

    for idx, ann in enumerate(annotations):
        category_id = ann['category_id']
        category_name = category_id_to_name.get(category_id, f"Unknown ({category_id})")
        color = colors[idx % len(colors)]

        bbox = ann['bbox']
        x, y, w, h = bbox

        # Handle negative width or height
        if w < 0:
            x += w
            w = abs(w)
        if h < 0:
            y += h
            h = abs(h)

        # Draw bounding box
        draw.rectangle([x, y, x+w, y+h], outline=color, width=2)

        # Add category name
        label = f"Kidney {idx + 1}"
        draw.text((x, y-15), label, fill=color)

    return image

def main(data_dir, annotation_file, output_dir):
    # Load COCO annotations
    coco_data = load_coco_annotations(annotation_file)

    # Create category ID to name mapping
    category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process only the first 20 images
    for img_info in coco_data['images'][:20]:
        img_id = img_info['id']
        img_file = img_info['file_name']
        img_path = os.path.join(data_dir, img_file)

        # Open the image (change to support .tif format)
        image = Image.open(img_path).convert('RGB')

        # Get annotations for this image
        img_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]
        print(f"Image {img_file} has {len(img_annotations)} annotations")

        if img_annotations:
            # Draw annotations on the image
            annotated_image = draw_annotations(image, img_annotations, category_id_to_name)
        else:
            annotated_image = image  # Use original image if no annotations

        # Save the annotated image
        output_path = os.path.join(output_dir, f"annotated_{img_file}")
        annotated_image.save(output_path)

        print(f"Processed {img_file}")

    print("Finished processing 20 images")

if __name__ == "__main__":
    data_dir = "/Users/ewern/Desktop/code/MetronMind/data/cat-dataset"
    annotation_file = "/Users/ewern/Desktop/code/MetronMind/data/cat-dataset/COCO_2/train_Data_coco_format.json"
    output_dir = "/Users/ewern/Desktop/code/MetronMind/data/cat-dataset-out"

    main(data_dir, annotation_file, output_dir)