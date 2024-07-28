import json
import numpy as np
from sklearn.cluster import KMeans
import os

def load_annotations(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data['annotations']

def extract_bbox_dimensions(annotations):
    widths = []
    heights = []
    for ann in annotations:
        # COCO bbox format: [x_min, y_min, width, height]
        bbox = ann['bbox']
        widths.append(bbox[2])
        heights.append(bbox[3])
    return np.array(widths), np.array(heights)

def calculate_aspect_ratios(widths, heights):
    return widths / heights

def perform_kmeans(data, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
    return kmeans.cluster_centers_

def process_file(json_path):
    print(f"Processing file: {json_path}")
    annotations = load_annotations(json_path)
    widths, heights = extract_bbox_dimensions(annotations)
    aspect_ratios = calculate_aspect_ratios(widths, heights)
    
    # Reshape data for k-means
    data = np.stack((widths, heights), axis=1)
    
    # Calculate optimal anchor sizes
    anchor_sizes = perform_kmeans(data)
    print("Optimal anchor sizes (width, height):")
    print(anchor_sizes)
    
    # Calculate optimal aspect ratios
    aspect_ratio_clusters = perform_kmeans(aspect_ratios.reshape(-1, 1))
    print("Optimal aspect ratios:")
    print(aspect_ratio_clusters.flatten())

def main():
    files = [
        '/Users/ewern/Desktop/code/MetronMind/c2/data/Train/only_with_bbox_Data_coco_format.json',
        '/Users/ewern/Desktop/code/MetronMind/c2/data/Train/all_images_Data_coco_format.json',
        '/Users/ewern/Desktop/code/MetronMind/c2/data/Test/only_with_bbox_Data_coco_format.json',
        '/Users/ewern/Desktop/code/MetronMind/c2/data/Test/all_images_Data_coco_format.json'
    ]
    
    for file_path in files:
        process_file(file_path)

if __name__ == "__main__":
    main()