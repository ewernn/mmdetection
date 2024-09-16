import json
import numpy as np
from sklearn.cluster import KMeans
import os
import signal
import argparse
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    # Handler for timeout during file reading
    raise TimeoutException("File reading timed out")

def load_annotations(json_path, timeout=120):
    # Load annotations from a JSON file with a timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        with open(json_path, 'r') as file:
            data = json.load(file)
        signal.alarm(0)  # Cancel the alarm
        return data['annotations']
    except TimeoutException:
        print(f"Timeout occurred while reading {json_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON in {json_path}")
        return None
    except Exception as e:
        print(f"Error reading {json_path}: {str(e)}")
        return None

def extract_bbox_dimensions(annotations):
    # Extract width and height from bounding box annotations
    widths = []
    heights = []
    for ann in annotations:
        bbox = ann['bbox']
        widths.append(bbox[2])
        heights.append(bbox[3])
    return np.array(widths), np.array(heights)

def calculate_scales(widths, heights, num_scales=5):
    sizes = np.sqrt(widths * heights)
    min_size, max_size = np.percentile(sizes, [5, 95])  # Use 5th and 95th percentiles to avoid outliers
    return np.exp(np.linspace(np.log(min_size), np.log(max_size), num_scales))

def calculate_aspect_ratios(widths, heights, num_ratios=5):
    ratios = widths / heights
    log_ratios = np.log(ratios)
    
    # Remove outliers
    q1, q3 = np.percentile(log_ratios, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    filtered_log_ratios = log_ratios[(log_ratios >= lower_bound) & (log_ratios <= upper_bound)]
    
    # Use K-means clustering to find representative ratios
    kmeans = KMeans(n_clusters=num_ratios).fit(filtered_log_ratios.reshape(-1, 1))
    centers = np.exp(kmeans.cluster_centers_.flatten())
    
    # Ensure 1:1 ratio is included
    if 1.0 not in centers:
        centers = np.sort(np.append(centers, 1.0))
        if len(centers) > num_ratios:
            # Remove the ratio farthest from 1:1
            distances = np.abs(np.log(centers))
            centers = centers[np.argsort(distances)[:num_ratios]]
    
    return np.sort(centers)

def generate_anchor_boxes(scales, aspect_ratios):
    anchors = []
    for scale in scales:
        for ratio in aspect_ratios:
            w = scale * np.sqrt(ratio)
            h = scale / np.sqrt(ratio)
            anchors.append([w, h])
    return np.array(anchors)

def evaluate_anchor_coverage(anchors, gt_boxes, iou_threshold=0.5):
    covered_boxes = 0
    for gt_box in gt_boxes:
        gt_w, gt_h = gt_box
        ious = calculate_ious(anchors, np.array([[gt_w, gt_h]]))
        if np.max(ious) > iou_threshold:
            covered_boxes += 1
    coverage = covered_boxes / len(gt_boxes)
    return coverage

def calculate_ious(anchors, gt_boxes):
    # Calculate IoU between anchors and ground truth boxes
    # This is a simplified version and assumes centered boxes
    min_sizes = np.minimum(anchors[:, np.newaxis, :], gt_boxes[np.newaxis, :, :])
    max_sizes = np.maximum(anchors[:, np.newaxis, :], gt_boxes[np.newaxis, :, :])
    intersections = np.prod(np.maximum(0, min_sizes), axis=2)
    unions = np.prod(anchors, axis=1)[:, np.newaxis] + np.prod(gt_boxes, axis=1) - intersections
    return intersections / unions

def calculate_line_of_best_fit(widths, heights):
    slope, intercept = np.polyfit(widths, heights, 1)
    return slope, intercept

def process_file(json_path):
    # Process a single JSON file to extract anchor boxes and aspect ratios
    print(f"Processing file: {json_path}")
    annotations = load_annotations(json_path)
    if annotations is None:
        print(f"Skipping file due to error: {json_path}")
        return None, None
    widths, heights = extract_bbox_dimensions(annotations)
    return widths, heights

def plot_anchor_coverage(gt_boxes, anchors, coverage, output_path):
    plt.figure(figsize=(10, 10))
    plt.scatter(gt_boxes[:, 0], gt_boxes[:, 1], alpha=0.5, label='Ground Truth')
    plt.scatter(anchors[:, 0], anchors[:, 1], color='red', s=50, marker='x', label='Anchors')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.title(f'Bounding Box Dimensions and Anchor Boxes\nCoverage: {coverage:.2%}')
    plt.legend()
    plt.savefig(output_path)
    print(f"Anchor coverage plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Calculate anchor boxes for Faster R-CNN')
    parser.add_argument('--num_scales', type=int, default=5, help='Number of scales')
    parser.add_argument('--num_ratios', type=int, default=5, help='Number of aspect ratios')
    args = parser.parse_args()

    files = [
        '/Users/ewern/Desktop/code/MetronMind/data/cat-dataset/labelme_to_coco/train_coco_annotations.json',
        '/Users/ewern/Desktop/code/MetronMind/data/cat-dataset/COCO_2/val_Data_coco_format.json'
    ]
    
    all_widths = []
    all_heights = []
    for file_path in files:
        widths, heights = process_file(file_path)
        if widths is not None and heights is not None:
            all_widths.extend(widths)
            all_heights.extend(heights)
    
    all_widths = np.array(all_widths)
    all_heights = np.array(all_heights)
    all_boxes = np.column_stack((all_widths, all_heights))
    
    scales = calculate_scales(all_widths, all_heights, args.num_scales)
    aspect_ratios = calculate_aspect_ratios(all_widths, all_heights, args.num_ratios)
    
    anchors = generate_anchor_boxes(scales, aspect_ratios)
    coverage = evaluate_anchor_coverage(anchors, all_boxes)
    
    print("\nScales:")
    print(scales)
    print("\nAspect Ratios:")
    print(aspect_ratios)
    print("\nAnchor Boxes (width, height):")
    print(anchors)
    print(f"\nAnchor Coverage: {coverage:.2%}")
    
    # Visualizations
    output_dir = '/Users/ewern/Desktop/code/MetronMind/mmdetection/configs/eric/anchor_graphs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Aspect ratio histogram
    plt.figure(figsize=(10, 5))
    plt.hist(all_widths / all_heights, bins=50)
    plt.title("Histogram of Aspect Ratios")
    plt.xlabel("Aspect Ratio (width/height)")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_dir, 'aspect_ratio_histogram.png'))
    plt.close()
    
    # Anchor coverage plot
    plot_anchor_coverage(all_boxes, anchors, coverage, os.path.join(output_dir, 'anchor_coverage_plot.png'))

if __name__ == "__main__":
    main()