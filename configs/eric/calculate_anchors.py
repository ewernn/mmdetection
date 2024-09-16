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

def calculate_aspect_ratios(widths, heights, num_ratios=5, coverage_percent=90):
    slope, intercept = calculate_line_of_best_fit(widths, heights)
    central_ratio = slope
    
    distances = np.abs(heights - (slope * widths + intercept)) / np.sqrt(1 + slope**2)
    max_distance = np.percentile(distances, coverage_percent)
    
    min_ratio = central_ratio - max_distance / np.mean(widths)
    max_ratio = central_ratio + max_distance / np.mean(widths)
    
    return np.linspace(min_ratio, max_ratio, num_ratios)

def calculate_anchor_boxes(widths, heights, num_sizes=5):
    avg_sizes = (widths + heights) / 2
    percentiles = np.linspace(0, 100, num_sizes)
    anchor_sizes = np.percentile(avg_sizes, percentiles)
    return np.column_stack((anchor_sizes, anchor_sizes))

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
    
    anchor_boxes = calculate_anchor_boxes(widths, heights)
    aspect_ratios = calculate_aspect_ratios(widths, heights)
    
    print("Anchor boxes:")
    print(anchor_boxes)
    print("Aspect ratios:")
    print(aspect_ratios)
    
    return anchor_boxes, aspect_ratios, widths, heights

def calculate_final_anchors(results):
    # Calculate final anchor boxes and aspect ratios from all results
    all_widths = np.concatenate([r[0][:, 0] for r in results if r[0] is not None])
    all_heights = np.concatenate([r[0][:, 1] for r in results if r[0] is not None])
    all_ratios = np.concatenate([r[1] for r in results if r[1] is not None])
    
    final_boxes = calculate_anchor_boxes(all_widths, all_heights)
    final_ratios = calculate_aspect_ratios(all_widths, all_heights)
    
    return final_boxes, final_ratios

def plot_scatter(all_widths, all_heights, final_boxes, output_path):
    # Create a scatter plot of bounding box dimensions and anchor boxes
    plt.figure(figsize=(10, 10))
    plt.scatter(all_widths, all_heights, alpha=0.5)
    plt.scatter(final_boxes[:, 0], final_boxes[:, 1], color='red', s=200, marker='x')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.title('Bounding Box Dimensions and Anchor Boxes')
    plt.savefig(output_path)
    print(f"Scatter plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Calculate anchor boxes and aspect ratios')
    parser.add_argument('--output', choices=['anchors', 'plot'], default='plot',
                        help='Output type: "anchors" for best anchor sizes or "plot" for scatter plot (default: plot)')
    parser.add_argument('--num_sizes', type=int, default=5, help='Number of anchor sizes')
    parser.add_argument('--num_ratios', type=int, default=5, help='Number of aspect ratios')
    parser.add_argument('--coverage_percent', type=float, default=90, help='Percentage of points to cover for aspect ratios')
    args = parser.parse_args()

    files = [
        '/Users/ewern/Desktop/code/MetronMind/data/cat-dataset/labelme_to_coco/train_coco_annotations.json',
        '/Users/ewern/Desktop/code/MetronMind/data/cat-dataset/COCO_2/val_Data_coco_format.json'
    ]
    # files = [
    #     '/Users/ewern/Desktop/code/MetronMind/data/cat-dataset/COCO_2/train_Data_coco_format.json',
    #     '/Users/ewern/Desktop/code/MetronMind/data/cat-dataset/COCO_2/val_Data_coco_format.json'
    # ]
    
    results = []
    all_widths = []
    all_heights = []
    for file_path in files:
        result = process_file(file_path)
        results.append(result[:2])  # Only append anchor_boxes and aspect_ratios
        all_widths.extend(result[2])  # Append widths
        all_heights.extend(result[3])  # Append heights
    
    final_boxes, final_ratios = calculate_final_anchors(results)
    
    if args.output == 'anchors':
        print("\nFinal anchor boxes (width, height):")
        print(final_boxes)
        print("\nFinal aspect ratios:")
        print(final_ratios)
        
        print("\nRounded final anchor boxes (width, height):")
        print(np.round(final_boxes).astype(int))
        print("\nRounded final aspect ratios:")
        print(np.round(final_ratios, 2))
    elif args.output == 'plot':
        output_path = '/Users/ewern/Desktop/code/MetronMind/mmdetection/configs/eric/anchor_graphs/anchor_scatter_plot.png'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plot_scatter(all_widths, all_heights, final_boxes, output_path)

if __name__ == "__main__":
    main()