import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def calculate_mean_std(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    
    if not image_files:
        print("No .tif images found in the specified folder.")
        return
    
    sum_pixels = 0
    sum_squared_pixels = 0
    num_pixels = 0
    
    for img_file in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(folder_path, img_file)
        with Image.open(img_path) as img:
            # Ensure the image is grayscale and 1000x1000
            if img.mode != 'L' or img.size != (1000, 1000):
                print(f"Skipping {img_file}: Not a 1000x1000 grayscale image")
                continue
            
            img_array = np.array(img, dtype=np.float64)
            sum_pixels += np.sum(img_array)
            sum_squared_pixels += np.sum(img_array ** 2)
            num_pixels += img_array.size
    
    if num_pixels == 0:
        print("No valid images processed.")
        return
    
    mean = sum_pixels / num_pixels
    std = np.sqrt((sum_squared_pixels / num_pixels) - (mean ** 2))
    
    return mean, std

folder_path = '/Users/ewern/Desktop/code/MetronMind/data/cat-kidneys'
result = calculate_mean_std(folder_path)

if result:
    mean, std = result
    print(f"Mean pixel value: {mean:.3f}")
    print(f"Standard deviation: {std:.3f}")



# correct brightness
train_pipeline = [
    dict(backend_args=None, color_type='grayscale', type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=True, scale=(
        1000,
        1000,
    ), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(
        max_rotate_degree=20,
        scaling_ratio_range=(
            0.8,
            1.2,
        ),
        type='RandomAffine'),
    dict(level=5, type='Brightness'),
    dict(mean=[
        123.675,
    ], std=[
        58.395,
    ], to_rgb=False, type='Normalize'),
    dict(size_divisor=32, type='Pad'),
    dict(type='PackDetInputs'),
]

# wrong
train_pipeline = [
    dict(backend_args=None, color_type='grayscale', type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=True, scale=(
        1000,
        1000,
    ), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(
        max_rotate_degree=20,
        scaling_ratio_range=(
            0.8,
            1.2,
        ),
        type='RandomAffine'),
    dict(level=5, type='Brightness'),
    dict(mean=[
        123.675,
    ], std=[
        58.395,
    ], to_rgb=False, type='Normalize'),
    dict(size_divisor=32, type='Pad'),
    dict(type='PackDetInputs'),
]