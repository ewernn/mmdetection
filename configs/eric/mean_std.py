# hi eric

image_dir = '/Users/ewern/Desktop/code/MetronMind/stacked_hour\
glass_point_localization/data/EqNeck/EqNeckData'

import os
from PIL import Image
import numpy as np

def calculate_mean_std(directory):
    mean_sum = 0
    sum_of_squared_error = 0
    num_pixels = 0
    for file_name in os.listdir(directory):
        if file_name.endswith('.tif'):
            file_path = os.path.join(directory, file_name)
            with Image.open(file_path) as img:
                img_array = np.array(img, dtype=np.float32) / 255.0  # Convert to float and scale to [0, 1]
                mean_sum += img_array.mean()
                sum_of_squared_error += ((img_array - img_array.mean())**2).sum()
                num_pixels += img_array.size

    mean = mean_sum / len(os.listdir(directory))  # Average mean across all images
    std = np.sqrt(sum_of_squared_error / num_pixels)  # Standard deviation across all images
    return mean, std

mean, std = calculate_mean_std(image_dir)
print(f"Mean: {mean}, Std: {std}")
