import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os

# Read the CSV file
df = pd.read_csv('/Users/ewern/Downloads/wetransfer_test-zip_2024-07-27_1944/Test/Data.csv')

# Convert columns to float, handling non-numeric values
df[['x1', 'y1', 'x2', 'y2']] = df[['x1', 'y1', 'x2', 'y2']].apply(pd.to_numeric, errors='coerce')

# Directory to save output images
output_dir = '/Users/ewern/Downloads/C2_imgs_out'
os.makedirs(output_dir, exist_ok=True)

# Function to draw bounding boxes and save images
def save_annotated_images(row):
    image_path = f"/Users/ewern/Downloads/wetransfer_test-zip_2024-07-27_1944/Test/{row['Image']}"  # Adjust path as necessary
    if not pd.isna(row['x1']):
        img = Image.open(image_path).convert('L')  # Convert image to grayscale
        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')  # Display the image in grayscale
        # Calculate bbox dimensions
        width, height = img.size
        rect = patches.Rectangle((row['x1'] * width, row['y1'] * height),
                                 (row['x2'] - row['x1']) * width,
                                 (row['y2'] - row['y1']) * height,
                                 linewidth=1, edgecolor='lightblue', facecolor='none')
        ax.add_patch(rect)
        ax.axis('off')  # Hide axes
        # Save the modified image
        plt.savefig(os.path.join(output_dir, row['Image']), bbox_inches='tight', pad_inches=0)
        plt.close()

# Apply the function to each row in the dataframe
# df.apply(save_annotated_images, axis=1)
print(df.head(15))