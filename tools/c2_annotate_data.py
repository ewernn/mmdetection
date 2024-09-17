import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
import numpy as np
from matplotlib.widgets import RectangleSelector

# Base directory path
base_dir = '/Users/ewern/Desktop/code/MetronMind/c2/data/Train'

# Read the CSV file
df = pd.read_csv(os.path.join(base_dir, 'Data.csv'))

# Convert columns to float, handling non-numeric values
df[['x1', 'y1', 'x2', 'y2']] = df[['x1', 'y1', 'x2', 'y2']].apply(pd.to_numeric, errors='coerce')

# # Directory to save output images
# output_dir = '/Users/ewern/Downloads/C2_imgs_out_test'
# os.makedirs(output_dir, exist_ok=True)

def onselect(eclick, erelease):
    global current_bbox
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    current_bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

def annotate_images(df):
    global current_bbox
    index = 0

    def on_key(event):
        nonlocal key_pressed
        key_pressed = event.key

    while index < len(df):
        row = df.iloc[index]
        image_path = os.path.join(base_dir, row['Image'])
        img = Image.open(image_path).convert('L')
        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')
        
        current_bbox = None
        rect = patches.Rectangle((0,0), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
        # Draw ground truth bbox if it exists
        if not pd.isna(row['x1']):
            width, height = img.size
            gt_rect = patches.Rectangle((row['x1']*width, row['y1']*height), 
                                        (row['x2']-row['x1'])*width, 
                                        (row['y2']-row['y1'])*height,
                                        linewidth=1, edgecolor='g', facecolor='none')
            ax.add_patch(gt_rect)
        
        selector = RectangleSelector(ax, onselect, useblit=True,
                                     button=[1], minspanx=5, minspany=5,
                                     spancoords='pixels', interactive=True)
        
        plt.title(f"Image: {row['Image']} ({index+1}/{len(df)})")
        plt.text(0.5, -0.1, "Controls: c (continue/save), d (delete), r (remove bbox), b (back)", 
                 ha='center', transform=ax.transAxes)
        
        key_pressed = None
        fig.canvas.mpl_connect('key_press_event', on_key)
        
        while True:
            plt.draw()
            plt.pause(0.1)  # Small pause to allow GUI events to process
            
            if key_pressed:
                if key_pressed == 'c':
                    if current_bbox:
                        width, height = img.size
                        df.at[df.index[index], 'x1'] = current_bbox[0] / width
                        df.at[df.index[index], 'y1'] = current_bbox[1] / height
                        df.at[df.index[index], 'x2'] = current_bbox[2] / width
                        df.at[df.index[index], 'y2'] = current_bbox[3] / height
                    plt.close()
                    index += 1
                    break
                elif key_pressed == 'd':
                    df = df.drop(df.index[index])
                    plt.close()
                    break
                elif key_pressed == 'r':
                    df.at[df.index[index], 'x1'] = np.nan
                    df.at[df.index[index], 'y1'] = np.nan
                    df.at[df.index[index], 'x2'] = np.nan
                    df.at[df.index[index], 'y2'] = np.nan
                    plt.close()
                    index += 1
                    break
                elif key_pressed == 'b':
                    if index > 0:
                        index -= 1
                    plt.close()
                    break
                
                key_pressed = None  # Reset the key press
            
            if current_bbox:
                rect.set_bounds(current_bbox[0], current_bbox[1], 
                                current_bbox[2]-current_bbox[0], 
                                current_bbox[3]-current_bbox[1])
                fig.canvas.draw_idle()
    
    return df

# Apply the annotation function
df = annotate_images(df)

# Save the updated DataFrame
df.to_csv(os.path.join(base_dir, 'Data_updated.csv'), index=False)

print("Annotation complete. Updated CSV saved.")