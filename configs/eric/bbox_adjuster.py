import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

def clean_repeated_values(value):
    if pd.isna(value):  # Check if the value is NaN
        return value
    if isinstance(value, str):
        try:
            # Try to convert to float directly first
            return float(value)
        except ValueError:
            # If that fails, try to clean repeated values
            try:
                # Find the first occurrence of a repeated pattern
                pattern = value[:8]
                end_index = value.index(pattern, 8)
                return float(value[:end_index])
            except ValueError:
                # If no repetition is found, return NaN silently
                return np.nan
    return value

def draw_dashed_line(img, start, end, color, thickness=1, dash_length=5):
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    distance = max(abs(dx), abs(dy))
    steps = int(distance / dash_length)
    for i in range(steps):
        start_point = (int(x1 + dx * i / steps), int(y1 + dy * i / steps))
        end_point = (int(x1 + dx * (i + 0.5) / steps), int(y1 + dy * (i + 0.5) / steps))
        cv2.line(img, start_point, end_point, color, thickness)

def adjust_bboxes(csv_path, images_dir, start_index=0):
    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"CSV file read. Shape: {df.shape}")
    print(f"Columns: {df.columns}")
    
    # Clean repeated values
    for col in ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']:
        df[col] = df[col].apply(clean_repeated_values)
    
    # Filter for images with both kidneys (8 non-null values)
    both_kidneys = df.dropna(subset=['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4'])
    total_images = len(both_kidneys)
    print(f"Number of images with both kidneys: {total_images}")
    
    drawing = False
    current_bbox = None
    start_point = None
    temp_bbox = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, current_bbox, start_point, temp_bbox, img_copy

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = (x, y)
            temp_bbox = None
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                img_copy = base_img.copy()
                draw_dashed_line(img_copy, start_point, (x, start_point[1]), (255, 0, 0), 1)
                draw_dashed_line(img_copy, (x, start_point[1]), (x, y), (255, 0, 0), 1)
                draw_dashed_line(img_copy, (x, y), (start_point[0], y), (255, 0, 0), 1)
                draw_dashed_line(img_copy, (start_point[0], y), start_point, (255, 0, 0), 1)
                cv2.imshow('Image', img_copy)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            temp_bbox = [min(start_point[0], x), min(start_point[1], y),
                         max(start_point[0], x), max(start_point[1], y)]
            img_copy = base_img.copy()
            cv2.rectangle(img_copy, (temp_bbox[0], temp_bbox[1]), (temp_bbox[2], temp_bbox[3]), (255, 0, 0), 1)
            cv2.imshow('Image', img_copy)

    idx = start_index - 1  # Start one before, as we increment at the beginning of the loop
    while idx < total_images - 1:
        idx += 1
        row = both_kidneys.iloc[idx]
        
        image_path = os.path.join(images_dir, row['Image'])
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to read image: {image_path}")
            continue
        h, w = img.shape[:2]
        
        # Switch bbox1 and bbox2 to correct left-right orientation
        bbox2 = [int(row['x1']*w), int(row['y1']*h), int(row['x2']*w), int(row['y2']*h)]  # Left kidney
        bbox1 = [int(row['x3']*w), int(row['y3']*h), int(row['x4']*w), int(row['y4']*h)]  # Right kidney
        
        cv2.namedWindow('Image')
        cv2.setMouseCallback('Image', mouse_callback)

        base_img = img.copy()
        cv2.rectangle(base_img, (bbox1[0], bbox1[1]), (bbox1[2], bbox1[3]), (0, 0, 255), 1)  # Right kidney (red)
        cv2.rectangle(base_img, (bbox2[0], bbox2[1]), (bbox2[2], bbox2[3]), (0, 255, 0), 1)  # Left kidney (green)
        cv2.putText(base_img, f"Image: {row['Image']}", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(base_img, f"Progress: {idx+1}/{total_images}", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        img_copy = base_img.copy()
        cv2.imshow('Image', img_copy)

        current_bbox = None
        temp_bbox = None
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):  # Quit
                print(f"Quitting. Last processed image index: {idx}")
                return df, idx
            elif key == ord('n'):  # Next image
                break
            elif key == ord('b'):  # Go back to previous image
                idx = max(0, idx - 2)  # Go back 2 because we increment at the start of the loop
                break
            elif key == ord('1'):  # Adjust bbox1 (right kidney)
                current_bbox = bbox1
                temp_bbox = None
                img_copy = base_img.copy()
                cv2.imshow('Image', img_copy)
            elif key == ord('2'):  # Adjust bbox2 (left kidney)
                current_bbox = bbox2
                temp_bbox = None
                img_copy = base_img.copy()
                cv2.imshow('Image', img_copy)
            elif key == ord('c'):  # Confirm new bbox
                if temp_bbox and current_bbox:
                    current_bbox[:] = temp_bbox[:]
                    temp_bbox = None
                    base_img = img.copy()
                    cv2.rectangle(base_img, (bbox1[0], bbox1[1]), (bbox1[2], bbox1[3]), (0, 0, 255), 1)
                    cv2.rectangle(base_img, (bbox2[0], bbox2[1]), (bbox2[2], bbox2[3]), (0, 255, 0), 1)
                    cv2.putText(base_img, f"Image: {row['Image']}", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    cv2.putText(base_img, f"Progress: {idx+1}/{total_images}", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    img_copy = base_img.copy()
                    cv2.imshow('Image', img_copy)
                    # Update DataFrame immediately
                    update_cols = ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']
                    df.loc[df['Image'] == row['Image'], update_cols] = [
                        bbox2[0]/w, bbox2[1]/h, bbox2[2]/w, bbox2[3]/h,  # Left kidney
                        bbox1[0]/w, bbox1[1]/h, bbox1[2]/w, bbox1[3]/h   # Right kidney
                    ]
                    df.to_csv(csv_path, index=False)  # Save changes immediately
            elif key == ord('r'):  # Reset/cancel new bbox
                temp_bbox = None
                img_copy = base_img.copy()
                cv2.imshow('Image', img_copy)
            elif key == ord('o'):  # Mark as outlier
                update_cols = ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']
                df.loc[df['Image'] == row['Image'], update_cols] = np.nan  # Set all bbox values to NaN
                df.to_csv(csv_path, index=False)  # Save changes immediately
                print(f"Image {row['Image']} marked as outlier.")
                break  # Move to next image
            elif key == ord('j'):  # Jump to specific image
                jump_to = input("Enter the index of the image you want to jump to: ")
                try:
                    jump_to = int(jump_to)
                    if 0 <= jump_to < total_images:
                        idx = jump_to - 1  # -1 because we increment at the start of the loop
                        break
                    else:
                        print("Invalid index. Continuing with current image.")
                except ValueError:
                    print("Invalid input. Continuing with current image.")
    
    cv2.destroyAllWindows()
    return df, idx

# Usage
csv_path = '/Users/ewern/Desktop/code/MetronMind/data/cat-dataset/Data.csv'
images_dir = '/Users/ewern/Desktop/code/MetronMind/data/cat-dataset'

while True:
    start_index = 0  # Always start from the beginning
    updated_df, last_index = adjust_bboxes(csv_path, images_dir, start_index)
    updated_df.to_csv('/Users/ewern/Desktop/code/MetronMind/data/cat-dataset/Updated_Data.csv', index=False)
    
    print(f"Last processed image index: {last_index}")
    print("Program finished. You can run it again to start from the beginning.")
    
    run_again = input("Do you want to run the program again? (y/n): ")
    if run_again.lower() != 'y':
        break

print("Program terminated.")