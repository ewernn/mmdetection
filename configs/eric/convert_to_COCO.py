

def split_data(df, test_size=0.1):
    # Split dataset into training and temp (for further splitting into validation and test)
    train_df, temp_df = train_test_split(df, test_size=test_size*2, random_state=42)
    # Split temp into validation and test
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    return train_df, val_df, test_df

def create_coco_format_subset(df, images_dir, output_json_path, subset_name):
    images_0_1_2 = []
    annotations_0_1_2 = []
    images_1_2 = []
    annotations_1_2 = []
    images_2 = []
    annotations_2 = []
    categories = [{'id': 1, 'name': 'vertebrae'}]

    annotation_id = 1
    for idx, row in df.iterrows():
        image_filename = row['Image']
        image_path = os.path.join(images_dir, image_filename)
        with Image.open(image_path) as img:
            width, height = img.size

        image_info = {
            'id': idx + 1,
            'width': width,
            'height': height,
            'file_name': image_filename,
        }

        num_objects = 0
        for bbox_idx in range(1, len(row), 2):
            if pd.isna(row[f'x{bbox_idx}']) or row[f'x{bbox_idx}'] == -1.0:
                break
            x_min = max(0, row[f'x{bbox_idx}'] * width)
            y_min = max(0, row[f'y{bbox_idx}'] * height)
            if bbox_idx+1 < len(row) and not pd.isna(row[f'x{bbox_idx+1}']) and row[f'x{bbox_idx+1}'] != -1.0:
                x_max = min(width, row[f'x{bbox_idx+1}'] * width)
                y_max = min(height, row[f'y{bbox_idx+1}'] * height)
                if x_min >= width or y_min >= height or x_max <= 0 or y_max <= 0 or (x_max - x_min) <= 0 or (y_max - y_min) <= 0:
                    continue
                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                annotation = {
                    'id': annotation_id,
                    'image_id': idx + 1,
                    'category_id': 1,
                    'bbox': bbox,
                    'area': (x_max - x_min) * (y_max - y_min),
                    'iscrowd': 0,
                    'segmentation': [],
                }
                annotation_id += 1
                num_objects += 1
                if num_objects <= 2:
                    annotations_0_1_2.append(annotation)
                    annotations_1_2.append(annotation)
                if num_objects == 2:
                    annotations_2.append(annotation)

        images_0_1_2.append(image_info)
        if num_objects >= 1:
            images_1_2.append(image_info)
        if num_objects == 2:
            images_2.append(image_info)

    for images, annotations, suffix in [
        (images_0_1_2, annotations_0_1_2, '0_1_2'),
        (images_1_2, annotations_1_2, '1_2'),
        (images_2, annotations_2, '2')
    ]:
        coco_format_json = {
            'images': images,
            'annotations': annotations,
            'categories': categories,
        }
        subset_output_path = os.path.join(output_json_path, f'{subset_name}_Data_coco_format_{suffix}.json')
        with open(subset_output_path, 'w') as f:
            json.dump(coco_format_json, f, indent=4)

def create_coco_format(data_csv_path, images_dir, output_json_path):
    # Load the dataset
    df = pd.read_csv(data_csv_path)
    
    # Split the DataFrame into train, val, and test
    train_df, val_df, test_df = split_data(df)
    
    # Generate COCO format JSON files for each subset
    for subset_name, subset_df in zip(['train', 'val', 'test'], [train_df, val_df, test_df]):
        create_coco_format_subset(subset_df, images_dir, output_json_path, subset_name)

# Usage
images_dir = '/Users/ewern/Desktop/code/MetronMind/data/cat-kidneys'               # Update with your actual path to images directory
data_csv_path = os.path.join(images_dir, 'Data.csv')  # Update with your actual path
output_json_path = '/Users/ewern/Desktop/code/MetronMind/data'  # Update with your desired output path

create_coco_format(data_csv_path, images_dir, output_json_path)
