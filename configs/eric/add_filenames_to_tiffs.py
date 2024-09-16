import os
from PIL import Image, ImageDraw, ImageFont

def add_filename_to_image(image_path, output_path):
    # Open the image
    with Image.open(image_path) as img:
        # Create a drawing object
        draw = ImageDraw.Draw(img)
        
        # Get the filename
        filename = os.path.basename(image_path)
        
        # Choose a font (you may need to specify a different font path)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        except IOError:
            font = ImageFont.load_default()
        
        # Add the text
        draw.text((10, 10), filename, fill="white", font=font)
        
        # Save the image
        img.save(output_path)

def process_images(input_dir, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each .tif file in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.tif'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            add_filename_to_image(input_path, output_path)
            print(f"Processed: {filename}")

if __name__ == "__main__":
    input_dir = '/Users/ewern/Desktop/code/MetronMind/data/cat-dataset'
    output_dir = '/Users/ewern/Desktop/code/MetronMind/data/cat-dataset-with-names'
    process_images(input_dir, output_dir)