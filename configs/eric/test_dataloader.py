import os
import torch
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Import your CocoDataset class and collate_fn
from tools.train_cat_kidney import CocoDataset, collate_fn, get_transform

def create_model(num_classes):
    backbone = resnet_fpn_backbone('resnet152', pretrained=False)
    model = FasterRCNN(backbone, num_classes=num_classes)
    return model

def visualize_sample(img, target, predictions, save_path=None):
    """Visualize an image with its ground truth and predicted bounding boxes."""
    fig, ax = plt.subplots(1, figsize=(12, 8))
    img = img.permute(1, 2, 0).cpu().numpy()
    ax.imshow(img, cmap='gray')  # Use grayscale colormap

    # Draw ground truth boxes
    for box, label in zip(target['boxes'], target['labels']):
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='green', linewidth=2)
        ax.add_patch(rect)
        ax.text(x1, y1, f"GT: Class {label}", color='green', fontsize=8, bbox=dict(facecolor='white', alpha=0.5))

    # Draw predicted boxes
    for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
        if score > 0.5:  # Only show predictions with confidence > 0.5
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, y1+20, f"Pred: Class {label} ({score:.2f})", color='red', fontsize=8, bbox=dict(facecolor='white', alpha=0.5))

    plt.title(f"Image ID: {target['image_id'].item()}")
    if save_path:
        plt.savefig(save_path)
    plt.close()

def load_model(model_path, num_classes):
    checkpoint = torch.load(model_path, map_location='mps')
    
    if 'model_state_dict' in checkpoint:
        # If the checkpoint contains a 'model_state_dict' key
        model = create_model(num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        # If the checkpoint is a dict with a 'state_dict' key
        model = create_model(num_classes)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # If the checkpoint is the state_dict itself
        model = create_model(num_classes)
        model.load_state_dict(checkpoint)
    
    return model

def test_dataloader():
    # Set your paths
    data_root = '/Users/ewern/Desktop/code/MetronMind/data/cat-dataset-with-names/'
    data_root = '/Users/ewern/Desktop/code/MetronMind/data/cat-dataset/'
    ann_file = data_root + 'COCO_2/val_Data_coco_format.json'
    model_path = '/Users/ewern/Downloads/best_model.pth'  # Update this to your model path

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dataset
    dataset = CocoDataset(data_root, ann_file, transforms=get_transform(train=False), preload=False)

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)

    # Load the model
    num_classes = 3  # Update this if your number of classes is different
    # model = create_model(num_classes)
    # model.load_state_dict(torch.load(model_path, map_location=device))
    model = load_model(model_path, num_classes)
    model.to(device)
    model.eval()

    # Test loop
    with torch.no_grad():
        for i, (images, targets) in enumerate(dataloader):
            if i >= 5:  # Test with 5 images
                break

            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Get predictions
            predictions = model(images)[0]

            # Filter predictions
            keep = torch.where(predictions['scores'] > 0.5)[0]  # Adjust threshold as needed
            filtered_predictions = {
                'boxes': predictions['boxes'][keep][:2],  # Keep top 2 predictions
                'labels': predictions['labels'][keep][:2],
                'scores': predictions['scores'][keep][:2]
            }

            image = images[0].cpu()  # Get the first (and only) image in the batch
            target = {k: v.cpu() for k, v in targets[0].items()}  # Get the first (and only) target in the batch

            print(f"Sample {i + 1}:")
            print(f"Image ID: {target['image_id'].item()}")
            print(f"Image shape: {image.shape}")
            print(f"Number of ground truth objects: {len(target['boxes'])}")
            print(f"Number of predicted objects (after filtering): {len(filtered_predictions['boxes'])}")
            print(f"Predicted scores: {filtered_predictions['scores']}")
            print("\n")

            # Visualize and save the image
            sp = '/Users/ewern/Desktop/code/MetronMind/data/'
            save_path = f'{sp}sample_with_predictions_{i+1}.png'
            visualize_sample(image, target, filtered_predictions, save_path)

    print("Dataloader test completed. Check the saved images for visualization.")

if __name__ == "__main__":
    test_dataloader()