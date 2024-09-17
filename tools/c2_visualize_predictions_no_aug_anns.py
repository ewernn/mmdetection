import numpy as np
import torch
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
from pycocotools.coco import COCO
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def create_model(device, num_classes):
    # Define the backbone and modifications as in train_c2.py
    backbone = resnet_fpn_backbone(backbone_name='resnet152', weights=None)
    anchor_sizes = (
        (200, 300),   # Small objects
        (300, 375),   # Medium-small objects
        (415, 450),   # Medium objects
        (550, 525),   # Medium-large objects
        (750, 615),   # Large objects
    )
    aspect_ratios = ((0.5, 0.7, 0.9, 1.0, 1.15, 1.35, 1.75, 2.4, 3.3),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    model = FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator)
    model.to(device)

    # Modify the model as done in training
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Apply other modifications as necessary
    model.rpn.nms_thresh = 0.7
    model.rpn.fg_iou_thresh = 0.7
    model.rpn.bg_iou_thresh = 0.3
    model.roi_heads.batch_size_per_image = 16 # 32
    model.roi_heads.positive_fraction = 0.25
    model.roi_heads.score_thresh = 0.6 # 0.3 # 0.5 or higher) to reduce false positives. higher threshold can help eliminate low-confidence predictions
    model.roi_heads.nms_thresh = 0.4 # 0.5 # lower --> if multiple boxes are detected, only the most confident one is kept
    model.roi_heads.detections_per_img = 1

    # Set pre_nms_top_n and post_nms_top_n
    model.rpn.pre_nms_top_n = lambda: 150
    model.rpn.post_nms_top_n = lambda: 20

    return model

def load_model(model_path, device, num_classes):
    # Create the model with the same architecture and modifications
    model = create_model(device, num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def get_dataset(coco_path, img_dir):
    coco = COCO(coco_path)
    img_ids = coco.getImgIds()
    return coco, img_ids

def visualize_and_save(image, gt_boxes, pred_boxes, pred_scores, image_id, save_dir):
    # Convert tensor image to PIL Image
    image_pil = to_pil_image(image.cpu())

    # Create figure and axis
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image_pil, cmap='gray')

    # Draw ground truth boxes in green
    for box in gt_boxes:
        x_min, y_min, width, height = box
        x_max = x_min + width
        y_max = y_min + height
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)

    # Draw predicted boxes in red and display scores
    for box, score in zip(pred_boxes, pred_scores):
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x_max, y_max, f'{score:.3f}', color='red', fontsize=10, verticalalignment='top', backgroundcolor='white')

    ax.set_xticks([])
    ax.set_yticks([])
    plt.title(f'Image ID: {image_id}')
    plt.savefig(os.path.join(save_dir, f'image_{image_id}.png'), bbox_inches='tight')
    plt.close(fig)

def main(model_path, coco_path, img_dir, save_dir, device):
    num_classes = 2
    model = load_model(model_path, device, num_classes)
    coco, img_ids = get_dataset(coco_path, img_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        path = img_info['file_name']
        image = Image.open(os.path.join(img_dir, path)).convert("L")  # Convert image to grayscale
        image = image.convert("RGB")  # Convert grayscale image to RGB by replicating channels
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        gt_boxes = [ann['bbox'] for ann in anns]  # [x, y, width, height]
        model.to(device)
        with torch.no_grad():
            prediction = model(image_tensor)[0]

        pred_boxes = prediction['boxes'].cpu().numpy()
        pred_scores = prediction['scores'].cpu().numpy()
        visualize_and_save(image_tensor.squeeze(0), gt_boxes, pred_boxes, pred_scores, img_id, save_dir)


if __name__ == "__main__":
    c2 = '/Users/ewern/Desktop/code/MetronMind/c2/'
    #model_path = '/content/drive/MyDrive/MM/c2/exps/u0iy3e9k/best_model.pth'
    # model_path = '/content/drive/MyDrive/MM/c2/exps/hdsu9uww/best_model.pth'
    model_path = '/content/drive/MyDrive/MM/c2/exps/jbuf7hg1/best_model.pth'
    coco_path = '/content/drive/MyDrive/MM/c2/data/Test/all_images_Data_coco_format.json'
    img_dir = '/content/drive/MyDrive/MM/c2/data/Test/'
    save_dir = '/content/drive/MyDrive/MM/c2/exps/aug12_preds_out-jbuf7hg1/'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(model_path, coco_path, img_dir, save_dir, device)