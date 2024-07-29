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
    backbone = resnet_fpn_backbone('resnet152', pretrained=False)  # Adjust the backbone model as per training configuration
    anchor_sizes = ((317, 428), (432, 528), (506, 536), (579, 544), (687, 543))
    aspect_ratios = ((0.5, 0.9, 1.0, 1.2, 1.4, 1.8),) * 5
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    model = FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator)
    model.to(device)

    # Modify the model as done in training
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Apply other modifications as necessary
    model.rpn.nms_thresh = 0.8
    model.rpn.fg_iou_thresh = 0.6
    model.rpn.bg_iou_thresh = 0.3
    model.roi_heads.batch_size_per_image = 256
    model.roi_heads.positive_fraction = 0.5
    model.roi_heads.score_thresh = 0.3
    model.roi_heads.nms_thresh = 0.4
    model.roi_heads.detections_per_img = 4

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

def visualize_and_save(image, gt_boxes, pred_boxes, image_id, save_dir):
    # Convert tensor image to PIL Image
    image_pil = to_pil_image(image.cpu())

    # Create figure and axis
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image_pil, cmap='gray')

    # Draw ground truth boxes in green
    for box in gt_boxes:
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)

    # Draw predicted boxes in red
    for box in pred_boxes:
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    ax.set_xticks([])
    ax.set_yticks([])
    plt.title(f'Image ID: {image_id}')
    plt.savefig(os.path.join(save_dir, f'image_{image_id}.png'), bbox_inches='tight')
    plt.close(fig)

def main(model_path, coco_path, img_dir, save_dir, device):
    num_classes = 3
    model = load_model(model_path, device, num_classes)
    coco, img_ids = get_dataset(coco_path, img_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        path = img_info['file_name']
        image = Image.open(os.path.join(img_dir, path)).convert("RGB")
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        gt_boxes = [ann['bbox'] for ann in anns]  # [x, y, width, height]

        with torch.no_grad():
            prediction = model(image_tensor)[0]

        pred_boxes = prediction['boxes'].cpu().numpy()
        visualize_and_save(image_tensor.squeeze(0), gt_boxes, pred_boxes, img_id, save_dir)

if __name__ == "__main__":
    c2 = '/Users/ewern/Desktop/code/MetronMind/c2/'
    model_path = '/content/drive/MyDrive/MM/c2/exps/u0iy3e9k/best_model.pth'
    coco_path = '/content/drive/MyDrive/MM/c2/data/Test/all_images_Data_coco_format.json'
    img_dir = '/content/drive/MyDrive/MM/c2/data/Test/'
    save_dir = '/content/drive/MyDrive/MM/c2/exps/july29_preds_out/'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(model_path, coco_path, img_dir, save_dir, device)