import os
import json
import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
import wandb
import sys
import math
import argparse
import random
import time
from torchvision.models import ResNet101_Weights, ResNet152_Weights, ResNet50_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.cuda.amp import GradScaler, autocast	
import torchvision
import psutil
import GPUtil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms.functional import to_pil_image
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models.detection.rpn import AnchorGenerator
import ast
from torch.nn.utils import clip_grad_norm_


# Initialize global variables
use_wandb = False
use_colab = False

class CocoDataset(Dataset):
    def __init__(self, root, annFile, transforms=None, preload=False, only_10=False):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        if only_10:
            random.shuffle(self.ids)
            self.ids = self.ids[:10]
        self.transforms = transforms
        self.preload = preload
        self.images = {}
        if self.preload:
            self._preload_images()
    def _preload_images(self):
        for img_id in self.ids:
            img_info = self.coco.loadImgs(img_id)[0]
            path = img_info['file_name']
            img = Image.open(os.path.join(self.root, path)).convert("L")  # Convert to grayscale
            self.images[img_id] = img

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        img_info = coco.loadImgs(img_id)[0]
        path = img_info['file_name']
        if self.preload:
            img = self.images[img_id]
        else:
            img = Image.open(os.path.join(self.root, path)).convert("L")  # Convert to grayscale
        num_objs = len(anns)
        boxes = []
        labels = []
        for i in range(num_objs):
            xmin = anns[i]['bbox'][0]
            ymin = anns[i]['bbox'][1]
            xmax = xmin + anns[i]['bbox'][2]
            ymax = ymin + anns[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])  # go from (x1,y1,w,h) to (x1,y1,x2,y2)
            labels.append(anns[i]['category_id'])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([img_id])  # Add image_id to target
        if self.transforms is not None:
            img = self.transforms(img)
        return img, target
    
    def __len__(self):
        return len(self.ids)

def adjust_brightness(img):
    return TF.adjust_brightness(img, brightness_factor=random.uniform(0.6, 1.4))

def expand_channels(img):
    return img.repeat(3, 1, 1) if img.shape[0] == 1 else img

def adjust_contrast(img):
    return TF.adjust_contrast(img, contrast_factor=random.uniform(0.5, 1.5))

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.extend([
            T.RandomAffine(
                degrees=(-20, 20),
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                fill=0
            ),
            T.Lambda(adjust_brightness),
            T.RandomAutocontrast(p=0.5),
            T.Lambda(adjust_contrast),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            T.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        ])
    # Expand grayscale to 3 channels
    transforms.append(T.Lambda(expand_channels))
    return T.Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))

def visualize_boxes(image, gt_boxes, gt_labels, pred_boxes, pred_labels, image_id, save_path):
    """
    Visualize ground truth and predicted bounding boxes on the image and save it.
    """

    if pred_boxes.size(0) == 0:
        print("No predicted boxes to visualize.")
        return

    # Convert tensor image to PIL Image
    image_pil = to_pil_image(image.cpu())
    
    # Move tensors to CPU for visualization
    gt_boxes = gt_boxes.cpu()
    gt_labels = gt_labels.cpu()
    pred_boxes = pred_boxes.cpu()
    pred_labels = pred_labels.cpu()

    # Create figure and axis
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image_pil)
    
    # Draw ground truth boxes in green
    for box, label in zip(gt_boxes, gt_labels):
        x, y, w, h = box[0].item(), box[1].item(), (box[2]-box[0]).item(), (box[3]-box[1]).item()
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y, f'GT: {label.item()}', color='g', fontsize=10, verticalalignment='top')
    
    # Draw predicted boxes in red
    if len(pred_boxes) > 0:
        for box, label in zip(pred_boxes, pred_labels):
            x, y, w, h = box[0].item(), box[1].item(), (box[2]-box[0]).item(), (box[3]-box[1]).item()
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y, f'Pred: {label.item()}', color='r', fontsize=10, verticalalignment='top')
    else:
        ax.text(10, 10, 'No predictions', color='r', fontsize=12, verticalalignment='top')
    
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title(f'Image ID: {image_id}')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

def filter_kidney_predictions(boxes, scores, labels, iou_threshold=0.5):
    '''
    boxes has shape torch.tensor((10,4))
    scores has shape torch.tensor((10,))
    labels has shape torch.tensor((10,))
    '''
    # Check if there are no predictions
    if boxes.shape[0] == 0:
        return boxes, scores, labels

    # Create a dictionary to store the best prediction for each class
    best_indices = {}

    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        label = label.item()
        if label not in best_indices or score > scores[best_indices[label]]:
            best_indices[label] = i

    # Select the best predictions
    best_indices = list(best_indices.values())
    best_boxes = boxes[best_indices]
    best_scores = scores[best_indices]
    best_labels = labels[best_indices]

    return best_boxes, best_scores, best_labels

def evaluate(model, data_loader, device, epoch):
    model.eval()
    coco = data_loader.dataset.coco
    coco_results = []
    # Create directory for saving images
    if use_colab:
        save_dir = f'/content/drive/MyDrive/MM/CatKidney/exps/imgs_out/epoch_{epoch}'
    else:
        save_dir = f'exps/images_with_predicted_bboxes/epoch_{epoch}'
    os.makedirs(save_dir, exist_ok=True)
    # loop thru eval set
    image_count = 0
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        
        with torch.no_grad():
            outputs = model(images)
        
        for i, (target, output) in enumerate(zip(targets, outputs)):  # for image 'i' in batch
            image_id = target["image_id"].item()
            boxes = output["boxes"]  # tensor(10,4)
            scores = output["scores"]  # tensor(10,)
            labels = output["labels"]  # tensor(10,)
            
            # Apply the filtering
            boxes, scores, labels = filter_kidney_predictions(boxes, scores, labels)
            
            # # Apply NMS
            # keep = torchvision.ops.nms(boxes, scores, iou_threshold=0.9);boxes = boxes[keep];scores = scores[keep];labels = labels[keep]

            print(f"BEFORE COCO APPEND: boxes: {boxes}, scores: {scores}, labels: {labels}")
            if len(boxes) == 0 or len(scores) == 0 or len(labels) == 0:
                print("No detections to process.")
            else:
                for box, score, label in zip(boxes, scores, labels):
                    coco_results.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])],
                        "score": float(score),
                    })
            
            if image_count < 2e2:
                #print(f"\nImage ID: {image_id}  ||  # filtered detections: {len(boxes)}  ||  Scores: {scores}  ||  Labels: {labels}")
                # Visualize and save image
                save_path = os.path.join(save_dir, f'image_{image_id}.png')
                #print(f"Visualizing Image ID: {image_id}  ||  # filtered detections: {len(boxes)}  ||  Scores: {scores}  ||  Labels: {labels}  ||  Save Path: {'/'.join(save_path.split('/')[-4:])}")
                #print(f"GT Boxes: {target['boxes']}, GT Labels: {target['labels']}\nPred Boxes: {boxes}, Pred Labels: {labels}")
                visualize_boxes(images[i], target["boxes"], target["labels"], boxes, labels, image_id, save_path)
                image_count += 1
    
    print(f"Total number of results: {len(coco_results)}")
    
    if len(coco_results) == 0:
        print("No valid detections found. Returning 0 mAP.")
        return {"mAP": 0.0, "AP_50": 0.0, "AP_75": 0.0}  # Return a dictionary with zero values
    
    # Print a sample result
    print("Sample detection result:")
    print(json.dumps(coco_results[0], indent=2))
    
    coco_dt = coco.loadRes(coco_results)
    coco_eval = COCOeval(coco, coco_dt, "bbox")
    
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Print more detailed evaluation metrics
    print(f"AP @ IoU=0.50:0.95: {coco_eval.stats[0]}")
    print(f"AP @ IoU=0.50: {coco_eval.stats[1]}")
    print(f"AP @ IoU=0.75: {coco_eval.stats[2]}")

    # Extract more detailed metrics
    metrics = {
        "mAP": coco_eval.stats[0],  # AP @ IoU=0.50:0.95
        "AP_50": coco_eval.stats[1],  # AP @ IoU=0.50
        "AP_75": coco_eval.stats[2],  # AP @ IoU=0.75
        "AP_small": coco_eval.stats[3],  # AP for small objects
        "AP_medium": coco_eval.stats[4],  # AP for medium objects
        "AP_large": coco_eval.stats[5],  # AP for large objects
        "AR_max_1": coco_eval.stats[6],  # AR given 1 detection per image
        "AR_max_10": coco_eval.stats[7],  # AR given 10 detections per image
        "AR_max_100": coco_eval.stats[8],  # AR given 100 detections per image
        "AR_small": coco_eval.stats[9],  # AR for small objects
        "AR_medium": coco_eval.stats[10],  # AR for medium objects
        "AR_large": coco_eval.stats[11],  # AR for large objects
    }

    # Log per-class AP if available
    if hasattr(coco_eval, 'eval') and 'precision' in coco_eval.eval:
        precisions = coco_eval.eval['precision']
        # precisions has shape (iou, recall, cls, area range, max dets)
        for idx, cat_id in enumerate(coco_eval.params.catIds):
            metrics[f"AP_class_{cat_id}"] = np.mean(precisions[:, :, idx, 0, -1])

    # Print more detailed evaluation metrics
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    if use_wandb:
        wandb.log(metrics)

    return metrics

def manual_gradient_clipping(parameters, max_norm):
    total_norm = 0
    for param in parameters:
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)  # Adding a small epsilon to avoid division by zero
        for param in parameters:
            if param.grad is not None:
                param.grad.data.mul_(scale)
    
    return total_norm

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, localization_weight=2.0, max_grad_norm=1.0):
    model.train()
    use_amp = device == 'cuda'  # Use automatic mixed precision only if CUDA is available
    use_amp = device == 'cum'  # Use automatic mixed precision only if CUDA is available
    scaler = GradScaler(enabled=use_amp)
    total_loss = 0
    num_batches = 0
    start_time = time.time()

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if use_amp:
            with autocast():
                loss_dict = model(images, targets)
                # Increase the weight of the localization losses
                loss_dict['loss_box_reg'] *= localization_weight
                loss_dict['loss_rpn_box_reg'] *= localization_weight
                losses = sum(loss for loss in loss_dict.values())
            scaler.scale(losses).backward()
            # Calculate and log gradient norms
            total_norm = manual_gradient_clipping(model.parameters(), max_grad_norm)
            print(f"Epoch {epoch}, Batch {batch_idx}, Gradient Norm: {total_norm:.4f}")
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            if batch_idx == 0:
                print(f"Loss Dict: {loss_dict}")
            # Increase the weight of the localization losses
            loss_dict['loss_box_reg'] *= localization_weight
            loss_dict['loss_rpn_box_reg'] *= localization_weight
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            parameters = list(model.parameters())
            total_norm_before = manual_gradient_clipping(parameters, float('inf'))
            total_norm = manual_gradient_clipping(parameters, max_grad_norm)
            # Calculate and log gradient norms before clipping
            #total_norm_before = manual_gradient_clipping(model.parameters(), float('inf'))
            print(f"Total norm before clipping: {total_norm_before:.4f}")
            # Apply gradient clipping
            print(f"max_grad_norm: {max_grad_norm}")
            print(f"model.parameters(): {model.parameters()}")
            #total_norm = manual_gradient_clipping(model.parameters(), max_grad_norm)
            print(f"Total norm after clipping: {total_norm:.4f}")
            optimizer.step()

        optimizer.zero_grad()  # Ensure gradients are zeroed after each batch
        total_loss += losses.item()
        num_batches += 1

        if batch_idx % print_freq == 0:
            avg_loss = total_loss / num_batches
            elapsed_time = time.time() - start_time
            images_per_sec = (batch_idx + 1) * len(images) / elapsed_time
            print(f"Epoch [{epoch}][{batch_idx}/{len(data_loader)}] "
                  f"Loss: {avg_loss:.4f} "
                  f"Images/sec: {images_per_sec:.1f}")

            for img_idx, (img, target) in enumerate(zip(images, targets)):
                print(f"Training on Image ID: {target['image_id'].item()} (Batch {batch_idx}, Item {img_idx})")

    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch} complete. Average Loss: {avg_loss:.4f}")

    return avg_loss


def parse_tuple(argument):
    try:
        return ast.literal_eval(argument)
    except ValueError as e:
        raise argparse.ArgumentTypeError("Invalid tuple: %s" % (e,))

def create_model(args, num_classes):
    weights = ResNet50_Weights.DEFAULT
    if args.backbone == 'resnet101': weights = ResNet101_Weights.DEFAULT
    if args.backbone == 'resnet152': weights = ResNet152_Weights.DEFAULT

    backbone = resnet_fpn_backbone(backbone_name=args.backbone, weights=weights, trainable_layers=3)

    # AnchorGenerator
    anchor_sizes = ((161,), (192,), (219,), (252,), (311,))
    aspect_ratios = ((1.5, 2.0, 2.5),) * 5
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    model = FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator)

    # Set the trainable layers based on args.freeze_layers
    for name, parameter in model.backbone.body.named_parameters():
        if 'layer1' in name and 'freeze_layer1' in args.freeze_layers:
            parameter.requires_grad = False
        elif 'layer2' in name and 'freeze_layer2' in args.freeze_layers:
            parameter.requires_grad = False
        else:
            parameter.requires_grad = True

    return model

def modify_model(model, num_classes):
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Modify other RPN and ROI parameters
    model.rpn.nms_thresh = 0.8  # increase from 0.5 to allow less overlap
    model.rpn.fg_iou_thresh = 0.8  # changed from .8
    model.rpn.bg_iou_thresh = 0.2  # Keep as is
    model.roi_heads.batch_size_per_image = 256  # Keep as is
    model.roi_heads.positive_fraction = 0.5  # Keep as is
    model.roi_heads.score_thresh = 0.5  # Lowered from 0.1 to allow lower confidence detections
    model.roi_heads.nms_thresh = 0.3  # Loosen from 0.3 to allow more overlap
    model.roi_heads.detections_per_img = 4  # Increase from 2 to 10

    # Set pre_nms_top_n and post_nms_top_n
    model.rpn.pre_nms_top_n = lambda: 200  # Decreased from 3000
    model.rpn.post_nms_top_n = lambda: 20  # Decreased from 1500
    return model

def load_checkpoint(filepath, model, optimizer):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    mAP = checkpoint.get('mAP', None)
    return model, optimizer, epoch, mAP

def setup_environment(args):
    if args.colab:
        data_root = '/content/drive/MyDrive/MM/CatKidney/data/cat-dataset/'
        checkpoint_dir = '/content/drive/MyDrive/MM/CatKidney/exps'
    else:
        data_root = '/Users/ewern/Desktop/code/MetronMind/data/cat-dataset-with-names'
        checkpoint_dir = '/Users/ewern/Desktop/code/MetronMind/cat_exps'
    
    if args.colab:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    return data_root, checkpoint_dir, device

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train Cat Kidney Detection Model')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--colab', action='store_true', help='Use Google Colab data path')
    parser.add_argument('--only_10', action='store_true', help='Use only 10 samples for quick testing')
    parser.add_argument('--backbone', type=str, default='resnet152', choices=['resnet50', 'resnet101', 'resnet152'],
                        help='Backbone architecture to use')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for training')
    parser.add_argument('--no_sweep', action='store_true', help='Disable wandb sweep and use specified hyperparameters')
    parser.add_argument('--no_preload', action='store_true', help='Preload images into memory')
    parser.add_argument('--freeze_layers', type=str, default='', help='Layers to freeze (comma-separated, e.g., "layer1,layer2")')
    return parser.parse_args()

def main():
    global use_wandb, use_colab

    args = parse_arguments()

    # Hyperparameters
    eval_every_n_epochs = 4
    num_classes = 3  # Background (0), left kidney (1), right kidney (2)
    num_epochs = 300
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    min_lr = args.learning_rate / 50
    data_root, checkpoint_dir, device = setup_environment(args)

    use_wandb = args.wandb
    use_colab = args.colab
    only_10 = args.only_10

    if use_wandb:
        wandb.init(project="kidney_detection", config=args)
    if only_10:
        num_epochs = min(num_epochs, 10)  # Limit to 10 epochs for quick testing

    print("Initializing datasets...")
    train_ann_file = os.path.join(data_root, 'COCO_2/train_Data_coco_format.json')
    val_ann_file = os.path.join(data_root, 'COCO_2/val_Data_coco_format.json')
    preload = not args.no_preload
    train_dataset = CocoDataset(data_root, train_ann_file, transforms=get_transform(train=True), preload=preload, only_10=only_10)
    val_dataset = CocoDataset(data_root, val_ann_file, transforms=get_transform(train=False), preload=preload, only_10=only_10)

    print("Creating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, collate_fn=collate_fn)
    print("Data loaders created.")

    print("Creating model...")
    model = create_model(args, num_classes)

    print("Modifying model parameters...")
    model = modify_model(model, num_classes)

    print("Printing trainable status of layers:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.requires_grad}")

    print(f"Using device: {device}")

    print("Moving model to device...")
    model.to(device)
    print("Model moved to device.")

    print("Creating optimizer and scheduler...")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.05)

    # Load checkpoint if it exists
    checkpoint_path = '/content/drive/MyDrive/MM/CatKidney/exps/model_epoch40/best_model.pth'
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        model, optimizer, start_epoch, best_mAP = load_checkpoint(checkpoint_path, model, optimizer)
        print(f"Resuming training from epoch {start_epoch} with mAP {best_mAP}")
    else:
        start_epoch = 0
        best_mAP = 0.0
        print("No checkpoint found, starting training from scratch.")

    # learning rate scheduler
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - start_epoch, eta_min=min_lr)

    print("Optimizer and scheduler created.")

    # Create a directory for checkpoints
    if use_wandb:
        run_id = wandb.run.id
        checkpoint_dir = os.path.join(checkpoint_dir, run_id)
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_epoch = -1

    print(f"Validation dataset size: {len(val_dataset)}")

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        # Adjust learning rate
        if epoch < start_epoch + 5:
            # Linear warmup
            lr = learning_rate * ((epoch - start_epoch + 1) / 5)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # Regular learning rate schedule
            lr_scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}, Current learning rate: {current_lr}")

        # Train for one epoch
        avg_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=50)

        # Evaluate on validation set every 5 epochs
        if (epoch + 1) % eval_every_n_epochs == 0:
            metrics = evaluate(model, val_loader, device, epoch)
            mAP = metrics.get("mAP", 0.0)

            print(f"Epoch {epoch + 1}: mAP = {mAP}, Avg Loss = {avg_loss}")

            if mAP == 0.0:
                print("Warning: Model failed to detect any objects. Consider adjusting model parameters or checking the dataset.")

            if use_wandb:
                # Log metrics
                wandb.log({
                    "epoch": epoch + 1,
                    "avg_loss": avg_loss,
                    "learning_rate": current_lr,
                    "mAP": mAP
                })

            # Update best_mAP and best_epoch, and save model if it's the best so far
            if mAP > best_mAP:
                best_mAP = mAP
                best_epoch = epoch + 1
                print(f"New best mAP: {best_mAP}")

                # Save the best model
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'mAP': mAP,
                }, os.path.join(checkpoint_dir, 'best_model.pth'))

                print(f"New best model saved in: {checkpoint_dir}/best_model.pth")
            else:
                print(f"mAP did not improve. Best is still {best_mAP} from epoch {best_epoch}")

    print(f"Training complete. Best mAP: {best_mAP} at epoch {best_epoch}")

    if use_wandb:
        wandb.log({"best_mAP": best_mAP, "best_epoch": best_epoch})
        wandb.finish()

if __name__ == "__main__":
    main()