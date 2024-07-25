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
import tools.utils as utils # Import your custom utils
import sys
import math  # Add this import
import argparse
import random  # Add this import
import time  # Add this import
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2 as fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights as FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torch.cuda.amp import GradScaler, autocast	
import torch.nn as nn
from collections import OrderedDict
import torchvision
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
import ast
from torchvision.models import ResNet101_Weights, ResNet152_Weights
import psutil
import GPUtil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image
from torch.optim.lr_scheduler import CosineAnnealingLR


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
            boxes.append([xmin, ymin, xmax, ymax])
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
            T.RandomAutocontrast(p=0.5),
            T.Lambda(lambda x: TF.adjust_brightness(x, brightness_factor=random.uniform(0.6, 1.4))),
            T.Lambda(lambda x: TF.adjust_contrast(x, contrast_factor=random.uniform(0.5, 1.5))),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            T.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            #T.RandomHorizontalFlip(p=0.5),
        ])
    
    # Expand grayscale to 3 channels
    transforms.append(T.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x))
    
    return T.Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))

def visualize_boxes(image, gt_boxes, gt_labels, pred_boxes, pred_labels, image_id, save_path):
    """
    Visualize ground truth and predicted bounding boxes on the image and save it.
    """
    # Debugging: Print shapes and types to understand the data structure at this point
    print(f"pred_boxes type: {type(pred_boxes)}, shape: {pred_boxes.shape}")
    print(f"pred_labels type: {type(pred_labels)}, shape: {pred_labels.shape if hasattr(pred_labels, 'shape') else 'N/A'}")

    # Convert tensor image to PIL Image
    image_pil = to_pil_image(image)
    
    # Create figure and axis
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image_pil)
    
    pred_labels = np.atleast_1d(pred_labels)
    # Draw ground truth boxes in green
    for box, label in zip(gt_boxes, gt_labels):
        rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                                 linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        ax.text(box[0], box[1], f'GT: {label}', color='g', fontsize=10, verticalalignment='top')
    
    # Draw predicted boxes in red
    if len(pred_boxes) > 0:
        for box, label in zip(pred_boxes, pred_labels):
            rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                                     linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(box[0], box[1]-20, f'Pred: {label}', color='r', fontsize=10, verticalalignment='top')
    else:
        ax.text(10, 10, 'No predictions', color='r', fontsize=12, verticalalignment='top')
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Set title
    plt.title(f'Image ID: {image_id}')
    
    # Save the figure
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

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
    
    image_count = 0

    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        
        with torch.no_grad():
            outputs = model(images)
        
        for i, (target, output) in enumerate(zip(targets, outputs)):
            image_id = target["image_id"].item()
            boxes = output["boxes"].detach().cpu().numpy()
            scores = output["scores"].detach().cpu().numpy()
            labels = output["labels"].detach().cpu().numpy()
            
            # Ensure boxes, scores, and labels are 2D arrays
            if len(boxes) == 0:
                boxes = np.empty((0, 4))
                scores = np.array([])
                labels = np.array([])
            else:
                boxes = np.atleast_2d(boxes)
                scores = np.atleast_1d(scores)
                labels = np.atleast_1d(labels)
            
            # Apply NMS
            keep = torchvision.ops.nms(torch.from_numpy(boxes), torch.from_numpy(scores), iou_threshold=0.9)  # Loosen NMS IoU threshold
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
            
            if image_count < 5:
                print(f"Image ID: {image_id}")
                print(f"Number of detections after filtering: {len(boxes)}")
                print(f"Scores: {scores}")
                print(f"Labels: {labels}")
                
                # Print ground truth for comparison
                gt_boxes = target["boxes"].cpu().numpy()
                gt_labels = target["labels"].cpu().numpy()
                print(f"Ground Truth - Number of objects: {len(gt_boxes)}")
                print(f"Ground Truth - Labels: {gt_labels}")
                
                # Visualize and save image
                image = images[i].cpu()  # Ensure the correct image is used
                save_path = os.path.join(save_dir, f'image_{image_id}.png')
                print(f"Visualizing Image ID: {image_id}, Save Path: {save_path}")
                print(f"Image Shape: {image.shape}, GT Boxes: {gt_boxes}, GT Labels: {gt_labels}, Pred Boxes: {boxes}, Pred Labels: {labels}")
                visualize_boxes(image, gt_boxes, gt_labels, boxes, labels, image_id, save_path)
                
                image_count += 1
            
            if len(boxes) == 0 or len(scores) == 0 or len(labels) == 0:
                print("No detections to process.")
            else: # Your existing code
                for box, score, label in zip(boxes, scores, labels):
                    coco_results.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])],
                        "score": float(score),
                    })
        
        if image_count >= 5:
            break
    
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

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, warmup_epochs=5, initial_lr=7e-5):
    model.train()
    scaler = GradScaler()
    total_loss = 0
    num_batches = 0
    start_time = time.time()

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with autocast():
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        scaler.scale(losses).backward()

        # Implement gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Adjust the learning rate
        if epoch < warmup_epochs:
            lr = initial_lr * ((epoch * len(data_loader) + batch_idx + 1) / (warmup_epochs * len(data_loader)))
        else:
            lr = optimizer.param_groups[0]['lr']  # Use the current lr from the optimizer outside the warmup period

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()
        total_loss += losses.item()
        num_batches += 1

        if batch_idx % print_freq == 0:
            avg_loss = total_loss / num_batches
            elapsed_time = time.time() - start_time
            images_per_sec = (batch_idx + 1) * len(images) / elapsed_time
            print(f"Epoch [{epoch}][{batch_idx}/{len(data_loader)}] "
                  f"Loss: {avg_loss:.4f} "
                  f"Images/sec: {images_per_sec:.1f} "
                  f"LR: {lr:.6f}")

    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch} complete. Average Loss: {avg_loss:.4f}")

    return avg_loss

def create_model(args, num_classes, anchor_generator):
    if args.backbone in ['resnet101', 'resnet152']:
        print(f"Using {args.backbone} as backbone")
        weights = ResNet101_Weights.IMAGENET1K_V1 if args.backbone == 'resnet101' else ResNet152_Weights.IMAGENET1K_V1
        backbone = resnet_fpn_backbone(backbone_name=args.backbone, weights=weights, trainable_layers=5)
        
        # For FPN backbones, we need 5 levels of anchor sizes
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
        
        model = FasterRCNN(backbone, num_classes=num_classes, 
                           rpn_anchor_generator=anchor_generator)
    else:
        print("Using default ResNet-50 as backbone")
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(weights=weights, 
                                        rpn_anchor_generator=anchor_generator)
    
    return model

def main():
    global use_wandb, use_colab
    parser = argparse.ArgumentParser(description='Train Cat Kidney Detection Model')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--colab', action='store_true', help='Use Google Colab data path')
    parser.add_argument('--only_10', action='store_true', help='Use only 10 samples for quick testing')
    parser.add_argument('--anchor_sizes', type=str, default="((161,), (192,), (219,), (252,), (311,))")
    parser.add_argument('--aspect_ratios', type=str, default="((1.5, 2.0, 2.5),)")
    parser.add_argument('--backbone', type=str, default='resnet152', choices=['resnet50', 'resnet101', 'resnet152'],
                        help='Backbone architecture to use')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate for training')
    parser.add_argument('--no_sweep', action='store_true', help='Disable wandb sweep and use specified hyperparameters')
    args = parser.parse_args()

    use_wandb = args.wandb
    use_colab = args.colab
    only_10 = args.only_10

    if use_wandb:
        wandb.init(project="cat_kidney_compiled", config=args)

    # Paths
    if use_colab:
        data_root = '/content/drive/MyDrive/MM/CatKidney/data/cat-dataset/'
        checkpoint_dir = '/content/drive/MyDrive/MM/CatKidney/exps'
    else:
        data_root = '/Users/ewern/Desktop/code/MetronMind/data/cat-dataset'
        checkpoint_dir = '/Users/ewern/Desktop/code/MetronMind/cat_exps'
    
    train_ann_file = os.path.join(data_root, 'COCO_2/train_Data_coco_format.json')
    val_ann_file = os.path.join(data_root, 'COCO_2/val_Data_coco_format.json')

    # Hyperparameters
    num_classes = 3  # Background (0), left kidney (1), right kidney (2)
    num_epochs = 500  # Set to 300 epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    min_lr = args.learning_rate / 100  # 1e-7

    if use_wandb and not args.no_sweep:
        learning_rate = wandb.config.learning_rate
        batch_size = wandb.config.batch_size
        args.anchor_sizes = wandb.config.anchor_sizes
        args.aspect_ratios = wandb.config.aspect_ratios

    print("Initializing datasets...")
    train_dataset = CocoDataset(data_root, train_ann_file, transforms=get_transform(train=True), preload=True, only_10=only_10)
    val_dataset = CocoDataset(data_root, val_ann_file, transforms=get_transform(train=False), preload=True, only_10=only_10)
    print("Datasets initialized.")

    # Adjust num_epochs if only_10 is True
    if only_10:
        num_epochs = min(num_epochs, 10)  # Limit to 10 epochs for quick testing
    print("Creating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, collate_fn=collate_fn)
    print("Data loaders created.")

    print("Creating model...")
    model = create_model(args, num_classes, None)  # Pass None for anchor_generator
    print("Model created.")

    print("Modifying model parameters...")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Set model parameters statically
    model.rpn.nms_thresh = 0.5
    model.rpn.fg_iou_thresh = 0.7
    model.rpn.bg_iou_thresh = 0.3
    model.roi_heads.batch_size_per_image = 256
    model.roi_heads.positive_fraction = 0.4
    model.roi_heads.score_thresh = 0.1
    model.roi_heads.nms_thresh = 0.1  # lowered from 0.2 to prevent overlap
    model.roi_heads.detections_per_img = 2  # Always exactly two kidneys

    # Set pre_nms_top_n and post_nms_top_n
    model.rpn.pre_nms_top_n = lambda: 3000
    model.rpn.post_nms_top_n = lambda: 1500
    print("Model parameters modified.")

    print("Printing trainable status of layers:")
    for name, param in model.named_parameters():
        if 'backbone' in name:
            print(f"{name}: {param.requires_grad}")

    if use_colab:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    print(f"Using device: {device}")

    print("Moving model to device...")
    model.to(device)
    print("Model moved to device.")

    print("Creating optimizer and scheduler...")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.005)

    lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)

    print("Optimizer and scheduler created.")

    # Create a directory for checkpoints
    if use_wandb:
        run_id = wandb.run.id
        checkpoint_dir = os.path.join(checkpoint_dir, run_id)
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_mAP = 0.0
    best_epoch = -1

    if use_wandb:
        wandb.config.update({
            "anchor_sizes": args.anchor_sizes,
            "aspect_ratios": args.aspect_ratios
        })

    print(f"Validation dataset size: {len(val_dataset)}")

    # After creating the optimizer and before the training loop, add this:
    initial_lr = learning_rate
    warmup_epochs = 5

    # In the training loop, replace the existing learning rate adjustment with this:
    for epoch in range(num_epochs):
        if epoch < warmup_epochs:
            # Linear warmup
            lr = initial_lr * ((epoch + 1) / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # Your existing learning rate schedule
            lr_scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}, Current learning rate: {current_lr}")

        # Train for one epoch
        avg_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, initial_lr=initial_lr, print_freq=50)

        # Evaluate on validation set every 5 epochs
        if (epoch + 1) % 5 == 0:
            metrics = evaluate(model, val_loader, device, epoch)
            mAP = metrics.get("mAP", 0.0)
            
            print(f"Epoch {epoch + 1}: mAP = {mAP}, Avg Loss = {avg_loss}")

            if mAP == 0.0:
                print("Warning: Model failed to detect any objects. Consider adjusting model parameters or checking the dataset.")

            if use_wandb:
                # Get CPU usage
                cpu_percent = psutil.cpu_percent()
                
                # Get memory usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                
                # Get GPU usage if available
                gpu_percent = 0
                gpu_memory_percent = 0
                if torch.cuda.is_available():
                    gpu = GPUtil.getGPUs()[0]
                    gpu_percent = gpu.load * 100
                    gpu_memory_percent = gpu.memoryUtil * 100

                wandb.log({
                    "epoch": epoch + 1,
                    "avg_loss": avg_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "gpu_percent": gpu_percent,
                    "gpu_memory_percent": gpu_memory_percent,
                    **metrics  # This will include all the metrics from the evaluate function
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
    try:
        main()
    except Exception as e:
        if use_wandb:
            wandb.log({"error": str(e)})
        print(f"An error occurred: {str(e)}")
        raise