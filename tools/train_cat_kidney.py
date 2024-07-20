import os
import json
import torch
import torchvision
import torchvision.transforms as T
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
from torchvision.models.detection import fasterrcnn_resnet152_fpn_v2, FasterRCNN_ResNet152_FPN_V2_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torch.cuda.amp import GradScaler, autocast	

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
        transforms.append(T.RandomHorizontalFlip(0.5))
        #transforms.append(T.RandomCrop((800, 800)))
        transforms.append(T.RandomAffine(degrees=20, scale=(0.8, 1.2), translate=(0.1, 0.1)))
        transforms.append(T.ColorJitter(brightness=0.5, contrast=0.5))#, sharpness=0.5))
    return T.Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))

def evaluate(model, data_loader, device):
    model.eval()
    coco = data_loader.dataset.coco
    coco_results = []
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        outputs = model(images)
        for target, output in zip(targets, outputs):
            image_id = target["image_id"].item()
            boxes = output["boxes"].detach().cpu().numpy()
            scores = output["scores"].detach().cpu().numpy()
            labels = output["labels"].detach().cpu().numpy()
            for box, score, label in zip(boxes, scores, labels):
                coco_results.append({
                    "image_id": image_id,
                    "category_id": label,
                    "bbox": box.tolist(),
                    "score": score,
                })
    coco_dt = coco.loadRes(coco_results)
    coco_eval = COCOeval(coco, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats[0]  # mAP

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    use_amp = device.type == 'cuda'
    scaler = GradScaler() if use_amp else None

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if use_amp:
            with autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
        else:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if use_amp:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # Log metrics to wandb
        if use_wandb:
            wandb.log({"loss": loss_value, "lr": optimizer.param_groups[0]["lr"]})

    # Evaluate and log mAP
    mAP = evaluate(model, data_loader, device)
    if use_wandb:
        wandb.log({"mAP": mAP})

    return metric_logger

def main():
    global use_wandb, use_colab
    parser = argparse.ArgumentParser(description='Train Cat Kidney Detection Model')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--colab', action='store_true', help='Use Google Colab data path')
    parser.add_argument('--only_10', action='store_true', help='Use only 10 samples for quick testing')
    args = parser.parse_args()

    use_wandb = args.wandb
    use_colab = args.colab
    only_10 = args.only_10

    if use_wandb:
        wandb.init(project="cat_kidney_detection")

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
    num_epochs = 120
    batch_size = 4  # Updated batch size
    learning_rate = 0.0001  # Default learning rate if not using wandb
    weight_decay = 0.0001
    momentum = 0.9

    if use_wandb:
        learning_rate = wandb.config.learning_rate  # Use wandb config for learning rate

    # Dataset and DataLoader
    train_dataset = CocoDataset(data_root, train_ann_file, transforms=get_transform(train=True), preload=True, only_10=only_10)
    val_dataset = CocoDataset(data_root, val_ann_file, transforms=get_transform(train=False), preload=True, only_10=only_10)

    # Adjust num_epochs if only_10 is True
    if only_10:
        num_epochs = min(num_epochs, 10)  # Limit to 10 epochs for quick testing

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, collate_fn=collate_fn)

    # Define custom anchor generator
    anchor_sizes = ((32,), (64,), (128,), (256,))
    aspect_ratios = ((1.0, 1.2, 1.5),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=aspect_ratios
    )

    # Model
    weights = FasterRCNN_ResNet152_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet152_fpn_v2(
        weights=weights,
        rpn_anchor_generator=anchor_generator,
        # RPN parameters
        rpn_pre_nms_top_n_train=200,
        rpn_pre_nms_top_n_test=100,
        rpn_post_nms_top_n_train=200,
        rpn_post_nms_top_n_test=100,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=128,
        rpn_positive_fraction=0.5,
        # ROI parameters
        box_batch_size_per_image=128,
        box_positive_fraction=0.5,
        box_score_thresh=0.1,
        box_nms_thresh=0.5,
        box_detections_per_img=5
    )

    # All layers are unfrozen by default, so no need to explicitly unfreeze

    # Print the trainable status of layers
    print("\nTrainable status of layers:")
    for name, param in model.named_parameters():
        if 'backbone' in name:
            print(f"{name}: {param.requires_grad}")

    if use_colab:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    model.to(device)

    # Optimizer with lower learning rate
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=1e-5, momentum=0.9, weight_decay=0.0005)

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Create a directory for checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_mAP = 0.0

    # Training loop
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        
        # Evaluate on validation set
        mAP = evaluate(model, val_loader, device)
        
        # Save the best model
        if mAP > best_mAP:
            best_mAP = mAP
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mAP': mAP,
            }, os.path.join(checkpoint_dir, 'best_model.pth'))
        
        # Save regular checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'mAP': mAP,
        }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'))

    print(f"Training complete. Best mAP: {best_mAP}")
    print(f"Checkpoints saved in: {checkpoint_dir}")

if __name__ == "__main__":
    main()