# eric, feb 10, 2024

_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# Dataset type and path adjustments
dataset_type = 'CocoDataset'
data_root = '/content/drive/MyDrive/EqNeck/'

# Since your images are black and white, you might consider converting them to 3 channels
# but without normalization. Adjust `img_norm_cfg` if you decide to normalize.
img_norm_cfg = dict(
    mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomAffine', scaling_ratio_range=(0.8, 1.2), max_rotate_degree=20),
    dict(type='PhotoMetricDistortion', brightness_delta=32, contrast_range=(0.8, 1.2), saturation_range=(0.8, 1.2)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug',
         flip=False,
         transforms=[
             # Removed Resize operation
             dict(type='Normalize', **img_norm_cfg),
             dict(type='Pad', size_divisor=32),
             dict(type='ImageToTensor', keys=['img']),
             dict(type='Collect', keys=['img']),
         ])
]

data = dict(
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train_Data_coco.json',
        img_prefix=data_root + 'EqNeckImages',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val_Data_coco.json',
        img_prefix=data_root + 'EqNeckImages',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/test_Data_coco.json',
        img_prefix=data_root + 'EqNeckImages',
        pipeline=test_pipeline)
)

# Adjust the number of classes for your dataset
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1 + 1)  # Adjust number of classes here
    )
)

# Adjust the learning rate policy
optimizer = dict(type='SGD', lr=0.003, momentum=0.9, weight_decay=0.0001)
# This example assumes 8 GPUs. Adjust the learning rate based on your actual setup.
optimizer_config = dict(grad_clip=None)
# Adjust the number of epochs, learning rate schedule, etc., according to your dataset size and desired training length.
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
total_epochs = 12  # Adjust based on your needs


# Choose appropriate work directory
work_dir = '/content/drive/mmdetection'

# Adjust log level and interval
log_config = dict(interval=50)

# Set up checkpoints saving strategy
checkpoint_config = dict(interval=1)
