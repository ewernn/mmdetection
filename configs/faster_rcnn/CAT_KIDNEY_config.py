_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_cat_kidney_grayscale.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# Model configuration changes
model = dict(
    data_preprocessor = dict(
        type='DetDataPreprocessor',
        mean=None,
        std=None,
        bgr_to_rgb=False,
        pad_size_divisor=32
    ),
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),  # Change to BN
        norm_eval=False,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101'),
        in_channels=1,
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5
    ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=512,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[2, 4, 8, 16],
            ratios=[1.0, 1.2, 1.5],
            strides=[4, 8, 16, 32, 64]
        ),
    ),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=2048,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='GIoULoss', loss_weight=10.0))),
)

# Custom hook to modify the first convolutional layer
custom_hooks = [
    dict(
        type='ModifyFirstConvHook',
        in_channels=1
    )
]

# Training schedule and learning rate changes
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200, val_interval=1)
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=2000
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=198,  # Total number of epochs - 2 (to account for the linear warmup)
        eta_min=1e-6,
        begin=2,  # Start after the linear warmup
        end=200,
        by_epoch=True
    )
]

# Optimizer settings
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0001)
)

# Early stopping
custom_hooks = [
    dict(type='EarlyStoppingHook', monitor='bbox_mAP', patience=10)
]

# Evaluation settings
val_evaluator = dict(
    type='CocoMetric',
    metric='bbox',
    format_only=False)

test_evaluator = val_evaluator

# Override dataloader settings
train_dataloader = dict(batch_size=4, num_workers=4)
val_dataloader = dict(batch_size=4, num_workers=4)
test_dataloader = dict(batch_size=4, num_workers=4)