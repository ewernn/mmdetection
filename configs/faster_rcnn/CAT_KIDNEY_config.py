_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_cat_kidney_grayscale.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]


model = dict(
    data_preprocessor = dict(
        type='DetDataPreprocessor',
        mean=None,  # [134.069] No normalization
        std=None,   # [98.622] No normalization
        bgr_to_rgb=False,
        pad_size_divisor=32
    ),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=None,#dict(type='Pretrained', checkpoint='/Users/ewern/Desktop/code/MetronMind/mmdetection/configs/eric/resnet50_grayscale_cleaned.pth'),
        in_channels=1,
        # Added: Attention mechanism to enhance feature extraction
        # plugins=[
        #     dict(
        #         cfg=dict(type='GeneralizedAttention', spatial_range=-1, num_heads=8, attention_type='0010', kv_stride=2),
        #         stages=(False, False, True, True),
        #         position='after_conv2'
        #     )
        # ]
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        # Added: Extra convolutions on input features
        add_extra_convs='on_input',
    ),
    rpn_head=dict(
        anchor_generator=dict(
            type='AnchorGenerator',
            # Modified: Adjusted scales for smaller objects
            scales=[4, 8, 16],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]
        ),
    ),
    roi_head=dict(
        bbox_head=dict(
            # Modified: Set to 1 for kidney detection (assuming single class)
            num_classes=1
        )
    )
)
