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
        frozen_stages=0, # 1
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=None,#dict(type='Pretrained', checkpoint='/Users/ewern/Desktop/code/MetronMind/mmdetection/configs/eric/resnet50_grayscale_cleaned.pth'),
        in_channels=1,
    ),
)
