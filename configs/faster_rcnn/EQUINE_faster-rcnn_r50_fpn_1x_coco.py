# Simplified Configuration for ONNX Export
model = dict(
    type='FasterRCNN',
    # Simplify the backbone, neck, and other components as required
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5
    ),
    # Specify roi_head, rpn_head, and other components as required
    roi_head=dict(
        bbox_head=dict(
            bbox_coder=dict(
                target_means=[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                target_stds=[
                    0.1,
                    0.1,
                    0.2,
                    0.2,
                ],
                type='DeltaXYWHBBoxCoder'),
            fc_out_channels=1024,
            in_channels=256,
            loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
            loss_cls=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            num_classes=2,
            reg_class_agnostic=False,
            roi_feat_size=7,
            type='Shared2FCBBoxHead'),
        bbox_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        type='StandardRoIHead'),
    rpn_head=dict(
        anchor_generator=dict(
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            scales=[
                8,
            ],
            strides=[
                4,
                8,
                16,
                32,
                64,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
        loss_cls=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        type='RPNHead'),
    test_cfg=dict(
        rcnn=dict(
            max_per_img=100,
            nms=dict(iou_threshold=0.5, type='nms'),
            score_thr=0.05),
        rpn=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=1000)),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=False,
                min_pos_iou=0.5,
                neg_iou_thr=0.5,
                pos_iou_thr=0.5,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=-1,
                num=512,
                pos_fraction=0.25,
                type='RandomSampler')),
        rpn=dict(
            allowed_border=-1,
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.3,
                neg_iou_thr=0.3,
                pos_iou_thr=0.7,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=False,
                neg_pos_ub=-1,
                num=256,
                pos_fraction=0.5,
                type='RandomSampler')),
        rpn_proposal=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=2000)))


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', keep_ratio=True, scale=(1000, 1000)),  # If specific resizing is needed
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='ImageToTensor', keys=['img']),
    #dict(type='Collect', keys=['img']),
    dict(type='Collect', keys=['img']),
]


# Ensure the test pipeline is compatible and simple for ONNX export
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         scale_factor=(1333, 800),  # Image scale
#         #flip=False,  # No flip for simplicity
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
#             dict(type='Pad', size_divisor=32),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img'])  # Ensure keys match expected values
#         ])
    
# ]


# Other configurations as required for ONNX export
data = dict(
    test=dict(
        ann_file='/home/eawern/EqNeckImagesSubset20/Data_coco.json',
        img_prefix='/home/eawern/EqNeckImagesSubset20/',
        pipeline=test_pipeline,
        type='CocoDataset')
    # Specify paths and other data configurations as required
)

# Specify other configurations as needed, such as dataset type, work directory, etc.


#auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
checkpoint_config = dict(interval=1)
data_root = '/home/eawern/EqNeckImagesSubset20/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(
        draw=True, test_out_dir='exps', type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_norm_cfg = dict(
    mean=[
        113.76166347104252,
        113.76166347104252,
        113.76166347104252,
    ],
    std=[
        75.57327894366065,
        75.57327894366065,
        75.57327894366065,
    ],
    to_rgb=True)
launcher = 'none'
load_from = 'bbox_equine_neck_model_pytorch.pth'
log_config = dict(interval=50)
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
lr_config = dict(
    policy='step',
    step=[
        8,
        11,
    ],
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001)
optim_wrapper = dict(
    optimizer=dict(lr=0.02, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
optimizer = dict(lr=0.003, momentum=0.9, type='SGD', weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=12,
        gamma=0.1,
        milestones=[
            8,
            11,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file='/home/eawern/EqNeckImagesSubset20/Data_coco.json',
        backend_args=None,
        data_root='/home/eawern/EqNeckImagesSubset20/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                transforms=[
                    dict(
                        mean=[
                            113.76166347104252,
                            113.76166347104252,
                            113.76166347104252,
                        ],
                        std=[
                            75.57327894366065,
                            75.57327894366065,
                            75.57327894366065,
                        ],
                        to_rgb=True,
                        type='Normalize'),
                    dict(size_divisor=32, type='Pad'),
                    dict(keys=[
                        'img',
                    ], type='ImageToTensor'),
                    dict(keys=[
                        'img',
                    ], type='Collect'),
                ],
                type='MultiScaleFlipAug'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='/home/eawern/EqNeckImagesSubset20/Data_coco.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         transforms=[
#             dict(
#                 mean=[
#                     113.76166347104252,
#                     113.76166347104252,
#                     113.76166347104252,
#                 ],
#                 std=[
#                     75.57327894366065,
#                     75.57327894366065,
#                     75.57327894366065,
#                 ],
#                 to_rgb=True,
#                 type='Normalize'),
#             dict(size_divisor=32, type='Pad'),
#             dict(keys=[
#                 'img',
#             ], type='ImageToTensor'),
#             dict(keys=[
#                 'img',
#             ], type='Collect'),
#         ],
#         type='MultiScaleFlipAug'),
# ]
total_epochs = 12
train_cfg = dict(max_epochs=12, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=4,
    dataset=dict(
        ann_file='/home/eawern/EqNeckImagesSubset20/Data_coco.json',
        backend_args=None,
        data_root='/home/eawern/EqNeckImagesSubset20/',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(to_float32=True, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                max_rotate_degree=20,
                scaling_ratio_range=(
                    0.8,
                    1.2,
                ),
                type='RandomAffine'),
            dict(
                brightness_delta=32,
                contrast_range=(
                    0.8,
                    1.2,
                ),
                saturation_range=(
                    0.8,
                    1.2,
                ),
                type='PhotoMetricDistortion'),
            dict(
                mean=[
                    113.76166347104252,
                    113.76166347104252,
                    113.76166347104252,
                ],
                std=[
                    75.57327894366065,
                    75.57327894366065,
                    75.57327894366065,
                ],
                to_rgb=True,
                type='Normalize'),
            dict(size_divisor=32, type='Pad'),
            dict(type='LoadAnnotations', with_bbox=True),
        ],
        type='CocoDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(to_float32=True, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        max_rotate_degree=20,
        scaling_ratio_range=(
            0.8,
            1.2,
        ),
        type='RandomAffine'),
    dict(
        brightness_delta=32,
        contrast_range=(
            0.8,
            1.2,
        ),
        saturation_range=(
            0.8,
            1.2,
        ),
        type='PhotoMetricDistortion'),
    dict(
        mean=[
            113.76166347104252,
            113.76166347104252,
            113.76166347104252,
        ],
        std=[
            75.57327894366065,
            75.57327894366065,
            75.57327894366065,
        ],
        to_rgb=True,
        type='Normalize'),
    dict(size_divisor=32, type='Pad'),
    dict(type='LoadAnnotations', with_bbox=True),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file='/home/eawern/EqNeckImagesSubset20/Data_coco.json',
        backend_args=None,
        data_root='/home/eawern/EqNeckImagesSubset20/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                transforms=[
                    dict(
                        mean=[
                            113.76166347104252,
                            113.76166347104252,
                            113.76166347104252,
                        ],
                        std=[
                            75.57327894366065,
                            75.57327894366065,
                            75.57327894366065,
                        ],
                        to_rgb=True,
                        type='Normalize'),
                    dict(size_divisor=32, type='Pad'),
                    dict(keys=[
                        'img',
                    ], type='ImageToTensor'),
                    dict(keys=[
                        'img',
                    ], type='Collect'),
                ],
                type='MultiScaleFlipAug'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='/home/eawern/EqNeckImagesSubset20/Data_coco.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')

work_dir = '/home/eawern/'
###############################################################################################################################
# data = dict(
#     test=dict(
#         ann_file='/home/eawern/EqNeckImagesSubset20/Data_coco.json',
#         img_prefix='/home/eawern/EqNeckImagesSubset20/',
#         pipeline=[
#             dict(type='LoadImageFromFile'),
#             dict(
#                 transforms=[
#                     dict(
#                         mean=[
#                             113.76166347104252,
#                             113.76166347104252,
#                             113.76166347104252,
#                         ],
#                         std=[
#                             75.57327894366065,
#                             75.57327894366065,
#                             75.57327894366065,
#                         ],
#                         to_rgb=True,
#                         type='Normalize'),
#                     dict(size_divisor=32, type='Pad'),
#                     dict(keys=[
#                         'img',
#                     ], type='ImageToTensor'),
#                     dict(keys=[
#                         'img',
#                     ], type='Collect'),
#                 ],
#                 type='MultiScaleFlipAug'),
#         ],
#         type='CocoDataset'),
#     train=dict(
#         ann_file='/home/eawern/EqNeckImagesSubset20/Data_coco.json',
#         img_prefix='/home/eawern/EqNeckImagesSubset20/',
#         pipeline=[
#             dict(to_float32=True, type='LoadImageFromFile'),
#             dict(type='LoadAnnotations', with_bbox=True),
#             dict(
#                 max_rotate_degree=20,
#                 scaling_ratio_range=(
#                     0.8,
#                     1.2,
#                 ),
#                 type='RandomAffine'),
#             dict(
#                 brightness_delta=32,
#                 contrast_range=(
#                     0.8,
#                     1.2,
#                 ),
#                 saturation_range=(
#                     0.8,
#                     1.2,
#                 ),
#                 type='PhotoMetricDistortion'),
#             dict(
#                 mean=[
#                     113.76166347104252,
#                     113.76166347104252,
#                     113.76166347104252,
#                 ],
#                 std=[
#                     75.57327894366065,
#                     75.57327894366065,
#                     75.57327894366065,
#                 ],
#                 to_rgb=True,
#                 type='Normalize'),
#             dict(size_divisor=32, type='Pad'),
#             dict(type='LoadAnnotations', with_bbox=True),
#         ],
#         type='CocoDataset'),
#     val=dict(
#         ann_file='/home/eawern/EqNeckImagesSubset20/Data_coco.json',
#         img_prefix='/home/eawern/EqNeckImagesSubset20/',
#         pipeline=[
#             dict(type='LoadImageFromFile'),
#             dict(
#                 transforms=[
#                     dict(
#                         mean=[
#                             113.76166347104252,
#                             113.76166347104252,
#                             113.76166347104252,
#                         ],
#                         std=[
#                             75.57327894366065,
#                             75.57327894366065,
#                             75.57327894366065,
#                         ],
#                         to_rgb=True,
#                         type='Normalize'),
#                     dict(size_divisor=32, type='Pad'),
#                     dict(keys=[
#                         'img',
#                     ], type='ImageToTensor'),
#                     dict(keys=[
#                         'img',
#                     ], type='Collect'),
#                 ],
#                 type='MultiScaleFlipAug'),
#         ],
#         type='CocoDataset'))