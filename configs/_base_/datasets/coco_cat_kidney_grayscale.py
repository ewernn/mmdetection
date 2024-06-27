# dataset settings
dataset_type = 'CocoDataset'
#data_root = 'data/coco/'
data_root = '/content/drive/MyDrive/MM/CatKidney/data/cat-dataset/'
data_version = 'COCO_2/'

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None

img_norm_cfg = dict(mean=[123.675], std=[58.395], to_rgb=False)  # Adjusted for grayscale
train_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    #dict(type='Resize', scale=(1000, 1000), keep_ratio=True),
    #dict(type='RandomFlip', prob=0.5),
    dict(type='RandomAffine', max_rotate_degree=20, scaling_ratio_range=(0.8, 1.2)),
    #dict(type='Contrast', level=5),  # Adjust contrast
    dict(type='Brightness', level=5),  # Adjust brightness
    #dict(type='Normalize', **img_norm_cfg),
    #dict(type='Pad', size_divisor=32),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale', backend_args=backend_args),
    dict(type='Resize', scale=(1000, 1000), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    #dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_version + 'train_Data_coco_format.json',
        #data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_version + 'val_Data_coco_format.json',
        #data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_version + 'test_Data_coco_format.json',
        #data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + data_version + 'val_Data_coco_format.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)

test_evaluator =  dict(
    type='CocoMetric',
    ann_file=data_root + data_version + 'test_Data_coco_format.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)

