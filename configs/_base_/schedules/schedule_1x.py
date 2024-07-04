# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=120, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=0.0001))

# learning rate
param_scheduler = [
    # Linear warm-up phase
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=1000
    ),
    # Cosine annealing phase
    dict(
        type='CosineAnnealingLR',
        T_max=119,  # Total number of epochs - 1
        eta_min=1e-5,  # Minimum learning rate
        begin=0,
        end=120,
        by_epoch=True
    )
]

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
