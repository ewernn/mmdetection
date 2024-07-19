## ERIC: turn resnet101 (color) into grayscale using this hook
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
import torch.nn as nn

from mmdet.registry import HOOKS

@HOOKS.register_module()
class ModifyFirstConvHook(Hook):

    def __init__(self, in_channels=1):
        self.in_channels = in_channels

    def before_train(self, runner):
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        if hasattr(model, 'backbone'):
            first_conv = model.backbone.conv1
            if isinstance(first_conv, nn.Conv2d):
                if first_conv.in_channels != self.in_channels:
                    weight = first_conv.weight.data
                    mean_weight = weight.mean(dim=1, keepdim=True)
                    new_weight = mean_weight.repeat(1, self.in_channels, 1, 1)
                    first_conv.weight.data = new_weight
                    first_conv.in_channels = self.in_channels
                    runner.logger.info(f'Modified first conv layer to {self.in_channels} input channels')