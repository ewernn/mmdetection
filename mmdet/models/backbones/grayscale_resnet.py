import torch.nn as nn
from mmdet.models import BACKBONES
from mmdet.models.backbones import ResNet
from mmcv.runner import load_checkpoint

@BACKBONES.register_module()
class GrayscaleResNet(ResNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Adjust the first conv layer for 1-channel input
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
    def init_weights(self, pretrained=None):
        super().init_weights(pretrained)
        if pretrained is not None:
            # Load pretrained weights
            checkpoint = load_checkpoint(self, pretrained, strict=False)
            
            # Average the weights of the first conv layer across the RGB channels
            rgb_weight = checkpoint['state_dict']['conv1.weight']
            self.conv1.weight.data = rgb_weight.sum(dim=1, keepdim=True) / 3.0