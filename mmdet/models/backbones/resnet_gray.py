from mmdet.registry import MODELS
from mmdet.models.backbones.resnet import ResNet
import torch.nn as nn

@MODELS.register_module()
class ResNetGray(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace the first conv layer with a 1-channel version
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def init_weights(self):
        super().init_weights()
        # Check if init_cfg is a dict and has the 'type' key set to 'Pretrained'
        if isinstance(self.init_cfg, dict) and self.init_cfg.get('type') == 'Pretrained':
            orig_conv1_weight = self.conv1.weight.data
            # Average the weights across the input channels and repeat for the single channel
            self.conv1.weight.data = orig_conv1_weight.mean(dim=1, keepdim=True).repeat(1, 1, 1, 1)