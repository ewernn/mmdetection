from mmdet.registry import MODELS
from mmdet.models.backbones.resnet import ResNet
import torch.nn as nn

@MODELS.register_module()
class ResNetGray(ResNet):
    def init_weights(self):
        super().init_weights()
        # Modify the first conv layer to accept 1-channel input
        if isinstance(self.conv1, nn.Conv2d) and self.conv1.in_channels != 1:
            new_conv = nn.Conv2d(1, self.conv1.out_channels, 
                                 kernel_size=self.conv1.kernel_size, 
                                 stride=self.conv1.stride, 
                                 padding=self.conv1.padding, 
                                 bias=False)
            # Average the weights across the input channels
            new_conv.weight.data = self.conv1.weight.data.mean(dim=1, keepdim=True)
            self.conv1 = new_conv