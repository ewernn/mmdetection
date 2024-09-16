eric = '/Users/ewern/Desktop/code/MetronMind/mmdetection/configs/eric/'
import torch
# from torchvision.models import resnet50, ResNet50_Weights

# model = resnet50(weights=ResNet50_Weights.DEFAULT)


# # Modify the first convolutional layer to accept a single channel
# model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

# # Initialize the weights of the new conv1 layer
# with torch.no_grad():
#     model.conv1.weight = torch.nn.Parameter(model.conv1.weight.sum(dim=1, keepdim=True))


# # Save the modified model
# torch.save(model.state_dict(), eric+'resnet50_grayscale.pth')

# Load the modified state dictionary
state_dict = torch.load(eric+'resnet50_grayscale.pth')

# Remove the keys related to the fully connected layer
state_dict.pop('fc.weight', None)
state_dict.pop('fc.bias', None)

# Save the cleaned state dictionary
torch.save(state_dict, eric+'resnet50_grayscale_cleaned.pth')