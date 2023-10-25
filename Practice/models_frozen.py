from torchvision import models
from efficientnet_pytorch import EfficientNet
import torch.nn as nn

def resnet18_frozen(num_classes):
    # Tải pre-trained ResNet-18
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    for name, param in model.named_parameters():
        if 'layer4' in name or 'fc' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model

def mobilenetv3_frozen(num_classes):
    # Tải pre-trained MobileNetV3
    model = models.mobilenet_v3_small(pretrained=True)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    for name, param in model.named_parameters():
        if 'features.12' in name or 'classifier' in name :
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model

def efficientnet_frozen(num_classes):
    # Chọn phiên bản EfficientNet (ví dụ: b3)
    model = EfficientNet.from_pretrained('efficientnet-b3')
    model._fc = nn.Linear(model._fc.in_features, num_classes)
    for name, param in model.named_parameters():
        if '_blocks.25' in name or '_fc' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model