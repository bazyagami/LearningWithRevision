import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ViT_B_16_Weights


# def get_pretrained_resnet(num_classes=100):
#     model = models.resnet18(pretrained=True)  
#     in_features = model.fc.in_features
#     model.fc = nn.Linear(in_features, num_classes)
#     return model

def mobilenet_v2(num_classes, pretrained):
    if pretrained:
        model = models.mobilenet_v2(pretrained=True)
    else:
        model = models.mobilenet_v2()
    model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
    return model

def mobilenet_v3(num_classes, pretrained):
    if pretrained: 
        model = models.mobilenet_v3_large(pretrained=True)
    else:
        model = models.mobilenet_v3_large()
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)
    return model

def resnet18(num_classes, pretrained):
    if pretrained:
        model = models.resnet18(pretrained=True)
    else:
        model = models.resnet18()
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def resnet34(num_classes, pretrained):
    if pretrained:
        model = models.resnet34(pretrained=True)
    else: 
        model = models.resnet34()
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def resnet50(num_classes, pretrained):
    if pretrained:
        model = models.resnet50(pretrained=True)
    else: 
        model = models.resnet50()
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def resnet101(num_classes, pretrained):
    if pretrained:
        model = models.resnet101(pretrained=True)
    else: 
        model = models.resnet101()
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def vit_b_16(num_classes, pretrained):
    if pretrained:
        weights = ViT_B_16_Weights.DEFAULT 
        model = models.vit_b_16(weights=weights)
    else:
        model = models.vit_b_16(weights=None)  

    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    
    return model

def efficientnet_b0(num_classes, pretrained):
    if pretrained:
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
    else: 
        model = models.efficientnet_b0()

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)