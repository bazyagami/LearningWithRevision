import torch
import torch.nn as nn
import torchvision.models as models


def get_pretrained_resnet(num_classes=100):
    model = models.resnet18(pretrained=True)  
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

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