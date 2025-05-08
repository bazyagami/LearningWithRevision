import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ViT_B_16_Weights
import timm
import torchvision.models.segmentation as seg_models

class ModelZoo:
    def __init__(self, num_classes, pretrained):
        self.num_classes = num_classes
        self.pretrained = pretrained

    def mobilenet_v2(self):
        if self.pretrained:
            model = models.mobilenet_v2(pretrained=True)
        else:
            model = models.mobilenet_v2()
        model.classifier[1] = torch.nn.Linear(model.last_channel, self.num_classes)
        return model

    def mobilenet_v3(self):
        if self.pretrained: 
            model = models.mobilenet_v3_large(pretrained=True)
        else:
            model = models.mobilenet_v3_large()
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, self.num_classes)
        return model

    def resnet18(self):
        if self.pretrained:
            model = models.resnet18(pretrained=True)
        else:
            model = models.resnet18()
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, self.num_classes)
        return model
    
    def resnet18_3d(self):
        if self.pretrained:
            model = models.video.r3d_18(pretrained=True)
        else:
            model = models.video.r3d_18()
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, self.num_classes)
        model.stem[0] = nn.Conv3d(
                        in_channels=1, 
                        out_channels=64,
                        kernel_size=(3, 7, 7),
                        stride=(1, 2, 2),
                        padding=(1, 3, 3),
                        bias=False
                    )
        return model

    def resnet34(self):
        if self.pretrained:
            model = models.resnet34(pretrained=True)
        else: 
            model = models.resnet34()
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, self.num_classes)
        return model

    def resnet50(self):
        if self.pretrained:
            model = models.resnet50(pretrained=True)
        else: 
            model = models.resnet50()
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, self.num_classes)
        return model

    def resnet101(self):
        if self.pretrained:
            model = models.resnet101(pretrained=True)
        else: 
            model = models.resnet101()
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, self.num_classes)
        return model

    def vit_b_16(self):
        if self.pretrained:
            weights = ViT_B_16_Weights.DEFAULT 
            model = models.vit_b_16(weights=weights)
        else:
            model = models.vit_b_16(weights=None)  

        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, self.num_classes)
        
        return model

    def efficientnet_b0(self):
        if self.pretrained:
            weights = models.EfficientNet_B0_Weights.DEFAULT
            model = models.efficientnet_b0(weights=weights)
        else: 
            model = models.efficientnet_b0()

        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, self.num_classes)
        return model
    
    def efficientnet_b7(self):
        if self.pretrained:
            weights = models.EfficientNet_B7_Weights.DEFAULT
            model = models.efficientnet_b7(weights=weights)
        else: 
            model = models.efficientnet_b7()

        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, self.num_classes)
        return model

    def efficientnet_b4(self):
        if self.pretrained:
            weights = models.EfficientNet_B4_Weights.DEFAULT
            model = models.efficientnet_b4(weights=weights)
        else: 
            model = models.efficientnet_b4()

        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, self.num_classes)
        return model


    def efficientformer(self):
        if self.pretrained:
            model = timm.create_model('efficientformer_l1', pretrained=self.pretrained)
        else:
            model = timm.create_model('efficientformer_l1')
        
        model.reset_classifier(self.num_classes)
        return model

    
