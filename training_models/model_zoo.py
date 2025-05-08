import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ViT_B_16_Weights
import timm
import torchvision.models.segmentation as seg_models
from transformers import SegformerForSemanticSegmentation, AutoImageProcessor
import segmentation_models_pytorch as smp

# import sys 
# sys.path.append('D:\LearningWithRevision')
# from mmsegmentation.mmseg.apis import init_model 

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


    def segformer_b2(self):
        """ SegFormer (Hugging Face) for segmentation """
        model_name = "nvidia/segformer-b2-finetuned-cityscapes"
        if self.pretrained:
            model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        else:
            model = SegformerForSemanticSegmentation.from_pretrained(model_name, num_labels=self.num_classes)

        return model
    
    # def mmseg_model(self, model_name="segformer_mit-b5", config_path=None, checkpoint_path=None, device="cuda:0"):
    #     """
    #     Load a segmentation model from MMSegmentation.

    #     Args:
    #         model_name (str): The model type (e.g., "segformer_mit-b5", "deeplabv3_r101").
    #         config_path (str): Path to the MMSeg configuration file.
    #         checkpoint_path (str): Path to the checkpoint file.
    #         device (str): Device to load the model on ("cuda:0" or "cpu").

    #     Returns:
    #         MMSegmentation model ready for inference.
    #     """
    #     # Default Config and Checkpoint for SegFormer on Cityscapes
    #     if model_name == "segformer_mit-b5" and not config_path:
    #         config_path = "configs/segformer/segformer_mit-b5_8xb2-160k_cityscapes-1024x1024.py"
    #         checkpoint_path = "https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b5_512x1024_160k_cityscapes/segformer_mit-b5_512x1024_160k_cityscapes_20220623_130509-afe909ac.pth"

    #     if not config_path or not checkpoint_path:
    #         raise ValueError("Config and checkpoint paths must be provided for custom models.")

    #     # Initialize the model
    #     model = init_model(config_path, checkpoint_path, device=device)

    #     return model
    
    def lraspp_mobilenet_v3_large(self):
        """ Lightweight segmentation with MobileNetV3 """
        if self.pretrained:
            model = seg_models.lraspp_mobilenet_v3_large(pretrained=True)
        else:
            model = seg_models.lraspp_mobilenet_v3_large(pretrained=False)

        model.classifier.low_classifier = nn.Conv2d(40, self.num_classes, kernel_size=1)
        model.classifier.high_classifier = nn.Conv2d(128, self.num_classes, kernel_size=1)
        return model
    
    def segformer(self):
        model = smp.Segformer("resnet34", encoder_weights="imagenet", classes=self.num_classes)
        return model
