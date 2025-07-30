import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ViT_B_16_Weights
import timm
import torchvision.models.segmentation as seg_models

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

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

    def mae_vit_b_16(self, checkpoint_path):
        print(f"Loading MAE ViT-B/16 from checkpoint: {checkpoint_path}")

        model = models.vit_b_16(weights=None)

        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            mae_state_dict = checkpoint.get('model', checkpoint)

            new_state_dict = {}
            key_mapping = {
                'cls_token': 'class_token',
                'pos_embed': 'encoder.pos_embedding',
                'patch_embed.proj.weight': 'conv_proj.weight',
                'patch_embed.proj.bias': 'conv_proj.bias',
                'norm.weight': 'encoder.ln.weight',
                'norm.bias': 'encoder.ln.bias',
            }

            for mae_key, v in mae_state_dict.items():
                if mae_key in key_mapping:
                    new_state_dict[key_mapping[mae_key]] = v
                elif mae_key.startswith('blocks.'):
                    parts = mae_key.split('.')
                    if len(parts) >= 3:
                        block_num = parts[1]
                        remaining = '.'.join(parts[2:])

                        mapping_rules = {
                            'norm1.weight': f'encoder.layers.encoder_layer_{block_num}.ln_1.weight',
                            'norm1.bias': f'encoder.layers.encoder_layer_{block_num}.ln_1.bias',
                            'norm2.weight': f'encoder.layers.encoder_layer_{block_num}.ln_2.weight',
                            'norm2.bias': f'encoder.layers.encoder_layer_{block_num}.ln_2.bias',
                            'attn.qkv.weight': f'encoder.layers.encoder_layer_{block_num}.self_attention.in_proj_weight',
                            'attn.qkv.bias': f'encoder.layers.encoder_layer_{block_num}.self_attention.in_proj_bias',
                            'attn.proj.weight': f'encoder.layers.encoder_layer_{block_num}.self_attention.out_proj.weight',
                            'attn.proj.bias': f'encoder.layers.encoder_layer_{block_num}.self_attention.out_proj.bias',
                            'mlp.fc1.weight': f'encoder.layers.encoder_layer_{block_num}.mlp.0.weight',
                            'mlp.fc1.bias': f'encoder.layers.encoder_layer_{block_num}.mlp.0.bias',
                            'mlp.fc2.weight': f'encoder.layers.encoder_layer_{block_num}.mlp.3.weight',
                            'mlp.fc2.bias': f'encoder.layers.encoder_layer_{block_num}.mlp.3.bias',
                        }

                        if remaining in mapping_rules:
                            new_state_dict[mapping_rules[remaining]] = v

            msg = model.load_state_dict(new_state_dict, strict=False)
            print(f"Loaded {len(new_state_dict) - len(msg.missing_keys)}/{len(model.state_dict())} weights")
        else:
            print("No checkpoint provided, using random initialization")

        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, self.num_classes)
        print(f"Classification head adapted to {self.num_classes} classes")

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

    def gpt2(self, model_name="gpt2"):
        """
        Load a GPT-2 model with a language modeling head.
        `model_name`: one of "gpt2", "gpt2-medium", etc.
        """
        if self.pretrained:
            model = GPT2LMHeadModel.from_pretrained(model_name)
        else:
            config = GPT2Config.from_pretrained(model_name)
            model = GPT2LMHeadModel(config)

        # GPT-2's vocab size is typically 50257; adapt output layer if needed.
        if self.num_classes and self.num_classes != model.config.vocab_size:
            model.resize_token_embeddings(self.num_classes)

        return model
