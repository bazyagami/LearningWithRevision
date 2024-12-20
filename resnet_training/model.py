import torch
import torch.nn as nn
import math


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=100):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def resnet18(num_classes=100):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, stride, kernel_size=3, se_ratio=0.25):
        super(MBConvBlock, self).__init__()
        hidden_dim = in_channels * expansion_factor
        self.use_residual = (stride == 1 and in_channels == out_channels)

        layers = []

        if expansion_factor != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.SiLU())

        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.SiLU())

        se_channels = max(1, int(hidden_dim * se_ratio))  # Use hidden_dim, not in_channels
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, se_channels, kernel_size=1),  # Squeeze to se_channels
            nn.SiLU(),
            nn.Conv2d(se_channels, hidden_dim, kernel_size=1),  # Excite back to hidden_dim
            nn.Sigmoid()
        )

        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        identity = x
        out = x

        for layer in self.block[:-2]:  
            out = layer(out)

        se = self.se(out)
        out = out * se 

        out = self.block[-2](out)  
        out = self.block[-1](out)  

        if self.use_residual:
            out += identity

        return out

class EfficientNet(nn.Module):
    def __init__(self, width_coefficient, depth_coefficient, num_classes=100, dropout_rate=0.2):
        super(EfficientNet, self).__init__()
        base_channels = 32
        stages = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 40, 2, 2),
            (6, 80, 3, 2),
            (6, 112, 3, 1),
            (6, 192, 4, 2),
            (6, 320, 1, 1),
        ]

        out_channels = math.ceil(base_channels * width_coefficient)
        self.stem = nn.Sequential(
            nn.Conv2d(3, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

        self.layers = []
        in_channels = out_channels
        for expansion_factor, channels, num_blocks, stride in stages:
            out_channels = math.ceil(channels * width_coefficient)
            for i in range(math.ceil(num_blocks * depth_coefficient)):
                self.layers.append(
                    MBConvBlock(
                        in_channels, out_channels, expansion_factor,
                        stride if i == 0 else 1
                    )
                )
                in_channels = out_channels

        self.layers = nn.Sequential(*self.layers)

        out_channels = math.ceil(1280 * width_coefficient)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = self.head(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def efficientnet_b0(num_classes=100):
    return EfficientNet(width_coefficient=1.0, depth_coefficient=1.0, num_classes=num_classes)