import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Network(nn.Module):
    def __init__(self, num_classes):
        super(Network, self).__init__()
        # Backbone
        self.backbone = nn.Sequential(
            ConvBlock(3, 32, 3, 1, 1),
            ConvBlock(32, 64, 3, 2, 1),
            ConvBlock(64, 128, 3, 2, 1),
            ConvBlock(128, 256, 3, 2, 1),
            ConvBlock(256, 512, 3, 2, 1),
        )
        # Neck
        self.neck = nn.Sequential(
            ConvBlock(512, 256, 1, 1, 0),
            ConvBlock(256, 512, 3, 1, 1),
            ConvBlock(512, 256, 1, 1, 0),
            ConvBlock(256, 512, 3, 1, 1),
        )
        # Head
        self.head = nn.Sequential(
            nn.Conv2d(512, num_classes, 1, 1, 0)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x

# Define the number of classes (e.g., 20 for VOC dataset)
num_classes = 80

# Initialize the model
model = Network(num_classes)

# Print the model architecture
print(model)