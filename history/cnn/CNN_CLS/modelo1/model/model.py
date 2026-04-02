import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, ch, dropout=0.05):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x):
        y = self.act(self.bn1(self.conv1(x)))
        y = self.drop(y)
        y = self.bn2(self.conv2(y))
        return self.act(y + x)


class CNN2DClassifier(nn.Module):
    """Pure CNN baseline: conv trunk + 1x1 conv classification head."""

    def __init__(self, in_ch=97, out_dim=3, dropout=0.2):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResBlock(64, dropout=0.05),
            ResBlock(64, dropout=0.05),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResBlock(128, dropout=0.05),
            ResBlock(128, dropout=0.05),
        )
        self.cls_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(64, out_dim, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, x):
        f = self.trunk(x)
        return self.cls_head(f)

