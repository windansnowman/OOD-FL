import torch
import torch.nn as nn
import torch.nn.functional as F
from models.simple import SimpleNet

class SimpleCNN(SimpleNet):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 8x8 -> 4x4
        )

        self.dense = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)  # Flatten
        out = self.dense(out)
        out = self.classifier(out)
        return out


def CNN(num_classes):
    return SimpleCNN(num_classes=num_classes)

