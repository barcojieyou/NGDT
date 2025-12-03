import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseModel

class TinyCNN(BaseModel):
    """A tiny CNN for quick experiments"""
    
    def __init__(self, num_classes=10, in_channels=3):
        super().__init__(num_classes)
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # Calculate feature size based on input
        if in_channels == 1:  # MNIST: 28x28 -> 3x3
            self.feature_size = 128 * 3 * 3
        else:  # CIFAR: 32x32 -> 4x4
            self.feature_size = 128 * 4 * 4
            
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
    
    def forward_features(self, x):
        """Extract features from conv layers"""
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return x
