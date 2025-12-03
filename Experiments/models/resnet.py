import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34
from .base import BaseModel

class ResNet18(BaseModel):
    """ResNet-18 wrapper with customizable features"""
    
    def __init__(self, num_classes=10, pretrained=False):
        super().__init__(num_classes)
        
        # Load pretrained ResNet-18
        self.model = resnet18(pretrained=pretrained)
        
        # Replace the final fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)
    
    def forward_features(self, x):
        """Extract features from the last convolutional layer"""
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class ResNet34(BaseModel):
    """ResNet-34 wrapper with customizable features"""
    
    def __init__(self, num_classes=10, pretrained=False):
        super().__init__(num_classes)
        
        # Load pretrained ResNet-34
        self.model = resnet34(pretrained=pretrained)
        
        # Replace the final fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)
    
    def forward_features(self, x):
        """Extract features from the last convolutional layer"""
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x
