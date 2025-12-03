import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    """Base class for all models with unified interface"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
    
    @abstractmethod
    def forward_features(self, x):
        """Extract features (for analysis)"""
        pass
    
    def get_num_params(self):
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_layer_info(self):
        """Get information about each layer"""
        info = []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                info.append({
                    'name': name,
                    'type': module.__class__.__name__,
                    'params': sum(p.numel() for p in module.parameters()),
                    'shape': module.weight.shape if hasattr(module, 'weight') else None
                })
        return info
