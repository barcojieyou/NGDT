from .tinycnn import TinyCNN
from .resnet import ResNet18, ResNet34

def get_model(model_name, num_classes=10, **kwargs):
    """Get model by name"""
    model_classes = {
        'tinycnn': TinyCNN,
        'resnet18': ResNet18,
        'resnet34': ResNet34,
    }
    
    if model_name not in model_classes:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_classes.keys())}")
    
    return model_classes[model_name](num_classes=num_classes, **kwargs)
