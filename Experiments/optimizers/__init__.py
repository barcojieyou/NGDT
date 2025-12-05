from .base import BaseOptimizer
from .sgd import SGDOptimizer
from .adam import AdamOptimizer
from .ngdt_emp import NGDT_EmpiricalOptimizer
from .ngdt_kfac import NGDT_KFACOptimizer

def get_optimizer(optimizer_name, model, **kwargs):
    """Get optimizer by name"""
    optimizer_classes = {
        'sgd': SGDOptimizer,
        'adam': AdamOptimizer,
        'ngdt_emp': NGDT_EmpiricalOptimizer,
        'ngdt_kfac': NGDT_KFACOptimizer,
    }
    
    if optimizer_name not in optimizer_classes:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. Available: {list(optimizer_classes.keys())}")

        # 根据优化器类型过滤参数
    if optimizer_name == 'sgd':
        # SGD不需要beta_mom参数
        filtered_kwargs = {k: v for k, v in kwargs.items()
                          if k not in ['beta_mom', 'beta_sqr']}
        return optimizer_classes[optimizer_name](model, **filtered_kwargs)
    else:
        return optimizer_classes[optimizer_name](model, **kwargs)
