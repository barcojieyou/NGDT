"""
SGD optimizer with unified interface
"""
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict
from .base import BaseOptimizer


class SGDOptimizer(BaseOptimizer):
    """SGD optimizer with momentum"""
    
    def __init__(self, 
                 model: nn.Module,
                 lr: float = 0.1,
                 weight_decay: float = 5e-4,
                 momentum: float = 0.9,
                 nesterov: bool = False,
                 damping: float = 1e-6,
                 beta_f: float = 0.95,
                 device: torch.device = None):
        
        super().__init__(model, lr, weight_decay, damping, beta_f, 0.0, device)
        
        # SGD specific
        self.momentum = momentum
        self.nesterov = nesterov
        
        # PyTorch optimizer
        self.optimizer = optim.SGD(
            self.params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov
        )
        
        # Learning rate scheduler
        self.scheduler = None
        
    def step(self, loss: torch.Tensor) -> Dict[str, float]:
        """Perform SGD step"""
        self.step_count += 1
        
        # Compute gradients
        loss.backward()
        
        # Update parameters
        self.optimizer.step()
        
        # Update empirical Fisher for diagnostics
        self.update_empirical_fisher()
        
        # Zero gradients
        self.zero_grad()
        
        # Step scheduler if exists
        if self.scheduler:
            self.scheduler.step()
        
        return self._collect_step_metrics()
    
    def zero_grad(self):
        """Zero out gradients"""
        self.optimizer.zero_grad()
    
    def set_scheduler(self, total_steps: int):
        """Set learning rate scheduler"""
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=total_steps,
            eta_min=self.lr * 0.01
        )
    
    def _collect_step_metrics(self) -> Dict[str, float]:
        """Collect metrics for current step"""
        diagnostics = self.compute_diagnostics()
        
        # Compute delta_f for comparison (even though SGD doesn't use it)
        delta_f, per_layer, _ = self.compute_delta_f()
        
        # Add SGD specific metrics
        current_lr = self.optimizer.param_groups[0]['lr']
        
        diagnostics.update({
            'delta_f': delta_f,
            'delta_f_norm': delta_f / max(diagnostics['param_count'], 1),
            'eta_t': current_lr,  # For consistency with NGD-T
            'q_pred': 0.0,  # SGD doesn't use Q_pred
            'condition_number': self.compute_condition_number(),
            'lr': current_lr,
        })
        
        # Add per-layer delta_f
        for i, val in enumerate(per_layer[:5]):
            diagnostics[f'delta_f_layer_{i}'] = val
        
        return diagnostics
    
    def _compute_optimizer_specific_diagnostics(self) -> Dict[str, float]:
        """Override for optimizer-specific diagnostics"""
        return {
            'momentum': self.momentum,
            'nesterov': float(self.nesterov),
        }
