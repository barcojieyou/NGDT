import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional
from .base import BaseOptimizer


class AdamOptimizer(BaseOptimizer):
    """Adam optimizer with unified interface"""
    
    def __init__(self, 
                 model: nn.Module,
                 lr: float = 0.001,
                 weight_decay: float = 0.0,
                 betas: tuple = (0.9, 0.999),
                 eps: float = 1e-8,
                 amsgrad: bool = False,
                 damping: float = 1e-6,
                 beta_f: float = 0.95,
                 device: torch.device = None):
        
        super().__init__(model, lr, weight_decay, damping, beta_f, 0.0, device)
        
        # Adam specific parameters
        self.betas = betas
        self.eps = eps
        self.amsgrad = amsgrad
        
        # PyTorch optimizer
        self.optimizer = optim.Adam(
            self.params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )
        
        # Learning rate scheduler
        self.scheduler = None
        
    def step(self, loss: torch.Tensor) -> Dict[str, float]:
        """Perform Adam step"""
        self.step_count += 1
        
        # Compute gradients
        loss.backward()
        
        # 在zero_grad前保存梯度范数
        self.compute_and_save_grad_norm()
        # Update parameters
        self.optimizer.step()
        
        # Update empirical Fisher for diagnostics
        self.update_empirical_fisher()
        
        # Zero gradients
        self.zero_grad()
        
        # Step scheduler if exists
        if self.scheduler:
            self.scheduler.step()
            # Update learning rate from scheduler
            self.lr = self.optimizer.param_groups[0]['lr']
        
        return self._collect_step_metrics()
    
    def zero_grad(self):
        """Zero out gradients"""
        self.optimizer.zero_grad()
    
    def set_scheduler(self, total_steps: int):
        """Set learning rate scheduler"""
        # OneCycleLR is good for Adam
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.lr,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy='cos'
        )
    
    def _collect_step_metrics(self) -> Dict[str, float]:
        """Collect metrics for current step"""
        diagnostics = self.compute_diagnostics()
        
        # Compute delta_f for comparison
        delta_f, per_layer, _ = self.compute_delta_f()
        
        # Add Adam specific metrics
        current_lr = self.get_learning_rate()
        
        diagnostics.update({
            'delta_f': delta_f,
            'delta_f_norm': delta_f / max(diagnostics['param_count'], 1),
            'eta_t': current_lr,  # For consistency with NGD-T
            'q_pred': 0.0,  # Adam doesn't use Q_pred
            'condition_number': self.compute_condition_number(),
            'lr': current_lr,
        })
        
        # Add per-layer delta_f (first 5 layers as example)
        for i, val in enumerate(per_layer[:5]):
            diagnostics[f'delta_f_layer_{i}'] = val
        
        return diagnostics
    
    def _compute_optimizer_specific_diagnostics(self) -> Dict[str, float]:
        """Override for optimizer-specific diagnostics"""
        return {
            'beta1': self.betas[0],
            'beta2': self.betas[1],
            'eps': self.eps,
            'amsgrad': float(self.amsgrad),
        }
    
    def get_learning_rate(self) -> float:
        """Get current learning rate from optimizer"""
        if self.optimizer.param_groups:
            return self.optimizer.param_groups[0]['lr']
        return self.lr
    
    def set_learning_rate(self, lr: float):
        """Set learning rate for all parameter groups"""
        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
