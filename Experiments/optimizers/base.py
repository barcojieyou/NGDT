"""
Base optimizer class for unified interface
"""
import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any


class BaseOptimizer(ABC):
    """Base class for all optimizers with unified interface"""
    
    def __init__(self, 
                 model: nn.Module,
                 lr: float = 0.01,
                 weight_decay: float = 0.0,
                 damping: float = 1e-6,
                 beta_f: float = 0.95,
                 beta_mom: float = 0.9,
                 device: torch.device = None):
        
        self.model = model
        self.params = list(model.parameters())
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.damping = damping
        self.beta_f = beta_f
        self.beta_mom = beta_mom
        
        # State tracking
        self.step_count = 0
        self.state_fisher = {}
        self.momentum_nat = {p: torch.zeros_like(p.data) for p in self.params}
        
        # Diagnostics
        self.metrics_history = []

    def compute_and_save_grad_norm(self) -> float:
        """计算并保存梯度范数，应在zero_grad前调用"""
        self.current_grad_norm = self._compute_current_grad_norm()
        return self.current_grad_norm
    
    def _compute_grad_norm(self) -> float:
        """重写：返回保存的梯度范数"""
        if hasattr(self, 'current_grad_norm'):
            return self.current_grad_norm
        # 后备：实时计算（可能为0）
        return super()._compute_current_grad_norm()
        
    @abstractmethod
    def step(self, loss: torch.Tensor) -> Dict[str, float]:
        """Perform one optimization step"""
        pass
    
    @abstractmethod
    def zero_grad(self):
        """Zero out gradients"""
        pass
    
    def compute_diagnostics(self) -> Dict[str, float]:
        """Compute unified diagnostic metrics"""
        diagnostics = {
            'step': self.step_count,
            'param_count': sum(p.numel() for p in self.params),
            'grad_norm': self._compute_grad_norm(),
            'weight_norm': self._compute_weight_norm(),
        }
        
        # Add optimizer-specific diagnostics
        diagnostics.update(self._compute_optimizer_specific_diagnostics())
        
        return diagnostics
    
    def _compute_current_grad_norm(self) -> float:
        """Compute gradient L2 norm"""
        total_norm = 0.0
        for p in self.params:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def _compute_weight_norm(self) -> float:
        """Compute parameter L2 norm"""
        total_norm = 0.0
        for p in self.params:
            param_norm = p.data.norm(2)
            total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def _compute_optimizer_specific_diagnostics(self) -> Dict[str, float]:
        """Override in subclasses for optimizer-specific metrics"""
        return {}
    
    def update_empirical_fisher(self):
        """Update empirical Fisher diagonal with exponential moving average"""
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad.detach()
            sq = g * g
            if p not in self.state_fisher:
                self.state_fisher[p] = sq.clone()
            else:
                self.state_fisher[p].mul_(self.beta_f).add_(sq * (1.0 - self.beta_f))
    
    def compute_delta_f(self) -> Tuple[float, List[float], List[torch.Tensor]]:
        """Compute Delta_F and natural gradients"""
        delta_f_total = 0.0
        per_layer = []
        g_nat_list = []
        
        for p in self.params:
            g = p.grad.detach() if p.grad is not None else None
            if g is None:
                per_layer.append(0.0)
                g_nat_list.append(None)
                continue
            
            Fdiag = self.state_fisher.get(p, torch.zeros_like(p.data))
            invF = 1.0 / (Fdiag + self.damping)
            g_nat = invF * g
            g_nat_list.append(g_nat)
            
            val = float((g * g_nat).sum().cpu().item())
            per_layer.append(val)
            delta_f_total += val
        
        return delta_f_total, per_layer, g_nat_list
    
    def compute_condition_number(self) -> float:
        """Approximate condition number from Fisher diagonal"""
        eigenvalues = []
        for p in self.params:
            Fdiag = self.state_fisher.get(p, torch.ones_like(p.data))
            # Avoid zeros and extreme values
            valid_vals = Fdiag[Fdiag > 1e-12]
            if len(valid_vals) > 0:
                log_vals = torch.log(valid_vals)
                eigenvalues.append(log_vals.flatten())
        
        if len(eigenvalues) == 0:
            return 0.0
        
        all_eigvals = torch.cat(eigenvalues)
        if len(all_eigvals) < 2:
            return 0.0
            
        condition_num = all_eigvals.max() - all_eigvals.min()
        return float(condition_num.item())
    
    def apply_weight_decay(self):
        """Apply weight decay to parameters"""
        if self.weight_decay > 0:
            for p in self.params:
                if p.grad is None:
                    continue
                p.grad.data.add_(self.weight_decay, p.data)
