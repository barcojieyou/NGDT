"""
NGD-T with empirical Fisher diagonal
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional
from .base import BaseOptimizer


class NGDT_EmpiricalOptimizer(BaseOptimizer):
    """Natural Gradient Descent with empirical Fisher diagonal"""
    
    def __init__(self, 
                 model: nn.Module,
                 lr: float = 1.0,
                 weight_decay: float = 5e-4,
                 damping: float = 1e-6,
                 beta_f: float = 0.95,
                 beta_mom: float = 0.9,
                 Q_budget: float = 1e-2,
                 eta_min: float = 1e-6,
                 eta_max: float = 1.0,
                 eta_null_ratio: float = 1e-3,
                 device: torch.device = None):
        
        super().__init__(model, lr, weight_decay, damping, beta_f, beta_mom, device)
        
        # NGD-T specific parameters
        self.Q_budget = Q_budget
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.eta_null_ratio = eta_null_ratio
        
        # State
        self.current_eta_t = lr
        self.current_delta_f = 0.0
        self.current_q_pred = 0.0
        
    def step(self, loss: torch.Tensor) -> Dict[str, float]:
        """Perform NGD-T step with empirical Fisher"""
        self.step_count += 1
        
        # Compute gradients
        loss.backward()
        
        # Apply weight decay
        self.apply_weight_decay()
        
        # Update empirical Fisher
        self.update_empirical_fisher()
        
        # Compute Delta_F and natural gradients
        delta_f, per_layer, g_nat_list = self.compute_delta_f()
        self.current_delta_f = delta_f
        
        # Compute adaptive learning rate
        self.current_eta_t = self._compute_adaptive_lr(delta_f)
        
        # Update parameters with natural gradient and momentum
        self._update_parameters(g_nat_list)
        
        # Compute Q_pred
        self.current_q_pred = 0.5 * (self.current_eta_t ** 2) * delta_f
        
        # Zero gradients for next step
        self.zero_grad()
        
        return self._collect_step_metrics()
    
    def zero_grad(self):
        """Zero out gradients"""
        self.model.zero_grad()
    
    def _compute_adaptive_lr(self, delta_f: float) -> float:
        """Compute adaptive learning rate based on Q-budget"""
        eps = 1e-12
        eta_t = self.lr * (self.Q_budget / (delta_f + eps))
        eta_t = max(min(eta_t, self.eta_max), self.eta_min)
        return eta_t
    
    def _update_parameters(self, g_nat_list: List[Optional[torch.Tensor]]):
        """Update parameters with natural gradient momentum"""
        for p, g_nat in zip(self.params, g_nat_list):
            if g_nat is None:
                # Fallback to raw gradient
                if p.grad is not None:
                    p.data.add_(-self.current_eta_t * p.grad.detach())
                continue
            
            # Update momentum
            m = self.momentum_nat[p]
            m.mul_(self.beta_mom).add_(g_nat * (1.0 - self.beta_mom))
            
            # Compute nullspace update
            eta_null = self.eta_null_ratio * self.current_eta_t
            raw_grad = p.grad.detach() if p.grad is not None else None
            
            # Apply update
            if raw_grad is None:
                p.data.add_(-self.current_eta_t * m)
            else:
                p.data.add_(-self.current_eta_t * m - eta_null * raw_grad)
    
    def _collect_step_metrics(self) -> Dict[str, float]:
        """Collect metrics for current step"""
        diagnostics = self.compute_diagnostics()
        
        # Add NGD-T specific metrics
        diagnostics.update({
            'delta_f': self.current_delta_f,
            'delta_f_norm': self.current_delta_f / max(diagnostics['param_count'], 1),
            'eta_t': self.current_eta_t,
            'q_pred': self.current_q_pred,
            'condition_number': self.compute_condition_number(),
            'lr': self.current_eta_t,  # For consistency with other optimizers
        })
        
        # Add per-layer delta_f (first 5 layers as example)
        delta_f, per_layer, _ = self.compute_delta_f()
        for i, val in enumerate(per_layer[:5]):
            diagnostics[f'delta_f_layer_{i}'] = val
        
        return diagnostics
    
    def _compute_optimizer_specific_diagnostics(self) -> Dict[str, float]:
        """Override for optimizer-specific diagnostics"""
        return {
            'Q_budget': self.Q_budget,
            'eta_min': self.eta_min,
            'eta_max': self.eta_max,
        }
