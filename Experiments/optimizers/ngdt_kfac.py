"""
NGD-T with K-FAC approximation (simplified version)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from .base import BaseOptimizer


class NGDT_KFACOptimizer(BaseOptimizer):
    """Natural Gradient Descent with K-FAC approximation (simplified)"""
    
    def __init__(self, 
                 model: nn.Module,
                 lr: float = 1.0,
                 weight_decay: float = 5e-4,
                 damping: float = 1e-3,
                 beta_f: float = 0.95,
                 beta_mom: float = 0.9,
                 Q_budget: float = 1e-2,
                 eta_min: float = 1e-6,
                 eta_max: float = 1.0,
                 eta_null_ratio: float = 1e-3,
                 update_freq: int = 10,
                 kl_clip: float = 0.001,
                 device: torch.device = None):
        
        super().__init__(model, lr, weight_decay, damping, beta_f, beta_mom, device)
        
        # K-FAC specific parameters
        self.Q_budget = Q_budget
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.eta_null_ratio = eta_null_ratio
        self.update_freq = update_freq
        self.kl_clip = kl_clip
        
        # Current state
        self.current_eta_t = lr
        self.current_delta_f = 0.0
        self.current_q_pred = 0.0
        
        # K-FAC statistics
        self.kfac_layers = []
        self.kfac_stats = {}
        self._setup_kfac_layers()
        
    def _setup_kfac_layers(self):
        """Identify layers for K-FAC approximation"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                self.kfac_layers.append({
                    'name': name,
                    'module': module,
                    'type': 'linear' if isinstance(module, nn.Linear) else 'conv2d'
                })
                
                # Initialize statistics
                if isinstance(module, nn.Linear):
                    in_features = module.in_features
                    out_features = module.out_features
                    self.kfac_stats[name] = {
                        'A': torch.zeros(in_features, in_features).to(self.device),
                        'G': torch.zeros(out_features, out_features).to(self.device),
                        'A_cov': torch.zeros(in_features, in_features).to(self.device),
                        'G_cov': torch.zeros(out_features, out_features).to(self.device),
                        'step_count': 0
                    }
                elif isinstance(module, nn.Conv2d):
                    # For conv layers, we flatten spatial dimensions
                    in_channels = module.in_channels
                    out_channels = module.out_channels
                    kernel_size = module.kernel_size[0] * module.kernel_size[1]
                    self.kfac_stats[name] = {
                        'A': torch.zeros(in_channels * kernel_size, in_channels * kernel_size).to(self.device),
                        'G': torch.zeros(out_channels, out_channels).to(self.device),
                        'A_cov': torch.zeros(in_channels * kernel_size, in_channels * kernel_size).to(self.device),
                        'G_cov': torch.zeros(out_channels, out_channels).to(self.device),
                        'step_count': 0
                    }
    
    def step(self, loss: torch.Tensor) -> Dict[str, float]:
        """Perform NGD-T step with K-FAC approximation"""
        self.step_count += 1
        
        # Compute gradients
        loss.backward()

        # 在zero_grad前保存梯度范数
        self.compute_and_save_grad_norm()
        
        # Apply weight decay
        self.apply_weight_decay()
        
        # Update K-FAC statistics
        if self.step_count % self.update_freq == 0:
            self._update_kfac_statistics()
        
        # Compute preconditioned gradients
        g_nat_list = self._precondition_gradients()
        
        # Compute Delta_F from preconditioned gradients
        delta_f = self._compute_delta_f_from_preconditioned(g_nat_list)
        self.current_delta_f = delta_f
        
        # Compute adaptive learning rate
        self.current_eta_t = self._compute_adaptive_lr(delta_f)
        
        # Update parameters with preconditioned gradients
        self._update_parameters_with_kfac(g_nat_list)
        
        # Compute Q_pred
        self.current_q_pred = 0.5 * (self.current_eta_t ** 2) * delta_f
        
        # Zero gradients for next step
        self.zero_grad()
        
        return self._collect_step_metrics()
    
    def zero_grad(self):
        """Zero out gradients"""
        self.model.zero_grad()
    
    def _update_kfac_statistics(self):
        """Update K-FAC covariance statistics"""
        # This is a simplified version - real K-FAC would be more complex
        for layer_info in self.kfac_layers:
            name = layer_info['name']
            module = layer_info['module']
            
            # Store activations and gradients (simplified)
            # In real implementation, we would use hooks to capture these
            
            # For now, we'll update empirical Fisher for these layers
            for p in module.parameters():
                if p.grad is not None:
                    g = p.grad.detach()
                    sq = g * g
                    if p not in self.state_fisher:
                        self.state_fisher[p] = sq.clone()
                    else:
                        self.state_fisher[p].mul_(self.beta_f).add_(sq * (1.0 - self.beta_f))
    
    def _precondition_gradients(self) -> List[Optional[torch.Tensor]]:
        """Precondition gradients using K-FAC approximation"""
        g_nat_list = []
        
        # Map parameters to their modules
        param_to_module = {}
        for layer_info in self.kfac_layers:
            module = layer_info['module']
            for p in module.parameters():
                param_to_module[p] = layer_info
        
        for p in self.params:
            if p.grad is None:
                g_nat_list.append(None)
                continue
            
            if p in param_to_module:
                # Use K-FAC preconditioning for supported layers
                layer_info = param_to_module[p]
                name = layer_info['name']
                
                if name in self.kfac_stats:
                    # Simplified K-FAC preconditioning
                    # In practice, this would use the stored A and G covariance matrices
                    Fdiag = self.state_fisher.get(p, torch.ones_like(p.data).to(self.device))
                    invF = 1.0 / (Fdiag + self.damping)
                    g_nat = invF * p.grad.detach()
                else:
                    # Fall back to empirical Fisher
                    Fdiag = self.state_fisher.get(p, torch.zeros_like(p.data).to(self.device))
                    invF = 1.0 / (Fdiag + self.damping)
                    g_nat = invF * p.grad.detach()
            else:
                # Use empirical Fisher for other layers
                Fdiag = self.state_fisher.get(p, torch.zeros_like(p.data).to(self.device))
                invF = 1.0 / (Fdiag + self.damping)
                g_nat = invF * p.grad.detach()
            
            g_nat_list.append(g_nat)
        
        return g_nat_list
    
    def _compute_delta_f_from_preconditioned(self, g_nat_list: List[Optional[torch.Tensor]]) -> float:
        """Compute Delta_F from preconditioned gradients"""
        delta_f = 0.0
        for p, g_nat in zip(self.params, g_nat_list):
            if p.grad is None or g_nat is None:
                continue
            g = p.grad.detach()
            delta_f += float((g * g_nat).sum().cpu().item())
        return delta_f
    
    def _compute_adaptive_lr(self, delta_f: float) -> float:
        """Compute adaptive learning rate based on Q-budget"""
        eps = 1e-12
        eta_t = self.lr * (self.Q_budget / (delta_f + eps))
        eta_t = max(min(eta_t, self.eta_max), self.eta_min)
        return eta_t
    
    def _update_parameters_with_kfac(self, g_nat_list: List[Optional[torch.Tensor]]):
        """Update parameters with K-FAC preconditioned gradients"""
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
        
        # Add K-FAC specific metrics
        diagnostics.update({
            'delta_f': self.current_delta_f,
            'delta_f_norm': self.current_delta_f / max(diagnostics['param_count'], 1),
            'eta_t': self.current_eta_t,
            'q_pred': self.current_q_pred,
            'condition_number': self.compute_condition_number(),
            'lr': self.current_eta_t,
            'kfac_layers': len(self.kfac_layers),
            'update_freq': self.update_freq,
        })
        
        return diagnostics
    
    def _compute_optimizer_specific_diagnostics(self) -> Dict[str, float]:
        """Override for optimizer-specific diagnostics"""
        return {
            'Q_budget': self.Q_budget,
            'eta_min': self.eta_min,
            'eta_max': self.eta_max,
            'eta_null_ratio': self.eta_null_ratio,
            'update_freq': self.update_freq,
            'kl_clip': self.kl_clip,
        }
