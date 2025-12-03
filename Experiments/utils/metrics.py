"""
Metrics and analysis utilities
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings


class ModelAnalyzer:
    """Analyze model complexity and optimization landscape"""
    
    def __init__(self, model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.model.eval()
    
    def analyze(self) -> Dict[str, float]:
        """Analyze model and dataset"""
        analysis = {}
        
        # Model statistics
        analysis['total_params'] = sum(p.numel() for p in self.model.parameters())
        analysis['trainable_params'] = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Compute layer-wise statistics
        layer_stats = self._compute_layer_statistics()
        analysis.update(layer_stats)
        
        # Estimate gradient statistics
        grad_stats = self._estimate_gradient_statistics(samples=100)
        analysis.update(grad_stats)
        
        # Estimate loss landscape curvature
        curvature_stats = self._estimate_curvature(samples=50)
        analysis.update(curvature_stats)
        
        return analysis
    
    def _compute_layer_statistics(self) -> Dict[str, float]:
        """Compute statistics for each layer"""
        stats = {}
        
        conv_layers = 0
        linear_layers = 0
        norm_layers = 0
        total_params_per_layer = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layers += 1
                layer_params = sum(p.numel() for p in module.parameters())
                total_params_per_layer.append(layer_params)
                
            elif isinstance(module, nn.Linear):
                linear_layers += 1
                layer_params = sum(p.numel() for p in module.parameters())
                total_params_per_layer.append(layer_params)
                
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                norm_layers += 1
        
        stats['conv_layers'] = conv_layers
        stats['linear_layers'] = linear_layers
        stats['norm_layers'] = norm_layers
        stats['total_layers'] = conv_layers + linear_layers + norm_layers
        
        if total_params_per_layer:
            stats['avg_params_per_layer'] = np.mean(total_params_per_layer)
            stats['std_params_per_layer'] = np.std(total_params_per_layer)
            stats['max_params_per_layer'] = np.max(total_params_per_layer)
            stats['min_params_per_layer'] = np.min(total_params_per_layer)
        
        return stats
    
    def _estimate_gradient_statistics(self, samples: int = 100) -> Dict[str, float]:
        """Estimate gradient statistics on a sample of data"""
        grad_norms = []
        grad_means = []
        grad_stds = []
        
        self.model.train()
        sample_count = 0
        
        with torch.enable_grad():
            for inputs, targets in self.dataloader:
                if sample_count >= samples:
                    break
                    
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, targets)
                
                # Backward pass
                loss.backward()
                
                # Compute gradient statistics
                total_norm = 0.0
                grad_values = []
                
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                        grad_values.append(p.grad.data.cpu().numpy().flatten())
                
                grad_norms.append(total_norm ** 0.5)
                
                if grad_values:
                    all_grads = np.concatenate(grad_values)
                    grad_means.append(np.mean(np.abs(all_grads)))
                    grad_stds.append(np.std(all_grads))
                
                # Zero gradients for next iteration
                self.model.zero_grad()
                sample_count += 1
        
        self.model.eval()
        
        stats = {}
        if grad_norms:
            stats['avg_grad_norm'] = np.mean(grad_norms)
            stats['std_grad_norm'] = np.std(grad_norms)
        
        if grad_means:
            stats['avg_grad_mean'] = np.mean(grad_means)
            stats['avg_grad_std'] = np.mean(grad_stds)
        
        return stats
    
    def _estimate_curvature(self, samples: int = 50) -> Dict[str, float]:
        """Estimate loss landscape curvature using finite differences"""
        curvatures = []
        
        self.model.train()
        sample_count = 0
        
        # Store original parameters
        original_params = [p.data.clone() for p in self.model.parameters()]
        
        with torch.enable_grad():
            for inputs, targets in self.dataloader:
                if sample_count >= samples:
                    break
                
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Compute gradient at current point
                outputs = self.model(inputs)
                loss1 = F.cross_entropy(outputs, targets)
                loss1.backward()
                
                # Store gradients
                gradients = []
                for p in self.model.parameters():
                    if p.grad is not None:
                        gradients.append(p.grad.data.clone())
                    else:
                        gradients.append(torch.zeros_like(p.data))
                
                self.model.zero_grad()
                
                # Take small step in gradient direction
                epsilon = 1e-4
                for p, g in zip(self.model.parameters(), gradients):
                    p.data.add_(epsilon * g)
                
                # Compute loss at new point
                outputs = self.model(inputs)
                loss2 = F.cross_entropy(outputs, targets)
                
                # Estimate curvature: (f(x + εg) - f(x)) / ε^2
                curvature = (loss2.item() - loss1.item()) / (epsilon ** 2)
                curvatures.append(curvature)
                
                # Restore original parameters
                for p, orig in zip(self.model.parameters(), original_params):
                    p.data.copy_(orig)
                
                sample_count += 1
        
        self.model.eval()
        
        stats = {}
        if curvatures:
            stats['avg_curvature'] = np.mean(curvatures)
            stats['std_curvature'] = np.std(curvatures)
            stats['max_curvature'] = np.max(curvatures)
            stats['min_curvature'] = np.min(curvatures)
        
        return stats


def compute_metrics(model: nn.Module, dataloader: torch.utils.data.DataLoader, 
                   criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    """Compute various metrics on a dataset"""
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Loss
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            
            # Accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Store for additional metrics
            all_predictions.append(outputs.cpu())
            all_targets.append(targets.cpu())
    
    # Basic metrics
    accuracy = 100. * correct / total
    avg_loss = total_loss / total
    
    # Additional metrics could be added here:
    # - Confidence calibration
    # - Per-class accuracy
    # - Entropy of predictions
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy / 100.0,  # Normalize to [0, 1]
        'error_rate': 1.0 - (correct / total),
        'samples': total,
    }
    
    return metrics
