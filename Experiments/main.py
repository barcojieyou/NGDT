#!/usr/bin/env python3
"""
Main experiment runner with unified interface
"""
import argparse
import os
import time
import json
import yaml
import csv
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import our modules
from models import get_model
from optimizers import get_optimizer
from utils.data_loader import get_dataset
from utils.logger import UnifiedLogger
from utils.metrics import ModelAnalyzer


def run_experiment(args):
    """Run a single experiment with unified logging"""
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.method}_{args.model}_{args.dataset}_run{args.run_id}"
    out_dir = Path(args.out_dir) / exp_name / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(out_dir / "config.json", 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Load dataset
    train_loader, test_loader, num_classes = get_dataset(
        args.dataset, args.batch_size, args.n_train
    )
    
    # Create model
    model = get_model(args.model, num_classes=num_classes)
    model.to(device)
    
    # Analyze model complexity
    analyzer = ModelAnalyzer(model, train_loader, device)
    model_info = analyzer.analyze()
    print(f"Model info: {model_info}")
    
    # Create optimizer
    optimizer = get_optimizer(
        args.method,
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        damping=args.damping,
        beta_f=args.beta_f,
        beta_mom=args.beta_mom,
        Q_budget=args.Q_budget,
        eta_min=args.eta_min,
        eta_max=args.eta_max,
        eta_null_ratio=args.eta_null_ratio,
        device=device
    )
    
    # Set scheduler for SGD/Adam
    if args.method in ['sgd', 'adam']:
        total_steps = len(train_loader) * args.epochs
        optimizer.set_scheduler(total_steps)
    
    # Create logger
    logger = UnifiedLogger(
        out_dir=out_dir,
        method=args.method,
        run_id=args.run_id,
        model_name=args.model
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_metrics = {}
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            step += 1
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Optimizer step
            step_metrics = optimizer.step(loss)
            step_metrics['train_loss'] = loss.item()
            step_metrics['epoch'] = epoch
            step_metrics['batch'] = batch_idx
            step_metrics['time'] = time.time()
            
            epoch_loss += loss.item()
            
            # Validation
            if step % args.eval_every == 0:
                val_metrics = evaluate(model, test_loader, criterion, device)
                step_metrics.update(val_metrics)
            
            # Log step metrics
            logger.log_step(step_metrics)
            
            # Print progress
            if step % args.log_every == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
        
        # End of epoch
        avg_loss = epoch_loss / len(train_loader)
        
        # Full validation
        val_metrics = evaluate(model, test_loader, criterion, device)
        epoch_metrics = {
            'epoch': epoch,
            'train_loss_avg': avg_loss,
            **val_metrics
        }
        
        # Log epoch metrics
        logger.log_epoch(epoch_metrics)
        
        # Save checkpoint
        if epoch % args.ckpt_freq == 0 or epoch == args.epochs - 1:
            checkpoint = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.optimizer.state_dict() if hasattr(optimizer, 'optimizer') else None,
                'val_acc': val_metrics['val_acc'],
            }
            torch.save(checkpoint, out_dir / f"checkpoint_epoch_{epoch}.pt")
        
        print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, Val Acc = {val_metrics['val_acc']:.4f}")
    
    # Final evaluation
    final_metrics = evaluate(model, test_loader, criterion, device, full=True)
    logger.log_final(final_metrics)
    
    # Save final model
    torch.save(model.state_dict(), out_dir / "final_model.pt")
    
    # Generate summary report
    logger.generate_summary()
    
    print(f"Experiment completed. Results saved to {out_dir}")
    return out_dir


def evaluate(model, dataloader, criterion, device, full=False):
    """Evaluate model on validation/test set"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / total
    
    metrics = {
        'val_loss': avg_loss,
        'val_acc': accuracy / 100.0,  # Normalize to [0, 1]
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Unified NGD-T Experiments")
    
    # Experiment config
    parser.add_argument('--dataset', choices=['mnist', 'cifar10', 'cifar100'], default='cifar10')
    parser.add_argument('--model', choices=['tinycnn', 'resnet18', 'resnet34'], default='resnet18')
    parser.add_argument('--method', choices=['sgd', 'adam', 'ngdt_emp', 'ngdt_kfac'], default='ngdt_emp')
    parser.add_argument('--run-id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-dir', type=str, default='results_unified')
    
    # Data config
    parser.add_argument('--n-train', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    
    # Optimizer config
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--damping', type=float, default=1e-6)
    parser.add_argument('--beta-f', type=float, default=0.95)
    parser.add_argument('--beta-mom', type=float, default=0.9)
    
    # NGD-T specific
    parser.add_argument('--Q-budget', type=float, default=1e-2)
    parser.add_argument('--eta-min', type=float, default=1e-6)
    parser.add_argument('--eta-max', type=float, default=1.0)
    parser.add_argument('--eta-null-ratio', type=float, default=1e-3)
    
    # Training config
    parser.add_argument('--log-every', type=int, default=50)
    parser.add_argument('--eval-every', type=int, default=200)
    parser.add_argument('--ckpt-freq', type=int, default=10)
    parser.add_argument('--cpu', action='store_true')
    
    # Additional analysis
    parser.add_argument('--analyze-model', action='store_true')
    parser.add_argument('--profile', action='store_true')
    
    args = parser.parse_args()
    
    # Run experiment
    result_dir = run_experiment(args)
    
    print(f"\nExperiment completed successfully!")
    print(f"Results saved to: {result_dir}")
    print(f"\nTo visualize results, run:")
    print(f"python visualize.py --result-dir {result_dir}")


if __name__ == "__main__":
    main()
