#!/usr/bin/env python3
"""
Main experiment runner for NGD-T experiments
"""
import argparse
import os
import sys
import time
import json
import warnings
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models import get_model
from optimizers import get_optimizer
from utils.data_loader import get_dataset
from utils.logger import UnifiedLogger
from utils.metrics import ModelAnalyzer, compute_metrics


def setup_experiment(args):
    """Setup experiment environment"""
    # Set device
    if args.cpu or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.method}_{args.model}_{args.dataset}_run{args.run_id}"
    out_dir = Path(args.out_dir) / exp_name / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save command line arguments
    with open(out_dir / "config.json", 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"Experiment: {exp_name}")
    print(f"Device: {device}")
    print(f"Output directory: {out_dir}")
    
    return device, out_dir


def run_experiment(args):
    """Run a single experiment"""
    # Setup
    device, out_dir = setup_experiment(args)
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    train_loader, val_loader, test_loader, num_classes = get_dataset(
        args.dataset, args.batch_size, args.n_train, args.val_split
    )
    
    # Create model
    print(f"Creating model: {args.model}")
    model = get_model(args.model, num_classes=num_classes, pretrained=args.pretrained)
    model.to(device)
    
    # Analyze model if requested
    if args.analyze_model:
        print("Analyzing model complexity...")
        analyzer = ModelAnalyzer(model, train_loader, device)
        model_info = analyzer.analyze()
        print(f"Model parameters: {model_info['total_params']:,}")
        print(f"Trainable parameters: {model_info['trainable_params']:,}")
        
        with open(out_dir / "model_analysis.json", 'w') as f:
            json.dump(model_info, f, indent=2)
    
    # Create optimizer
    print(f"Creating optimizer: {args.method}")
    optimizer_kwargs = {
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'damping': args.damping,
        'beta_f': args.beta_f,
        'beta_mom': args.beta_mom,
        'device': device,
    }
    
    # Add method-specific arguments
    if args.method == 'sgd':
        optimizer_kwargs.update({
            'momentum': args.momentum,
            'nesterov': args.nesterov,
        })
    elif args.method == 'adam':
        optimizer_kwargs.update({
            'betas': (args.beta1, args.beta2),
            'eps': args.eps,
            'amsgrad': args.amsgrad,
        })
    elif args.method in ['ngdt_emp', 'ngdt_kfac']:
        optimizer_kwargs.update({
            'Q_budget': args.Q_budget,
            'eta_min': args.eta_min,
            'eta_max': args.eta_max,
            'eta_null_ratio': args.eta_null_ratio,
        })
        if args.method == 'ngdt_kfac':
            optimizer_kwargs.update({
                'update_freq': args.update_freq,
                'kl_clip': args.kl_clip,
            })
    
    optimizer = get_optimizer(args.method, model, **optimizer_kwargs)
    
    # Set scheduler for SGD/Adam
    if args.method in ['sgd', 'adam'] and args.scheduler:
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
    print(f"Starting training for {args.epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Optimizer step
            step_metrics = optimizer.step(loss)
            
            # Training metrics
            _, predicted = outputs.max(1)
            correct = predicted.eq(targets).sum().item()
            
            # Update epoch statistics
            epoch_loss += loss.item() * inputs.size(0)
            epoch_correct += correct
            epoch_total += inputs.size(0)
            
            # Prepare step metrics
            step_metrics.update({
                'train_loss': loss.item(),
                'train_acc': correct / inputs.size(0),
                'epoch': epoch,
                'batch': batch_idx,
                'time': time.time() - start_time,
            })
            
            # Validation
            if (batch_idx + 1) % args.eval_every == 0:
                val_metrics = compute_metrics(model, val_loader, criterion, device)
                step_metrics.update({
                    'val_loss': val_metrics['loss'],
                    'val_acc': val_metrics['accuracy'],
                })
            
            # Log step metrics
            logger.log_step(step_metrics)
            
            # Print progress
            if (batch_idx + 1) % args.log_every == 0:
                print(f"Epoch {epoch}, Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, Acc: {100.*correct/inputs.size(0):.2f}%")
        
        # End of epoch
        epoch_train_loss = epoch_loss / epoch_total
        epoch_train_acc = 100. * epoch_correct / epoch_total
        
        # Full validation
        val_metrics = compute_metrics(model, val_loader, criterion, device)
        test_metrics = compute_metrics(model, test_loader, criterion, device)
        
        # Prepare epoch metrics
        epoch_metrics = {
            'epoch': epoch,
            'train_loss_avg': epoch_train_loss,
            'train_acc_avg': epoch_train_acc / 100.0,  # Normalize to [0, 1]
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
            'test_loss': test_metrics['loss'],
            'test_acc': test_metrics['accuracy'],
            'time': time.time() - start_time,
        }
        
        # Log epoch metrics
        logger.log_epoch(epoch_metrics)
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Val Acc: {100.*val_metrics['accuracy']:.2f}%")
        print(f"  Test Loss: {test_metrics['loss']:.4f}, Test Acc: {100.*test_metrics['accuracy']:.2f}%")
        
        # Save checkpoint
        if (epoch + 1) % args.ckpt_freq == 0 or epoch == args.epochs - 1:
            checkpoint = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.optimizer.state_dict() if hasattr(optimizer, 'optimizer') else None,
                'val_acc': val_metrics['accuracy'],
                'test_acc': test_metrics['accuracy'],
            }
            torch.save(checkpoint, out_dir / f"checkpoint_epoch_{epoch}.pt")
            print(f"  Checkpoint saved")
        
        print(f"  Time elapsed: {time.time() - start_time:.2f}s")
        print("-" * 80)
    
    # Final evaluation and summary
    print("\nFinal evaluation...")
    final_val_metrics = compute_metrics(model, val_loader, criterion, device, full=True)
    final_test_metrics = compute_metrics(model, test_loader, criterion, device, full=True)
    
    final_metrics = {
        'final_val_loss': final_val_metrics['loss'],
        'final_val_acc': final_val_metrics['accuracy'],
        'final_test_loss': final_test_metrics['loss'],
        'final_test_acc': final_test_metrics['accuracy'],
        'total_training_time': time.time() - start_time,
        'total_epochs': args.epochs,
        'total_steps': len(train_loader) * args.epochs,
    }
    
    logger.log_final(final_metrics)
    
    # Save final model
    torch.save(model.state_dict(), out_dir / "final_model.pt")
    
    # Generate summary
    summary = logger.generate_summary()
    
    print(f"\nExperiment completed!")
    print(f"Final validation accuracy: {100.*final_val_metrics['accuracy']:.2f}%")
    print(f"Final test accuracy: {100.*final_test_metrics['accuracy']:.2f}%")
    print(f"Total training time: {time.time() - start_time:.2f}s")
    print(f"\nResults saved to: {out_dir}")
    
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="NGD-T Experiments Runner")
    
    # Experiment configuration
    parser.add_argument('--dataset', choices=['mnist', 'cifar10', 'cifar100'], default='cifar10')
    parser.add_argument('--model', choices=['tinycnn', 'resnet18', 'resnet34'], default='resnet18')
    parser.add_argument('--method', choices=['sgd', 'adam', 'ngdt_emp', 'ngdt_kfac'], default='ngdt_emp')
    parser.add_argument('--run-id', type=int, default=0, help='Run identifier for multiple runs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--out-dir', type=str, default='results', help='Output directory')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    
    # Data configuration
    parser.add_argument('--n-train', type=int, default=None, help='Number of training samples (for quick tests)')
    parser.add_argument('--val-split', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    
    # Model configuration
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained model')
    parser.add_argument('--analyze-model', action='store_true', help='Analyze model complexity')
    
    # Optimizer configuration (common)
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--damping', type=float, default=1e-6, help='Damping term')
    parser.add_argument('--beta-f', type=float, default=0.95, help='Fisher update coefficient')
    parser.add_argument('--beta-mom', type=float, default=0.9, help='Momentum coefficient')
    
    # SGD specific
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--nesterov', action='store_true', help='Use Nesterov momentum')
    
    # Adam specific
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam beta2')
    parser.add_argument('--eps', type=float, default=1e-8, help='Adam epsilon')
    parser.add_argument('--amsgrad', action='store_true', help='Use AMSGrad variant')
    
    # NGD-T specific
    parser.add_argument('--Q-budget', type=float, default=1e-2, help='Q-budget for adaptive learning rate')
    parser.add_argument('--eta-min', type=float, default=1e-6, help='Minimum learning rate')
    parser.add_argument('--eta-max', type=float, default=1.0, help='Maximum learning rate')
    parser.add_argument('--eta-null-ratio', type=float, default=1e-3, help='Nullspace update ratio')
    
    # K-FAC specific
    parser.add_argument('--update-freq', type=int, default=10, help='K-FAC update frequency')
    parser.add_argument('--kl-clip', type=float, default=0.001, help='KL clipping for K-FAC')
    
    # Training configuration
    parser.add_argument('--scheduler', action='store_true', help='Use learning rate scheduler')
    parser.add_argument('--log-every', type=int, default=50, help='Log frequency in batches')
    parser.add_argument('--eval-every', type=int, default=200, help='Evaluation frequency in batches')
    parser.add_argument('--ckpt-freq', type=int, default=10, help='Checkpoint frequency in epochs')
    
    args = parser.parse_args()
    
    # Run experiment
    try:
        result_dir = run_experiment(args)
        
        print(f"\nTo visualize results, run:")
        print(f"python visualize.py --result-dir {result_dir}")
        
    except Exception as e:
        print(f"Experiment failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
