#!/usr/bin/env python3
"""
Batch experiment runner
"""
import subprocess
import itertools
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse


def create_experiment_grid():
    """Create grid of experiment configurations"""
    grid = {
        'dataset': ['mnist', 'cifar10'],
        'model': ['tinycnn', 'resnet18'],
        'method': ['sgd', 'adam', 'ngdt_emp'],
        'lr': [0.1, 0.01, 0.001],
        'batch_size': [64, 128],
        'epochs': [50, 100],
        'Q_budget': [1e-3, 1e-2, 1e-1],  # For NGD-T
    }
    return grid


def generate_configs(grid, num_runs=3):
    """Generate all experiment configurations"""
    configs = []
    
    # Get all combinations
    keys = list(grid.keys())
    values = list(grid.values())
    
    for combination in itertools.product(*values):
        config = dict(zip(keys, combination))
        
        # Skip incompatible configs
        if config['method'] not in ['ngdt_emp', 'ngdt_kfac']:
            config['Q_budget'] = None
        
        # Generate multiple runs with different seeds
        for run_id in range(num_runs):
            config_copy = config.copy()
            config_copy['run_id'] = run_id
            config_copy['seed'] = 42 + run_id * 100
            configs.append(config_copy)
    
    return configs


def run_single_experiment(config, out_dir="batch_results"):
    """Run a single experiment"""
    cmd = [
        'python', 'main.py',
        '--dataset', config['dataset'],
        '--model', config['model'],
        '--method', config['method'],
        '--run-id', str(config['run_id']),
        '--seed', str(config['seed']),
        '--batch-size', str(config['batch_size']),
        '--epochs', str(config['epochs']),
        '--lr', str(config['lr']),
        '--out-dir', out_dir,
    ]
    
    # Add NGD-T specific args
    if config['method'] in ['ngdt_emp', 'ngdt_kfac'] and config['Q_budget']:
        cmd.extend(['--Q-budget', str(config['Q_budget'])])
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Success: {config}")
        return {
            'config': config,
            'success': True,
            'output': result.stdout
        }
    except subprocess.CalledProcessError as e:
        print(f"Failed: {config}, Error: {e.stderr}")
        return {
            'config': config,
            'success': False,
            'error': e.stderr
        }


def run_batch_experiments(num_workers=2, out_dir="batch_results"):
    """Run batch experiments in parallel"""
    grid = create_experiment_grid()
    configs = generate_configs(grid, num_runs=3)
    
    print(f"Total experiments: {len(configs)}")
    print(f"Running with {num_workers} workers")
    
    # Create output directory
    Path(out_dir).mkdir(exist_ok=True)
    
    # Save experiment grid
    with open(Path(out_dir) / "experiment_grid.json", 'w') as f:
        json.dump({'grid': grid, 'configs': configs}, f, indent=2)
    
    # Run experiments
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for config in configs:
            future = executor.submit(run_single_experiment, config, out_dir)
            futures.append(future)
        
        for i, future in enumerate(futures):
            results.append(future.result())
            print(f"Completed {i+1}/{len(configs)}")
    
    # Save results summary
    successes = [r for r in results if r['success']]
    failures = [r for r in results if not r['success']]
    
    summary = {
        'total': len(results),
        'successful': len(successes),
        'failed': len(failures),
        'success_rate': len(successes) / len(results) * 100,
        'failures': failures[:10]  # Save first 10 failures
    }
    
    with open(Path(out_dir) / "batch_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nBatch experiment summary:")
    print(f"Total: {summary['total']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Success rate: {summary['success_rate']:.1f}%")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--out-dir', type=str, default='batch_results')
    parser.add_argument('--dry-run', action='store_true')
    
    args = parser.parse_args()
    
    if args.dry_run:
        grid = create_experiment_grid()
        configs = generate_configs(grid, num_runs=2)
        print(f"Would run {len(configs)} experiments")
        print("First 5 configs:")
        for config in configs[:5]:
            print(config)
    else:
        run_batch_experiments(args.workers, args.out_dir)
