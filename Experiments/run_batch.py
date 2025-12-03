#!/usr/bin/env python3
"""
Batch experiment runner for NGD-T experiments
"""
import argparse
import subprocess
import itertools
import json
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any


def create_experiment_grid() -> Dict[str, List[Any]]:
    """Create grid of experiment configurations"""
    grid = {
        'dataset': ['mnist', 'cifar10'],
        'model': ['tinycnn', 'resnet18'],
        'method': ['sgd', 'adam', 'ngdt_emp'],
        'lr': [0.1, 0.01, 0.001],
        'batch_size': [64, 128],
        'epochs': [50, 100],
    }
    return grid


def generate_configs(grid: Dict[str, List[Any]], num_runs: int = 3) -> List[Dict[str, Any]]:
    """Generate all experiment configurations"""
    configs = []
    
    # Get all combinations
    keys = list(grid.keys())
    values = list(grid.values())
    
    for combination in itertools.product(*values):
        config = dict(zip(keys, combination))
        
        # Set method-specific defaults
        if config['method'] in ['ngdt_emp', 'ngdt_kfac']:
            config['Q_budget'] = 1e-2
        else:
            config['Q_budget'] = None
        
        # Generate multiple runs with different seeds
        for run_id in range(num_runs):
            config_copy = config.copy()
            config_copy['run_id'] = run_id
            config_copy['seed'] = 42 + run_id * 100
            configs.append(config_copy)
    
    return configs


def run_single_experiment(config: Dict[str, Any], out_dir: str = "batch_results") -> Dict[str, Any]:
    """Run a single experiment"""
    cmd = [
        'python', 'main.py',
        '--dataset', str(config['dataset']),
        '--model', str(config['model']),
        '--method', str(config['method']),
        '--run-id', str(config['run_id']),
        '--seed', str(config['seed']),
        '--batch-size', str(config['batch_size']),
        '--epochs', str(config['epochs']),
        '--lr', str(config['lr']),
        '--out-dir', out_dir,
        '--log-every', '100',
        '--eval-every', '200',
        '--ckpt-freq', '10',
    ]
    
    # Add method-specific arguments
    if config['method'] in ['ngdt_emp', 'ngdt_kfac'] and config.get('Q_budget'):
        cmd.extend(['--Q-budget', str(config['Q_budget'])])
    
    # Add scheduler for SGD/Adam
    if config['method'] in ['sgd', 'adam']:
        cmd.append('--scheduler')
    
    experiment_name = f"{config['method']}_{config['model']}_{config['dataset']}_run{config['run_id']}"
    print(f"Running: {experiment_name}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=3600 * 2  # 2 hour timeout
        )
        
        return {
            'config': config,
            'success': True,
            'output': result.stdout[-1000:],  # Last 1000 chars
            'error': '',
        }
    except subprocess.CalledProcessError as e:
        return {
            'config': config,
            'success': False,
            'output': '',
            'error': e.stderr[-1000:] if e.stderr else str(e),
        }
    except subprocess.TimeoutExpired:
        return {
            'config': config,
            'success': False,
            'output': '',
            'error': 'Timeout expired (2 hours)',
        }
    except Exception as e:
        return {
            'config': config,
            'success': False,
            'output': '',
            'error': str(e),
        }


def run_batch_experiments(num_workers: int = 2, out_dir: str = "batch_results"):
    """Run batch experiments in parallel"""
    grid = create_experiment_grid()
    configs = generate_configs(grid, num_runs=2)  # 2 runs per config
    
    print(f"Total experiments: {len(configs)}")
    print(f"Running with {num_workers} workers")
    print(f"Output directory: {out_dir}")
    
    # Create output directory
    Path(out_dir).mkdir(exist_ok=True)
    
    # Save experiment grid
    grid_file = Path(out_dir) / "experiment_grid.json"
    with open(grid_file, 'w') as f:
        json.dump({'grid': grid, 'configs': configs}, f, indent=2)
    
    # Run experiments
    results = []
    successes = []
    failures = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_config = {
            executor.submit(run_single_experiment, config, out_dir): config
            for config in configs
        }
        
        # Process results as they complete
        for i, future in enumerate(as_completed(future_to_config), 1):
            config = future_to_config[future]
            result = future.result()
            results.append(result)
            
            if result['success']:
                successes.append(result)
                status = "✓"
            else:
                failures.append(result)
                status = "✗"
            
            print(f"{status} Completed {i}/{len(configs)}: "
                  f"{config['method']}_{config['model']}_{config['dataset']}_run{config['run_id']}")
            
            if not result['success']:
                print(f"  Error: {result['error'][:200]}...")
    
    # Save results summary
    summary = {
        'total': len(results),
        'successful': len(successes),
        'failed': len(failures),
        'success_rate': len(successes) / len(results) * 100 if results else 0,
        'failures': [
            {
                'config': f['config'],
                'error': f['error'][:500] if f['error'] else 'Unknown error'
            }
            for f in failures[:10]  # Save first 10 failures
        ]
    }
    
    summary_file = Path(out_dir) / "batch_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("BATCH EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total experiments: {summary['total']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Success rate: {summary['success_rate']:.1f}%")
    
    if failures:
        print(f"\nFirst 3 failures:")
        for i, failure in enumerate(failures[:3], 1):
            config = failure['config']
            print(f"{i}. {config['method']}_{config['model']}_{config['dataset']}_run{config['run_id']}")
            print(f"   Error: {failure['error'][:200]}...")
    
    # Generate comparison report
    print(f"\n{'='*60}")
    print("GENERATING COMPARISON REPORT")
    print(f"{'='*60}")
    
    # Find all result directories
    result_dirs = []
    for exp_dir in Path(out_dir).glob('*/*'):  # pattern: method_model_dataset_runX/YYYYMMDD_HHMMSS
        if (exp_dir / 'config.json').exists():
            result_dirs.append(exp_dir)
    
    if result_dirs:
        try:
            from visualize import generate_comparison_report
            report = generate_comparison_report(
                result_dirs,
                output_file=Path(out_dir) / "comparison_report.txt"
            )
            print("Comparison report generated successfully")
        except ImportError:
            print("Could not import visualization module")
        except Exception as e:
            print(f"Error generating report: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Batch experiment runner")
    parser.add_argument('--workers', type=int, default=2, help='Number of parallel workers')
    parser.add_argument('--out-dir', type=str, default='batch_results', help='Output directory')
    parser.add_argument('--dry-run', action='store_true', help='Show configs without running')
    parser.add_argument('--max-experiments', type=int, help='Maximum number of experiments to run')
    
    args = parser.parse_args()
    
    if args.dry_run:
        grid = create_experiment_grid()
        configs = generate_configs(grid, num_runs=2)
        
        if args.max_experiments:
            configs = configs[:args.max_experiments]
        
        print(f"Would run {len(configs)} experiments")
        print("\nFirst 5 configs:")
        for config in configs[:5]:
            print(f"  {config['method']} on {config['model']}/{config['dataset']} "
                  f"(run {config['run_id']}, lr={config['lr']}, bs={config['batch_size']})")
    else:
        run_batch_experiments(args.workers, args.out_dir)


if __name__ == "__main__":
    main()
