#!/usr/bin/env python3
"""
Run comparison between different optimizers
"""
import subprocess
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_comparison():
    """Run comparison between SGD, Adam, and NGD-T"""
    methods = ['sgd', 'adam', 'ngdt_emp', 'ngdt_kfac']
    results = []
    
    print("Running optimizer comparison...")
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Running {method.upper()}...")
        print(f"{'='*60}")
        
        cmd = [
            'python', 'main.py',
            '--dataset', 'cifar10',
            '--model', 'resnet18',
            '--method', method,
            '--run-id', '0',
            '--epochs', '100',  # Reduced for demo
            '--batch-size', '128',
            '--lr', '0.0001',
            '--log-every', '50',
            '--eval-every', '100',
            '--out-dir', 'comparison_results',
            '--scheduler',  # Use scheduler for SGD/Adam
        ]
        
        if method == 'ngdt_emp' or method == 'ngdt_kfac':
            cmd.extend(['--Q-budget', '0.01'])
        
        print(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, text=True, capture_output=True)
            results.append({
                'method': method,
                'success': True,
                'output': result.stdout[-500:],  # Last 500 chars
            })
            print(f"{method.upper()} completed successfully")
        except subprocess.CalledProcessError as e:
            results.append({
                'method': method,
                'success': False,
                'error': e.stderr[-500:] if e.stderr else str(e),
            })
            print(f"{method.upper()} failed: {e}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    for result in results:
        if result['success']:
            print(f"✓ {result['method'].upper()}: Success")
        else:
            print(f"✗ {result['method'].upper()}: Failed")
            print(f"  Error: {result['error'][:200]}...")
    
    # Generate comparison report
    print(f"\n{'='*60}")
    print("GENERATING COMPARISON REPORT")
    print(f"{'='*60}")
    
    try:
        result_dirs = list(Path('comparison_results').glob('*/*'))
        if result_dirs:
            cmd = [
                'python', 'visualize.py',
                '--result-dirs',
                *[str(d) for d in result_dirs],
                '--output-dir', 'comparison_visualizations',
                '--all',
            ]
            
            subprocess.run(cmd, check=True, text=True)
            print(f"\nComparison report generated in 'comparison_visualizations/'")
    except Exception as e:
        print(f"Error generating report: {e}")
    
    return all(r['success'] for r in results)


if __name__ == "__main__":
    run_comparison()
