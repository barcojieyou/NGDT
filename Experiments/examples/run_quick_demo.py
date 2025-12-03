#!/usr/bin/env python3
"""
Quick demo script to run a simple experiment
"""
import subprocess
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_quick_demo():
    """Run a quick demo experiment"""
    print("Running quick demo experiment...")
    
    # Run a simple experiment
    cmd = [
        'python', 'main.py',
        '--dataset', 'mnist',
        '--model', 'tinycnn',
        '--method', 'sgd',
        '--epochs', '5',
        '--batch-size', '64',
        '--lr', '0.1',
        '--log-every', '10',
        '--eval-every', '20',
        '--out-dir', 'demo_results',
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, text=True)
        print("\nDemo completed successfully!")
        print(f"Results saved to: demo_results/")
        
        # Try to visualize results
        try:
            import visualize
            # Find the result directory
            result_dirs = list(Path('demo_results').glob('*/*'))
            if result_dirs:
                print("\nGenerating visualizations...")
                vis_cmd = ['python', 'visualize.py', '--result-dir', str(result_dirs[0])]
                subprocess.run(vis_cmd, check=True, text=True)
        except ImportError:
            print("Visualization module not available")
            
    except subprocess.CalledProcessError as e:
        print(f"Demo failed with error: {e}")
        return False
    
    return True


if __name__ == "__main__":
    run_quick_demo()
