"""
Analysis utilities for experiment results
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_results(result_dir: Path) -> Dict[str, Any]:
    """Analyze experiment results from a directory"""
    result_dir = Path(result_dir)
    
    if not result_dir.exists():
        raise ValueError(f"Result directory does not exist: {result_dir}")
    
    analysis = {
        'directory': str(result_dir),
        'files': [],
        'metrics': {},
        'config': None,
    }
    
    # List all files
    for file_path in result_dir.glob('*'):
        analysis['files'].append(file_path.name)
    
    # Load config if exists
    config_file = result_dir / 'config.json'
    if config_file.exists():
        with open(config_file) as f:
            analysis['config'] = json.load(f)
    
    # Load summary if exists
    summary_files = list(result_dir.glob('summary_*.json'))
    if summary_files:
        with open(summary_files[0]) as f:
            analysis['summary'] = json.load(f)
    
    # Load step logs if exists
    step_files = list(result_dir.glob('step_logs_*.csv'))
    if step_files:
        step_df = pd.read_csv(step_files[0])
        analysis['step_data'] = {
            'rows': len(step_df),
            'columns': list(step_df.columns),
            'steps': step_df['step'].max() if 'step' in step_df.columns else 0,
            'final_loss': step_df['train_loss'].iloc[-1] if 'train_loss' in step_df.columns else None,
            'final_delta_f': step_df['delta_f'].iloc[-1] if 'delta_f' in step_df.columns else None,
        }
    
    # Load epoch logs if exists
    epoch_files = list(result_dir.glob('epoch_logs_*.csv'))
    if epoch_files:
        epoch_df = pd.read_csv(epoch_files[0])
        analysis['epoch_data'] = {
            'rows': len(epoch_df),
            'columns': list(epoch_df.columns),
            'epochs': epoch_df['epoch'].max() if 'epoch' in epoch_df.columns else 0,
            'best_val_acc': epoch_df['val_acc'].max() if 'val_acc' in epoch_df.columns else None,
            'final_val_acc': epoch_df['val_acc'].iloc[-1] if 'val_acc' in epoch_df.columns else None,
        }
    
    return analysis


def compare_methods(result_dirs: List[Path]) -> pd.DataFrame:
    """Compare multiple experiment results"""
    comparison_data = []
    
    for result_dir in result_dirs:
        result_dir = Path(result_dir)
        
        if not result_dir.exists():
            continue
        
        # Get basic info from directory name
        dir_name = result_dir.name
        parts = dir_name.split('_')
        
        if len(parts) >= 3:
            method = parts[0]
            model = parts[1]
            dataset = parts[2]
        else:
            method = model = dataset = 'unknown'
        
        # Load config
        config_file = result_dir / 'config.json'
        config = {}
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
        
        # Load epoch logs
        epoch_files = list(result_dir.glob('epoch_logs_*.csv'))
        if not epoch_files:
            continue
        
        epoch_df = pd.read_csv(epoch_files[0])
        
        # Extract key metrics
        if 'val_acc' in epoch_df.columns:
            best_val_acc = epoch_df['val_acc'].max()
            final_val_acc = epoch_df['val_acc'].iloc[-1]
            convergence_epoch = epoch_df[epoch_df['val_acc'] >= 0.9 * best_val_acc]['epoch'].min()
        else:
            best_val_acc = final_val_acc = convergence_epoch = np.nan
        
        if 'val_loss' in epoch_df.columns:
            best_val_loss = epoch_df['val_loss'].min()
            final_val_loss = epoch_df['val_loss'].iloc[-1]
        else:
            best_val_loss = final_val_loss = np.nan
        
        if 'train_loss_avg' in epoch_df.columns:
            final_train_loss = epoch_df['train_loss_avg'].iloc[-1]
        else:
            final_train_loss = np.nan
        
        # Add to comparison
        comparison_data.append({
            'method': method,
            'model': model,
            'dataset': dataset,
            'run_id': config.get('run_id', 0),
            'seed': config.get('seed', 0),
            'lr': config.get('lr', np.nan),
            'batch_size': config.get('batch_size', np.nan),
            'epochs': config.get('epochs', np.nan),
            'best_val_acc': best_val_acc,
            'final_val_acc': final_val_acc,
            'best_val_loss': best_val_loss,
            'final_val_loss': final_val_loss,
            'final_train_loss': final_train_loss,
            'convergence_epoch': convergence_epoch,
            'result_dir': str(result_dir),
        })
    
    return pd.DataFrame(comparison_data)


def generate_comparison_report(result_dirs: List[Path], output_file: Optional[Path] = None) -> str:
    """Generate a comprehensive comparison report"""
    df = compare_methods(result_dirs)
    
    if df.empty:
        return "No valid results to compare"
    
    report = []
    report.append("=" * 80)
    report.append("EXPERIMENT COMPARISON REPORT")
    report.append("=" * 80)
    report.append(f"\nTotal experiments: {len(df)}")
    
    # Group by method
    methods = df['method'].unique()
    report.append(f"\nMethods: {', '.join(methods)}")
    
    # Overall statistics
    report.append("\n" + "=" * 80)
    report.append("OVERALL STATISTICS")
    report.append("=" * 80)
    
    for method in methods:
        method_df = df[df['method'] == method]
        report.append(f"\n{method.upper()}: {len(method_df)} experiments")
        
        if 'best_val_acc' in method_df.columns:
            report.append(f"  Best accuracy: {method_df['best_val_acc'].mean():.4f} ± {method_df['best_val_acc'].std():.4f}")
            report.append(f"  Final accuracy: {method_df['final_val_acc'].mean():.4f} ± {method_df['final_val_acc'].std():.4f}")
        
        if 'convergence_epoch' in method_df.columns:
            conv_epochs = method_df['convergence_epoch'].dropna()
            if len(conv_epochs) > 0:
                report.append(f"  Convergence epoch: {conv_epochs.mean():.1f} ± {conv_epochs.std():.1f}")
    
    # Best performing experiments
    report.append("\n" + "=" * 80)
    report.append("TOP 5 EXPERIMENTS BY VALIDATION ACCURACY")
    report.append("=" * 80)
    
    top5 = df.nlargest(5, 'best_val_acc')
    for i, (_, row) in enumerate(top5.iterrows()):
        report.append(f"\n{i+1}. {row['method']} on {row['model']}/{row['dataset']} (run {row['run_id']})")
        report.append(f"   Best accuracy: {row['best_val_acc']:.4f}")
        report.append(f"   Final accuracy: {row['final_val_acc']:.4f}")
        report.append(f"   Learning rate: {row['lr']}")
        report.append(f"   Convergence epoch: {row['convergence_epoch']:.1f}")
    
    # Method comparison by dataset
    report.append("\n" + "=" * 80)
    report.append("METHOD COMPARISON BY DATASET")
    report.append("=" * 80)
    
    datasets = df['dataset'].unique()
    for dataset in datasets:
        dataset_df = df[df['dataset'] == dataset]
        report.append(f"\n{dataset.upper()}:")
        
        for method in dataset_df['method'].unique():
            method_df = dataset_df[dataset_df['method'] == method]
            if len(method_df) > 0:
                report.append(f"  {method}: {method_df['best_val_acc'].mean():.4f} ± {method_df['best_val_acc'].std():.4f}")
    
    # Save report
    report_text = '\n'.join(report)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
    
    return report_text


def plot_comparison_charts(result_dirs: List[Path], output_dir: Path):
    """Generate comparison charts for multiple experiments"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all data
    all_data = []
    for result_dir in result_dirs:
        result_dir = Path(result_dir)
        
        # Load epoch logs
        epoch_files = list(result_dir.glob('epoch_logs_*.csv'))
        if not epoch_files:
            continue
        
        epoch_df = pd.read_csv(epoch_files[0])
        
        # Add experiment info
        dir_name = result_dir.name
        parts = dir_name.split('_')
        
        if len(parts) >= 3:
            method = parts[0]
            model = parts[1]
            dataset = parts[2]
        else:
            method = model = dataset = 'unknown'
        
        epoch_df['method'] = method
        epoch_df['model'] = model
        epoch_df['dataset'] = dataset
        epoch_df['experiment'] = dir_name
        
        all_data.append(epoch_df)
    
    if not all_data:
        print("No data to plot")
        return
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Set plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    # Plot 1: Validation accuracy over epochs by method
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy curves
    for method in combined_df['method'].unique():
        method_df = combined_df[combined_df['method'] == method]
        for dataset in method_df['dataset'].unique():
            dataset_df = method_df[method_df['dataset'] == dataset]
            if 'epoch' in dataset_df.columns and 'val_acc' in dataset_df.columns:
                axes[0, 0].plot(
                    dataset_df['epoch'],
                    dataset_df['val_acc'],
                    label=f'{method} ({dataset})',
                    alpha=0.7
                )
    
    axes[0, 0].set_title('Validation Accuracy by Method and Dataset')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Loss curves
    for method in combined_df['method'].unique():
        method_df = combined_df[combined_df['method'] == method]
        for dataset in method_df['dataset'].unique():
            dataset_df = method_df[method_df['dataset'] == dataset]
            if 'epoch' in dataset_df.columns and 'val_loss' in dataset_df.columns:
                axes[0, 1].plot(
                    dataset_df['epoch'],
                    dataset_df['val_loss'],
                    label=f'{method} ({dataset})',
                    alpha=0.7
                )
    
    axes[0, 1].set_title('Validation Loss by Method and Dataset')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Final accuracy comparison
    final_accuracies = []
    for method in combined_df['method'].unique():
        method_df = combined_df[combined_df['method'] == method]
        for dataset in method_df['dataset'].unique():
            dataset_df = method_df[method_df['dataset'] == dataset]
            if 'val_acc' in dataset_df.columns:
                final_acc = dataset_df['val_acc'].iloc[-1] if len(dataset_df) > 0 else 0
                final_accuracies.append({
                    'method': method,
                    'dataset': dataset,
                    'final_accuracy': final_acc
                })
    
    if final_accuracies:
        final_acc_df = pd.DataFrame(final_accuracies)
        sns.barplot(data=final_acc_df, x='method', y='final_accuracy', hue='dataset', ax=axes[1, 0])
        axes[1, 0].set_title('Final Validation Accuracy Comparison')
        axes[1, 0].set_xlabel('Method')
        axes[1, 0].set_ylabel('Final Accuracy')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 4: Convergence speed
    convergence_data = []
    for method in combined_df['method'].unique():
        method_df = combined_df[combined_df['method'] == method]
        for dataset in method_df['dataset'].unique():
            dataset_df = method_df[method_df['dataset'] == dataset]
            if 'epoch' in dataset_df.columns and 'val_acc' in dataset_df.columns:
                max_acc = dataset_df['val_acc'].max()
                convergence_epoch = dataset_df[dataset_df['val_acc'] >= 0.9 * max_acc]['epoch'].min()
                if not pd.isna(convergence_epoch):
                    convergence_data.append({
                        'method': method,
                        'dataset': dataset,
                        'convergence_epoch': convergence_epoch
                    })
    
    if convergence_data:
        conv_df = pd.DataFrame(convergence_data)
        sns.barplot(data=conv_df, x='method', y='convergence_epoch', hue='dataset', ax=axes[1, 1])
        axes[1, 1].set_title('Convergence Speed (Epoch to 90% of Max Accuracy)')
        axes[1, 1].set_xlabel('Method')
        axes[1, 1].set_ylabel('Convergence Epoch')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_charts.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Charts saved to {output_dir / 'comparison_charts.png'}")
