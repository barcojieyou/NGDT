#!/usr/bin/env python3
"""
Visualization module for NGD-T experiment results
"""
import argparse
import json
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D


class ExperimentVisualizer:
    """Visualize experiment results"""
    
    def __init__(self, result_dirs: List[Path], output_dir: Optional[Path] = None):
        self.result_dirs = result_dirs
        self.output_dir = output_dir or Path("visualizations")
        self.output_dir.mkdir(exist_ok=True)
        
        self.data = {}
        self.summary_data = []
        
    def load_all_data(self):
        """Load data from all experiment directories"""
        print(f"Loading data from {len(self.result_dirs)} directories...")
        
        for result_dir in self.result_dirs:
            try:
                # Load config
                config_file = result_dir / "config.json"
                if not config_file.exists():
                    print(f"Warning: No config file in {result_dir}")
                    continue
                
                with open(config_file) as f:
                    config = json.load(f)
                
                # Load step logs
                step_files = list(result_dir.glob("step_logs_*.csv"))
                if not step_files:
                    print(f"Warning: No step logs in {result_dir}")
                    continue
                
                step_df = pd.read_csv(step_files[0])
                
                # Load epoch logs
                epoch_files = list(result_dir.glob("epoch_logs_*.csv"))
                if not epoch_files:
                    print(f"Warning: No epoch logs in {result_dir}")
                    continue
                
                epoch_df = pd.read_csv(epoch_files[0])
                
                # Create experiment identifier
                exp_id = f"{config['method']}_{config['model']}_{config['dataset']}_run{config['run_id']}"
                
                self.data[exp_id] = {
                    'step_df': step_df,
                    'epoch_df': epoch_df,
                    'config': config,
                    'directory': result_dir,
                }
                
                # Extract summary
                summary = self._extract_summary(exp_id, step_df, epoch_df, config)
                self.summary_data.append(summary)
                
                print(f"  Loaded: {exp_id}")
                
            except Exception as e:
                print(f"Error loading {result_dir}: {e}")
        
        print(f"Successfully loaded {len(self.data)} experiments")
    
    def _extract_summary(self, exp_id: str, step_df: pd.DataFrame, 
                        epoch_df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract summary statistics from experiment data"""
        summary = {
            'exp_id': exp_id,
            'method': config['method'],
            'model': config['model'],
            'dataset': config['dataset'],
            'run_id': config['run_id'],
            'lr': config.get('lr', 0),
            'batch_size': config.get('batch_size', 0),
            'epochs': config.get('epochs', 0),
        }
        
        # Extract from epoch data
        if 'val_acc' in epoch_df.columns and len(epoch_df) > 0:
            summary['final_val_acc'] = epoch_df['val_acc'].iloc[-1]
            summary['best_val_acc'] = epoch_df['val_acc'].max()
            summary['best_epoch'] = int(epoch_df.loc[epoch_df['val_acc'].idxmax(), 'epoch'])
        
        if 'val_loss' in epoch_df.columns and len(epoch_df) > 0:
            summary['final_val_loss'] = epoch_df['val_loss'].iloc[-1]
            summary['best_val_loss'] = epoch_df['val_loss'].min()
        
        if 'train_loss_avg' in epoch_df.columns and len(epoch_df) > 0:
            summary['final_train_loss'] = epoch_df['train_loss_avg'].iloc[-1]
        
        # Extract from step data
        if 'delta_f' in step_df.columns and len(step_df) > 0:
            summary['avg_delta_f'] = step_df['delta_f'].mean()
            summary['final_delta_f'] = step_df['delta_f'].iloc[-1] if len(step_df) > 0 else 0
        
        if 'eta_t' in step_df.columns and len(step_df) > 0:
            summary['avg_eta_t'] = step_df['eta_t'].mean()
        
        if 'grad_norm' in step_df.columns and len(step_df) > 0:
            summary['avg_grad_norm'] = step_df['grad_norm'].mean()
        
        return summary
    
    def plot_training_curves(self, figsize=(15, 10)):
        """Plot training curves for all experiments"""
        if not self.data:
            print("No data to plot")
            return
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Color map by method
        methods = list(set(exp_data['config']['method'] for exp_data in self.data.values()))
        colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
        method_colors = dict(zip(methods, colors))
        
        # Plot each experiment
        for exp_id, exp_data in self.data.items():
            method = exp_data['config']['method']
            color = method_colors[method]
            alpha = 0.7
            
            step_df = exp_data['step_df']
            epoch_df = exp_data['epoch_df']
            
            # Plot 1: Training loss (steps)
            if 'step' in step_df.columns and 'train_loss' in step_df.columns:
                axes[0, 0].plot(step_df['step'], step_df['train_loss'], 
                               color=color, alpha=alpha*0.5, linewidth=0.5)
            
            # Plot 2: Validation accuracy (epochs)
            if 'epoch' in epoch_df.columns and 'val_acc' in epoch_df.columns:
                axes[0, 1].plot(epoch_df['epoch'], epoch_df['val_acc'], 
                               color=color, alpha=alpha, linewidth=2)
            
            # Plot 3: Validation loss (epochs)
            if 'epoch' in epoch_df.columns and 'val_loss' in epoch_df.columns:
                axes[0, 2].plot(epoch_df['epoch'], epoch_df['val_loss'], 
                               color=color, alpha=alpha, linewidth=2)
            
            # Plot 4: Delta_F (steps) - if available
            if 'step' in step_df.columns and 'delta_f' in step_df.columns:
                axes[1, 0].plot(step_df['step'], step_df['delta_f'], 
                               color=color, alpha=alpha*0.5, linewidth=0.5)
                axes[1, 0].set_yscale('log')
            
            # Plot 5: Learning rate (eta_t) (steps) - if available
            if 'step' in step_df.columns and 'eta_t' in step_df.columns:
                axes[1, 1].plot(step_df['step'], step_df['eta_t'], 
                               color=color, alpha=alpha, linewidth=1)
                axes[1, 1].set_yscale('log')
            
            # Plot 6: Gradient norm (steps) - if available
            if 'step' in step_df.columns and 'grad_norm' in step_df.columns:
                axes[1, 2].plot(step_df['step'], step_df['grad_norm'], 
                               color=color, alpha=alpha*0.5, linewidth=0.5)
                axes[1, 2].set_yscale('log')
        
        # Set titles and labels
        axes[0, 0].set_title('Training Loss (Steps)')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_yscale('log')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Validation Accuracy (Epochs)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[0, 2].set_title('Validation Loss (Epochs)')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].set_yscale('log')
        axes[0, 2].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('Delta_F (Steps)')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Delta_F')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_title('Learning Rate (eta_t)')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[1, 2].set_title('Gradient Norm (Steps)')
        axes[1, 2].set_xlabel('Step')
        axes[1, 2].set_ylabel('Gradient Norm')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Add legend
        legend_elements = [Line2D([0], [0], color=method_colors[method], 
                                 lw=2, label=method) for method in methods]
        fig.legend(handles=legend_elements, loc='lower center', 
                  ncol=min(len(methods), 4), bbox_to_anchor=(0.5, -0.05))
        
        plt.suptitle('Training Curves Comparison', fontsize=16, y=1.02)
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / 'training_curves.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved training curves to {output_path}")
    
    def plot_method_comparison(self, figsize=(12, 8)):
        """Plot method comparison summary"""
        if not self.summary_data:
            print("No summary data to plot")
            return
        
        summary_df = pd.DataFrame(self.summary_data)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Final validation accuracy by method
        if 'final_val_acc' in summary_df.columns:
            sns.boxplot(data=summary_df, x='method', y='final_val_acc', ax=axes[0, 0])
            axes[0, 0].set_title('Final Validation Accuracy by Method')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Best validation accuracy by method
        if 'best_val_acc' in summary_df.columns:
            sns.boxplot(data=summary_df, x='method', y='best_val_acc', ax=axes[0, 1])
            axes[0, 1].set_title('Best Validation Accuracy by Method')
            axes[0, 1].set_ylabel('Best Accuracy')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Convergence speed
        if 'best_epoch' in summary_df.columns:
            sns.boxplot(data=summary_df, x='method', y='best_epoch', ax=axes[1, 0])
            axes[1, 0].set_title('Convergence Speed (Epoch to Best Accuracy)')
            axes[1, 0].set_ylabel('Epoch')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Average Delta_F by method
        if 'avg_delta_f' in summary_df.columns:
            method_delta_f = summary_df.groupby('method')['avg_delta_f'].mean()
            axes[1, 1].bar(method_delta_f.index, method_delta_f.values)
            axes[1, 1].set_title('Average Delta_F by Method')
            axes[1, 1].set_ylabel('Delta_F')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].set_yscale('log')
        
        plt.suptitle('Method Comparison Summary', fontsize=16, y=1.02)
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / 'method_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved method comparison to {output_path}")
    
    def plot_dataset_comparison(self, figsize=(12, 8)):
        """Plot dataset comparison"""
        if not self.summary_data:
            print("No summary data to plot")
            return
        
        summary_df = pd.DataFrame(self.summary_data)
        
        # Filter to have multiple datasets
        datasets = summary_df['dataset'].unique()
        if len(datasets) < 2:
            print(f"Only one dataset found: {datasets[0]}")
            return
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot accuracy by dataset and method
        if 'final_val_acc' in summary_df.columns:
            sns.barplot(data=summary_df, x='method', y='final_val_acc', 
                       hue='dataset', ax=axes[0, 0])
            axes[0, 0].set_title('Final Accuracy by Method and Dataset')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].legend(title='Dataset')
        
        # Plot convergence speed by dataset
        if 'best_epoch' in summary_df.columns:
            sns.barplot(data=summary_df, x='method', y='best_epoch', 
                       hue='dataset', ax=axes[0, 1])
            axes[0, 1].set_title('Convergence Speed by Method and Dataset')
            axes[0, 1].set_ylabel('Epoch to Best Accuracy')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].legend(title='Dataset')
        
        # Plot training loss by dataset
        if 'final_train_loss' in summary_df.columns:
            sns.barplot(data=summary_df, x='method', y='final_train_loss', 
                       hue='dataset', ax=axes[1, 0])
            axes[1, 0].set_title('Final Training Loss by Method and Dataset')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].set_yscale('log')
            axes[1, 0].legend(title='Dataset')
        
        # Plot validation loss by dataset
        if 'final_val_loss' in summary_df.columns:
            sns.barplot(data=summary_df, x='method', y='final_val_loss', 
                       hue='dataset', ax=axes[1, 1])
            axes[1, 1].set_title('Final Validation Loss by Method and Dataset')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].set_yscale('log')
            axes[1, 1].legend(title='Dataset')
        
        plt.suptitle('Dataset Comparison', fontsize=16, y=1.02)
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / 'dataset_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved dataset comparison to {output_path}")
    
    def generate_summary_table(self):
        """Generate summary table of all experiments"""
        if not self.summary_data:
            print("No data to generate summary")
            return
        
        summary_df = pd.DataFrame(self.summary_data)
        
        # Group by method and dataset
        grouped = summary_df.groupby(['method', 'dataset'])
        
        summary_stats = grouped.agg({
            'final_val_acc': ['mean', 'std', 'count'],
            'best_val_acc': ['mean', 'std'],
            'best_epoch': ['mean', 'std'],
        }).round(4)
        
        # Format for readability
        summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]
        
        # Save to CSV
        csv_path = self.output_dir / 'experiment_summary.csv'
        summary_stats.to_csv(csv_path)
        
        # Save to Markdown
        md_path = self.output_dir / 'experiment_summary.md'
        with open(md_path, 'w') as f:
            f.write("# Experiment Summary\n\n")
            f.write(summary_stats.to_markdown())
        
        print(f"Saved summary table to {csv_path}")
        print(f"Saved markdown summary to {md_path}")
        
        return summary_stats
    
    def generate_report(self):
        """Generate comprehensive report"""
        if not self.summary_data:
            print("No data to generate report")
            return
        
        summary_df = pd.DataFrame(self.summary_data)
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("NGD-T EXPERIMENT REPORT")
        report_lines.append("=" * 80)
        
        # Basic statistics
        report_lines.append(f"\nTotal experiments: {len(summary_df)}")
        report_lines.append(f"Methods: {', '.join(summary_df['method'].unique())}")
        report_lines.append(f"Models: {', '.join(summary_df['model'].unique())}")
        report_lines.append(f"Datasets: {', '.join(summary_df['dataset'].unique())}")
        
        # Method comparison
        report_lines.append("\n" + "=" * 80)
        report_lines.append("METHOD COMPARISON")
        report_lines.append("=" * 80)
        
        for method in summary_df['method'].unique():
            method_df = summary_df[summary_df['method'] == method]
            
            report_lines.append(f"\n{method.upper()}:")
            report_lines.append(f"  Experiments: {len(method_df)}")
            
            if 'final_val_acc' in method_df.columns:
                mean_acc = method_df['final_val_acc'].mean()
                std_acc = method_df['final_val_acc'].std()
                report_lines.append(f"  Final accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
            
            if 'best_val_acc' in method_df.columns:
                best_acc = method_df['best_val_acc'].max()
                report_lines.append(f"  Best accuracy: {best_acc:.4f}")
            
            if 'best_epoch' in method_df.columns:
                mean_epoch = method_df['best_epoch'].mean()
                report_lines.append(f"  Avg convergence epoch: {mean_epoch:.1f}")
        
        # Best performing experiments
        report_lines.append("\n" + "=" * 80)
        report_lines.append("TOP 5 EXPERIMENTS")
        report_lines.append("=" * 80)
        
        if 'best_val_acc' in summary_df.columns:
            top5 = summary_df.nlargest(5, 'best_val_acc')
            for i, (_, row) in enumerate(top5.iterrows(), 1):
                report_lines.append(f"\n{i}. {row['method']} on {row['model']}/{row['dataset']} (run {row['run_id']})")
                report_lines.append(f"   Best accuracy: {row['best_val_acc']:.4f}")
                report_lines.append(f"   Final accuracy: {row['final_val_acc']:.4f}")
                report_lines.append(f"   Learning rate: {row['lr']}")
                report_lines.append(f"   Batch size: {row['batch_size']}")
        
        # Dataset comparison
        report_lines.append("\n" + "=" * 80)
        report_lines.append("DATASET COMPARISON")
        report_lines.append("=" * 80)
        
        for dataset in summary_df['dataset'].unique():
            dataset_df = summary_df[summary_df['dataset'] == dataset]
            
            report_lines.append(f"\n{dataset.upper()}:")
            for method in dataset_df['method'].unique():
                method_df = dataset_df[dataset_df['method'] == method]
                if len(method_df) > 0 and 'final_val_acc' in method_df.columns:
                    mean_acc = method_df['final_val_acc'].mean()
                    report_lines.append(f"  {method}: {mean_acc:.4f}")
        
        # Save report
        report_text = '\n'.join(report_lines)
        
        report_path = self.output_dir / 'experiment_report.txt'
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(f"Saved report to {report_path}")
        
        # Print to console
        print("\n" + report_text)
        
        return report_text


def main():
    parser = argparse.ArgumentParser(description="Visualize NGD-T experiment results")
    parser.add_argument('--result-dir', type=str, help='Single result directory')
    parser.add_argument('--result-dirs', type=str, nargs='+', help='Multiple result directories')
    parser.add_argument('--pattern', type=str, help='Pattern to match result directories')
    parser.add_argument('--batch-dir', type=str, help='Batch results directory')
    parser.add_argument('--output-dir', type=str, default='visualizations', help='Output directory')
    parser.add_argument('--all', action='store_true', help='Generate all visualizations')
    parser.add_argument('--curves', action='store_true', help='Generate training curves')
    parser.add_argument('--comparison', action='store_true', help='Generate method comparison')
    parser.add_argument('--dataset-comp', action='store_true', help='Generate dataset comparison')
    parser.add_argument('--summary', action='store_true', help='Generate summary table')
    parser.add_argument('--report', action='store_true', help='Generate comprehensive report')
    
    args = parser.parse_args()
    
    # Find result directories
    result_dirs = []
    
    if args.result_dir:
        result_dirs.append(Path(args.result_dir))
    elif args.result_dirs:
        result_dirs.extend([Path(d) for d in args.result_dirs])
    elif args.pattern:
        result_dirs.extend([Path(p) for p in glob.glob(args.pattern)])
    elif args.batch_dir:
        batch_dir = Path(args.batch_dir)
        if batch_dir.exists():
            # Find all experiment directories
            for exp_dir in batch_dir.glob('*/*'):  # pattern: method_model_dataset_runX/YYYYMMDD_HHMMSS
                if (exp_dir / 'config.json').exists():
                    result_dirs.append(exp_dir)
    else:
        # Default: look for individual experiment directories
        default_patterns = ['results/*/*', 'batch_results/*/*']
        for pattern in default_patterns:
            result_dirs.extend([Path(p) for p in glob.glob(pattern)])
    
    if not result_dirs:
        print("No result directories found")
        return
    
    print(f"Found {len(result_dirs)} result directories")
    
    # Create visualizer
    visualizer = ExperimentVisualizer(result_dirs, Path(args.output_dir))
    visualizer.load_all_data()
    
    # Generate visualizations
    if args.all or args.curves:
        print("\nGenerating training curves...")
        visualizer.plot_training_curves()
    
    if args.all or args.comparison:
        print("\nGenerating method comparison...")
        visualizer.plot_method_comparison()
    
    if args.all or args.dataset_comp:
        print("\nGenerating dataset comparison...")
        visualizer.plot_dataset_comparison()
    
    if args.all or args.summary:
        print("\nGenerating summary table...")
        visualizer.generate_summary_table()
    
    if args.all or args.report:
        print("\nGenerating comprehensive report...")
        visualizer.generate_report()
    
    if not any([args.all, args.curves, args.comparison, args.dataset_comp, args.summary, args.report]):
        # Default: generate all
        print("\nGenerating all visualizations (default)...")
        visualizer.plot_training_curves()
        visualizer.plot_method_comparison()
        visualizer.plot_dataset_comparison()
        visualizer.generate_summary_table()
        visualizer.generate_report()
    
    print(f"\nAll visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()
