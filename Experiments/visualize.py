"""
Visualization module for experiment results
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import glob
from typing import List, Dict, Any
import argparse

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12


class ExperimentVisualizer:
    """Visualize experiment results"""
    
    def __init__(self, result_dirs: List[Path]):
        self.result_dirs = result_dirs
        self.data = {}
        self.summary_data = []
        
    def load_all_data(self):
        """Load data from all experiment directories"""
        for result_dir in self.result_dirs:
            try:
                # Load step logs
                step_csv = next(result_dir.glob("step_logs_*.csv"))
                step_df = pd.read_csv(step_csv)
                
                # Load epoch logs
                epoch_csv = next(result_dir.glob("epoch_logs_*.csv"))
                epoch_df = pd.read_csv(epoch_csv)
                
                # Load config
                config_file = result_dir / "config.json"
                with open(config_file) as f:
                    config = json.load(f)
                
                # Create experiment identifier
                exp_id = f"{config['method']}_{config['model']}_{config['dataset']}_run{config['run_id']}"
                
                self.data[exp_id] = {
                    'step_df': step_df,
                    'epoch_df': epoch_df,
                    'config': config
                }
                
                # Extract summary
                summary = self._extract_summary(exp_id, step_df, epoch_df, config)
                self.summary_data.append(summary)
                
            except Exception as e:
                print(f"Error loading {result_dir}: {e}")
    
    def _extract_summary(self, exp_id, step_df, epoch_df, config):
        """Extract summary statistics"""
        best_epoch = epoch_df.loc[epoch_df['val_acc'].idxmax()]
        
        return {
            'exp_id': exp_id,
            'method': config['method'],
            'model': config['model'],
            'dataset': config['dataset'],
            'run_id': config['run_id'],
            'final_train_loss': epoch_df['train_loss_avg'].iloc[-1],
            'final_val_loss': epoch_df['val_loss'].iloc[-1],
            'final_val_acc': epoch_df['val_acc'].iloc[-1],
            'best_val_acc': best_epoch['val_acc'],
            'best_epoch': int(best_epoch['epoch']),
            'avg_delta_f': step_df['delta_f'].mean() if 'delta_f' in step_df.columns else 0,
            'avg_eta_t': step_df['eta_t'].mean() if 'eta_t' in step_df.columns else config.get('lr', 0),
            'total_steps': len(step_df),
        }
    
    def plot_training_curves(self, save_path=None):
        """Plot training curves for all experiments"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Group by method for better comparison
        methods = set()
        for exp_data in self.data.values():
            methods.add(exp_data['config']['method'])
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
        method_colors = dict(zip(methods, colors))
        
        for exp_id, exp_data in self.data.items():
            method = exp_data['config']['method']
            color = method_colors[method]
            
            # Plot training loss
            axes[0, 0].plot(
                exp_data['step_df']['step'],
                exp_data['step_df']['train_loss'],
                color=color, alpha=0.5, linewidth=0.5
            )
            
            # Plot validation accuracy
            axes[0, 1].plot(
                exp_data['epoch_df']['epoch'],
                exp_data['epoch_df']['val_acc'],
                color=color, alpha=0.7, linewidth=2
            )
            
            # Plot validation loss
            axes[0, 2].plot(
                exp_data['epoch_df']['epoch'],
                exp_data['epoch_df']['val_loss'],
                color=color, alpha=0.7, linewidth=2
            )
            
            # Plot Delta_F if available
            if 'delta_f' in exp_data['step_df'].columns:
                axes[1, 0].plot(
                    exp_data['step_df']['step'],
                    exp_data['step_df']['delta_f'],
                    color=color, alpha=0.5, linewidth=0.5
                )
            
            # Plot eta_t if available
            if 'eta_t' in exp_data['step_df'].columns:
                axes[1, 1].plot(
                    exp_data['step_df']['step'],
                    exp_data['step_df']['eta_t'],
                    color=color, alpha=0.7, linewidth=1
                )
            
            # Plot gradient norm
            if 'grad_norm' in exp_data['step_df'].columns:
                axes[1, 2].plot(
                    exp_data['step_df']['step'],
                    exp_data['step_df']['grad_norm'],
                    color=color, alpha=0.5, linewidth=0.5
                )
        
        # Set titles and labels
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_yscale('log')
        
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        
        axes[0, 2].set_title('Validation Loss')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].set_yscale('log')
        
        axes[1, 0].set_title('Delta_F')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Delta_F')
        axes[1, 0].set_yscale('log')
        
        axes[1, 1].set_title('Learning Rate (eta_t)')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        
        axes[1, 2].set_title('Gradient Norm')
        axes[1, 2].set_xlabel('Step')
        axes[1, 2].set_ylabel('Gradient L2 Norm')
        axes[1, 2].set_yscale('log')
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color=method_colors[method], lw=2, label=method) 
                          for method in methods]
        fig.legend(handles=legend_elements, loc='lower center', ncol=len(methods), bbox_to_anchor=(0.5, -0.05))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.show()
    
    def plot_comparison_summary(self, save_path=None):
        """Plot comparison summary across methods"""
        if not self.summary_data:
            return
        
        summary_df = pd.DataFrame(self.summary_data)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Final validation accuracy by method
        sns.boxplot(data=summary_df, x='method', y='final_val_acc', ax=axes[0, 0])
        axes[0, 0].set_title('Final Validation Accuracy by Method')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Best validation accuracy by method
        sns.boxplot(data=summary_df, x='method', y='best_val_acc', ax=axes[0, 1])
        axes[0, 1].set_title('Best Validation Accuracy by Method')
        axes[0, 1].set_ylabel('Best Accuracy')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Convergence speed (epoch to best accuracy)
        sns.boxplot(data=summary_df, x='method', y='best_epoch', ax=axes[0, 2])
        axes[0, 2].set_title('Convergence Speed by Method')
        axes[0, 2].set_ylabel('Epoch to Best Accuracy')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Average Delta_F by method
        if 'avg_delta_f' in summary_df.columns:
            summary_df_filtered = summary_df[summary_df['avg_delta_f'] > 0]
            if len(summary_df_filtered) > 0:
                sns.boxplot(data=summary_df_filtered, x='method', y='avg_delta_f', ax=axes[1, 0])
                axes[1, 0].set_title('Average Delta_F by Method')
                axes[1, 0].set_ylabel('Delta_F')
                axes[1, 0].tick_params(axis='x', rotation=45)
                axes[1, 0].set_yscale('log')
        
        # 5. Average learning rate by method
        if 'avg_eta_t' in summary_df.columns:
            summary_df_filtered = summary_df[summary_df['avg_eta_t'] > 0]
            if len(summary_df_filtered) > 0:
                sns.boxplot(data=summary_df_filtered, x='method', y='avg_eta_t', ax=axes[1, 1])
                axes[1, 1].set_title('Average Learning Rate by Method')
                axes[1, 1].set_ylabel('Learning Rate')
                axes[1, 1].tick_params(axis='x', rotation=45)
                axes[1, 1].set_yscale('log')
        
        # 6. Training stability (loss variance)
        # Calculate loss variance from step data
        loss_variances = []
        for exp_id, exp_data in self.data.items():
            method = exp_data['config']['method']
            loss_var = exp_data['step_df']['train_loss'].var()
            loss_variances.append({'method': method, 'loss_variance': loss_var})
        
        if loss_variances:
            loss_var_df = pd.DataFrame(loss_variances)
            sns.boxplot(data=loss_var_df, x='method', y='loss_variance', ax=axes[1, 2])
            axes[1, 2].set_title('Training Loss Variance by Method')
            axes[1, 2].set_ylabel('Loss Variance')
            axes[1, 2].tick_params(axis='x', rotation=45)
            axes[1, 2].set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved comparison plot to {save_path}")
        
        plt.show()
    
    def generate_report(self, save_path=None):
        """Generate comprehensive report"""
        if not self.summary_data:
            print("No data to generate report")
            return
        
        summary_df = pd.DataFrame(self.summary_data)
        
        report = []
        report.append("=" * 80)
        report.append("EXPERIMENT RESULTS SUMMARY")
        report.append("=" * 80)
        
        # Overall statistics
        report.append(f"\nTotal experiments: {len(summary_df)}")
        report.append(f"Methods: {', '.join(summary_df['method'].unique())}")
        report.append(f"Models: {', '.join(summary_df['model'].unique())}")
        report.append(f"Datasets: {', '.join(summary_df['dataset'].unique())}")
        
        # Method comparison
        report.append("\n" + "=" * 80)
        report.append("METHOD COMPARISON")
        report.append("=" * 80)
        
        for method in summary_df['method'].unique():
            method_df = summary_df[summary_df['method'] == method]
            report.append(f"\n{method.upper()}:")
            report.append(f"  Experiments: {len(method_df)}")
            report.append(f"  Average final accuracy: {method_df['final_val_acc'].mean():.4f} ± {method_df['final_val_acc'].std():.4f}")
            report.append(f"  Best accuracy: {method_df['best_val_acc'].max():.4f}")
            report.append(f"  Average convergence epoch: {method_df['best_epoch'].mean():.1f}")
            
            if 'avg_delta_f' in method_df.columns and method_df['avg_delta_f'].mean() > 0:
                report.append(f"  Average Delta_F: {method_df['avg_delta_f'].mean():.2e}")
        
        # Best performing experiments
        report.append("\n" + "=" * 80)
        report.append("TOP 5 EXPERIMENTS BY VALIDATION ACCURACY")
        report.append("=" * 80)
        
        top5 = summary_df.nlargest(5, 'best_val_acc')
        for i, (_, row) in enumerate(top5.iterrows()):
            report.append(f"\n{i+1}. {row['exp_id']}")
            report.append(f"   Best accuracy: {row['best_val_acc']:.4f}")
            report.append(f"   Final accuracy: {row['final_val_acc']:.4f}")
            report.append(f"   Converged at epoch: {row['best_epoch']}")
        
        # Save report
        report_text = '\n'.join(report)
        print(report_text)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"\nReport saved to {save_path}")
        
        return report_text


def main():
    parser = argparse.ArgumentParser(description="Visualize experiment results")
    parser.add_argument('--result-dir', type=str, help='Single result directory')
    parser.add_argument('--result-dirs', type=str, nargs='+', help='Multiple result directories')
    parser.add_argument('--pattern', type=str, help='Pattern to match result directories')
    parser.add_argument('--out-dir', type=str, default='visualizations')
    
    args = parser.parse_args()
    
    # Find result directories
    if args.result_dir:
        result_dirs = [Path(args.result_dir)]
    elif args.result_dirs:
        result_dirs = [Path(d) for d in args.result_dirs]
    elif args.pattern:
        result_dirs = [Path(p) for p in glob.glob(args.pattern)]
    else:
        # Default: look in batch_results
        result_dirs = list(Path('batch_results').glob('*/*'))
    
    print(f"Found {len(result_dirs)} result directories")
    
    # Create visualizer
    vis = ExperimentVisualizer(result_dirs)
    vis.load_all_data()
    
    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    
    # Generate plots
    if vis.data:
        # Training curves
        vis.plot_training_curves(save_path=out_dir / 'training_curves.png')
        
        # Comparison summary
        vis.plot_comparison_summary(save_path=out_dir / 'comparison_summary.png')
        
        # Generate report
        vis.generate_report(save_path=out_dir / 'experiment_report.txt')
    else:
        print("No valid experiment data found")


if __name__ == "__main__":
    main()
