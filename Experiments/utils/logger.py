"""
Unified logging system for experiments
"""
import csv
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd


class UnifiedLogger:
    """Unified logger for all experiments"""
    
    def __init__(self, out_dir: Path, method: str, run_id: int, model_name: str):
        self.out_dir = Path(out_dir)
        self.method = method
        self.run_id = run_id
        self.model_name = model_name
        
        # Create directories
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize log files
        self.step_log_file = self.out_dir / f"step_logs_{method}_run{run_id}.csv"
        self.epoch_log_file = self.out_dir / f"epoch_logs_{method}_run{run_id}.csv"
        self.summary_file = self.out_dir / f"summary_{method}_run{run_id}.json"
        
        # Initialize CSV writers
        self.step_writer = None
        self.epoch_writer = None
        self.step_fieldnames = None
        self.epoch_fieldnames = None
        
        # Store all data for later analysis
        self.step_data = []
        self.epoch_data = []
        
        # Track best metrics
        self.best_metrics = {
            'val_acc': 0.0,
            'val_loss': float('inf'),
            'step': 0,
            'epoch': 0
        }
        
        # Start time
        self.start_time = time.time()
    
    def log_step(self, step_metrics: Dict[str, Any]):
        """Log step-level metrics"""
        # Add timestamp
        step_metrics['timestamp'] = time.time() - self.start_time
        
        # Store in memory
        self.step_data.append(step_metrics.copy())
        
        # Write to CSV (append)
        self._write_step_to_csv(step_metrics)
        
        # Update best metrics
        if 'val_acc' in step_metrics and step_metrics['val_acc'] > self.best_metrics['val_acc']:
            self.best_metrics['val_acc'] = step_metrics['val_acc']
            self.best_metrics['step'] = step_metrics.get('step', 0)
        
        if 'val_loss' in step_metrics and step_metrics['val_loss'] < self.best_metrics['val_loss']:
            self.best_metrics['val_loss'] = step_metrics['val_loss']
    
    def log_epoch(self, epoch_metrics: Dict[str, Any]):
        """Log epoch-level metrics"""
        # Add timestamp
        epoch_metrics['timestamp'] = time.time() - self.start_time
        
        # Store in memory
        self.epoch_data.append(epoch_metrics.copy())
        
        # Write to CSV (append)
        self._write_epoch_to_csv(epoch_metrics)
        
        # Update best metrics
        if 'val_acc' in epoch_metrics and epoch_metrics['val_acc'] > self.best_metrics['val_acc']:
            self.best_metrics['val_acc'] = epoch_metrics['val_acc']
            self.best_metrics['epoch'] = epoch_metrics.get('epoch', 0)
        
        if 'val_loss' in epoch_metrics and epoch_metrics['val_loss'] < self.best_metrics['val_loss']:
            self.best_metrics['val_loss'] = epoch_metrics['val_loss']
    
    def log_final(self, final_metrics: Dict[str, Any]):
        """Log final experiment metrics"""
        final_metrics['total_time'] = time.time() - self.start_time
        final_metrics['best_metrics'] = self.best_metrics
        
        # Save final metrics
        with open(self.summary_file, 'w') as f:
            json.dump(final_metrics, f, indent=2)
    
    def _write_step_to_csv(self, metrics: Dict[str, Any]):
        """Write step metrics to CSV file"""
        # Ensure all fields are written in correct order
        if self.step_fieldnames is None:
            self.step_fieldnames = list(metrics.keys())
            with open(self.step_log_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.step_fieldnames)
                writer.writeheader()
        
        # Write row
        with open(self.step_log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.step_fieldnames)
            # Ensure all fields are present
            row = {field: metrics.get(field, '') for field in self.step_fieldnames}
            writer.writerow(row)
    
    def _write_epoch_to_csv(self, metrics: Dict[str, Any]):
        """Write epoch metrics to CSV file"""
        # Ensure all fields are written in correct order
        if self.epoch_fieldnames is None:
            self.epoch_fieldnames = list(metrics.keys())
            with open(self.epoch_log_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.epoch_fieldnames)
                writer.writeheader()
        
        # Write row
        with open(self.epoch_log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.epoch_fieldnames)
            # Ensure all fields are present
            row = {field: metrics.get(field, '') for field in self.epoch_fieldnames}
            writer.writerow(row)
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics from logged data"""
        if not self.step_data or not self.epoch_data:
            return {}
        
        # Convert to DataFrames for easier analysis
        step_df = pd.DataFrame(self.step_data)
        epoch_df = pd.DataFrame(self.epoch_data)
        
        summary = {
            'experiment_info': {
                'method': self.method,
                'run_id': self.run_id,
                'model': self.model_name,
                'out_dir': str(self.out_dir),
                'total_time': time.time() - self.start_time,
            },
            'best_metrics': self.best_metrics,
            'final_metrics': {
                'final_train_loss': epoch_df['train_loss_avg'].iloc[-1] if 'train_loss_avg' in epoch_df.columns else None,
                'final_val_loss': epoch_df['val_loss'].iloc[-1] if 'val_loss' in epoch_df.columns else None,
                'final_val_acc': epoch_df['val_acc'].iloc[-1] if 'val_acc' in epoch_df.columns else None,
            },
            'convergence_analysis': self._analyze_convergence(step_df, epoch_df),
            'optimization_analysis': self._analyze_optimization(step_df),
        }
        
        # Save summary
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def _analyze_convergence(self, step_df: pd.DataFrame, epoch_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze convergence characteristics"""
        analysis = {}
        
        if 'val_acc' in epoch_df.columns:
            # Find epoch where 90% of max accuracy is reached
            max_acc = epoch_df['val_acc'].max()
            target_acc = 0.9 * max_acc
            convergence_epoch = epoch_df[epoch_df['val_acc'] >= target_acc]['epoch'].min()
            if pd.notna(convergence_epoch):
                analysis['convergence_epoch_90pct'] = float(convergence_epoch)
        
        if 'train_loss' in step_df.columns:
            # Analyze loss decrease
            initial_loss = step_df['train_loss'].iloc[:100].mean()
            final_loss = step_df['train_loss'].iloc[-100:].mean()
            analysis['loss_decrease_ratio'] = float(final_loss / initial_loss) if initial_loss > 0 else 0.0
        
        return analysis
    
    def _analyze_optimization(self, step_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze optimization characteristics"""
        analysis = {}
        
        if 'delta_f' in step_df.columns:
            analysis['avg_delta_f'] = float(step_df['delta_f'].mean())
            analysis['std_delta_f'] = float(step_df['delta_f'].std())
            analysis['delta_f_trend'] = float(step_df['delta_f'].iloc[-100:].mean() / step_df['delta_f'].iloc[:100].mean())
        
        if 'eta_t' in step_df.columns:
            analysis['avg_eta_t'] = float(step_df['eta_t'].mean())
            analysis['std_eta_t'] = float(step_df['eta_t'].std())
        
        if 'grad_norm' in step_df.columns:
            analysis['avg_grad_norm'] = float(step_df['grad_norm'].mean())
            analysis['grad_norm_trend'] = float(step_df['grad_norm'].iloc[-100:].mean() / step_df['grad_norm'].iloc[:100].mean())
        
        return analysis
    
    def get_step_dataframe(self) -> pd.DataFrame:
        """Get step data as DataFrame"""
        return pd.DataFrame(self.step_data)
    
    def get_epoch_dataframe(self) -> pd.DataFrame:
        """Get epoch data as DataFrame"""
        return pd.DataFrame(self.epoch_data)
