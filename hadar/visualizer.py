"""
HADAR Visualizer Module

Generates visualizations for HADAR analysis results including:
- Consistency score distributions
- Hallucination detection heatmaps  
- Correction efficiency plots
- Debater comparison charts
"""

import gc
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns


class HadarVisualizer:
    """Generates visualizations for HADAR analysis results."""
    
    def __init__(self, df: pd.DataFrame, metrics_data: List[Dict], base_dir: str = "hadar_results"):
        self.df = df
        self.metrics_data = metrics_data
        self.base_dir = Path(base_dir)
        
        # Create output directories
        self.analysis_dir = self.base_dir / "consistency_analysis"
        self.hallucination_dir = self.base_dir / "hallucination_analysis"
        self.aggregate_dir = self.base_dir / "aggregate_metrics"
        
        for dir_path in [self.analysis_dir, self.hallucination_dir, self.aggregate_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Set style
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except OSError:
            plt.style.use("seaborn-whitegrid")
    
    def generate_all_plots(self, aggregate_metrics: Dict[str, Any]) -> None:
        """Generate all visualization plots."""
        print("\n📊 Generating Visualizations:")
        
        if len(self.df) == 0:
            print("  ⚠️  No data available for visualization")
            return
        
        try:
            self._plot_consistency_distribution()
            self._plot_hallucination_rates()
            self._plot_correction_efficiency()
            self._plot_debater_comparison()
            self._plot_round_progression()
            self._plot_aggregate_summary(aggregate_metrics)
        except Exception as e:
            print(f"  ⚠️  Error generating plots: {e}")
        finally:
            plt.close('all')
            gc.collect()
    
    def _plot_consistency_distribution(self) -> None:
        """Plot distribution of consistency scores."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.histplot(data=self.df, x='mean_consistency', hue='debater', 
                    kde=True, ax=ax, alpha=0.6)
        
        ax.axvline(x=0.65, color='red', linestyle='--', label='Threshold (0.65)')
        ax.set_xlabel('Mean Consistency Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Consistency Scores by Debater')
        ax.legend()
        
        output_path = self.analysis_dir / 'consistency_distribution.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved: {output_path}")
    
    def _plot_hallucination_rates(self) -> None:
        """Plot hallucination rates across rounds."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if 'round' in self.df.columns and 'hallucination_rate' in self.df.columns:
            for debater in self.df['debater'].unique():
                debater_data = self.df[self.df['debater'] == debater]
                ax.plot(debater_data['round'], debater_data['hallucination_rate'], 
                       marker='o', label=f'Debater {debater}', linewidth=2)
        
        ax.set_xlabel('Round')
        ax.set_ylabel('Hallucination Rate')
        ax.set_title('Hallucination Rate Progression by Round')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        output_path = self.hallucination_dir / 'hallucination_rates.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved: {output_path}")
    
    def _plot_correction_efficiency(self) -> None:
        """Plot correction efficiency over time."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if 'correction_efficiency' in self.df.columns:
            for debater in self.df['debater'].unique():
                debater_data = self.df[self.df['debater'] == debater]
                if 'round' in debater_data.columns:
                    ax.bar(debater_data['round'] + (0.2 if debater == 'B' else -0.2), 
                          debater_data['correction_efficiency'],
                          width=0.4, label=f'Debater {debater}', alpha=0.7)
        
        ax.set_xlabel('Round')
        ax.set_ylabel('Correction Efficiency (tanh)')
        ax.set_title('Correction Efficiency by Round')
        ax.legend()
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        
        output_path = self.analysis_dir / 'correction_efficiency.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved: {output_path}")
    
    def _plot_debater_comparison(self) -> None:
        """Plot comparison between debaters."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        metrics_to_plot = [
            ('mean_consistency', 'Mean Consistency'),
            ('hallucination_rate', 'Hallucination Rate'),
            ('corrected_count', 'Corrections Applied'),
            ('acceptance_ratio', 'Acceptance Ratio')
        ]
        
        for ax, (metric, title) in zip(axes.flat, metrics_to_plot):
            if metric in self.df.columns:
                debater_means = self.df.groupby('debater')[metric].mean()
                debater_means.plot(kind='bar', ax=ax, color=['#3498db', '#e74c3c'])
                ax.set_title(title)
                ax.set_xlabel('Debater')
                ax.set_ylabel(title)
                ax.tick_params(axis='x', rotation=0)
        
        plt.suptitle('Debater Performance Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.aggregate_dir / 'debater_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved: {output_path}")
    
    def _plot_round_progression(self) -> None:
        """Plot metrics progression across rounds."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if 'round' in self.df.columns:
            round_means = self.df.groupby('round').agg({
                'mean_consistency': 'mean',
                'hallucination_rate': 'mean'
            }).reset_index()
            
            ax.plot(round_means['round'], round_means['mean_consistency'], 
                   marker='s', label='Mean Consistency', linewidth=2, color='#2ecc71')
            ax.plot(round_means['round'], round_means['hallucination_rate'], 
                   marker='^', label='Hallucination Rate', linewidth=2, color='#e74c3c')
        
        ax.set_xlabel('Round')
        ax.set_ylabel('Score')
        ax.set_title('Metrics Progression Across Rounds')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        output_path = self.aggregate_dir / 'round_progression.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved: {output_path}")
    
    def _plot_aggregate_summary(self, aggregate_metrics: Dict[str, Any]) -> None:
        """Plot aggregate summary metrics."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Select key metrics for visualization
        key_metrics = {
            'Overall Hallucination Rate': aggregate_metrics.get('overall_hallucination_rate', 0),
            'Mean Consistency': aggregate_metrics.get('mean_consistency_score', 0),
            'Mean HII': aggregate_metrics.get('mean_hii', 0),
            'Correction Efficiency': aggregate_metrics.get('mean_correction_efficiency', 0),
            'Acceptance Ratio': aggregate_metrics.get('avg_acceptance_ratio', 0),
        }
        
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
        bars = ax.barh(list(key_metrics.keys()), list(key_metrics.values()), color=colors)
        
        # Add value labels
        for bar, value in zip(bars, key_metrics.values()):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{value:.3f}', va='center', fontsize=10)
        
        ax.set_xlabel('Score')
        ax.set_title('HADAR Aggregate Metrics Summary')
        ax.set_xlim(0, 1.1)
        
        output_path = self.aggregate_dir / 'aggregate_summary.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved: {output_path}")


# Backward compatibility alias
FinchZkVisualizer = HadarVisualizer

__all__ = ["HadarVisualizer", "FinchZkVisualizer"]
