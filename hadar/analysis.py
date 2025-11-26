import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict
from pathlib import Path
from .visualizer import HadarVisualizer


class HadarAnalyzer:    
    def __init__(self, metrics_data: List[Dict], output_dir: str = "hadar_results"):
        self.metrics_data = metrics_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.metrics_dir = self.output_dir / "metrics"
        self.metrics_dir.mkdir(exist_ok=True)
        
        self.df = pd.DataFrame(metrics_data)
        self._compute_derived_metrics()
        
        self.visualizer = HadarVisualizer(self.df, metrics_data, output_dir)
        
    def _compute_derived_metrics(self):
        if len(self.df) == 0:
            return
        
        self.df['hallucination_rate'] = self.df.apply(
            lambda row: row['hallucinated_count'] / row['total_sentences'] 
            if row['total_sentences'] > 0 else 0.0,
            axis=1
        )
        
        if 'correction_efficiency' not in self.df.columns:
            self.df['correction_efficiency'] = self.df.apply(
                lambda row: np.tanh((row.get('post_mean_consistency', row['mean_consistency']) - row['mean_consistency']) / 
                                   (row.get('hallucinated_count', 1) + 1e-6))
                if row.get('corrected_count', 0) > 0 else 0.0,
                axis=1
            )
        
        self.df['delta_consistency'] = (
            self.df.get('post_mean_consistency', self.df['mean_consistency']) - self.df['mean_consistency']
        )
        
        self.df['hii'] = (1 - self.df['hallucination_rate']) * self.df['mean_consistency']
        self.df['fzii'] = self.df['hii']
        
        if 'acceptance_ratio' not in self.df.columns:
            self.df['acceptance_ratio'] = 0.0
        
    def get_aggregate_metrics(self) -> Dict:
        if len(self.df) == 0:
            return {}
            
        total_sentences = self.df['total_sentences'].sum()
        total_hallucinated = self.df['hallucinated_count'].sum()
        total_corrected = self.df['corrected_count'].sum()
        
        metrics = {
            'total_rounds': len(self.df['round'].unique()) if 'round' in self.df.columns else len(self.df),
            'total_sentences_analyzed': int(total_sentences),
            'total_hallucinations_detected': int(total_hallucinated),
            'total_corrections_applied': int(total_corrected),
            'overall_hallucination_rate': float(total_hallucinated / total_sentences) if total_sentences > 0 else 0,
            'mean_consistency_score': float(self.df['mean_consistency'].mean()),
            'std_consistency_score': float(self.df['mean_consistency'].std()),
            'mean_hii': float(self.df['hii'].mean()),
            'mean_fzii': float(self.df['fzii'].mean()),
            'mean_correction_efficiency': float(self.df['correction_efficiency'].mean()),
            'mean_delta_consistency': float(self.df['delta_consistency'].mean()) if 'delta_consistency' in self.df.columns else 0.0,
            'avg_acceptance_ratio': float(self.df['acceptance_ratio'].mean()) if 'acceptance_ratio' in self.df.columns else 0.0,
            'debater_a_mean_consistency': float(self.df[self.df['debater'] == 'A']['mean_consistency'].mean()) if 'A' in self.df['debater'].values else None,
            'debater_b_mean_consistency': float(self.df[self.df['debater'] == 'B']['mean_consistency'].mean()) if 'B' in self.df['debater'].values else None,
            'debater_a_hallucination_rate': float(self.df[self.df['debater'] == 'A']['hallucination_rate'].mean()) if 'A' in self.df['debater'].values else None,
            'debater_b_hallucination_rate': float(self.df[self.df['debater'] == 'B']['hallucination_rate'].mean()) if 'B' in self.df['debater'].values else None,
        }
        
        return metrics
    
    def save_metrics_to_csv(self, filename: str = 'aggregate_metrics.csv'):
        metrics = self.get_aggregate_metrics()
        if not metrics:
            print("No metrics to save.")
            return
        
        df_metrics = pd.DataFrame([metrics])
        output_path = self.metrics_dir / filename
        df_metrics.to_csv(output_path, index=False)
        print(f"✓ Saved: {output_path}")
    
    def save_metrics_to_json(self, filename: str = 'aggregate_metrics.json'):
        metrics = self.get_aggregate_metrics()
        if not metrics:
            print("No metrics to save.")
            return
        
        output_path = self.metrics_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        print(f"✓ Saved: {output_path}")
    
    def save_detailed_metrics_csv(self, filename: str = 'detailed_metrics.csv'):
        if len(self.df) == 0:
            print("No detailed metrics to save.")
            return
        
        output_path = self.metrics_dir / filename
        self.df.to_csv(output_path, index=False)
        print(f"✓ Saved: {output_path}")
    
    def generate_all_visualizations(self):
        print("\n" + "="*60)
        print("GENERATING HADAR VISUALIZATIONS & METRICS")
        print("="*60 + "\n")
        
        aggregate_metrics = self.get_aggregate_metrics()
        self.visualizer.generate_all_plots(aggregate_metrics)
        
        print("\n🧾 Saving Metrics:")
        self.save_metrics_to_csv()
        self.save_metrics_to_json()
        self.save_detailed_metrics_csv()
        
        print("\n" + "="*60)
        print(f"✅ All outputs saved to: {self.output_dir.absolute()}")
        print("="*60 + "\n")


def analyze_debate_metrics(metrics_data: List[Dict], output_dir: str = "hadar_results"):
    """Analyze debate metrics and generate visualizations."""
    analyzer = HadarAnalyzer(metrics_data, output_dir)
    analyzer.generate_all_visualizations()
    return analyzer


# Backward compatibility aliases
FinchZkAnalyzer = HadarAnalyzer
