"""
Evaluation Module (Novelty 5: Real-World Generalization Study)
==============================================================

Comprehensive evaluation harness for comparing model performance
on synthetic vs. real-world degradations.

Computes: PSNR, SSIM, IFS (Instruction Faithfulness Score)
Generates: comparison tables, bar charts, analysis

Usage:
    evaluator = Evaluator(pipeline, ifs_scorer)
    results = evaluator.evaluate_dataset(test_images, instructions)
    evaluator.print_summary(results)
"""

import numpy as np
import os
from typing import List, Dict, Any, Optional, Tuple
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt


class Evaluator:
    """
    Evaluation harness for instruction-driven image restoration.
    """
    
    def __init__(self, pipeline=None, ifs_scorer=None):
        """
        Args:
            pipeline: RestorationPipeline instance.
            ifs_scorer: InstructionFaithfulnessScore instance.
        """
        self.pipeline = pipeline
        self.ifs_scorer = ifs_scorer
    
    def evaluate_single(
        self, degraded: np.ndarray, instruction: str,
        ground_truth: Optional[np.ndarray] = None,
        restored: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single image-instruction pair.
        
        Args:
            degraded: Degraded input image (H, W, 3).
            instruction: Text instruction.
            ground_truth: Optional clean ground truth.
            restored: Pre-computed restored image (if None, uses pipeline).
            
        Returns:
            Dictionary with metrics and images.
        """
        if restored is None and self.pipeline is not None:
            restored = self.pipeline.process_single(degraded, instruction)
        
        result = {
            'instruction': instruction,
            'degraded': degraded,
            'restored': restored,
            'ground_truth': ground_truth,
        }
        
        # Compute reference metrics if ground truth is available
        if ground_truth is not None and restored is not None:
            result['psnr_degraded'] = float(psnr(ground_truth, degraded, data_range=1.0))
            result['psnr_restored'] = float(psnr(ground_truth, restored, data_range=1.0))
            result['ssim_degraded'] = float(ssim(ground_truth, degraded, 
                                                  data_range=1.0, channel_axis=2))
            result['ssim_restored'] = float(ssim(ground_truth, restored, 
                                                  data_range=1.0, channel_axis=2))
            result['psnr_gain'] = result['psnr_restored'] - result['psnr_degraded']
            result['ssim_gain'] = result['ssim_restored'] - result['ssim_degraded']
        
        # Compute IFS
        if self.ifs_scorer is not None and restored is not None:
            ifs_scores = self.ifs_scorer.compute(
                degraded, restored, instruction, ground_truth
            )
            result['ifs_scores'] = ifs_scores
            result['ifs'] = ifs_scores['IFS']
        
        return result
    
    def evaluate_dataset(
        self, test_cases: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Evaluate a dataset of test cases.
        
        Args:
            test_cases: List of dicts with 'degraded', 'instruction', 
                       optionally 'ground_truth', 'category'.
                       
        Returns:
            List of evaluation results.
        """
        results = []
        for i, case in enumerate(test_cases):
            print(f"  Evaluating [{i+1}/{len(test_cases)}]: {case['instruction'][:50]}...")
            result = self.evaluate_single(
                degraded=case['degraded'],
                instruction=case['instruction'],
                ground_truth=case.get('ground_truth'),
                restored=case.get('restored'),
            )
            result['category'] = case.get('category', 'unknown')
            result['domain'] = case.get('domain', 'unknown')  # 'synthetic' or 'real_world'
            result['image_name'] = case.get('name', f'image_{i}')
            results.append(result)
        
        return results
    
    def compute_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute summary statistics from evaluation results."""
        summary = {
            'total_images': len(results),
            'categories': {},
            'domains': {},
            'overall': {},
        }
        
        # Collect metrics
        all_metrics = {'psnr_gain': [], 'ssim_gain': [], 'ifs': []}
        
        for r in results:
            category = r.get('category', 'unknown')
            domain = r.get('domain', 'unknown')
            
            if category not in summary['categories']:
                summary['categories'][category] = {'psnr_gain': [], 'ssim_gain': [], 'ifs': []}
            if domain not in summary['domains']:
                summary['domains'][domain] = {'psnr_gain': [], 'ssim_gain': [], 'ifs': []}
            
            for metric in ['psnr_gain', 'ssim_gain', 'ifs']:
                val = r.get(metric)
                if val is not None:
                    summary['categories'][category][metric].append(val)
                    summary['domains'][domain][metric].append(val)
                    all_metrics[metric].append(val)
        
        # Compute means
        for group_name in ['categories', 'domains']:
            for group_key in summary[group_name]:
                for metric in ['psnr_gain', 'ssim_gain', 'ifs']:
                    vals = summary[group_name][group_key][metric]
                    if vals:
                        summary[group_name][group_key][f'{metric}_mean'] = np.mean(vals)
                        summary[group_name][group_key][f'{metric}_std'] = np.std(vals)
        
        for metric in all_metrics:
            if all_metrics[metric]:
                summary['overall'][f'{metric}_mean'] = np.mean(all_metrics[metric])
                summary['overall'][f'{metric}_std'] = np.std(all_metrics[metric])
        
        return summary
    
    def print_summary(self, results: List[Dict[str, Any]]):
        """Print a formatted evaluation summary."""
        summary = self.compute_summary(results)
        
        print("\n" + "=" * 70)
        print("        EVALUATION SUMMARY — Instruction-Driven IR")
        print("=" * 70)
        print(f"  Total images evaluated: {summary['total_images']}")
        
        # Overall metrics
        print("\n  ─── Overall Metrics ───")
        for metric_key, metric_name in [('psnr_gain', 'PSNR Gain (dB)'), 
                                          ('ssim_gain', 'SSIM Gain'),
                                          ('ifs', 'IFS Score')]:
            mean_key = f'{metric_key}_mean'
            std_key = f'{metric_key}_std'
            if mean_key in summary['overall']:
                print(f"  {metric_name:20s}: {summary['overall'][mean_key]:.4f} "
                      f"± {summary['overall'][std_key]:.4f}")
        
        # By category
        if len(summary['categories']) > 1:
            print("\n  ─── By Degradation Category ───")
            print(f"  {'Category':15s} {'PSNR Gain':>12s} {'SSIM Gain':>12s} {'IFS':>12s}")
            print(f"  {'─' * 51}")
            for cat, metrics in sorted(summary['categories'].items()):
                psnr_str = f"{metrics.get('psnr_gain_mean', 0):.2f}" if metrics.get('psnr_gain_mean') is not None else "N/A"
                ssim_str = f"{metrics.get('ssim_gain_mean', 0):.4f}" if metrics.get('ssim_gain_mean') is not None else "N/A"
                ifs_str = f"{metrics.get('ifs_mean', 0):.4f}" if metrics.get('ifs_mean') is not None else "N/A"
                print(f"  {cat:15s} {psnr_str:>12s} {ssim_str:>12s} {ifs_str:>12s}")
        
        # By domain (synthetic vs real-world)
        if len(summary['domains']) > 1:
            print("\n  ─── By Domain (Synthetic vs Real-World) ───")
            print(f"  {'Domain':15s} {'PSNR Gain':>12s} {'SSIM Gain':>12s} {'IFS':>12s}")
            print(f"  {'─' * 51}")
            for dom, metrics in sorted(summary['domains'].items()):
                psnr_str = f"{metrics.get('psnr_gain_mean', 0):.2f}" if metrics.get('psnr_gain_mean') is not None else "N/A"
                ssim_str = f"{metrics.get('ssim_gain_mean', 0):.4f}" if metrics.get('ssim_gain_mean') is not None else "N/A"
                ifs_str = f"{metrics.get('ifs_mean', 0):.4f}" if metrics.get('ifs_mean') is not None else "N/A"
                print(f"  {dom:15s} {psnr_str:>12s} {ssim_str:>12s} {ifs_str:>12s}")
        
        print("\n" + "=" * 70)
    
    def plot_domain_comparison(self, results: List[Dict[str, Any]], 
                                save_path: Optional[str] = None):
        """
        Bar chart comparing metrics across synthetic vs real-world domains.
        """
        summary = self.compute_summary(results)
        domains = summary['domains']
        
        if len(domains) < 2:
            print("Need both synthetic and real-world results for comparison plot.")
            return
        
        metrics = ['ifs']
        metric_labels = ['IFS Score']
        
        # Check if reference metrics available
        has_psnr = any('psnr_gain_mean' in v for v in domains.values() 
                       if v.get('psnr_gain_mean') is not None)
        if has_psnr:
            metrics.insert(0, 'ssim_gain')
            metric_labels.insert(0, 'SSIM Gain')
            metrics.insert(0, 'psnr_gain')
            metric_labels.insert(0, 'PSNR Gain (dB)')
        
        domain_names = sorted(domains.keys())
        n_metrics = len(metrics)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]
        
        colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            means = []
            stds = []
            for dom in domain_names:
                mean_val = domains[dom].get(f'{metric}_mean', 0) or 0
                std_val = domains[dom].get(f'{metric}_std', 0) or 0
                means.append(mean_val)
                stds.append(std_val)
            
            bars = axes[i].bar(domain_names, means, yerr=stds, 
                              color=colors[:len(domain_names)], capsize=5, 
                              edgecolor='black', linewidth=0.5)
            axes[i].set_title(label, fontsize=12, fontweight='bold')
            axes[i].set_ylabel(label)
            
            # Add value labels on bars
            for bar, mean in zip(bars, means):
                axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                           f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle("Synthetic vs Real-World Generalization", 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Plot saved to: {save_path}")
        plt.show()
