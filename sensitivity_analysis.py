#!/usr/bin/env python3
"""
Sensitivity Analysis Script (m2, m3 from tasks.md)

m2: Jitter scale sensitivity analysis - σ ∈ {0.01, 0.05, 0.10, 0.15}
m3: K selection threshold sensitivity - δ ∈ {0.03, 0.05, 0.10}

Usage:
    python sensitivity_analysis.py --analysis all
    python sensitivity_analysis.py --analysis m2_jitter
    python sensitivity_analysis.py --analysis m3_threshold
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import sys
from typing import Dict, List

import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from consensus import run_consensus
from data import load_dataset
from eval import stability_summary
from preprocess import prepare_representations


# Datasets to analyze
DATASETS = ["obscare", "heart_disease", "breast_cancer", "pima_diabetes", "hcv", "liver_disorders", "early_diabetes"]


def load_config(dataset: str) -> dict:
    """Load configuration for dataset."""
    config_path = Path(f"configs/{dataset}.yml")
    if not config_path.exists():
        config_path = Path("configs/default.yml")
    
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    # Ensure consensus config exists with defaults
    if "consensus" not in cfg:
        cfg["consensus"] = {}
    cfg["consensus"].setdefault("n_bootstraps", 50)
    cfg["consensus"].setdefault("numeric_jitter_scale", 0.05)
    cfg["consensus"].setdefault("sample_fraction", 1.0)
    cfg["consensus"].setdefault("feature_bootstrap", False)
    
    return cfg


def get_features(df, cfg):
    """Get feature matrix and representation type."""
    reps = prepare_representations(df, cfg)
    representation = cfg.get("representation", "numeric")
    
    if representation == "numeric" and reps.numeric is not None:
        X = reps.numeric
    elif representation == "mixed_encoded" and reps.mixed_encoded is not None:
        X = reps.mixed_encoded
    else:
        X = reps.numeric if reps.numeric is not None else reps.mixed_encoded
        representation = "numeric" if reps.numeric is not None else "mixed_encoded"
    
    return X, representation


# =============================================================================
# m2: Jitter Scale Sensitivity
# =============================================================================

def run_jitter_sensitivity(
    jitter_scales: List[float] = [0.0, 0.01, 0.05, 0.10, 0.15],
    k_values: List[int] = [2, 3, 4],
    seed: int = 42
) -> pd.DataFrame:
    """
    m2: Test sensitivity to jitter scale across all datasets.
    """
    print("\n" + "="*60)
    print("m2: JITTER SCALE SENSITIVITY")
    print(f"σ values: {jitter_scales}")
    print("="*60)
    
    results = []
    
    for dataset in DATASETS:
        print(f"\n[{dataset}]")
        cfg = load_config(dataset)
        df = load_dataset(cfg)
        X, representation = get_features(df, cfg)
        print(f"  N={len(df)}, Features={X.shape[1]}, Repr={representation}")
        
        original_jitter = cfg["consensus"].get("numeric_jitter_scale", 0.05)
        
        for sigma in jitter_scales:
            cfg["consensus"]["numeric_jitter_scale"] = sigma
            
            for k in k_values:
                try:
                    consensus_result = run_consensus(
                        X=X,
                        representation=representation,
                        cfg=cfg,
                        k=k,
                        seed=seed
                    )
                    
                    stability_df = stability_summary(
                        ari_scores=consensus_result.ari_scores,
                        labels=consensus_result.labels,
                        confidence=consensus_result.confidence,
                        X=X
                    )
                    stability = stability_df.iloc[0].to_dict()
                    
                    results.append({
                        'analysis': 'm2_jitter',
                        'dataset': dataset,
                        'N': len(df),
                        'jitter_scale': sigma,
                        'k': k,
                        'ari': stability['mean_ari'],
                        'ari_std': stability['std_ari'],
                        'confidence': stability['mean_confidence'],
                        'silhouette': stability.get('silhouette', np.nan)
                    })
                    print(f"    σ={sigma:.2f}, K={k}: ARI={stability['mean_ari']:.3f}")
                except Exception as e:
                    print(f"    σ={sigma:.2f}, K={k}: ERROR - {str(e)[:50]}")
        
        # Restore original
        cfg["consensus"]["numeric_jitter_scale"] = original_jitter
    
    return pd.DataFrame(results)


def plot_jitter_sensitivity(df: pd.DataFrame, output_path: str):
    """Create visualization: ARI vs. jitter scale across datasets."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    datasets = df['dataset'].unique()
    
    for i, dataset in enumerate(datasets):
        if i >= len(axes):
            break
        ax = axes[i]
        data = df[df['dataset'] == dataset]
        
        for k in sorted(data['k'].unique()):
            k_data = data[data['k'] == k].sort_values('jitter_scale')
            ax.plot(k_data['jitter_scale'], k_data['ari'], 
                   marker='o', label=f'K={k}', linewidth=2)
        
        N = data['N'].iloc[0]
        ax.set_title(f'{dataset} (N={N})', fontsize=12, fontweight='bold')
        ax.set_xlabel('Jitter Scale (σ)')
        ax.set_ylabel('Mean ARI')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    # Hide empty subplots
    for j in range(len(datasets), len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('m2: Jitter Scale Sensitivity Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Figure saved: {output_path}")


# =============================================================================
# m3: K Selection Threshold Sensitivity
# =============================================================================

def select_k_with_delta(k_summary: pd.DataFrame, delta: float, min_size: int = 2) -> int:
    """Select K using stability plateau with custom delta."""
    max_ari = k_summary["mean_ari"].max()
    plateau = k_summary[k_summary["mean_ari"] >= max_ari - delta]
    plateau = plateau[plateau["min_cluster_size"] >= min_size]
    if plateau.empty:
        plateau = k_summary
    # Tie-break by recall (if available), then mean ARI
    if "mean_recall" in plateau.columns:
        best = plateau.sort_values(
            by=["mean_recall", "mean_ari", "k"], ascending=[False, False, True]
        ).iloc[0]
    else:
        best = plateau.sort_values(
            by=["mean_ari", "k"], ascending=[False, True]
        ).iloc[0]
    return int(best["k"])


def run_k_threshold_sensitivity(
    delta_values: List[float] = [0.03, 0.05, 0.10],
    k_range: List[int] = [2, 3, 4, 5, 6],
    seed: int = 42
) -> pd.DataFrame:
    """
    m3: Test sensitivity to K selection threshold δ.
    """
    print("\n" + "="*60)
    print("m3: K SELECTION THRESHOLD SENSITIVITY")
    print(f"δ values: {delta_values}")
    print("="*60)
    
    results = []
    
    for dataset in DATASETS:
        print(f"\n[{dataset}]")
        cfg = load_config(dataset)
        df = load_dataset(cfg)
        X, representation = get_features(df, cfg)
        
        # Run K sweep to get stability metrics
        k_sweep_results = []
        for k in k_range:
            try:
                consensus_result = run_consensus(
                    X=X,
                    representation=representation,
                    cfg=cfg,
                    k=k,
                    seed=seed
                )
                
                stability_df = stability_summary(
                    ari_scores=consensus_result.ari_scores,
                    labels=consensus_result.labels,
                    confidence=consensus_result.confidence,
                    X=X
                )
                stability = stability_df.iloc[0].to_dict()
                stability['k'] = k
                k_sweep_results.append(stability)
            except Exception as e:
                print(f"    K={k}: SKIPPED - {str(e)[:30]}")
        
        if not k_sweep_results:
            continue
            
        k_summary = pd.DataFrame(k_sweep_results)
        
        # Test each delta value
        for delta in delta_values:
            selected_k = select_k_with_delta(k_summary, delta)
            selected_ari = k_summary[k_summary['k'] == selected_k]['mean_ari'].values[0]
            
            results.append({
                'analysis': 'm3_threshold',
                'dataset': dataset,
                'delta': delta,
                'selected_k': selected_k,
                'selected_ari': selected_ari,
                'max_ari': k_summary['mean_ari'].max(),
                'ari_gap': k_summary['mean_ari'].max() - selected_ari
            })
            print(f"  δ={delta:.2f}: Selected K={selected_k}, ARI={selected_ari:.3f}")
    
    return pd.DataFrame(results)


def plot_k_threshold_sensitivity(df: pd.DataFrame, output_path: str):
    """Create visualization: Selected K by delta threshold."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Selected K by delta for each dataset
    ax1 = axes[0]
    datasets = df['dataset'].unique()
    x = np.arange(len(datasets))
    width = 0.25
    
    for i, delta in enumerate(sorted(df['delta'].unique())):
        delta_data = df[df['delta'] == delta].set_index('dataset')
        k_values = [delta_data.loc[d, 'selected_k'] if d in delta_data.index else 0 for d in datasets]
        ax1.bar(x + i*width, k_values, width, label=f'δ={delta}', alpha=0.8)
    
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('Selected K')
    ax1.set_title('Selected K by Threshold δ', fontweight='bold')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(datasets, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: ARI gap from max by delta
    ax2 = axes[1]
    for delta in sorted(df['delta'].unique()):
        delta_data = df[df['delta'] == delta]
        ax2.scatter([delta] * len(delta_data), delta_data['ari_gap'], 
                   s=100, alpha=0.7, label=f'δ={delta}')
    
    # Add mean line
    mean_gaps = df.groupby('delta')['ari_gap'].mean()
    ax2.plot(mean_gaps.index, mean_gaps.values, 'k-o', linewidth=2, 
             markersize=10, label='Mean gap')
    
    ax2.set_xlabel('Threshold δ')
    ax2.set_ylabel('ARI Gap from Maximum')
    ax2.set_title('ARI Loss by Threshold Choice', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('m3: K Selection Threshold Sensitivity', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Figure saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Sensitivity analysis (m2, m3)")
    parser.add_argument("--analysis", default="all", 
                       choices=["all", "m2_jitter", "m3_threshold"],
                       help="Which analysis to run")
    args = parser.parse_args()
    
    output_dir = Path("reports/sensitivity")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path("../benchmark-data-paper/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    if args.analysis in ["all", "m2_jitter"]:
        results = run_jitter_sensitivity()
        results.to_csv(output_dir / "m2_jitter_sensitivity.csv", index=False)
        print(f"\n✓ Results saved: {output_dir / 'm2_jitter_sensitivity.csv'}")
        
        plot_jitter_sensitivity(results, str(figures_dir / "jitter_sensitivity.png"))
        all_results.append(results)
    
    if args.analysis in ["all", "m3_threshold"]:
        results = run_k_threshold_sensitivity()
        results.to_csv(output_dir / "m3_k_threshold_sensitivity.csv", index=False)
        print(f"\n✓ Results saved: {output_dir / 'm3_k_threshold_sensitivity.csv'}")
        
        plot_k_threshold_sensitivity(results, str(figures_dir / "k_threshold_sensitivity.png"))
        all_results.append(results)
    
    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
