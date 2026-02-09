#!/usr/bin/env python3
"""
Sensitivity Analysis Script (m2, m3, h4 from tasks.md)

m2: Jitter scale sensitivity analysis - σ ∈ {0.01, 0.05, 0.10, 0.15}
m3: K selection threshold sensitivity - δ ∈ {0.03, 0.05, 0.10}
h4: Alpha (mixed_weight) sensitivity analysis - α ∈ {0.2, 0.3, 0.5, 0.7, 0.8}

Usage:
    python sensitivity_analysis.py --analysis all
    python sensitivity_analysis.py --analysis m2_jitter
    python sensitivity_analysis.py --analysis m3_threshold
    python sensitivity_analysis.py --analysis h4_alpha
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

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.consensus import run_consensus
from src.data import load_dataset
from src.eval import stability_summary
from src.preprocess import prepare_representations


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
# h4: Alpha (Mixed Weight) Sensitivity
# =============================================================================

def run_alpha_sensitivity(
    alpha_values: List[float] = [0.2, 0.3, 0.5, 0.7, 0.8],
    k_values: List[int] = [2, 3, 4],
    seed: int = 42
) -> pd.DataFrame:
    """
    h4: Test sensitivity to alpha (mixed_weight) parameter for mixed_separated representation.

    Alpha controls the blending of numeric and categorical co-assignment matrices:
        C = α * C_num + (1 - α) * C_cat

    α = 1.0 → pure numeric clustering
    α = 0.0 → pure categorical clustering
    α = 0.5 → equal weighting (default)
    """
    print("\n" + "="*60)
    print("h4: ALPHA (MIXED_WEIGHT) SENSITIVITY")
    print(f"α values: {alpha_values}")
    print("Testing mixed_separated representation vs mixed_encoded baseline")
    print("="*60)

    results = []

    for dataset in DATASETS:
        print(f"\n[{dataset}]")
        cfg = load_config(dataset)
        df = load_dataset(cfg)

        # Get representations - need both numeric and categorical
        reps = prepare_representations(df, cfg)

        # Check if dataset has both numeric and categorical features
        has_numeric = reps.numeric is not None and reps.numeric.shape[1] > 0
        has_categorical = reps.categorical is not None and reps.categorical.shape[1] > 0

        if not (has_numeric and has_categorical):
            print(f"  SKIPPED - needs both numeric ({has_numeric}) and categorical ({has_categorical}) features")
            continue

        X_num = reps.numeric
        X_cat = reps.categorical
        X_mixed_encoded = reps.mixed_encoded

        print(f"  N={len(df)}, Numeric={X_num.shape[1]}, Categorical={X_cat.shape[1]}")

        # First, run mixed_encoded baseline for comparison
        print("  Baseline (mixed_encoded):")
        for k in k_values:
            try:
                consensus_result = run_consensus(
                    X=X_mixed_encoded,
                    representation="mixed_encoded",
                    cfg=cfg,
                    k=k,
                    seed=seed
                )

                stability_df = stability_summary(
                    ari_scores=consensus_result.ari_scores,
                    labels=consensus_result.labels,
                    confidence=consensus_result.confidence,
                    X=X_mixed_encoded
                )
                stability = stability_df.iloc[0].to_dict()

                results.append({
                    'analysis': 'h4_alpha',
                    'dataset': dataset,
                    'N': len(df),
                    'representation': 'mixed_encoded',
                    'alpha': None,  # Not applicable for mixed_encoded
                    'k': k,
                    'ari': stability['mean_ari'],
                    'ari_std': stability['std_ari'],
                    'confidence': stability['mean_confidence'],
                    'silhouette': stability.get('silhouette', np.nan),
                    'n_numeric_features': X_num.shape[1],
                    'n_categorical_features': X_cat.shape[1]
                })
                print(f"    K={k}: ARI={stability['mean_ari']:.3f}")
            except Exception as e:
                print(f"    K={k}: ERROR - {str(e)[:50]}")

        # Now test mixed_separated with different alpha values
        print("  Mixed_separated:")
        for alpha in alpha_values:
            cfg["consensus"]["mixed_weight"] = alpha

            for k in k_values:
                try:
                    consensus_result = run_consensus(
                        X=(X_num, X_cat),
                        representation="mixed_separated",
                        cfg=cfg,
                        k=k,
                        seed=seed
                    )

                    # For silhouette, we need a single feature matrix - use mixed_encoded
                    stability_df = stability_summary(
                        ari_scores=consensus_result.ari_scores,
                        labels=consensus_result.labels,
                        confidence=consensus_result.confidence,
                        X=X_mixed_encoded
                    )
                    stability = stability_df.iloc[0].to_dict()

                    results.append({
                        'analysis': 'h4_alpha',
                        'dataset': dataset,
                        'N': len(df),
                        'representation': 'mixed_separated',
                        'alpha': alpha,
                        'k': k,
                        'ari': stability['mean_ari'],
                        'ari_std': stability['std_ari'],
                        'confidence': stability['mean_confidence'],
                        'silhouette': stability.get('silhouette', np.nan),
                        'n_numeric_features': X_num.shape[1],
                        'n_categorical_features': X_cat.shape[1]
                    })
                    print(f"    α={alpha:.1f}, K={k}: ARI={stability['mean_ari']:.3f}")
                except Exception as e:
                    print(f"    α={alpha:.1f}, K={k}: ERROR - {str(e)[:50]}")

        # Restore default alpha
        cfg["consensus"]["mixed_weight"] = 0.5

    return pd.DataFrame(results)


def plot_alpha_sensitivity(df: pd.DataFrame, output_path: str):
    """Create visualization: ARI vs. alpha across datasets."""
    # Filter to mixed_separated results only for alpha comparison
    alpha_df = df[df['representation'] == 'mixed_separated']
    baseline_df = df[df['representation'] == 'mixed_encoded']

    if alpha_df.empty:
        print("  No mixed_separated results to plot")
        return

    datasets = alpha_df['dataset'].unique()
    n_datasets = len(datasets)

    if n_datasets == 0:
        print("  No datasets with mixed features found")
        return

    # Create subplot grid
    n_cols = min(3, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows), squeeze=False)
    axes = axes.flatten()

    for i, dataset in enumerate(datasets):
        ax = axes[i]

        # Plot mixed_separated results for each K
        data = alpha_df[alpha_df['dataset'] == dataset]
        baseline = baseline_df[baseline_df['dataset'] == dataset]

        for k in sorted(data['k'].unique()):
            k_data = data[data['k'] == k].sort_values('alpha')
            ax.plot(k_data['alpha'], k_data['ari'],
                   marker='o', label=f'mixed_sep K={k}', linewidth=2)

            # Add horizontal line for mixed_encoded baseline
            if not baseline.empty:
                baseline_ari = baseline[baseline['k'] == k]['ari'].values
                if len(baseline_ari) > 0:
                    ax.axhline(y=baseline_ari[0], linestyle='--',
                              alpha=0.5, label=f'mixed_enc K={k}' if k == sorted(data['k'].unique())[0] else '')

        N = data['N'].iloc[0]
        n_num = data['n_numeric_features'].iloc[0]
        n_cat = data['n_categorical_features'].iloc[0]
        ax.set_title(f'{dataset}\n(N={N}, num={n_num}, cat={n_cat})',
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('α (numeric weight)')
        ax.set_ylabel('Mean ARI')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)

        # Add vertical line at α=0.5 (default)
        ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)

    # Hide empty subplots
    for j in range(len(datasets), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('h4: Alpha (Mixed Weight) Sensitivity Analysis\n'
                 'α=1.0 → numeric only, α=0.0 → categorical only, α=0.5 → equal (default)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Figure saved: {output_path}")


def summarize_alpha_results(df: pd.DataFrame) -> str:
    """Generate text summary of alpha sensitivity findings."""
    if df.empty:
        return "No results to summarize."

    summary_lines = [
        "\n" + "="*60,
        "h4: ALPHA SENSITIVITY SUMMARY",
        "="*60,
    ]

    # Filter to mixed_separated
    alpha_df = df[df['representation'] == 'mixed_separated']
    baseline_df = df[df['representation'] == 'mixed_encoded']

    if alpha_df.empty:
        return "No mixed_separated results available."

    # For each dataset, find the best alpha
    for dataset in alpha_df['dataset'].unique():
        data = alpha_df[alpha_df['dataset'] == dataset]
        baseline = baseline_df[baseline_df['dataset'] == dataset]

        # Aggregate across K values
        alpha_means = data.groupby('alpha').agg({
            'ari': 'mean',
            'confidence': 'mean'
        }).reset_index()

        best_alpha = alpha_means.loc[alpha_means['ari'].idxmax(), 'alpha']
        best_ari = alpha_means.loc[alpha_means['ari'].idxmax(), 'ari']
        default_ari = alpha_means[alpha_means['alpha'] == 0.5]['ari'].values
        default_ari = default_ari[0] if len(default_ari) > 0 else np.nan

        baseline_ari = baseline['ari'].mean() if not baseline.empty else np.nan

        summary_lines.append(f"\n{dataset}:")
        summary_lines.append(f"  Best α: {best_alpha:.1f} (ARI={best_ari:.3f})")
        summary_lines.append(f"  Default α=0.5: ARI={default_ari:.3f}")
        summary_lines.append(f"  mixed_encoded baseline: ARI={baseline_ari:.3f}")

        if best_ari > default_ari + 0.02:
            summary_lines.append(f"  → Optimal α differs from default by {best_ari - default_ari:.3f}")
        else:
            summary_lines.append(f"  → Default α=0.5 is near-optimal")

    # Overall recommendation
    all_best_alphas = []
    for dataset in alpha_df['dataset'].unique():
        data = alpha_df[alpha_df['dataset'] == dataset]
        alpha_means = data.groupby('alpha').agg({'ari': 'mean'}).reset_index()
        best = alpha_means.loc[alpha_means['ari'].idxmax(), 'alpha']
        all_best_alphas.append(best)

    summary_lines.append("\n" + "-"*40)
    summary_lines.append("RECOMMENDATION:")
    mean_best = np.mean(all_best_alphas)
    std_best = np.std(all_best_alphas)
    summary_lines.append(f"  Mean optimal α across datasets: {mean_best:.2f} (±{std_best:.2f})")

    if std_best < 0.15:
        summary_lines.append(f"  Consistent optimal α ≈ {mean_best:.1f} across datasets")
    else:
        summary_lines.append("  Optimal α varies by dataset - consider domain-specific tuning")

    return "\n".join(summary_lines)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Sensitivity analysis (m2, m3, h4)")
    parser.add_argument("--analysis", default="all",
                       choices=["all", "m2_jitter", "m3_threshold", "h4_alpha"],
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

    if args.analysis in ["all", "h4_alpha"]:
        results = run_alpha_sensitivity()
        results.to_csv(output_dir / "h4_alpha_sensitivity.csv", index=False)
        print(f"\n✓ Results saved: {output_dir / 'h4_alpha_sensitivity.csv'}")

        plot_alpha_sensitivity(results, str(figures_dir / "alpha_sensitivity.png"))

        # Print summary
        summary = summarize_alpha_results(results)
        print(summary)

        # Save summary to file
        with open(output_dir / "h4_alpha_summary.txt", "w") as f:
            f.write(summary)
        print(f"✓ Summary saved: {output_dir / 'h4_alpha_summary.txt'}")

        all_results.append(results)

    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
