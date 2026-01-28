#!/usr/bin/env python3
"""
Visualization Script: Generate figures for paper based on code_tasks.md and tasks.md.

Figures to create:
1. P1.2: ARI vs. epsilon by dataset (privacy analysis)
2. P3.1/m7: Peer percentile vs. global percentile scatter
3. m2: ARI vs. jitter scale sensitivity
4. m6: Single-run vs consensus comparison

Usage:
    python generate_figures.py --dataset all
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import seaborn as sns
from pathlib import Path
import sys
import yaml
import warnings

warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from consensus import run_consensus
from data import load_dataset
from eval import stability_summary
from preprocess import prepare_representations


# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

FIGURES_DIR = Path("../benchmark-data-paper/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_config(dataset: str) -> dict:
    """Load configuration for dataset."""
    config_path = Path(f"configs/{dataset}.yml")
    if not config_path.exists():
        config_path = Path("configs/default.yml")
    
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    if "consensus" not in cfg:
        cfg["consensus"] = {}
    cfg["consensus"].setdefault("n_bootstraps", 50)
    cfg["consensus"].setdefault("numeric_jitter_scale", 0.05)
    cfg["consensus"].setdefault("sample_fraction", 1.0)
    cfg["consensus"].setdefault("feature_bootstrap", False)
    
    return cfg


# =============================================================================
# Figure 1: Privacy Analysis (P1.2)
# =============================================================================

def plot_privacy_sensitivity(datasets: list = None):
    """
    P1.2: Line plot showing ARI vs. epsilon by dataset.
    """
    print("\n=== Generating Privacy Sensitivity Figure ===")
    
    if datasets is None:
        datasets = ["obscare", "heart_disease", "breast_cancer", "pima_diabetes"]
    
    all_data = []
    for dataset in datasets:
        csv_path = Path(f"reports/{dataset}/privacy_pilot.csv")
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df['dataset'] = dataset
            all_data.append(df)
    
    if not all_data:
        print("  No privacy data found. Run privacy_pilot.py first.")
        return
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Aggregate by dataset and epsilon
    summary = combined.groupby(['dataset', 'epsilon_label']).agg({
        'ari_mean': 'mean',
        'confidence': 'mean'
    }).reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for dataset in summary['dataset'].unique():
        data = summary[summary['dataset'] == dataset]
        # Sort by epsilon value
        data = data.sort_values('epsilon_label')
        ax.plot(data['epsilon_label'], data['ari_mean'], 'o-', 
                label=dataset, linewidth=2, markersize=8)
    
    ax.set_xlabel('Privacy Budget (ε)', fontsize=12)
    ax.set_ylabel('Mean ARI', fontsize=12)
    ax.set_title('Privacy-Utility Trade-off: ARI vs. Epsilon', fontsize=14)
    ax.legend(title='Dataset', loc='best')
    ax.set_ylim(0, 0.7)
    
    plt.tight_layout()
    output_path = FIGURES_DIR / "privacy_sensitivity.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# Figure 2: Peer vs Global Percentile (P3.1/m7)
# =============================================================================

def plot_peer_vs_global(dataset: str = "obscare"):
    """
    m7: Scatter plot comparing peer percentile vs. global percentile.
    """
    print(f"\n=== Generating Peer vs Global Figure ({dataset}) ===")
    
    cfg = load_config(dataset)
    df = load_dataset(cfg)
    reps = prepare_representations(df, cfg)
    
    representation = cfg.get("representation", "numeric")
    X = reps.numeric if reps.numeric is not None else reps.mixed_encoded
    representation = "numeric" if reps.numeric is not None else "mixed_encoded"
    
    # Run consensus to get peer groups
    consensus_result = run_consensus(
        X=X,
        representation=representation,
        cfg=cfg,
        k=3,
        seed=42
    )
    
    # Compute benchmarks
    features = cfg.get("features", {})
    kpi_cols = features.get("kpi", [])
    
    if not kpi_cols and "numeric" in features:
        kpi_cols = features["numeric"][:1]  # Use first numeric as proxy
    
    if not kpi_cols:
        print("  No KPI columns found. Skipping.")
        return
    
    kpi_col = kpi_cols[0]
    kpi_values = df[kpi_col].values
    
    # Global percentiles
    global_percentiles = np.array([
        np.sum(kpi_values < v) / len(kpi_values) * 100 
        for v in kpi_values
    ])
    
    # Peer percentiles
    peer_percentiles = np.zeros(len(kpi_values))
    for peer_id in np.unique(consensus_result.labels):
        mask = consensus_result.labels == peer_id
        peer_kpis = kpi_values[mask]
        for i in np.where(mask)[0]:
            peer_percentiles[i] = np.sum(peer_kpis < kpi_values[i]) / len(peer_kpis) * 100
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    scatter = ax.scatter(global_percentiles, peer_percentiles, 
                        c=consensus_result.labels, cmap='Set2',
                        s=100, alpha=0.7, edgecolors='black')
    
    # Diagonal line
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.5, label='Equal ranking')
    
    ax.set_xlabel('Global Percentile', fontsize=12)
    ax.set_ylabel('Peer Percentile', fontsize=12)
    ax.set_title(f'Context-Dependent Benchmarking: {dataset}', fontsize=14)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.legend()
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Peer Group', fontsize=10)
    
    plt.tight_layout()
    output_path = FIGURES_DIR / f"peer_vs_global_{dataset}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# Figure 3: Stability Comparison (m6)
# =============================================================================

def plot_stability_comparison(datasets: list = None):
    """
    m6: Bar chart comparing consensus vs single-run stability.
    """
    print("\n=== Generating Stability Comparison Figure ===")
    
    if datasets is None:
        datasets = ["obscare", "heart_disease"]
    
    all_data = []
    for dataset in datasets:
        csv_path = Path(f"reports/{dataset}/liability_analysis.csv")
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df['dataset'] = dataset
            all_data.append(df)
    
    if not all_data:
        print("  No liability data found. Run liability_analysis.py first.")
        return
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Filter to L4 baseline analysis
    l4_data = combined[combined['analysis'] == 'L4_baseline'].copy()
    
    if l4_data.empty:
        print("  No L4 baseline data found.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(l4_data))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, l4_data['consensus_ari'], width, 
                   label='Consensus', color='steelblue')
    bars2 = ax.bar(x + width/2, l4_data['single_pairwise_ari'], width,
                   label='Single-Run Pairwise', color='coral')
    
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('ARI', fontsize=12)
    ax.set_title('Consensus vs. Single-Run Clustering Stability', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row['dataset']}\nK={row['k']}" for _, row in l4_data.iterrows()])
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    output_path = FIGURES_DIR / "stability_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# Figure 4: Multi-Seed Variance (L1)
# =============================================================================

def plot_multiseed_variance(datasets: list = None):
    """
    L1: Box plot showing ARI distribution across seeds.
    """
    print("\n=== Generating Multi-Seed Variance Figure ===")
    
    if datasets is None:
        datasets = ["obscare", "heart_disease"]
    
    all_data = []
    for dataset in datasets:
        csv_path = Path(f"reports/{dataset}/liability_analysis.csv")
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df['dataset'] = dataset
            all_data.append(df)
    
    if not all_data:
        print("  No liability data found.")
        return
    
    combined = pd.concat(all_data, ignore_index=True)
    l1_data = combined[combined['analysis'] == 'L1_multiseed'].copy()
    
    if l1_data.empty:
        print("  No L1 multiseed data found.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.boxplot(data=l1_data, x='dataset', y='ari', hue='k', ax=ax)
    
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('ARI (across 5 seeds)', fontsize=12)
    ax.set_title('Reproducibility: ARI Variance Across Random Seeds', fontsize=14)
    ax.legend(title='K')
    
    plt.tight_layout()
    output_path = FIGURES_DIR / "multiseed_variance.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--dataset", default="all", help="Dataset or 'all'")
    parser.add_argument("--figures", default="all",
                       choices=["all", "privacy", "peer_global", "stability", "multiseed"],
                       help="Which figures to generate")
    args = parser.parse_args()
    
    print(f"Output directory: {FIGURES_DIR.absolute()}")
    
    datasets = ["obscare", "heart_disease"]
    if args.dataset != "all":
        datasets = [args.dataset]
    
    if args.figures in ["all", "privacy"]:
        plot_privacy_sensitivity()
    
    if args.figures in ["all", "peer_global"]:
        for ds in datasets:
            try:
                plot_peer_vs_global(ds)
            except Exception as e:
                print(f"  Error for {ds}: {e}")
    
    if args.figures in ["all", "stability"]:
        plot_stability_comparison(datasets)
    
    if args.figures in ["all", "multiseed"]:
        plot_multiseed_variance(datasets)
    
    print(f"\n✓ All figures saved to {FIGURES_DIR.absolute()}")


if __name__ == "__main__":
    main()
