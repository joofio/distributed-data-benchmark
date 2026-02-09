#!/usr/bin/env python3
"""
Privacy Visualization Script (P1.2): Generate enhanced privacy analysis figures.

Creates:
1. Line plot: ARI vs. epsilon by dataset (faceted by N)
2. Highlight N-dependence pattern (small N benefits, large N suffers)
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

FIGURES_DIR = Path("../benchmark-data-paper/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_all_privacy_data():
    """Load privacy pilot results from all datasets."""
    datasets = ["obscare", "heart_disease", "breast_cancer", "pima_diabetes"]
    n_values = {"obscare": 9, "heart_disease": 30, "breast_cancer": 30, "pima_diabetes": 30}
    
    all_data = []
    for dataset in datasets:
        csv_path = Path(f"reports/{dataset}/privacy_pilot.csv")
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df['dataset'] = dataset
            df['N'] = n_values.get(dataset, 30)
            df['N_category'] = 'Small (N=9)' if n_values[dataset] < 15 else 'Large (N=30)'
            all_data.append(df)
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return None


def plot_privacy_by_n_dependence(data: pd.DataFrame):
    """
    P1.2: Create faceted plot showing N-dependence of privacy-utility trade-off.
    """
    print("\n=== Creating N-Dependence Privacy Figure ===")
    
    # Aggregate by dataset and epsilon
    summary = data.groupby(['dataset', 'epsilon_label', 'N', 'N_category']).agg({
        'ari_mean': 'mean',
        'ari_std': 'mean',
        'confidence': 'mean'
    }).reset_index()
    
    # Create epsilon order for proper sorting
    epsilon_order = {'0.5': 0, '1.0': 1, '2.0': 2, '5.0': 3, 'inf': 4}
    summary['epsilon_sort'] = summary['epsilon_label'].map(epsilon_order)
    summary = summary.sort_values(['dataset', 'epsilon_sort'])
    
    # Create figure with two panels
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel 1: Small N (obscare)
    ax1 = axes[0]
    small_n = summary[summary['N_category'] == 'Small (N=9)']
    for dataset in small_n['dataset'].unique():
        ds_data = small_n[small_n['dataset'] == dataset]
        ax1.plot(ds_data['epsilon_label'], ds_data['ari_mean'], 'o-', 
                 label=f'{dataset}', linewidth=2.5, markersize=10)
        ax1.fill_between(ds_data['epsilon_label'], 
                        ds_data['ari_mean'] - ds_data['ari_std'],
                        ds_data['ari_mean'] + ds_data['ari_std'],
                        alpha=0.2)
    
    ax1.set_xlabel('Privacy Budget (ε)', fontsize=12)
    ax1.set_ylabel('Mean ARI', fontsize=12)
    ax1.set_title('Small N (N=9): Noise as Regularizer', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.set_ylim(0, 0.7)
    ax1.axhline(y=0.427, color='gray', linestyle='--', alpha=0.5, label='Baseline (ε=∞)')
    
    # Add annotation for improvement
    ax1.annotate('Noise IMPROVES\nstability', xy=(0.5, 0.15), xytext=(0.5, 0.15),
                fontsize=10, color='green', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Panel 2: Large N (30 datasets)
    ax2 = axes[1]
    large_n = summary[summary['N_category'] == 'Large (N=30)']
    for dataset in large_n['dataset'].unique():
        ds_data = large_n[large_n['dataset'] == dataset]
        ax2.plot(ds_data['epsilon_label'], ds_data['ari_mean'], 'o-', 
                 label=f'{dataset}', linewidth=2.5, markersize=10)
        ax2.fill_between(ds_data['epsilon_label'], 
                        ds_data['ari_mean'] - ds_data['ari_std'],
                        ds_data['ari_mean'] + ds_data['ari_std'],
                        alpha=0.2)
    
    ax2.set_xlabel('Privacy Budget (ε)', fontsize=12)
    ax2.set_ylabel('Mean ARI', fontsize=12)
    ax2.set_title('Large N (N=30): Noise Degrades Clustering', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.set_ylim(0, 0.7)
    
    # Add annotation for degradation
    ax2.annotate('Noise DEGRADES\nstability', xy=(0.5, 0.08), xytext=(0.5, 0.08),
                fontsize=10, color='red', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightsalmon', alpha=0.5))
    
    plt.tight_layout()
    output_path = FIGURES_DIR / "privacy_n_dependence.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_privacy_delta(data: pd.DataFrame):
    """
    Create bar chart showing ARI change (delta) from baseline at each epsilon.
    """
    print("\n=== Creating Privacy Delta Figure ===")
    
    # Calculate delta from baseline (inf)
    summary = data.groupby(['dataset', 'epsilon_label']).agg({
        'ari_mean': 'mean'
    }).reset_index()
    
    # Get baseline for each dataset
    baselines = summary[summary['epsilon_label'] == 'inf'].set_index('dataset')['ari_mean'].to_dict()
    
    summary['baseline'] = summary['dataset'].map(baselines)
    summary['delta_pct'] = ((summary['ari_mean'] - summary['baseline']) / summary['baseline']) * 100
    summary = summary[summary['epsilon_label'] != 'inf']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Group by epsilon
    epsilons = ['0.5', '1.0', '2.0', '5.0']
    x = np.arange(len(epsilons))
    width = 0.2
    
    datasets = ['obscare', 'heart_disease', 'breast_cancer', 'pima_diabetes']
    colors = ['#2ecc71', '#e74c3c', '#e74c3c', '#e74c3c']  # Green for obscare, red for others
    
    for i, dataset in enumerate(datasets):
        ds_data = summary[summary['dataset'] == dataset]
        deltas = []
        for eps in epsilons:
            row = ds_data[ds_data['epsilon_label'] == eps]
            if not row.empty:
                deltas.append(row['delta_pct'].values[0])
            else:
                deltas.append(0)
        
        ax.bar(x + i * width, deltas, width, label=f'{dataset}', color=colors[i], alpha=0.8)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Privacy Budget (ε)', fontsize=12)
    ax.set_ylabel('ARI Change from Baseline (%)', fontsize=12)
    ax.set_title('Privacy-Utility Trade-off: ARI Change by Dataset', fontsize=14)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f'ε={e}' for e in epsilons])
    ax.legend()
    
    # Add annotations
    ax.text(0.05, 0.95, 'Above 0%: Noise helps\nBelow 0%: Noise hurts',
           transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = FIGURES_DIR / "privacy_delta.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    print("Loading privacy pilot data...")
    data = load_all_privacy_data()
    
    if data is None or data.empty:
        print("ERROR: No privacy pilot data found. Run privacy_pilot.py first.")
        return
    
    print(f"Loaded {len(data)} rows from {data['dataset'].nunique()} datasets")
    
    plot_privacy_by_n_dependence(data)
    plot_privacy_delta(data)
    
    print(f"\n✓ Privacy figures saved to {FIGURES_DIR.absolute()}")


if __name__ == "__main__":
    main()
