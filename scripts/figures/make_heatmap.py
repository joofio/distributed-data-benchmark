#!/usr/bin/env python3
"""Generate benchmark percentiles heatmap."""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Read the data
csv_path = Path('results/obscare/tables/benchmark_results.csv')
df = pd.read_csv(csv_path)

# Filter to target_rate only
df = df[df['kpi'] == 'target_rate'].copy()

# Pivot to create matrix: institutions x methods
pivot = df.pivot_table(index='institution_id', columns='method', values='percentile')

# Rename columns for clarity
rename_map = {
    'peer': 'Peer',
    'global': 'Global', 
    'knn_k3': 'kNN-3',
    'knn_k5': 'kNN-5',
    'knn_k7': 'kNN-7',
    'rule_based': 'Rule'
}
pivot.columns = [rename_map.get(c, c) for c in pivot.columns]

# Reorder columns logically
col_order = ['Peer', 'Global', 'kNN-3', 'kNN-5', 'kNN-7', 'Rule']
pivot = pivot[[c for c in col_order if c in pivot.columns]]

# Sort by peer percentile
pivot = pivot.sort_values('Peer')

# Create heatmap
fig, ax = plt.subplots(figsize=(10, 6))

# Use diverging colormap centered at 50 - REVERSED so high=bad (red), low=good (green)
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='RdYlGn_r', center=50,
            vmin=0, vmax=100, linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'Percentile Rank', 'shrink': 0.8},
            annot_kws={'size': 11, 'weight': 'bold'}, ax=ax)

ax.set_xlabel('Benchmarking Method', fontsize=12)
ax.set_ylabel('Institution', fontsize=12)
ax.set_title('Benchmark Percentiles: Target Rate (obscare, N=9)', 
             fontsize=14, fontweight='bold')

# Rotate labels for readability
plt.yticks(rotation=0)
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
output_path = Path('../benchmark-data-paper/figures/benchmark_percentiles_target_rate.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved heatmap to {output_path}")
plt.close()
