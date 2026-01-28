#!/usr/bin/env python3
"""
Run privacy experiments (P1.1) on all datasets with multiple epsilon values.
Generates summary table with privacy results including ARI comparison.
"""

import subprocess
import pandas as pd
from pathlib import Path
import json
import sys

# Datasets and epsilon values to test
DATASETS = [
    "obscare", "heart_disease", "breast_cancer", "pima_diabetes",
    "hcv", "liver_disorders", "early_diabetes"
]

# inf = baseline (no noise), 5.0 = low noise, 2.0 = high noise
EPSILON_VALUES = [float('inf'), 5.0, 2.0]


def run_privacy_experiment(dataset: str, epsilon: float) -> dict:
    """Run experiment with given privacy epsilon and extract results."""
    config_path = f"configs/{dataset}.yml"
    
    # Build command
    cmd = ["python", "-m", "src.run_experiments", "--config", config_path]
    
    # Only add --privacy-epsilon for finite epsilon values
    if epsilon != float('inf'):
        cmd.extend(["--privacy-epsilon", str(epsilon)])
    
    epsilon_label = "∞ (baseline)" if epsilon == float('inf') else str(epsilon)
    print(f"  Running {dataset} with ε={epsilon_label}...", end=" ", flush=True)
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
    
    if result.returncode != 0:
        print("FAILED")
        print(f"    Error: {result.stderr[:200]}" if result.stderr else "")
        return None
    
    # Read summary to get selected K and ARI
    summary_path = Path(__file__).parent / f"reports/{dataset}/summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        
        selected_k = summary['selected_k']
        
        # Find mean_ari for selected_k from k_sweep
        mean_ari = None
        for k_data in summary.get('k_sweep', []):
            if k_data['k'] == selected_k:
                mean_ari = k_data.get('mean_ari')
                break
        
        print(f"K={selected_k}, ARI={mean_ari:.3f}" if mean_ari else f"K={selected_k}")
        
        return {
            'dataset': dataset,
            'epsilon': epsilon if epsilon != float('inf') else 999.0,
            'epsilon_label': epsilon_label,
            'selected_k': selected_k,
            'mean_ari': mean_ari,
            'representation': summary.get('representation')
        }
    
    print("OK (no summary)")
    return {'dataset': dataset, 'epsilon': epsilon, 'epsilon_label': epsilon_label}


def compute_ari_changes(df: pd.DataFrame) -> pd.DataFrame:
    """Add ARI change percentage relative to baseline for each dataset."""
    # Get baseline (epsilon=999.0 which represents infinity)
    baseline_aris = df[df['epsilon'] == 999.0].set_index('dataset')['mean_ari']
    
    def calc_change(row):
        baseline = baseline_aris.get(row['dataset'])
        if baseline is None or row['mean_ari'] is None or baseline == 0:
            return None
        return ((row['mean_ari'] - baseline) / baseline) * 100
    
    df['ari_change_pct'] = df.apply(calc_change, axis=1)
    return df


def main():
    print("=" * 60)
    print("P1.1: Running Privacy Experiments on All Datasets")
    print("=" * 60)
    print(f"Datasets: {len(DATASETS)}")
    print(f"Epsilon values: {[str(e) if e != float('inf') else '∞' for e in EPSILON_VALUES]}")
    print()
    
    results = []
    
    for dataset in DATASETS:
        print(f"\n[{dataset}]")
        for epsilon in EPSILON_VALUES:
            result = run_privacy_experiment(dataset, epsilon)
            if result:
                results.append(result)
    
    # Save summary
    if results:
        df = pd.DataFrame(results)
        
        # Compute ARI change percentages
        df = compute_ari_changes(df)
        
        # Reorder columns for clarity
        cols = ['dataset', 'epsilon_label', 'epsilon', 'selected_k', 'mean_ari', 'ari_change_pct', 'representation']
        df = df[[c for c in cols if c in df.columns]]
        
        output_path = Path(__file__).parent / "reports/privacy_summary.csv"
        df.to_csv(output_path, index=False)
        
        print("\n" + "=" * 60)
        print("Results Summary")
        print("=" * 60)
        print(df.to_string(index=False))
        print(f"\n✓ Summary saved to {output_path}")
        
        # Print key findings
        print("\n" + "=" * 60)
        print("Key Findings")
        print("=" * 60)
        for dataset in DATASETS:
            subset = df[df['dataset'] == dataset]
            if len(subset) > 0:
                baseline = subset[subset['epsilon'] == 999.0]['mean_ari'].values
                noise = subset[subset['epsilon'] != 999.0]
                if len(baseline) > 0 and len(noise) > 0:
                    baseline_val = baseline[0]
                    for _, row in noise.iterrows():
                        change = row.get('ari_change_pct', 0)
                        if change is not None:
                            direction = "↑" if change > 0 else "↓" if change < 0 else "→"
                            print(f"  {dataset} (ε={row['epsilon_label']}): ARI {direction} {abs(change):.1f}%")


if __name__ == "__main__":
    main()
