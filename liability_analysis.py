#!/usr/bin/env python3
"""
Liability Analysis Script: Address L1-L4 experimental concerns.

L1: Multi-seed validation (seeds=42,123,456,789,1000)
L2: Bootstrap sensitivity (B=25,50,100)
L3: Threshold sensitivity (percentile, z-score variations)
L4: Single-run vs consensus baseline

Usage:
    python liability_analysis.py --dataset obscare --analysis all
    python liability_analysis.py --dataset obscare --analysis L1_multiseed
"""

import argparse
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import sys
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from consensus import run_consensus
from data import load_dataset
from eval import stability_summary
from preprocess import prepare_representations
from clustering import cluster_data


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


def run_single_kmeans(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Run single K-means without consensus (baseline for L4)."""
    return cluster_data(X, k, seed, "kmeans")


def compute_ari_single_vs_reference(labels: np.ndarray, ref_labels: np.ndarray) -> float:
    """Compute ARI between single run and reference labels."""
    from sklearn.metrics import adjusted_rand_score
    return adjusted_rand_score(ref_labels, labels)


# =============================================================================
# L1: Multi-Seed Validation
# =============================================================================

def run_multiseed_analysis(
    df: pd.DataFrame,
    X: np.ndarray,
    cfg: dict,
    representation: str,
    k_values: List[int] = [2, 3, 4],
    seeds: List[int] = [42, 123, 456, 789, 1000]
) -> pd.DataFrame:
    """
    L1: Run experiments with multiple seeds to validate reproducibility.
    """
    print("\n" + "="*60)
    print("L1: MULTI-SEED VALIDATION")
    print(f"Seeds: {seeds}")
    print("="*60)
    
    results = []
    
    for seed in seeds:
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
                    'analysis': 'L1_multiseed',
                    'seed': seed,
                    'k': k,
                    'ari': stability['mean_ari'],
                    'ari_std': stability['std_ari'],
                    'confidence': stability['mean_confidence'],
                    'silhouette': stability.get('silhouette', np.nan)
                })
                print(f"  Seed={seed}, K={k}: ARI={stability['mean_ari']:.3f}")
            except ValueError as e:
                if "n_samples" in str(e) and "n_clusters" in str(e):
                    print(f"  Seed={seed}, K={k}: SKIPPED (K > bootstrap sample size)")
                else:
                    raise
    
    results_df = pd.DataFrame(results)
    
    # Aggregate across seeds
    print("\nAggregated Results (mean ± std across seeds):")
    for k in k_values:
        k_data = results_df[results_df['k'] == k]
        mean_ari = k_data['ari'].mean()
        std_ari = k_data['ari'].std()
        print(f"  K={k}: ARI = {mean_ari:.3f} ± {std_ari:.3f}")
    
    return results_df


# =============================================================================
# L2: Bootstrap Count Sensitivity
# =============================================================================

def run_bootstrap_sensitivity(
    df: pd.DataFrame,
    X: np.ndarray,
    cfg: dict,
    representation: str,
    k_values: List[int] = [2, 3, 4],
    bootstrap_counts: List[int] = [25, 50, 100],
    seed: int = 42
) -> pd.DataFrame:
    """
    L2: Test sensitivity to number of bootstrap iterations.
    """
    print("\n" + "="*60)
    print("L2: BOOTSTRAP COUNT SENSITIVITY")
    print(f"B values: {bootstrap_counts}")
    print("="*60)
    
    results = []
    original_B = cfg["consensus"]["n_bootstraps"]
    
    for B in bootstrap_counts:
        cfg["consensus"]["n_bootstraps"] = B
        
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
                    'analysis': 'L2_bootstrap',
                    'B': B,
                    'k': k,
                    'ari': stability['mean_ari'],
                    'ari_std': stability['std_ari'],
                    'confidence': stability['mean_confidence']
                })
                print(f"  B={B}, K={k}: ARI={stability['mean_ari']:.3f} ± {stability['std_ari']:.3f}")
            except ValueError as e:
                if "n_samples" in str(e) and "n_clusters" in str(e):
                    print(f"  B={B}, K={k}: SKIPPED")
                else:
                    raise
    
    # Restore original
    cfg["consensus"]["n_bootstraps"] = original_B
    
    return pd.DataFrame(results)


# =============================================================================
# L3: Outlier Threshold Sensitivity
# =============================================================================

def run_threshold_sensitivity(
    X: np.ndarray,
    labels: np.ndarray,
    kpi_values: np.ndarray = None,
    percentile_thresholds: List[Tuple[int, int]] = [(5, 95), (3, 97), (10, 90)],
    zscore_thresholds: List[float] = [1.5, 2.0, 2.5]
) -> pd.DataFrame:
    """
    L3: Test sensitivity to outlier detection thresholds.
    """
    print("\n" + "="*60)
    print("L3: OUTLIER THRESHOLD SENSITIVITY")
    print("="*60)
    
    # Use synthetic KPI if not provided
    if kpi_values is None:
        np.random.seed(42)
        kpi_values = np.random.randn(len(labels))
    
    results = []
    
    # Percentile-based thresholds
    for low_pct, high_pct in percentile_thresholds:
        low_thresh = np.percentile(kpi_values, low_pct)
        high_thresh = np.percentile(kpi_values, high_pct)
        
        n_flagged = np.sum((kpi_values < low_thresh) | (kpi_values > high_thresh))
        flag_rate = n_flagged / len(kpi_values)
        
        results.append({
            'analysis': 'L3_threshold',
            'method': 'percentile',
            'threshold': f'{low_pct}/{high_pct}',
            'n_flagged': n_flagged,
            'flag_rate': flag_rate
        })
        print(f"  Percentile {low_pct}/{high_pct}: {n_flagged} flagged ({flag_rate*100:.1f}%)")
    
    # Z-score thresholds
    z_scores = (kpi_values - np.mean(kpi_values)) / np.std(kpi_values)
    
    for z_thresh in zscore_thresholds:
        n_flagged = np.sum(np.abs(z_scores) > z_thresh)
        flag_rate = n_flagged / len(kpi_values)
        
        results.append({
            'analysis': 'L3_threshold',
            'method': 'zscore',
            'threshold': f'z={z_thresh}',
            'n_flagged': n_flagged,
            'flag_rate': flag_rate
        })
        print(f"  Z-score |z|>{z_thresh}: {n_flagged} flagged ({flag_rate*100:.1f}%)")
    
    return pd.DataFrame(results)


# =============================================================================
# L4: Single-Run vs Consensus Baseline
# =============================================================================

def run_single_vs_consensus(
    df: pd.DataFrame,
    X: np.ndarray,
    cfg: dict,
    representation: str,
    k_values: List[int] = [2, 3, 4],
    n_single_runs: int = 10,
    seed: int = 42
) -> pd.DataFrame:
    """
    L4: Compare single K-means runs to consensus clustering.
    """
    print("\n" + "="*60)
    print("L4: SINGLE-RUN VS CONSENSUS BASELINE")
    print(f"Comparing {n_single_runs} single runs to consensus")
    print("="*60)
    
    results = []
    
    for k in k_values:
        # Run consensus
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
        consensus_ari = stability_df.iloc[0]['mean_ari']
        consensus_labels = consensus_result.labels
        
        # Run multiple single K-means and measure stability
        single_aris = []
        for i in range(n_single_runs):
            labels_i = run_single_kmeans(X, k, seed + i)
            # Compare to consensus as reference
            ari_i = compute_ari_single_vs_reference(labels_i, consensus_labels)
            single_aris.append(ari_i)
        
        single_mean = np.mean(single_aris)
        single_std = np.std(single_aris)
        
        # Also compute pairwise ARI among single runs
        pairwise_aris = []
        for i in range(n_single_runs):
            labels_i = run_single_kmeans(X, k, seed + i)
            for j in range(i + 1, n_single_runs):
                labels_j = run_single_kmeans(X, k, seed + j)
                pairwise_aris.append(compute_ari_single_vs_reference(labels_i, labels_j))
        
        pairwise_mean = np.mean(pairwise_aris) if pairwise_aris else 0
        
        results.append({
            'analysis': 'L4_baseline',
            'k': k,
            'consensus_ari': consensus_ari,
            'single_vs_consensus_mean': single_mean,
            'single_vs_consensus_std': single_std,
            'single_pairwise_ari': pairwise_mean
        })
        
        print(f"  K={k}:")
        print(f"    Consensus internal ARI: {consensus_ari:.3f}")
        print(f"    Single vs Consensus:    {single_mean:.3f} ± {single_std:.3f}")
        print(f"    Single pairwise ARI:    {pairwise_mean:.3f}")
    
    return pd.DataFrame(results)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Liability analysis")
    parser.add_argument("--dataset", default="obscare", help="Dataset name")
    parser.add_argument("--analysis", default="all", 
                       choices=["all", "L1_multiseed", "L2_bootstrap", "L3_threshold", "L4_baseline"],
                       help="Which analysis to run")
    parser.add_argument("--k-values", default="2,3,4", help="K values to test")
    args = parser.parse_args()
    
    k_values = [int(k) for k in args.k_values.split(",")]
    
    # Load config and data
    cfg = load_config(args.dataset)
    print(f"Loading dataset: {args.dataset}")
    df = load_dataset(cfg)
    
    # Prepare features
    reps = prepare_representations(df, cfg)
    representation = cfg.get("representation", "numeric")
    
    if representation == "numeric" and reps.numeric is not None:
        X = reps.numeric
    elif representation == "mixed_encoded" and reps.mixed_encoded is not None:
        X = reps.mixed_encoded
    else:
        X = reps.numeric if reps.numeric is not None else reps.mixed_encoded
        representation = "numeric" if reps.numeric is not None else "mixed_encoded"
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Representation: {representation}")
    
    all_results = []
    
    # Run analyses
    if args.analysis in ["all", "L1_multiseed"]:
        results = run_multiseed_analysis(df, X, cfg, representation, k_values)
        all_results.append(results)
    
    if args.analysis in ["all", "L2_bootstrap"]:
        results = run_bootstrap_sensitivity(df, X, cfg, representation, k_values)
        all_results.append(results)
    
    if args.analysis in ["all", "L3_threshold"]:
        # Get consensus labels first
        consensus_result = run_consensus(X=X, representation=representation, cfg=cfg, k=3, seed=42)
        results = run_threshold_sensitivity(X, consensus_result.labels)
        all_results.append(results)
    
    if args.analysis in ["all", "L4_baseline"]:
        results = run_single_vs_consensus(df, X, cfg, representation, k_values)
        all_results.append(results)
    
    # Save all results
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        output_path = Path(f"reports/{args.dataset}/liability_analysis.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_path, index=False)
        print(f"\nAll results saved to: {output_path}")


if __name__ == "__main__":
    main()
