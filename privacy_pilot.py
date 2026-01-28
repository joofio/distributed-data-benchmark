#!/usr/bin/env python3
"""
Privacy Pilot Experiment: Test LDP noise impact on clustering stability.

This script tests the privacy-utility trade-off by adding Laplace noise
to institutional feature profiles before consensus clustering.

Usage:
    python privacy_pilot.py --dataset obscare --epsilons 0.5,1,2,5
"""

import argparse
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from consensus import run_consensus
from data import load_dataset
from eval import stability_summary
from preprocess import prepare_representations


def add_laplace_noise(X: np.ndarray, epsilon: float, sensitivity: np.ndarray = None) -> np.ndarray:
    """
    Add Laplace noise to features for local differential privacy.
    
    Args:
        X: Feature matrix (N x F)
        epsilon: Privacy budget (lower = more noise, more privacy)
        sensitivity: Per-feature sensitivity. If None, uses feature range.
    
    Returns:
        Noisy feature matrix
    """
    if epsilon == float('inf'):
        return X.copy()  # No noise
    
    if sensitivity is None:
        # Estimate sensitivity as feature range (conservative for bounded data)
        sensitivity = np.ptp(X, axis=0)  # max - min per feature
        # Avoid zero sensitivity for constant features
        sensitivity = np.maximum(sensitivity, 1e-6)
    
    # Laplace scale = sensitivity / epsilon
    scale = sensitivity / epsilon
    
    # Add Laplace noise
    noise = np.random.laplace(loc=0, scale=scale, size=X.shape)
    X_noisy = X + noise
    
    return X_noisy


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


def run_privacy_experiment(
    df: pd.DataFrame,
    X_original: np.ndarray,
    cfg: dict,
    epsilon: float,
    k_values: list,
    representation: str = "numeric",
    seed: int = 42
) -> dict:
    """
    Run consensus clustering on noisy features and measure stability.
    """
    np.random.seed(seed)
    
    # Add LDP noise to features BEFORE clustering
    X_noisy = add_laplace_noise(X_original, epsilon)
    
    results = {}
    for k in k_values:
        try:
            # Run consensus clustering on noisy data
            consensus_result = run_consensus(
                X=X_noisy,
                representation=representation,
                cfg=cfg,
                k=k,
                seed=seed
            )
            
            # Get stability metrics using correct API
            stability_df = stability_summary(
                ari_scores=consensus_result.ari_scores,
                labels=consensus_result.labels,
                confidence=consensus_result.confidence,
                X=X_noisy
            )
            
            # Extract metrics from DataFrame
            stability = stability_df.iloc[0].to_dict()
            
            results[k] = {
                'ari_mean': stability['mean_ari'],
                'ari_std': stability['std_ari'],
                'confidence': stability['mean_confidence'],
                'silhouette': stability.get('silhouette', np.nan)
            }
        except Exception as e:
            print(f"  Warning: K={k} failed: {e}")
            results[k] = {
                'ari_mean': np.nan,
                'ari_std': np.nan,
                'confidence': np.nan,
                'silhouette': np.nan
            }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Privacy pilot experiment")
    parser.add_argument("--dataset", default="obscare", help="Dataset name")
    parser.add_argument("--epsilons", default="0.5,1,2,5,inf", help="Comma-separated epsilon values")
    parser.add_argument("--k-values", default="2,3,4", help="Comma-separated K values")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Parse epsilon values
    epsilons = []
    for e in args.epsilons.split(","):
        if e.lower() == "inf":
            epsilons.append(float('inf'))
        else:
            epsilons.append(float(e))
    
    k_values = [int(k) for k in args.k_values.split(",")]
    
    # Load config
    cfg = load_config(args.dataset)
    
    # Load dataset (load_dataset takes only cfg)
    print(f"Loading dataset: {args.dataset}")
    df = load_dataset(cfg)
    
    # Prepare features
    reps = prepare_representations(df, cfg)
    representation = cfg.get("representation", "numeric")
    
    # Get feature matrix
    if representation == "numeric" and reps.numeric is not None:
        X = reps.numeric
    elif representation == "mixed_encoded" and reps.mixed_encoded is not None:
        X = reps.mixed_encoded
    else:
        X = reps.numeric if reps.numeric is not None else reps.mixed_encoded
        representation = "numeric" if reps.numeric is not None else "mixed_encoded"
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Representation: {representation}")
    print(f"Testing epsilon values: {epsilons}")
    print(f"Testing K values: {k_values}")
    print()
    
    # Run experiments
    all_results = []
    
    for epsilon in epsilons:
        epsilon_label = "∞ (baseline)" if epsilon == float('inf') else str(epsilon)
        print(f"\n=== Epsilon = {epsilon_label} ===")
        
        results = run_privacy_experiment(
            df=df,
            X_original=X,
            cfg=cfg,
            epsilon=epsilon,
            k_values=k_values,
            representation=representation,
            seed=args.seed
        )
        
        for k, metrics in results.items():
            if not np.isnan(metrics['ari_mean']):
                print(f"  K={k}: ARI={metrics['ari_mean']:.3f} ± {metrics['ari_std']:.3f}, "
                      f"Conf={metrics['confidence']:.3f}")
            
            all_results.append({
                'epsilon': epsilon if epsilon != float('inf') else 999,
                'epsilon_label': epsilon_label,
                'k': k,
                **metrics
            })
    
    # Summary table
    print("\n" + "="*60)
    print("SUMMARY: Mean ARI across K values")
    print("="*60)
    
    results_df = pd.DataFrame(all_results)
    summary = results_df.groupby('epsilon_label')['ari_mean'].mean()
    
    # Get baseline ARI
    if "∞ (baseline)" in summary.index:
        baseline_ari = summary["∞ (baseline)"]
    else:
        baseline_ari = summary.iloc[0]
    
    for eps_label in summary.index:
        ari = summary[eps_label]
        pct_drop = (1 - ari/baseline_ari) * 100 if baseline_ari > 0 else 0
        print(f"  ε = {eps_label:15s}: ARI = {ari:.3f}  ({pct_drop:+.1f}% vs baseline)")
    
    # Save results
    output_path = Path(f"reports/{args.dataset}/privacy_pilot.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
