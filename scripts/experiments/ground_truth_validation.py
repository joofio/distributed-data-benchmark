#!/usr/bin/env python3
"""
Ground-truth validation experiment for consensus clustering.

This script generates synthetic institutional data with known cluster structure
and evaluates how well consensus clustering recovers the true peer groups.

Addresses Major Reviewer Point #4: Lack of ground-truth validation.

Outputs:
- results/synthetic/ground_truth_evaluation.csv: ARI/NMI metrics per configuration
- results/synthetic/figures/: Visualization of recovery performance
"""
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from src.consensus import run_consensus
from src.preprocess import prepare_representations
from src.synthetic import (
    SyntheticDataset,
    evaluate_against_ground_truth,
    generate_config_for_synthetic,
    generate_synthetic_institutions,
)


def run_ground_truth_experiment(
    n_institutions: int,
    n_clusters: int,
    cluster_separation: float,
    k_values: List[int],
    methods: List[str],
    seed: int,
) -> List[Dict]:
    """Run experiment for a single synthetic configuration.

    Parameters
    ----------
    n_institutions : int
        Number of institutions to generate.
    n_clusters : int
        True number of clusters.
    cluster_separation : float
        How distinct clusters are (higher = easier).
    k_values : List[int]
        K values to test (including the true K).
    methods : List[str]
        Clustering methods to compare.
    seed : int
        Random seed.

    Returns
    -------
    List[Dict]
        Results for each method/K combination.
    """
    # Generate synthetic data
    data = generate_synthetic_institutions(
        n_institutions=n_institutions,
        n_clusters=n_clusters,
        cluster_separation=cluster_separation,
        seed=seed,
    )

    # Generate config for the pipeline
    config = generate_config_for_synthetic(data)
    config["consensus"]["n_bootstraps"] = 50  # Faster for sweep

    # Use synthetic data directly (already in memory, no need to save/load)
    df = data.df
    reps = prepare_representations(df, config)

    results = []

    for method in methods:
        # Select appropriate representation
        if method == "mixed_encoded":
            X = reps.mixed_encoded
            representation = "mixed_encoded"
        elif method == "gower_pam":
            X = (reps.mixed_separated_numeric, reps.mixed_separated_categorical)
            representation = "gower_pam"
        elif method == "kprototypes":
            X = (reps.mixed_separated_numeric, reps.mixed_separated_categorical)
            representation = "kprototypes"
        elif method == "hierarchical_gower":
            X = (reps.mixed_separated_numeric, reps.mixed_separated_categorical)
            representation = "hierarchical_gower"
        else:
            continue

        for k in k_values:
            try:
                # Run consensus clustering
                consensus_result = run_consensus(
                    X, representation, config, k, seed
                )

                # Evaluate against ground truth
                eval_metrics = evaluate_against_ground_truth(
                    consensus_result.labels,
                    data.true_labels,
                )

                results.append({
                    "n_institutions": n_institutions,
                    "true_k": n_clusters,
                    "cluster_separation": cluster_separation,
                    "method": method,
                    "k": k,
                    "k_is_true": k == n_clusters,
                    "ari": eval_metrics["ari"],
                    "nmi": eval_metrics["nmi"],
                    "cluster_purity": eval_metrics["cluster_purity"],
                    "mean_ari_bootstrap": float(np.mean(consensus_result.ari_scores)),
                    "mean_confidence": float(np.mean(consensus_result.confidence)),
                    "seed": seed,
                })
            except Exception as e:
                print(f"  Error with {method} K={k}: {e}")
                results.append({
                    "n_institutions": n_institutions,
                    "true_k": n_clusters,
                    "cluster_separation": cluster_separation,
                    "method": method,
                    "k": k,
                    "k_is_true": k == n_clusters,
                    "ari": np.nan,
                    "nmi": np.nan,
                    "cluster_purity": np.nan,
                    "mean_ari_bootstrap": np.nan,
                    "mean_confidence": np.nan,
                    "seed": seed,
                })

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ground-truth validation for consensus clustering"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/synthetic_groundtruth.yml"),
        help="Configuration file for sweep parameters",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/synthetic/ground_truth_evaluation.csv"),
        help="Output CSV path",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Load config or use defaults
    if args.config.exists():
        with open(args.config) as f:
            sweep_config = yaml.safe_load(f)
    else:
        sweep_config = {
            "n_institutions": [20, 30, 50],
            "n_clusters": [3, 4, 5],
            "cluster_separation": [1.0, 2.0, 3.0],
            "methods": ["mixed_encoded", "gower_pam", "kprototypes", "hierarchical_gower"],
        }

    n_institutions_list = sweep_config.get("n_institutions", [30])
    n_clusters_list = sweep_config.get("n_clusters", [4])
    separation_list = sweep_config.get("cluster_separation", [2.0])
    methods = sweep_config.get("methods", ["gower_pam"])

    print("Ground-Truth Validation Experiment")
    print("=" * 60)
    print(f"Institutions: {n_institutions_list}")
    print(f"Clusters: {n_clusters_list}")
    print(f"Separations: {separation_list}")
    print(f"Methods: {methods}")
    print("=" * 60)

    all_results = []
    run_id = 0

    for n_inst in n_institutions_list:
        for n_clust in n_clusters_list:
            if n_clust >= n_inst // 2:
                continue  # Skip if too many clusters for sample size

            for sep in separation_list:
                print(f"\nConfig: N={n_inst}, K_true={n_clust}, sep={sep}")

                # Test K values around the true K
                k_values = list(range(
                    max(2, n_clust - 1),
                    min(n_clust + 3, n_inst // 3)
                ))

                results = run_ground_truth_experiment(
                    n_institutions=n_inst,
                    n_clusters=n_clust,
                    cluster_separation=sep,
                    k_values=k_values,
                    methods=methods,
                    seed=args.seed + run_id,
                )

                all_results.extend(results)
                run_id += 1

                # Print summary for this config
                for method in methods:
                    method_results = [r for r in results if r["method"] == method and r["k_is_true"]]
                    if method_results:
                        ari = method_results[0]["ari"]
                        print(f"  {method}: ARI={ari:.3f}")

    # Save results
    df_results = pd.DataFrame(all_results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(args.output, index=False)
    print(f"\nResults saved to: {args.output}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY: Mean ARI at True K by Method")
    print("=" * 60)
    true_k_results = df_results[df_results["k_is_true"]]
    summary = true_k_results.groupby("method")["ari"].agg(["mean", "std", "count"])
    print(summary.to_string())

    # Summary by cluster separation
    print("\n" + "=" * 60)
    print("SUMMARY: Mean ARI at True K by Cluster Separation")
    print("=" * 60)
    sep_summary = true_k_results.groupby("cluster_separation")["ari"].agg(["mean", "std"])
    print(sep_summary.to_string())


if __name__ == "__main__":
    main()
