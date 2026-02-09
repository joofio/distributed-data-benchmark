#!/usr/bin/env python3
"""
Compare clustering methods on mixed-type datasets.

This script compares three approaches for clustering mixed-type data:
1. mixed_encoded + K-means (current default): ordinal-encodes categoricals
   and uses Euclidean distance
2. Gower distance + PAM: properly handles mixed types with Gower distance
   and Partitioning Around Medoids
3. K-prototypes: extends K-means with categorical handling via simple matching

Metrics computed:
- ARI (Adjusted Rand Index): cluster stability across bootstraps
- Silhouette: cluster separation quality
- Confidence: mean within-cluster co-assignment probability
- Runtime: wall-clock time per method
"""
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import silhouette_score

from src.consensus import ConsensusResult, run_consensus
from src.data import load_dataset
from src.preprocess import prepare_representations


@dataclass
class ComparisonResult:
    """Results from a single method/dataset/K comparison."""

    dataset: str
    method: str
    k: int
    mean_ari: float
    std_ari: float
    silhouette: float
    mean_confidence: float
    runtime_seconds: float


def compute_silhouette_mixed(
    X_num: np.ndarray,
    X_cat: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute silhouette score for mixed-type data using Gower distance.

    For fair comparison, we compute silhouette on Gower distance for all methods,
    since this properly handles mixed-type features.
    """
    from src.clustering import gower_distance_matrix

    if len(np.unique(labels)) < 2:
        return 0.0

    D = gower_distance_matrix(X_num, X_cat)
    return float(silhouette_score(D, labels, metric="precomputed"))


def run_single_comparison(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    method: str,
    k: int,
    seed: int,
) -> ComparisonResult:
    """Run a single method comparison and collect metrics."""
    reps = prepare_representations(df, cfg)
    gamma = cfg["consensus"].get("kprototypes_gamma", 0.5)

    # Prepare data based on method
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
        raise ValueError(f"Unknown method: {method}")

    # Run consensus clustering with timing
    start_time = time.perf_counter()
    result: ConsensusResult = run_consensus(
        X, representation, cfg, k, seed, gamma=gamma
    )
    elapsed = time.perf_counter() - start_time

    # Compute silhouette on Gower distance for fair comparison
    sil = compute_silhouette_mixed(
        reps.mixed_separated_numeric,
        reps.mixed_separated_categorical,
        result.labels,
    )

    return ComparisonResult(
        dataset=cfg.get("dataset_name", "unknown"),
        method=method,
        k=k,
        mean_ari=float(np.mean(result.ari_scores)),
        std_ari=float(np.std(result.ari_scores)),
        silhouette=sil,
        mean_confidence=float(np.mean(result.confidence)),
        runtime_seconds=elapsed,
    )


def run_baseline_comparison(
    datasets: List[Tuple[str, Dict[str, Any]]],
    methods: List[str],
    k_values: List[int],
    seed: int = 42,
) -> pd.DataFrame:
    """Run full baseline comparison across datasets, methods, and K values.

    Parameters
    ----------
    datasets : List[Tuple[str, Dict[str, Any]]]
        List of (dataset_name, config_dict) tuples.
    methods : List[str]
        Methods to compare: "mixed_encoded", "gower_pam", "kprototypes".
    k_values : List[int]
        Number of clusters to test.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Comparison results with columns: dataset, method, k, mean_ari, std_ari,
        silhouette, mean_confidence, runtime_seconds.
    """
    results: List[ComparisonResult] = []

    for dataset_name, cfg in datasets:
        cfg["dataset_name"] = dataset_name
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")

        # Load data
        df = load_dataset(cfg)

        for k in k_values:
            print(f"\n  K={k}")
            for method in methods:
                print(f"    Running {method}...", end=" ", flush=True)
                try:
                    result = run_single_comparison(df, cfg, method, k, seed)
                    results.append(result)
                    print(
                        f"ARI={result.mean_ari:.3f} "
                        f"Sil={result.silhouette:.3f} "
                        f"({result.runtime_seconds:.1f}s)"
                    )
                except Exception as e:
                    print(f"FAILED: {e}")

    # Convert to DataFrame
    df_results = pd.DataFrame(
        [
            {
                "dataset": r.dataset,
                "method": r.method,
                "k": r.k,
                "mean_ari": r.mean_ari,
                "std_ari": r.std_ari,
                "silhouette": r.silhouette,
                "mean_confidence": r.mean_confidence,
                "runtime_seconds": r.runtime_seconds,
            }
            for r in results
        ]
    )

    return df_results


def format_comparison_table(df: pd.DataFrame) -> str:
    """Format comparison results as a markdown table."""
    lines = [
        "| Dataset | Method | K | ARI (mean±std) | Silhouette | Confidence | Runtime |",
        "|---------|--------|---|----------------|------------|------------|---------|",
    ]

    for _, row in df.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['k']} | "
            f"{row['mean_ari']:.3f}±{row['std_ari']:.3f} | "
            f"{row['silhouette']:.3f} | "
            f"{row['mean_confidence']:.3f} | "
            f"{row['runtime_seconds']:.1f}s |"
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare mixed-type clustering methods"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yml"),
        help="Base configuration file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/tables/baseline_comparison.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["mixed_encoded", "gower_pam", "kprototypes", "hierarchical_gower"],
        help="Methods to compare",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[2, 3, 4],
        help="K values to test",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Load base config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Prepare dataset list (single dataset from config by default)
    dataset_name = Path(cfg["dataset"]["path"]).stem
    datasets: List[Tuple[str, Dict[str, Any]]] = [(dataset_name, cfg)]

    print("Baseline Clustering Comparison")
    print(f"Methods: {args.methods}")
    print(f"K values: {args.k_values}")
    print(f"Seed: {args.seed}")

    # Run comparison
    df_results = run_baseline_comparison(
        datasets, args.methods, args.k_values, args.seed
    )

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(args.output, index=False)
    print(f"\nResults saved to: {args.output}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(format_comparison_table(df_results))

    # Print key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    # Best method per dataset/K
    for dataset in df_results["dataset"].unique():
        for k in df_results["k"].unique():
            subset = df_results[(df_results["dataset"] == dataset) & (df_results["k"] == k)]
            if len(subset) > 0:
                best_idx = int(subset["mean_ari"].idxmax())  # type: ignore[arg-type]
                best = subset.loc[best_idx]
                print(
                    f"{dataset} (K={k}): Best ARI = {best['method']} "
                    f"({best['mean_ari']:.3f})"
                )


if __name__ == "__main__":
    main()
