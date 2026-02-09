"""
Synthetic data generator for ground-truth validation experiments.

Generates synthetic institutional data with known cluster structure to validate
that consensus clustering can recover the true peer group assignments.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


@dataclass
class SyntheticDataset:
    """Container for synthetic data with ground truth labels."""

    df: pd.DataFrame
    true_labels: np.ndarray
    metadata: Dict[str, any]


def generate_synthetic_institutions(
    n_institutions: int = 30,
    n_clusters: int = 4,
    n_numeric_features: int = 10,
    n_categorical_features: int = 5,
    n_categories_per_feature: int = 4,
    cluster_separation: float = 2.0,
    within_cluster_std: float = 1.0,
    kpi_cluster_effect: float = 1.5,
    seed: int = 42,
) -> SyntheticDataset:
    """Generate synthetic institutional data with known cluster structure.

    Creates a dataset where institutions belong to distinct clusters with:
    - Numeric features: cluster centroids separated by `cluster_separation` std
    - Categorical features: cluster-correlated category probabilities
    - KPIs: cluster-dependent means with within-cluster variation

    Parameters
    ----------
    n_institutions : int
        Number of institutions to generate.
    n_clusters : int
        Number of true peer groups (clusters).
    n_numeric_features : int
        Number of continuous features.
    n_categorical_features : int
        Number of categorical features.
    n_categories_per_feature : int
        Number of categories per categorical feature.
    cluster_separation : float
        Distance between cluster centroids in standard deviation units.
        Higher values make clusters more distinct.
    within_cluster_std : float
        Standard deviation of features within each cluster.
        Lower values make clusters tighter.
    kpi_cluster_effect : float
        How much KPIs differ between clusters (in std units).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    SyntheticDataset
        Contains DataFrame with features and KPIs, true cluster labels,
        and metadata about generation parameters.

    Examples
    --------
    >>> data = generate_synthetic_institutions(n_institutions=50, n_clusters=5)
    >>> data.df.shape
    (50, ...)
    >>> len(np.unique(data.true_labels))
    5
    """
    rng = np.random.default_rng(seed)

    # Assign institutions to clusters (approximately balanced)
    base_size = n_institutions // n_clusters
    remainder = n_institutions % n_clusters
    cluster_sizes = [base_size + (1 if i < remainder else 0) for i in range(n_clusters)]
    true_labels = np.concatenate(
        [np.full(size, cluster_id) for cluster_id, size in enumerate(cluster_sizes)]
    )
    # Shuffle to avoid ordering bias
    shuffle_idx = rng.permutation(n_institutions)
    true_labels = true_labels[shuffle_idx]

    # Generate cluster centroids for numeric features
    # Each cluster centroid is offset from origin by cluster_separation
    centroids = np.zeros((n_clusters, n_numeric_features))
    for c in range(n_clusters):
        # Random direction for each cluster centroid
        direction = rng.standard_normal(n_numeric_features)
        direction = direction / np.linalg.norm(direction)
        centroids[c] = direction * cluster_separation * (c + 1) / n_clusters * 2

    # Generate numeric features
    X_numeric = np.zeros((n_institutions, n_numeric_features))
    for i in range(n_institutions):
        cluster = true_labels[i]
        X_numeric[i] = centroids[cluster] + rng.normal(
            0, within_cluster_std, n_numeric_features
        )

    # Generate categorical features with cluster-dependent probabilities
    X_categorical = np.zeros((n_institutions, n_categorical_features), dtype=int)
    for f in range(n_categorical_features):
        # Each cluster has a preferred category distribution
        cluster_probs = np.zeros((n_clusters, n_categories_per_feature))
        for c in range(n_clusters):
            # Dominant category for this cluster
            dominant = (c + f) % n_categories_per_feature
            probs = rng.dirichlet(np.ones(n_categories_per_feature) * 0.5)
            probs[dominant] += 2.0  # Boost dominant category
            probs = probs / probs.sum()
            cluster_probs[c] = probs

        for i in range(n_institutions):
            cluster = true_labels[i]
            X_categorical[i, f] = rng.choice(
                n_categories_per_feature, p=cluster_probs[cluster]
            )

    # Generate KPIs with cluster effects
    # target_rate: probability-like KPI (0-1 range)
    # target_std: variability KPI (positive)
    cluster_kpi_means = {
        "target_rate": rng.uniform(0.1, 0.9, n_clusters),
        "target_std": rng.uniform(0.05, 0.3, n_clusters),
    }

    kpis = {}
    for kpi_name, cluster_means in cluster_kpi_means.items():
        values = np.zeros(n_institutions)
        for i in range(n_institutions):
            cluster = true_labels[i]
            base_mean = cluster_means[cluster]
            # Add within-cluster variation
            noise = rng.normal(0, kpi_cluster_effect * 0.1)
            values[i] = np.clip(base_mean + noise, 0.01, 0.99)
        kpis[kpi_name] = values

    # Build DataFrame
    data = {"institution_id": [f"INST_{i:03d}" for i in range(n_institutions)]}

    # Add numeric features
    for f in range(n_numeric_features):
        data[f"numeric_{f}"] = X_numeric[:, f]

    # Add categorical features
    for f in range(n_categorical_features):
        data[f"categorical_{f}"] = X_categorical[:, f]

    # Add KPIs
    data["target_rate"] = kpis["target_rate"]
    data["target_std"] = kpis["target_std"]

    # Add institution size (correlated with cluster)
    base_sizes = rng.integers(100, 1000, n_clusters)
    data["n_patients"] = [
        int(base_sizes[true_labels[i]] + rng.integers(-50, 50))
        for i in range(n_institutions)
    ]

    df = pd.DataFrame(data)

    metadata = {
        "n_institutions": n_institutions,
        "n_clusters": n_clusters,
        "n_numeric_features": n_numeric_features,
        "n_categorical_features": n_categorical_features,
        "cluster_separation": cluster_separation,
        "within_cluster_std": within_cluster_std,
        "kpi_cluster_effect": kpi_cluster_effect,
        "seed": seed,
        "cluster_sizes": dict(zip(range(n_clusters), cluster_sizes)),
    }

    return SyntheticDataset(df=df, true_labels=true_labels, metadata=metadata)


def evaluate_against_ground_truth(
    predicted_labels: np.ndarray,
    true_labels: np.ndarray,
) -> Dict[str, float]:
    """Evaluate clustering accuracy against known ground truth.

    Computes standard clustering evaluation metrics that compare
    predicted cluster assignments to true labels.

    Parameters
    ----------
    predicted_labels : np.ndarray
        Cluster labels from consensus clustering.
    true_labels : np.ndarray
        True cluster labels from synthetic data generation.

    Returns
    -------
    Dict[str, float]
        Dictionary with:
        - ari: Adjusted Rand Index (-1 to 1, 1 = perfect match)
        - nmi: Normalized Mutual Information (0 to 1, 1 = perfect match)
        - cluster_purity: Fraction of samples in correct majority cluster
        - n_predicted_clusters: Number of clusters found
        - n_true_clusters: Number of true clusters

    Notes
    -----
    ARI is preferred for comparing clusterings as it adjusts for chance.
    NMI is useful for comparing clusterings with different numbers of clusters.
    """
    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)

    # Compute cluster purity
    n_samples = len(true_labels)
    purity = 0.0
    for cluster in np.unique(predicted_labels):
        mask = predicted_labels == cluster
        true_in_cluster = true_labels[mask]
        if len(true_in_cluster) > 0:
            # Count most common true label in this predicted cluster
            unique, counts = np.unique(true_in_cluster, return_counts=True)
            purity += counts.max()
    purity = purity / n_samples

    return {
        "ari": float(ari),
        "nmi": float(nmi),
        "cluster_purity": float(purity),
        "n_predicted_clusters": int(len(np.unique(predicted_labels))),
        "n_true_clusters": int(len(np.unique(true_labels))),
    }


def generate_config_for_synthetic(
    synthetic_data: SyntheticDataset,
    output_path: str = "data/synthetic.csv",
) -> Dict[str, any]:
    """Generate a configuration dict compatible with the pipeline.

    Parameters
    ----------
    synthetic_data : SyntheticDataset
        Generated synthetic dataset.
    output_path : str
        Path where the synthetic CSV will be saved.

    Returns
    -------
    Dict
        Configuration dictionary for run_experiments.py
    """
    df = synthetic_data.df
    meta = synthetic_data.metadata

    numeric_cols = [c for c in df.columns if c.startswith("numeric_")]
    categorical_cols = [c for c in df.columns if c.startswith("categorical_")]

    config = {
        "seed": meta["seed"],
        "dataset": {
            "type": "synthetic",
            "path": output_path,
            "mapping": {},
        },
        "features": {
            "id": "institution_id",
            "numeric": numeric_cols + ["n_patients"],
            "categorical": categorical_cols,
        },
        "targets": {
            "kpis": ["target_rate", "target_std"],
            "descriptors": ["n_patients"],
        },
        "preprocessing": {
            "categorical_encoding": "ordinal",
        },
        "representation": "mixed_encoded",
        "consensus": {
            "n_bootstraps": 100,
            "numeric_jitter_scale": 0.05,
            "feature_bootstrap": True,
            "sample_fraction": 1.0,
            "mixed_weight": 0.5,
        },
        "benchmark": {
            "outlier_percentile_low": 5,
            "outlier_percentile_high": 95,
            "outlier_zscore_abs": 2.0,
            "knn_k": [3, 5, 7],
        },
        "perturbation": {
            "enabled": True,
            "n_runs": 50,
            "n_institutions": min(5, meta["n_institutions"] // 3),
            "kpis": ["target_rate", "target_std"],
            "shift": {
                "type": "additive",
                "magnitudes": [0.1, 0.2, 0.3],
            },
        },
        "selection": {
            "k_values": list(
                range(
                    max(2, meta["n_clusters"] - 1),
                    min(meta["n_clusters"] + 3, meta["n_institutions"] // 3),
                )
            ),
            "min_cluster_size": 2,
            "stability_plateau_delta": 0.05,
        },
        "output": {
            "tables_dir": "reports/synthetic/tables",
            "figures_dir": "reports/synthetic/figures",
            "summary_path": "reports/synthetic/summary.json",
        },
    }

    return config


def run_ground_truth_sweep(
    n_institutions_list: List[int],
    n_clusters_list: List[int],
    cluster_separation_list: List[float],
    seed: int = 42,
) -> pd.DataFrame:
    """Run parameter sweep for ground truth validation.

    Generates synthetic data across parameter combinations and stores
    results for later validation with consensus clustering.

    Parameters
    ----------
    n_institutions_list : List[int]
        List of institution counts to test.
    n_clusters_list : List[int]
        List of cluster counts to test.
    cluster_separation_list : List[float]
        List of separation values to test.
    seed : int
        Base random seed.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: n_institutions, n_clusters, cluster_separation,
        plus the generated data path for each combination.
    """
    results = []
    run_id = 0

    for n_inst in n_institutions_list:
        for n_clust in n_clusters_list:
            if n_clust >= n_inst:
                continue  # Skip invalid combinations
            for sep in cluster_separation_list:
                data = generate_synthetic_institutions(
                    n_institutions=n_inst,
                    n_clusters=n_clust,
                    cluster_separation=sep,
                    seed=seed + run_id,
                )
                results.append(
                    {
                        "run_id": run_id,
                        "n_institutions": n_inst,
                        "n_clusters": n_clust,
                        "cluster_separation": sep,
                        "seed": seed + run_id,
                        "metadata": data.metadata,
                    }
                )
                run_id += 1

    return pd.DataFrame(results)
