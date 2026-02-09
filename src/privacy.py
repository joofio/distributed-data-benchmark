"""
Privacy module for Local Differential Privacy (LDP) noise injection
and re-identification risk analysis.

Provides functions to:
- Add calibrated Laplace noise to feature profiles for LDP
- Compute re-identification risk metrics (k-anonymity, singletons)
- Simulate membership inference attacks for privacy evaluation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import Counter


def add_laplace_noise(
    X: np.ndarray,
    epsilon: float,
    sensitivity: Optional[np.ndarray] = None,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Add Laplace noise calibrated for epsilon-differential privacy.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    epsilon : float
        Privacy budget. Higher = less noise = less privacy.
        Use np.inf for no noise (baseline).
    sensitivity : np.ndarray, optional
        Per-feature sensitivity (scale for Laplace noise).
        If None, uses feature range (max - min) as sensitivity.
    seed : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    X_noisy : np.ndarray
        Feature matrix with Laplace noise added.
    """
    if epsilon == np.inf or epsilon <= 0:
        return X.copy()
    
    if seed is not None:
        np.random.seed(seed)
    
    # Estimate sensitivity as feature range if not provided
    if sensitivity is None:
        sensitivity = np.ptp(X, axis=0)  # max - min per feature
        # Avoid zero sensitivity (constant features)
        sensitivity = np.maximum(sensitivity, 1e-10)
    
    # Laplace scale = sensitivity / epsilon
    scale = sensitivity / epsilon
    
    # Generate Laplace noise
    noise = np.random.laplace(loc=0, scale=scale, size=X.shape)
    
    return X + noise


def estimate_sensitivity(X: np.ndarray, method: str = "range") -> np.ndarray:
    """
    Estimate per-feature sensitivity for Laplace mechanism.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    method : str
        Estimation method:
        - "range": max - min (conservative)
        - "iqr": interquartile range (robust to outliers)
        - "std": 2 standard deviations
        
    Returns
    -------
    sensitivity : np.ndarray
        Per-feature sensitivity estimates.
    """
    if method == "range":
        sensitivity = np.ptp(X, axis=0)
    elif method == "iqr":
        q75, q25 = np.percentile(X, [75, 25], axis=0)
        sensitivity = q75 - q25
    elif method == "std":
        sensitivity = 2 * np.std(X, axis=0)
    else:
        raise ValueError(f"Unknown sensitivity method: {method}")
    
    # Ensure non-zero sensitivity
    return np.maximum(sensitivity, 1e-10)


def compute_noise_stats(
    X_original: np.ndarray,
    X_noisy: np.ndarray
) -> dict:
    """
    Compute statistics comparing original and noisy features.
    
    Returns
    -------
    stats : dict
        Dictionary with noise statistics.
    """
    noise = X_noisy - X_original
    
    return {
        "mean_abs_noise": np.mean(np.abs(noise)),
        "max_noise": np.max(np.abs(noise)),
        "noise_to_signal_ratio": np.mean(np.abs(noise)) / np.mean(np.abs(X_original)),
        "per_feature_noise_std": np.std(noise, axis=0).tolist()
    }


def compute_reidentification_risk(
    df: pd.DataFrame,
    quasi_identifiers: List[str],
    sensitive_attributes: Optional[List[str]] = None,
) -> Dict[str, any]:
    """
    Compute re-identification risk metrics based on quasi-identifiers.

    Analyzes how unique records are based on combinations of quasi-identifier
    attributes, which could be used to re-identify institutions.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with institutional records.
    quasi_identifiers : List[str]
        Column names that could be used to identify institutions
        (e.g., size, region, specialty mix).
    sensitive_attributes : List[str], optional
        Columns containing sensitive information to protect.

    Returns
    -------
    Dict[str, any]
        Dictionary containing:
        - n_records: Total number of records
        - n_unique_combinations: Number of unique quasi-identifier patterns
        - singleton_count: Number of records with unique patterns (k=1)
        - singleton_rate: Fraction of records that are singletons
        - min_k_anonymity: Minimum k-anonymity level
        - mean_k_anonymity: Average equivalence class size
        - median_k_anonymity: Median equivalence class size
        - k_distribution: Distribution of equivalence class sizes
        - disclosure_risk_score: Overall risk score (0-1, higher = riskier)

    Notes
    -----
    K-anonymity of k means each record shares its quasi-identifier pattern
    with at least k-1 other records. Lower k values indicate higher risk.
    """
    n_records = len(df)

    # Create quasi-identifier combinations
    if not quasi_identifiers:
        return {
            "n_records": n_records,
            "n_unique_combinations": 1,
            "singleton_count": 0,
            "singleton_rate": 0.0,
            "min_k_anonymity": n_records,
            "mean_k_anonymity": float(n_records),
            "median_k_anonymity": float(n_records),
            "k_distribution": {n_records: 1},
            "disclosure_risk_score": 0.0,
        }

    # Discretize numeric columns for k-anonymity analysis
    qi_data = df[quasi_identifiers].copy()
    for col in quasi_identifiers:
        if pd.api.types.is_numeric_dtype(qi_data[col]):
            # Bin into quartiles for numeric features
            qi_data[col] = pd.qcut(qi_data[col], q=4, labels=False, duplicates="drop")

    # Count equivalence class sizes
    qi_tuples = qi_data.apply(tuple, axis=1)
    class_counts = Counter(qi_tuples)
    class_sizes = list(class_counts.values())

    # Compute metrics
    n_unique = len(class_counts)
    singleton_count = sum(1 for size in class_sizes if size == 1)
    singleton_rate = singleton_count / n_records if n_records > 0 else 0.0
    min_k = min(class_sizes) if class_sizes else 0
    mean_k = np.mean(class_sizes) if class_sizes else 0.0
    median_k = np.median(class_sizes) if class_sizes else 0.0

    # Distribution of k values
    k_dist = Counter(class_sizes)

    # Disclosure risk score: weighted combination of metrics
    # Higher weight on singletons and small classes
    risk_score = 0.0
    for size, count in k_dist.items():
        if size == 1:
            risk_score += count * 1.0
        elif size == 2:
            risk_score += count * 0.5
        elif size <= 5:
            risk_score += count * 0.2
        else:
            risk_score += count * 0.05
    risk_score = risk_score / n_records if n_records > 0 else 0.0
    risk_score = min(risk_score, 1.0)

    return {
        "n_records": n_records,
        "n_unique_combinations": n_unique,
        "singleton_count": singleton_count,
        "singleton_rate": float(singleton_rate),
        "min_k_anonymity": min_k,
        "mean_k_anonymity": float(mean_k),
        "median_k_anonymity": float(median_k),
        "k_distribution": dict(k_dist),
        "disclosure_risk_score": float(risk_score),
    }


def simulate_membership_inference_attack(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_shadow_models: int = 10,
    k_neighbors: int = 5,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """
    Simulate a membership inference attack using kNN-based approach.

    Tests whether an adversary can determine if a specific record was
    used in the clustering/benchmarking by comparing its distance to
    known in-sample vs out-of-sample records.

    Parameters
    ----------
    X_train : np.ndarray
        Feature matrix of records used in clustering (members).
    X_test : np.ndarray
        Feature matrix of records NOT used in clustering (non-members).
    n_shadow_models : int
        Number of shadow model iterations for stability.
    k_neighbors : int
        Number of nearest neighbors for distance computation.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - attack_accuracy: Fraction of correct membership predictions
        - baseline_accuracy: Random guess accuracy (should be ~0.5)
        - attack_advantage: attack_accuracy - baseline_accuracy
        - member_confidence: Mean confidence score for true members
        - nonmember_confidence: Mean confidence score for non-members
        - auc_roc: Area under ROC curve for membership prediction

    Notes
    -----
    Attack accuracy significantly above 0.5 indicates privacy leakage.
    This simulates a realistic adversary with access to aggregate outputs.
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics import roc_auc_score

    if seed is not None:
        np.random.seed(seed)

    n_train = len(X_train)
    n_test = len(X_test)

    if n_train == 0 or n_test == 0:
        return {
            "attack_accuracy": 0.5,
            "baseline_accuracy": 0.5,
            "attack_advantage": 0.0,
            "member_confidence": 0.5,
            "nonmember_confidence": 0.5,
            "auc_roc": 0.5,
        }

    # Combine all data
    X_all = np.vstack([X_train, X_test])
    y_true = np.concatenate([np.ones(n_train), np.zeros(n_test)])

    attack_scores = []

    for _ in range(n_shadow_models):
        # Fit kNN on training data only
        k_eff = min(k_neighbors, n_train - 1)
        if k_eff < 1:
            k_eff = 1

        nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean")
        nn.fit(X_train)

        # Compute distances for all points
        distances, _ = nn.kneighbors(X_all)

        # Use mean distance to k neighbors as membership score
        # Lower distance to training set = more likely to be a member
        mean_distances = distances[:, 1:].mean(axis=1)  # Exclude self

        # Normalize to [0, 1] confidence scores
        # Invert so higher score = more likely member
        max_dist = mean_distances.max()
        if max_dist > 0:
            confidence = 1 - (mean_distances / max_dist)
        else:
            confidence = np.ones_like(mean_distances) * 0.5

        attack_scores.append(confidence)

    # Average across shadow models
    avg_scores = np.mean(attack_scores, axis=0)

    # Binary predictions at 0.5 threshold
    predictions = (avg_scores > 0.5).astype(int)
    accuracy = np.mean(predictions == y_true)

    # Compute AUC
    try:
        auc = roc_auc_score(y_true, avg_scores)
    except ValueError:
        auc = 0.5

    # Member vs non-member confidence
    member_conf = avg_scores[:n_train].mean()
    nonmember_conf = avg_scores[n_train:].mean()

    return {
        "attack_accuracy": float(accuracy),
        "baseline_accuracy": 0.5,
        "attack_advantage": float(accuracy - 0.5),
        "member_confidence": float(member_conf),
        "nonmember_confidence": float(nonmember_conf),
        "auc_roc": float(auc),
    }


def compute_privacy_risk_summary(
    df: pd.DataFrame,
    feature_cols: List[str],
    quasi_identifiers: List[str],
    holdout_fraction: float = 0.2,
    seed: int = 42,
) -> Dict[str, any]:
    """
    Compute comprehensive privacy risk summary for a dataset.

    Combines k-anonymity analysis and membership inference attack
    simulation to provide an overall privacy risk assessment.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with institutional records.
    feature_cols : List[str]
        Columns used for clustering/analysis.
    quasi_identifiers : List[str]
        Columns that could be used for re-identification.
    holdout_fraction : float
        Fraction of data to use as non-members for attack simulation.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, any]
        Combined privacy risk metrics from both analyses.
    """
    # K-anonymity analysis
    reid_risk = compute_reidentification_risk(df, quasi_identifiers)

    # Prepare data for membership inference
    rng = np.random.default_rng(seed)
    n = len(df)
    n_holdout = max(1, int(n * holdout_fraction))

    if n_holdout >= n:
        # Not enough data for meaningful split
        mia_risk = {
            "attack_accuracy": 0.5,
            "baseline_accuracy": 0.5,
            "attack_advantage": 0.0,
            "member_confidence": 0.5,
            "nonmember_confidence": 0.5,
            "auc_roc": 0.5,
        }
    else:
        # Split into members and non-members
        holdout_idx = rng.choice(n, size=n_holdout, replace=False)
        member_idx = np.setdiff1d(np.arange(n), holdout_idx)

        # Filter to numeric columns only for membership inference
        numeric_cols = []
        for col in feature_cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)

        if not numeric_cols:
            # No numeric features, skip membership inference
            mia_risk = {
                "attack_accuracy": 0.5,
                "baseline_accuracy": 0.5,
                "attack_advantage": 0.0,
                "member_confidence": 0.5,
                "nonmember_confidence": 0.5,
                "auc_roc": 0.5,
            }
        else:
            X = df[numeric_cols].values.astype(float)

            # Handle missing values
            X = np.nan_to_num(X, nan=0.0)

            X_train = X[member_idx]
            X_test = X[holdout_idx]

            mia_risk = simulate_membership_inference_attack(
                X_train, X_test, seed=seed
            )

    # Combine results
    return {
        "reidentification": reid_risk,
        "membership_inference": mia_risk,
        "overall_risk_score": (
            reid_risk["disclosure_risk_score"] * 0.5
            + mia_risk["attack_advantage"] * 0.5
        ),
    }
