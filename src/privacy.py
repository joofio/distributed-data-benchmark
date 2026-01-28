"""
Privacy module for Local Differential Privacy (LDP) noise injection.

Provides functions to add calibrated Laplace noise to feature profiles
before clustering, enabling privacy-utility trade-off analysis.
"""

import numpy as np
from typing import Optional, Tuple


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
