from __future__ import annotations

from typing import Literal

import numpy as np
from kmodes.kmodes import KModes
from sklearn.cluster import KMeans


def kmeans_cluster(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Run k-means clustering with a fixed seed."""
    # Use multiple initializations for stability.
    model = KMeans(n_clusters=k, n_init=10, random_state=seed)
    return model.fit_predict(X)


def kmodes_cluster(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Run k-modes clustering with a fixed seed."""
    # Huang initialization is the default for categorical data.
    model = KModes(n_clusters=k, init="Huang", n_init=5, random_state=seed)
    return model.fit_predict(X)


def cluster_data(
    X: np.ndarray, k: int, seed: int, method: Literal["kmeans", "kmodes"]
) -> np.ndarray:
    """Dispatch clustering based on method name."""
    if method == "kmeans":
        return kmeans_cluster(X, k, seed)
    if method == "kmodes":
        return kmodes_cluster(X, k, seed)
    raise ValueError(f"Unsupported clustering method: {method}")
