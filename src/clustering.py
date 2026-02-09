from __future__ import annotations

from typing import List, Literal, Tuple

try:
    import gower
except ImportError:
    gower = None

import numpy as np
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids


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


def gower_distance_matrix(
    X_num: np.ndarray | None,
    X_cat: np.ndarray | None,
    cat_features: List[int] | None = None,
) -> np.ndarray:
    """Compute Gower distance matrix for mixed numeric/categorical data.

    The Gower distance properly handles mixed-type data by computing:
    - Manhattan distance (scaled by range) for numeric features
    - Simple matching (0/1) for categorical features
    Each feature contributes equally to the final distance in [0, 1].

    Parameters
    ----------
    X_num : np.ndarray or None
        Numeric features array of shape (N, n_numeric).
    X_cat : np.ndarray or None
        Categorical features array of shape (N, n_categorical).
    cat_features : List[int] or None
        Indices of categorical columns in a combined array. If provided,
        X_num and X_cat are ignored and a combined array is expected.

    Returns
    -------
    np.ndarray
        N×N symmetric distance matrix with values in [0, 1].
    """
    if gower is None:
        raise ImportError("gower package is required for Gower distance. Install with 'pip install gower'")

    if X_num is None and X_cat is None:
        raise ValueError("At least one of X_num or X_cat must be provided")

    # Combine arrays for gower library
    # gower expects cat_features as a boolean array where True = categorical
    # Also ensure float dtype to avoid casting issues in gower library
    if X_num is not None and X_cat is not None:
        X_combined = np.hstack([X_num.astype(float), X_cat.astype(float)])
        n_num = X_num.shape[1]
        n_cat = X_cat.shape[1]
        n_total = n_num + n_cat
        # Create boolean mask: False for numeric, True for categorical
        cat_mask = np.zeros(n_total, dtype=bool)
        cat_mask[n_num:] = True
    elif X_num is not None:
        X_combined = X_num.astype(float)
        n_total = X_num.shape[1]
        cat_mask = np.zeros(n_total, dtype=bool)  # All numeric
    else:
        X_combined = X_cat.astype(float)
        n_total = X_cat.shape[1]
        cat_mask = np.ones(n_total, dtype=bool)  # All categorical

    # Compute Gower distance matrix
    # gower.gower_matrix returns distances (not similarities)
    D = gower.gower_matrix(X_combined, cat_features=cat_mask)

    # Ensure symmetry (numerical precision) and valid range
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    D = np.clip(D, 0.0, 1.0)

    return D


def pam_cluster(D: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Partitioning Around Medoids on precomputed distance matrix.

    PAM is more robust than k-means for non-Euclidean distances as it
    selects actual data points (medoids) as cluster centers rather than
    computing means that may lie outside the data space.

    Parameters
    ----------
    D : np.ndarray
        N×N precomputed distance matrix (symmetric, zero diagonal).
    k : int
        Number of clusters.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        1D array of cluster labels (0 to k-1).
    """
    # Use sklearn-extra's KMedoids with precomputed metric
    model = KMedoids(
        n_clusters=k,
        metric="precomputed",
        init="k-medoids++",
        random_state=seed,
        max_iter=300,
    )
    return model.fit_predict(D)


def hierarchical_gower_cluster(
    D: np.ndarray,
    k: int,
    linkage_method: Literal["average", "complete", "single"] = "average",
) -> np.ndarray:
    """Hierarchical clustering on precomputed Gower distance matrix.

    Hierarchical clustering is well-suited for mixed-type data as it operates
    directly on pairwise distances without assuming cluster geometry. This
    implementation uses agglomerative clustering with configurable linkage.

    Parameters
    ----------
    D : np.ndarray
        N×N precomputed distance matrix (symmetric, zero diagonal).
        Typically computed via gower_distance_matrix().
    k : int
        Number of clusters to extract from the dendrogram.
    linkage_method : {"average", "complete", "single"}
        Linkage criterion for merging clusters:
        - "average" (UPGMA): robust, produces balanced dendrograms
        - "complete": maximizes compactness, sensitive to outliers
        - "single": minimizes distances, prone to chaining

    Returns
    -------
    np.ndarray
        1D array of cluster labels (0 to k-1).

    Notes
    -----
    The average linkage method is recommended for healthcare benchmarking
    as it balances sensitivity to outliers (complete) and chaining (single).
    """
    n = D.shape[0]
    if n == 1:
        return np.array([0])

    # Convert distance matrix to condensed form for scipy
    # squareform expects symmetric matrix with zero diagonal
    condensed = squareform(D, checks=False)

    # Build linkage matrix
    Z = linkage(condensed, method=linkage_method)

    # Cut dendrogram to get k clusters
    # fcluster returns 1-indexed labels, convert to 0-indexed
    labels = fcluster(Z, k, criterion="maxclust")
    return labels.astype(int) - 1


def kprototypes_cluster(
    X_num: np.ndarray,
    X_cat: np.ndarray,
    k: int,
    seed: int,
    gamma: float = 0.5,
) -> np.ndarray:
    """K-prototypes clustering for mixed numeric/categorical data.

    K-prototypes extends k-means to handle categorical features by:
    - Using Euclidean distance for numeric features
    - Using simple matching (Hamming) distance for categorical features
    - Combining both with a weighting factor gamma

    Parameters
    ----------
    X_num : np.ndarray
        Numeric features array of shape (N, n_numeric).
    X_cat : np.ndarray
        Categorical features array of shape (N, n_categorical).
    k : int
        Number of clusters.
    seed : int
        Random seed for reproducibility.
    gamma : float
        Weight for categorical distance. Higher values give more importance
        to categorical features. If None, the library computes a default
        based on the mean std of numeric features.

    Returns
    -------
    np.ndarray
        1D array of cluster labels (0 to k-1).
    """
    # K-prototypes expects categorical indices in the combined array
    n_num = X_num.shape[1]
    X_combined = np.hstack([X_num, X_cat])
    cat_indices = list(range(n_num, n_num + X_cat.shape[1]))

    model = KPrototypes(
        n_clusters=k,
        init="Huang",
        n_init=5,
        gamma=gamma,
        random_state=seed,
    )
    return model.fit_predict(X_combined, categorical=cat_indices)


def cluster_data(
    X: np.ndarray | Tuple[np.ndarray, np.ndarray],
    k: int,
    seed: int,
    method: Literal["kmeans", "kmodes", "pam", "kprototypes", "hierarchical"],
    distance_matrix: np.ndarray | None = None,
    gamma: float = 0.5,
    linkage_method: Literal["average", "complete", "single"] = "average",
) -> np.ndarray:
    """Dispatch clustering based on method name.

    Parameters
    ----------
    X : np.ndarray or Tuple[np.ndarray, np.ndarray]
        Feature array. For pam/kprototypes/hierarchical, can be a tuple (X_num, X_cat).
    k : int
        Number of clusters.
    seed : int
        Random seed for reproducibility (not used for hierarchical method).
    method : str
        Clustering method: "kmeans", "kmodes", "pam", "kprototypes", or "hierarchical".
    distance_matrix : np.ndarray or None
        Precomputed distance matrix for PAM/hierarchical. If None and method
        requires distances, Gower distance will be computed from X.
    gamma : float
        Weight for categorical distance in kprototypes. Default 0.5.
    linkage_method : {"average", "complete", "single"}
        Linkage criterion for hierarchical clustering. Default "average".

    Returns
    -------
    np.ndarray
        1D array of cluster labels.
    """
    if method == "kmeans":
        if isinstance(X, tuple):
            raise ValueError("kmeans expects a single array, not tuple")
        return kmeans_cluster(X, k, seed)

    if method == "kmodes":
        if isinstance(X, tuple):
            raise ValueError("kmodes expects a single array, not tuple")
        return kmodes_cluster(X, k, seed)

    if method == "pam":
        if distance_matrix is not None:
            return pam_cluster(distance_matrix, k, seed)
        # Compute Gower distance if not provided
        if isinstance(X, tuple):
            X_num, X_cat = X
            D = gower_distance_matrix(X_num, X_cat)
        else:
            # Assume all numeric if single array
            D = gower_distance_matrix(X, None)
        return pam_cluster(D, k, seed)

    if method == "kprototypes":
        if not isinstance(X, tuple):
            raise ValueError("kprototypes requires a tuple (X_num, X_cat)")
        X_num, X_cat = X
        return kprototypes_cluster(X_num, X_cat, k, seed, gamma)

    if method == "hierarchical":
        if distance_matrix is not None:
            return hierarchical_gower_cluster(distance_matrix, k, linkage_method)
        # Compute Gower distance if not provided
        if isinstance(X, tuple):
            X_num, X_cat = X
            D = gower_distance_matrix(X_num, X_cat)
        else:
            # Assume all numeric if single array
            D = gower_distance_matrix(X, None)
        return hierarchical_gower_cluster(D, k, linkage_method)

    raise ValueError(f"Unsupported clustering method: {method}")
