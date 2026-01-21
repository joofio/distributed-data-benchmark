from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

from src.clustering import cluster_data


@dataclass
class ConsensusResult:
    labels: np.ndarray
    coassignment: np.ndarray
    ari_scores: List[float]
    confidence: np.ndarray


def _bootstrap_indices(rng: np.random.Generator, n: int, fraction: float) -> np.ndarray:
    """Return bootstrap sample indices for a given fraction."""
    # Sample with replacement for snapshot perturbations.
    size = max(1, int(round(n * fraction)))
    return rng.choice(n, size=size, replace=True)


def _feature_bootstrap(rng: np.random.Generator, n_features: int) -> np.ndarray:
    """Return bootstrap indices for feature columns."""
    if n_features == 0:
        return np.array([], dtype=int)
    return rng.choice(n_features, size=n_features, replace=True)


def _apply_numeric_jitter(
    X: np.ndarray, rng: np.random.Generator, jitter_scale: float, iqr: np.ndarray
) -> np.ndarray:
    """Apply Gaussian jitter scaled by feature IQR."""
    if jitter_scale <= 0:
        return X
    # Scale noise per feature to preserve relative magnitudes.
    noise = rng.normal(0.0, jitter_scale * iqr, size=X.shape)
    return X + noise


def _update_coassignment(
    coassign: np.ndarray, counts: np.ndarray, labels: np.ndarray, indices: np.ndarray
) -> None:
    """Update co-assignment counts for a bootstrap sample."""
    # Increment pairwise counts for sampled indices.
    m = len(indices)
    for i in range(m):
        idx_i = indices[i]
        for j in range(m):
            idx_j = indices[j]
            counts[idx_i, idx_j] += 1
            if labels[i] == labels[j]:
                coassign[idx_i, idx_j] += 1


def _consensus_from_coassignment(C: np.ndarray, k: int) -> np.ndarray:
    """Derive consensus labels from a co-assignment matrix."""
    if C.shape[0] == 1:
        return np.array([0])
    # Cluster on distance defined by 1 - co-assignment.
    D = 1.0 - C
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, k, criterion="maxclust")
    return labels.astype(int) - 1


def _compute_confidence(C: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute per-institution confidence as mean in-cluster co-assignment."""
    n = len(labels)
    confidence = np.zeros(n, dtype=float)
    for i in range(n):
        members = np.where(labels == labels[i])[0]
        if len(members) <= 1:
            confidence[i] = 1.0
        else:
            others = members[members != i]
            confidence[i] = float(np.mean(C[i, others]))
    return confidence


def _categorical_method(cfg: Dict[str, Any]) -> str:
    """Choose clustering method for categorical data based on encoding."""
    encoding = cfg["preprocessing"]["categorical_encoding"]
    return "kmeans" if encoding == "onehot" else "kmodes"


def run_consensus(
    X: np.ndarray | Tuple[np.ndarray, np.ndarray],
    representation: str,
    cfg: Dict[str, Any],
    k: int,
    seed: int,
) -> ConsensusResult:
    """Run snapshot consensus clustering and return stability artifacts."""
    consensus_cfg = cfg["consensus"]
    n_boot = consensus_cfg["n_bootstraps"]
    jitter_scale = consensus_cfg.get("numeric_jitter_scale", 0.0)
    feature_bootstrap = consensus_cfg.get("feature_bootstrap", False)
    sample_fraction = consensus_cfg.get("sample_fraction", 1.0)

    if representation == "mixed_separated":
        X_num, X_cat = X
        n = X_num.shape[0]
    else:
        n = X.shape[0]

    coassign = np.zeros((n, n), dtype=float)
    counts = np.zeros((n, n), dtype=float)
    coassign_num = np.zeros((n, n), dtype=float)
    counts_num = np.zeros((n, n), dtype=float)
    coassign_cat = np.zeros((n, n), dtype=float)
    counts_cat = np.zeros((n, n), dtype=float)
    assignments: List[Tuple[np.ndarray, np.ndarray]] = []

    # Precompute IQR for jittering numeric representations.
    if representation == "numeric":
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        iqr = np.where(iqr == 0, 1.0, iqr)
    elif representation == "mixed_encoded":
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        iqr = np.where(iqr == 0, 1.0, iqr)
    elif representation == "mixed_separated":
        iqr = np.subtract(*np.percentile(X_num, [75, 25], axis=0))
        iqr = np.where(iqr == 0, 1.0, iqr)
    else:
        iqr = None

    for b in range(n_boot):
        # Bootstrap rows (and optionally features) for each run.
        rng = np.random.default_rng(seed + b)
        sample_idx = _bootstrap_indices(rng, n, sample_fraction)
        unique_idx = np.unique(sample_idx)

        if representation == "numeric":
            Xb = X[unique_idx].copy()
            if feature_bootstrap:
                cols = _feature_bootstrap(rng, Xb.shape[1])
                Xb = Xb[:, cols]
            Xb = _apply_numeric_jitter(Xb, rng, jitter_scale, iqr)
            labels = cluster_data(Xb, k, seed + b, "kmeans")
            assignments.append((unique_idx, labels))
            # Update co-assignment for the sampled institutions.
            _update_coassignment(coassign, counts, labels, unique_idx)
        elif representation == "categorical":
            Xb = X[unique_idx].copy()
            if feature_bootstrap:
                cols = _feature_bootstrap(rng, Xb.shape[1])
                Xb = Xb[:, cols]
            labels = cluster_data(Xb, k, seed + b, _categorical_method(cfg))
            assignments.append((unique_idx, labels))
            _update_coassignment(coassign, counts, labels, unique_idx)
        elif representation == "mixed_encoded":
            Xb = X[unique_idx].copy()
            if feature_bootstrap:
                cols = _feature_bootstrap(rng, Xb.shape[1])
                Xb = Xb[:, cols]
            Xb = _apply_numeric_jitter(Xb, rng, jitter_scale, iqr)
            labels = cluster_data(Xb, k, seed + b, "kmeans")
            assignments.append((unique_idx, labels))
            _update_coassignment(coassign, counts, labels, unique_idx)
        elif representation == "mixed_separated":
            Xb_num = X_num[unique_idx].copy()
            Xb_cat = X_cat[unique_idx].copy()
            if feature_bootstrap:
                cols_num = _feature_bootstrap(rng, Xb_num.shape[1])
                cols_cat = _feature_bootstrap(rng, Xb_cat.shape[1])
                Xb_num = Xb_num[:, cols_num]
                Xb_cat = Xb_cat[:, cols_cat]
            Xb_num = _apply_numeric_jitter(Xb_num, rng, jitter_scale, iqr)
            labels_num = cluster_data(Xb_num, k, seed + b, "kmeans")
            labels_cat = cluster_data(Xb_cat, k, seed + b, _categorical_method(cfg))
            assignments.append((unique_idx, labels_num))
            assignments.append((unique_idx, labels_cat))
            _update_coassignment(coassign_num, counts_num, labels_num, unique_idx)
            _update_coassignment(coassign_cat, counts_cat, labels_cat, unique_idx)
        else:
            raise ValueError(f"Unsupported representation: {representation}")

    if representation == "mixed_separated":
        # Blend numeric and categorical co-assignment matrices.
        weight = consensus_cfg.get("mixed_weight", 0.5)
        C_num = np.divide(
            coassign_num,
            counts_num,
            out=np.zeros_like(coassign_num),
            where=counts_num > 0,
        )
        C_cat = np.divide(
            coassign_cat,
            counts_cat,
            out=np.zeros_like(coassign_cat),
            where=counts_cat > 0,
        )
        np.fill_diagonal(C_num, 1.0)
        np.fill_diagonal(C_cat, 1.0)
        C = np.clip(weight * C_num + (1.0 - weight) * C_cat, 0.0, 1.0)
    else:
        # Normalize counts into co-assignment probabilities.
        C = np.divide(coassign, counts, out=np.zeros_like(coassign), where=counts > 0)
        np.fill_diagonal(C, 1.0)

    # Convert co-assignment into final consensus labels.
    labels = _consensus_from_coassignment(C, k)

    ari_scores: List[float] = []
    for idx, run_labels in assignments:
        consensus_subset = labels[idx]
        if len(np.unique(run_labels)) < 2 or len(np.unique(consensus_subset)) < 2:
            ari_scores.append(0.0)
        else:
            ari_scores.append(adjusted_rand_score(consensus_subset, run_labels))

    # Confidence is mean co-assignment within the assigned cluster.
    confidence = _compute_confidence(C, labels)
    return ConsensusResult(labels=labels, coassignment=C, ari_scores=ari_scores, confidence=confidence)
