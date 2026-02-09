from __future__ import annotations

import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata, spearmanr
from sklearn.metrics import pairwise_distances


# Minimum recommended cluster size for reliable within-peer statistics
MIN_CLUSTER_SIZE_WARNING = 3


def _percentiles(values: np.ndarray) -> np.ndarray:
    """Compute percentile ranks in [0, 100] for a vector."""
    n = len(values)
    if n == 1:
        return np.array([50.0])
    # Rank-based percentiles for small-N robustness.
    ranks = rankdata(values, method="average")
    return 100.0 * (ranks - 1) / (n - 1)


def _zscores(values: np.ndarray) -> np.ndarray:
    """Compute z-scores with safeguards for small groups."""
    n = len(values)
    if n < 3:
        return np.full(n, np.nan)
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    if std == 0:
        return np.full(n, np.nan)
    return (values - mean) / std


def _zscores_mad(values: np.ndarray) -> np.ndarray:
    """
    Compute robust z-scores using Median Absolute Deviation (MAD).

    MAD-based z-scores are more robust to outliers than standard z-scores,
    making them preferable for skewed healthcare data where extreme values
    are common.

    The MAD is scaled by 1.4826 to be consistent with standard deviation
    for normally distributed data.

    Parameters
    ----------
    values : np.ndarray
        Array of values to compute z-scores for.

    Returns
    -------
    np.ndarray
        MAD-based z-scores, or NaN array if computation not possible.
    """
    n = len(values)
    if n < 3:
        return np.full(n, np.nan)

    median = np.median(values)
    mad = np.median(np.abs(values - median))

    # Scale factor for consistency with standard deviation (normal distribution)
    mad_scaled = mad * 1.4826

    if mad_scaled == 0:
        return np.full(n, np.nan)

    return (values - median) / mad_scaled


def _outlier_flags(
    percentiles: np.ndarray, zscores: np.ndarray, cfg: Dict[str, Any]
) -> np.ndarray:
    """Flag outliers based on percentile and z-score thresholds."""
    low = cfg["benchmark"]["outlier_percentile_low"]
    high = cfg["benchmark"]["outlier_percentile_high"]
    zthr = cfg["benchmark"]["outlier_zscore_abs"]
    flags = (percentiles <= low) | (percentiles >= high)
    if zthr > 0:
        flags = flags | (np.abs(zscores) >= zthr)
    return flags


def _knn_fallback_benchmark(
    df: pd.DataFrame,
    X: np.ndarray,
    idx: int,
    kpi: str,
    cfg: Dict[str, Any],
    k: int = 5,
) -> Dict[str, Any]:
    """
    Compute benchmarking for a singleton institution using kNN fallback.

    For institutions in singleton peer groups (n=1), we use k-nearest neighbors
    to provide meaningful comparison context rather than defaulting to 50th percentile.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with all institutions.
    X : np.ndarray
        Feature matrix for distance computation.
    idx : int
        Index of the singleton institution.
    kpi : str
        KPI column name.
    cfg : Dict[str, Any]
        Configuration dictionary.
    k : int
        Number of nearest neighbors (default 5).

    Returns
    -------
    Dict[str, Any]
        Benchmark result dictionary with kNN-based percentile and z-score.
    """
    id_col = cfg["features"]["id"]
    k_eff = min(k, len(df) - 1)

    if k_eff <= 0:
        # No neighbors available, return default 50th percentile
        return {
            "institution_id": df.loc[idx, id_col],
            "kpi": kpi,
            "method": "peer",
            "peer_group": "singleton_knn_fallback",
            "peer_size": 1,
            "percentile": 50.0,
            "zscore": np.nan,
            "outlier": False,
        }

    # Compute distances to all other institutions
    distances = pairwise_distances(X[idx : idx + 1], X, metric="euclidean")[0]
    distances[idx] = np.inf  # Exclude self

    # Find k nearest neighbors
    neighbor_indices = np.argsort(distances)[:k_eff]
    peer_idx = np.concatenate([[idx], neighbor_indices])

    # Compute metrics within this kNN-defined peer group
    values = df.iloc[peer_idx][kpi].to_numpy()
    percentiles = _percentiles(values)
    zscores = _zscores(values)
    flags = _outlier_flags(percentiles, zscores, cfg)

    return {
        "institution_id": df.loc[idx, id_col],
        "kpi": kpi,
        "method": "peer",
        "peer_group": "singleton_knn_fallback",
        "peer_size": int(len(peer_idx)),
        "percentile": float(percentiles[0]),
        "zscore": float(zscores[0]) if not np.isnan(zscores[0]) else np.nan,
        "outlier": bool(flags[0]),
    }


def within_peer_benchmark(
    df: pd.DataFrame,
    labels: np.ndarray,
    cfg: Dict[str, Any],
    X: Optional[np.ndarray] = None,
    singleton_fallback: str = "default",
) -> pd.DataFrame:
    """
    Compute within-peer percentiles, z-scores, and outlier flags.

    Parameters
    ----------
    df : pd.DataFrame
        Institution-level data.
    labels : np.ndarray
        Peer group assignments.
    cfg : Dict[str, Any]
        Configuration dictionary.
    X : np.ndarray, optional
        Feature matrix for kNN fallback (required if singleton_fallback='knn').
    singleton_fallback : str
        Strategy for singleton clusters: 'default' (50th percentile) or 'knn'
        (use k-nearest neighbors for comparison).

    Returns
    -------
    pd.DataFrame
        Benchmark results with percentiles, z-scores, and outlier flags.
    """
    id_col = cfg["features"]["id"]
    kpis = cfg["targets"]["kpis"]
    output = []
    group = pd.Series(labels, index=df.index, name="peer_group")

    # Check for small clusters and warn
    cluster_sizes = group.value_counts()
    small_clusters = cluster_sizes[cluster_sizes < MIN_CLUSTER_SIZE_WARNING]
    if len(small_clusters) > 0:
        warnings.warn(
            f"Found {len(small_clusters)} peer group(s) with fewer than "
            f"{MIN_CLUSTER_SIZE_WARNING} members: {small_clusters.to_dict()}. "
            f"Within-peer statistics may be unreliable for these groups. "
            f"Consider using singleton_fallback='knn' or increasing minimum cluster size.",
            UserWarning,
        )

    # Compute metrics per KPI within each peer group.
    for kpi in kpis:
        for g, idx in group.groupby(group).groups.items():
            cluster_size = len(idx)

            # Handle singletons with fallback strategy
            if cluster_size == 1 and singleton_fallback == "knn":
                if X is None:
                    raise ValueError(
                        "Feature matrix X required for singleton_fallback='knn'"
                    )
                single_idx = idx[0]
                result = _knn_fallback_benchmark(df, X, single_idx, kpi, cfg)
                output.append(result)
                continue

            values = df.loc[idx, kpi].to_numpy()
            percentiles = _percentiles(values)
            zscores = _zscores(values)
            flags = _outlier_flags(percentiles, zscores, cfg)
            for row_idx, pct, z, flag in zip(idx, percentiles, zscores, flags):
                output.append(
                    {
                        "institution_id": df.loc[row_idx, id_col],
                        "kpi": kpi,
                        "method": "peer",
                        "peer_group": int(g),
                        "peer_size": int(cluster_size),
                        "percentile": float(pct),
                        "zscore": float(z) if not np.isnan(z) else np.nan,
                        "outlier": bool(flag),
                    }
                )
    return pd.DataFrame(output)


def global_benchmark(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Compute global benchmarking without peer groups."""
    id_col = cfg["features"]["id"]
    kpis = cfg["targets"]["kpis"]
    output = []
    # Treat all institutions as a single group.
    for kpi in kpis:
        values = df[kpi].to_numpy()
        percentiles = _percentiles(values)
        zscores = _zscores(values)
        flags = _outlier_flags(percentiles, zscores, cfg)
        for i, pct, z, flag in zip(df.index, percentiles, zscores, flags):
            output.append(
                {
                    "institution_id": df.loc[i, id_col],
                    "kpi": kpi,
                    "method": "global",
                    "peer_group": "global",
                    "peer_size": int(len(df)),
                    "percentile": float(pct),
                    "zscore": float(z) if not np.isnan(z) else np.nan,
                    "outlier": bool(flag),
                }
            )
    return pd.DataFrame(output)


def rule_based_benchmark(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Compute benchmarking within rule-based descriptor groups."""
    descriptors = cfg["targets"].get("descriptors", [])
    if not descriptors:
        return pd.DataFrame()
    id_col = cfg["features"]["id"]
    kpis = cfg["targets"]["kpis"]
    output = []
    # Group institutions by descriptor columns.
    for kpi in kpis:
        for group_vals, group_df in df.groupby(descriptors):
            values = group_df[kpi].to_numpy()
            percentiles = _percentiles(values)
            zscores = _zscores(values)
            flags = _outlier_flags(percentiles, zscores, cfg)
            for i, pct, z, flag in zip(group_df.index, percentiles, zscores, flags):
                output.append(
                    {
                        "institution_id": group_df.loc[i, id_col],
                        "kpi": kpi,
                        "method": "rule_based",
                        "peer_group": str(group_vals),
                        "peer_size": int(len(group_df)),
                        "percentile": float(pct),
                        "zscore": float(z) if not np.isnan(z) else np.nan,
                        "outlier": bool(flag),
                    }
                )
    return pd.DataFrame(output)


def knn_benchmark(
    df: pd.DataFrame,
    X: np.ndarray,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """Compute benchmarking within kNN-defined peer groups."""
    id_col = cfg["features"]["id"]
    kpis = cfg["targets"]["kpis"]
    k_list = cfg["benchmark"].get("knn_k", [3])
    # Use Euclidean distances in the chosen representation.
    distances = pairwise_distances(X, metric="euclidean")
    np.fill_diagonal(distances, np.inf)
    output = []
    for k in k_list:
        k_eff = min(k, len(df) - 1)
        if k_eff <= 0:
            continue
        # Pick k nearest neighbors for each institution.
        neighbors = np.argsort(distances, axis=1)[:, :k_eff]
        for idx in range(len(df)):
            peer_idx = np.concatenate([[idx], neighbors[idx]])
            for kpi in kpis:
                values = df.iloc[peer_idx][kpi].to_numpy()
                percentiles = _percentiles(values)
                zscores = _zscores(values)
                flags = _outlier_flags(percentiles, zscores, cfg)
                self_pos = 0
                output.append(
                    {
                        "institution_id": df.loc[idx, id_col],
                        "kpi": kpi,
                        "method": f"knn_k{k_eff}",
                        "peer_group": f"knn_k{k_eff}",
                        "peer_size": int(len(peer_idx)),
                        "percentile": float(percentiles[self_pos]),
                        "zscore": float(zscores[self_pos]) if not np.isnan(zscores[self_pos]) else np.nan,
                        "outlier": bool(flags[self_pos]),
                    }
                )
    return pd.DataFrame(output)


def combine_benchmarks(parts: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate non-empty benchmark tables."""
    frames = [p for p in parts if not p.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def multi_kpi_peer_agreement(
    df: pd.DataFrame,
    labels: np.ndarray,
    kpis: List[str],
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """
    Compute rank correlation between KPIs within each peer group.

    Measures how well peer group rankings generalize across multiple
    performance indicators. High correlation suggests the peer structure
    is valid for multi-dimensional benchmarking.

    Parameters
    ----------
    df : pd.DataFrame
        Institution-level data with KPI columns.
    labels : np.ndarray
        Peer group assignments from clustering.
    kpis : List[str]
        List of KPI column names to compare.
    cfg : Dict[str, Any]
        Configuration dictionary (for id column).

    Returns
    -------
    pd.DataFrame
        Results with columns:
        - peer_group: Cluster ID
        - peer_size: Number of institutions in group
        - kpi_1, kpi_2: KPI pair being compared
        - kendall_tau: Kendall's tau correlation coefficient
        - kendall_pvalue: P-value for the correlation
        - spearman_rho: Spearman's rho correlation coefficient
        - spearman_pvalue: P-value for Spearman correlation
        - rank_agreement_score: Fraction of concordant pairs

    Notes
    -----
    Kendall's tau is preferred for small samples (typical in healthcare
    benchmarking) as it has better statistical properties than Pearson's r.
    Values range from -1 (perfect disagreement) to +1 (perfect agreement).
    """
    id_col = cfg["features"]["id"]
    results = []
    group = pd.Series(labels, index=df.index, name="peer_group")

    # Generate all KPI pairs
    kpi_pairs = []
    for i, kpi1 in enumerate(kpis):
        for kpi2 in kpis[i + 1 :]:
            kpi_pairs.append((kpi1, kpi2))

    if not kpi_pairs:
        return pd.DataFrame()

    # Compute correlations within each peer group
    for g, idx in group.groupby(group).groups.items():
        peer_size = len(idx)

        for kpi1, kpi2 in kpi_pairs:
            values1 = df.loc[idx, kpi1].to_numpy()
            values2 = df.loc[idx, kpi2].to_numpy()

            # Need at least 3 observations for meaningful correlation
            if peer_size < 3:
                results.append(
                    {
                        "peer_group": int(g),
                        "peer_size": peer_size,
                        "kpi_1": kpi1,
                        "kpi_2": kpi2,
                        "kendall_tau": np.nan,
                        "kendall_pvalue": np.nan,
                        "spearman_rho": np.nan,
                        "spearman_pvalue": np.nan,
                        "rank_agreement_score": np.nan,
                    }
                )
                continue

            # Remove NaN values for correlation computation
            valid_mask = ~(np.isnan(values1) | np.isnan(values2))
            v1 = values1[valid_mask]
            v2 = values2[valid_mask]

            if len(v1) < 3:
                tau, tau_p = np.nan, np.nan
                rho, rho_p = np.nan, np.nan
                agreement = np.nan
            else:
                # Kendall's tau
                tau, tau_p = kendalltau(v1, v2)

                # Spearman's rho
                rho, rho_p = spearmanr(v1, v2)

                # Compute rank agreement score
                # Fraction of pairs where ranking order agrees
                ranks1 = rankdata(v1)
                ranks2 = rankdata(v2)
                n = len(v1)
                concordant = 0
                total_pairs = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        diff1 = ranks1[i] - ranks1[j]
                        diff2 = ranks2[i] - ranks2[j]
                        if diff1 * diff2 > 0:
                            concordant += 1
                        total_pairs += 1
                agreement = concordant / total_pairs if total_pairs > 0 else np.nan

            results.append(
                {
                    "peer_group": int(g),
                    "peer_size": peer_size,
                    "kpi_1": kpi1,
                    "kpi_2": kpi2,
                    "kendall_tau": float(tau) if not np.isnan(tau) else np.nan,
                    "kendall_pvalue": float(tau_p) if not np.isnan(tau_p) else np.nan,
                    "spearman_rho": float(rho) if not np.isnan(rho) else np.nan,
                    "spearman_pvalue": float(rho_p) if not np.isnan(rho_p) else np.nan,
                    "rank_agreement_score": float(agreement)
                    if not np.isnan(agreement)
                    else np.nan,
                }
            )

    return pd.DataFrame(results)


def global_multi_kpi_agreement(
    df: pd.DataFrame,
    kpis: List[str],
) -> pd.DataFrame:
    """
    Compute rank correlation between KPIs across all institutions.

    Baseline comparison to see if KPIs are inherently correlated
    before peer group adjustment.

    Parameters
    ----------
    df : pd.DataFrame
        Institution-level data with KPI columns.
    kpis : List[str]
        List of KPI column names to compare.

    Returns
    -------
    pd.DataFrame
        Results with Kendall's tau and Spearman's rho for each KPI pair.
    """
    results = []

    # Generate all KPI pairs
    for i, kpi1 in enumerate(kpis):
        for kpi2 in kpis[i + 1 :]:
            values1 = df[kpi1].to_numpy()
            values2 = df[kpi2].to_numpy()

            # Remove NaN values
            valid_mask = ~(np.isnan(values1) | np.isnan(values2))
            v1 = values1[valid_mask]
            v2 = values2[valid_mask]

            if len(v1) < 3:
                tau, tau_p = np.nan, np.nan
                rho, rho_p = np.nan, np.nan
            else:
                tau, tau_p = kendalltau(v1, v2)
                rho, rho_p = spearmanr(v1, v2)

            results.append(
                {
                    "method": "global",
                    "kpi_1": kpi1,
                    "kpi_2": kpi2,
                    "n_institutions": len(v1),
                    "kendall_tau": float(tau) if not np.isnan(tau) else np.nan,
                    "kendall_pvalue": float(tau_p) if not np.isnan(tau_p) else np.nan,
                    "spearman_rho": float(rho) if not np.isnan(rho) else np.nan,
                    "spearman_pvalue": float(rho_p) if not np.isnan(rho_p) else np.nan,
                }
            )

    return pd.DataFrame(results)
