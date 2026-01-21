from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import pairwise_distances


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


def within_peer_benchmark(
    df: pd.DataFrame,
    labels: np.ndarray,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """Compute within-peer percentiles, z-scores, and outlier flags."""
    id_col = cfg["features"]["id"]
    kpis = cfg["targets"]["kpis"]
    output = []
    group = pd.Series(labels, index=df.index, name="peer_group")
    # Compute metrics per KPI within each peer group.
    for kpi in kpis:
        for g, idx in group.groupby(group).groups.items():
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
                        "peer_size": int(len(idx)),
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
