"""
Detection Analysis Module (P2.1)

Functions for analyzing detection sensitivity, including:
- Minimum N computation for percentile-based detection
- Z-score-only outlier detection for small N
- Fixed peer assignment detection (no re-clustering)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.benchmark import within_peer_benchmark


def compute_min_n_for_percentile_threshold(
    percentile_threshold: float,
    confidence_level: float = 0.95,
) -> int:
    """
    Compute the minimum N required to reliably detect outliers at a given percentile.
    
    For percentile-based detection, with N institutions, the finest granularity for
    percentiles is 100/(N-1). To reliably flag an institution at the `percentile_threshold`
    (e.g., 5th or 95th percentile), N must be large enough that the percentile bin
    width is smaller than the threshold distance from the median.
    
    Parameters
    ----------
    percentile_threshold : float
        Target percentile threshold (e.g., 5.0 for 5th percentile, 95.0 for 95th).
    confidence_level : float
        Desired confidence level for detection (default 0.95).
        
    Returns
    -------
    min_n : int
        Minimum number of institutions required.
        
    Examples
    --------
    >>> compute_min_n_for_percentile_threshold(5.0)  # 5th percentile
    21
    >>> compute_min_n_for_percentile_threshold(10.0)  # 10th percentile  
    11
    """
    # Distance from median (50) to threshold
    threshold_distance = min(percentile_threshold, 100 - percentile_threshold)
    
    if threshold_distance <= 0:
        return 2  # Edge case
    
    # Percentile granularity: 100 / (N-1)
    # To distinguish threshold from median: granularity <= threshold_distance
    # => N-1 >= 100 / threshold_distance
    # => N >= 100 / threshold_distance + 1
    min_n_percentile = int(np.ceil(100.0 / threshold_distance)) + 1
    
    # Statistical power adjustment: with binomial distribution,
    # probability of observing an outlier at extreme percentile requires sufficient N
    # For 95% confidence that we can distinguish the extreme position:
    # Using inverse binomial, we need N such that P(rank=1) is distinguishable
    min_n_statistical = int(np.ceil(np.log(1 - confidence_level) / np.log(0.5))) + 2
    
    return max(min_n_percentile, min_n_statistical, 3)


def compute_min_n_table(
    percentile_thresholds: List[float] = [5.0, 10.0, 15.0, 20.0, 25.0],
    confidence_levels: List[float] = [0.90, 0.95, 0.99],
) -> pd.DataFrame:
    """
    Generate a table showing minimum N for various percentile thresholds.
    
    Parameters
    ----------
    percentile_thresholds : list of float
        Percentile values to analyze (e.g., [5, 10, 15, 20]).
    confidence_levels : list of float
        Confidence levels to consider.
        
    Returns
    -------
    df : pd.DataFrame
        Table with min_n for each threshold/confidence combination.
    """
    rows = []
    for pct in percentile_thresholds:
        row = {"percentile_threshold": pct}
        for conf in confidence_levels:
            min_n = compute_min_n_for_percentile_threshold(pct, conf)
            row[f"min_n_{int(conf*100)}%_conf"] = min_n
        rows.append(row)
    return pd.DataFrame(rows)


def zscore_only_outlier_flags(
    values: np.ndarray,
    zscore_threshold: float = 2.0,
    min_group_size: int = 3,
) -> np.ndarray:
    """
    Flag outliers using z-score criterion only (for small N where percentiles fail).
    
    For peer groups with N < ~15, percentile-based detection has poor granularity.
    Z-score detection can identify outliers with fewer institutions, though it
    requires normally-distributed KPIs and is sensitive to small-sample variance.
    
    Parameters
    ----------
    values : np.ndarray
        KPI values for the peer group.
    zscore_threshold : float
        Absolute z-score threshold for outlier flagging (default 2.0).
    min_group_size : int
        Minimum group size for valid z-score computation (default 3).
        
    Returns
    -------
    flags : np.ndarray
        Boolean array indicating outliers.
    """
    n = len(values)
    
    if n < min_group_size:
        return np.zeros(n, dtype=bool)
    
    mean = np.mean(values)
    std = np.std(values, ddof=1)  # Sample std
    
    if std == 0 or np.isnan(std):
        return np.zeros(n, dtype=bool)
    
    zscores = (values - mean) / std
    
    return np.abs(zscores) >= zscore_threshold


def evaluate_zscore_detection_small_n(
    df: pd.DataFrame,
    labels: np.ndarray,
    cfg: Dict[str, Any],
    zscore_thresholds: List[float] = [1.5, 2.0, 2.5, 3.0],
) -> pd.DataFrame:
    """
    Evaluate z-score-only detection across different thresholds for small N.
    
    Compares detection rates using z-score vs percentile criteria,
    particularly useful for assessing detection viability in small peer networks.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with institutions.
    labels : np.ndarray
        Peer group assignments.
    cfg : dict
        Pipeline configuration.
    zscore_thresholds : list of float
        Z-score thresholds to evaluate.
        
    Returns
    -------
    results : pd.DataFrame
        Comparison of detection methods across thresholds.
    """
    id_col = cfg["features"]["id"]
    kpis = cfg["targets"]["kpis"]
    
    # Get standard peer benchmark results
    peer_bench = within_peer_benchmark(df, labels, cfg)
    
    results = []
    for kpi in kpis:
        kpi_bench = peer_bench[peer_bench["kpi"] == kpi]
        
        for peer_group in kpi_bench["peer_group"].unique():
            group_data = kpi_bench[kpi_bench["peer_group"] == peer_group]
            n = len(group_data)
            
            # Get KPI values for this group
            group_ids = group_data["institution_id"].tolist()
            mask = df[id_col].isin(group_ids)
            values = df.loc[mask, kpi].values
            
            # Percentile-based outliers (from within_peer_benchmark)
            pct_outliers = group_data["outlier"].sum()
            
            for z_thresh in zscore_thresholds:
                z_flags = zscore_only_outlier_flags(values, z_thresh)
                z_outliers = z_flags.sum()
                
                results.append({
                    "kpi": kpi,
                    "peer_group": peer_group,
                    "n": n,
                    "zscore_threshold": z_thresh,
                    "zscore_outliers": int(z_outliers),
                    "percentile_outliers": int(pct_outliers),
                    "zscore_rate": z_outliers / n if n > 0 else 0,
                    "percentile_rate": pct_outliers / n if n > 0 else 0,
                })
    
    return pd.DataFrame(results)


def run_fixed_peer_detection(
    df: pd.DataFrame,
    fixed_labels: np.ndarray,
    cfg: Dict[str, Any],
    perturbed_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Run detection using fixed peer assignments (no re-clustering).
    
    This mode evaluates detection performance when peer groups are held constant,
    isolating the effect of KPI perturbations from clustering instability.
    
    Parameters
    ----------
    df : pd.DataFrame
        Original dataset.
    fixed_labels : np.ndarray
        Pre-computed peer group assignments to use.
    cfg : dict
        Pipeline configuration.
    perturbed_df : pd.DataFrame, optional
        Perturbed dataset. If None, uses original df.
        
    Returns
    -------
    benchmark_df : pd.DataFrame
        Benchmark results using fixed peer assignments.
    """
    target_df = perturbed_df if perturbed_df is not None else df
    return within_peer_benchmark(target_df, fixed_labels, cfg)


def run_fixed_peer_perturbation_eval(
    df: pd.DataFrame,
    fixed_labels: np.ndarray,
    cfg: Dict[str, Any],
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run perturbation evaluation with fixed peer assignments.
    
    Unlike the standard perturbation eval that re-clusters after perturbation,
    this mode keeps peer groups fixed, measuring pure detection sensitivity
    without clustering instability confounds.
    
    Parameters
    ----------
    df : pd.DataFrame
        Original dataset.
    fixed_labels : np.ndarray
        Fixed peer group assignments.
    cfg : dict
        Pipeline configuration.
    seed : int
        Random seed.
        
    Returns
    -------
    results : pd.DataFrame
        Detection metrics under fixed-peer mode.
    """
    from src.perturbation import apply_kpi_shift
    
    pert_cfg = cfg["perturbation"]
    if not pert_cfg.get("enabled", False):
        return pd.DataFrame()
    
    n_runs = pert_cfg["n_runs"]
    id_col = cfg["features"]["id"]
    kpis = pert_cfg.get("kpis", cfg["targets"]["kpis"])
    shift_type = pert_cfg["shift"]["type"]
    magnitudes = pert_cfg["shift"].get("magnitudes")
    mag_min = pert_cfg["shift"].get("magnitude_min", 0.0)
    mag_max = pert_cfg["shift"].get("magnitude_max", 0.0)
    
    # Baseline benchmarks with fixed labels
    baseline_bench = within_peer_benchmark(df, fixed_labels, cfg)
    baseline_lookup = {
        (row["institution_id"], row["kpi"]): (row["percentile"], row["peer_size"])
        for _, row in baseline_bench.iterrows()
    }
    
    output = []
    for run_id in range(n_runs):
        rng = np.random.default_rng(seed + run_id)
        
        # Sample institutions
        if pert_cfg.get("institutions"):
            institution_ids = pert_cfg["institutions"]
        else:
            n_inst = min(pert_cfg.get("n_institutions", 1), len(df))
            institution_ids = rng.choice(
                df[id_col].tolist(), size=n_inst, replace=False
            ).tolist()
        
        # Sample magnitude
        if magnitudes:
            magnitude = float(rng.choice(magnitudes))
        else:
            magnitude = float(rng.uniform(mag_min, mag_max))
        
        # Apply perturbation
        perturbed = apply_kpi_shift(
            df, institution_ids, kpis, shift_type, magnitude, id_col
        )
        
        # Run benchmark with FIXED labels (no re-clustering)
        perturbed_bench = run_fixed_peer_detection(
            df, fixed_labels, cfg, perturbed_df=perturbed
        )
        
        # Compute detection metrics
        non_perturbed_ids = set(df[id_col]) - set(institution_ids)
        
        for kpi in kpis:
            kpi_bench = perturbed_bench[perturbed_bench["kpi"] == kpi]
            pert_rows = kpi_bench[kpi_bench["institution_id"].isin(institution_ids)]
            non_pert_rows = kpi_bench[kpi_bench["institution_id"].isin(non_perturbed_ids)]
            
            fpr = float(non_pert_rows["outlier"].mean()) if len(non_pert_rows) > 0 else 0.0
            
            for _, row in pert_rows.iterrows():
                base_pct, base_n = baseline_lookup.get(
                    (row["institution_id"], row["kpi"]), (np.nan, 0)
                )
                pct_shift = abs(float(row["percentile"]) - base_pct) if not np.isnan(base_pct) else np.nan
                
                output.append({
                    "run_id": run_id,
                    "kpi": kpi,
                    "institution_id": row["institution_id"],
                    "shift_type": shift_type,
                    "magnitude": magnitude,
                    "mode": "fixed_peer",
                    "recall": float(bool(row["outlier"])),
                    "false_positive_rate": fpr,
                    "percentile_shift": pct_shift,
                })
    
    return pd.DataFrame(output)


def compare_detection_modes(
    standard_results: pd.DataFrame,
    fixed_peer_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare detection performance between standard and fixed-peer modes.
    
    Parameters
    ----------
    standard_results : pd.DataFrame
        Results from standard perturbation evaluation (with re-clustering).
    fixed_peer_results : pd.DataFrame
        Results from fixed-peer evaluation.
        
    Returns
    -------
    comparison : pd.DataFrame
        Side-by-side comparison of metrics.
    """
    def summarize(df: pd.DataFrame, mode: str) -> Dict:
        if df.empty:
            return {"mode": mode, "mean_recall": 0, "mean_fpr": 0, "n_runs": 0}
        return {
            "mode": mode,
            "mean_recall": df["recall"].mean(),
            "mean_fpr": df["false_positive_rate"].mean(),
            "std_recall": df["recall"].std(),
            "n_runs": len(df),
        }
    
    return pd.DataFrame([
        summarize(standard_results, "standard (re-cluster)"),
        summarize(fixed_peer_results, "fixed_peer"),
    ])
