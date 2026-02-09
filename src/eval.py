from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import silhouette_score


def compute_silhouette(
    X: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], labels: np.ndarray
) -> float:
    """Compute the silhouette score for a clustering assignment."""
    if np.unique(labels).size < 2:
        return 0.0
    if isinstance(X, tuple):
        X = np.hstack(X)
    if X.size == 0 or X.shape[0] < 2:
        return 0.0
    return float(silhouette_score(X, labels))


def stability_summary(
    ari_scores: List[float],
    labels: np.ndarray,
    confidence: np.ndarray,
    X: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Summarize stability metrics for a given consensus run."""
    # Cluster sizes drive minimum size constraints.
    counts = pd.Series(labels).value_counts()
    summary = {
        "mean_ari": float(np.mean(ari_scores)) if ari_scores else 0.0,
        "std_ari": float(np.std(ari_scores)) if ari_scores else 0.0,
        "min_cluster_size": int(counts.min()) if not counts.empty else 0,
        "mean_confidence": float(np.mean(confidence)) if len(confidence) else 0.0,
        "silhouette": compute_silhouette(X, labels),
    }
    return pd.DataFrame([summary])


def utility_summary(perturb_df: pd.DataFrame) -> Dict[str, float]:
    """Summarize perturbation detection utility metrics."""
    if perturb_df.empty:
        return {"mean_recall": 0.0, "mean_false_positive_rate": 0.0}
    # Aggregate detection metrics over Monte Carlo runs.
    return {
        "mean_recall": float(perturb_df["recall"].mean()),
        "mean_false_positive_rate": float(perturb_df["false_positive_rate"].mean()),
    }


def bootstrap_ci(
    data: np.ndarray,
    statistic: str = "mean",
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.

    Parameters
    ----------
    data : np.ndarray
        1D array of values to compute CI for.
    statistic : str
        Statistic to compute: 'mean', 'median', or 'std'.
    confidence : float
        Confidence level (default 0.95 for 95% CI).
    n_bootstrap : int
        Number of bootstrap resamples.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Tuple[float, float, float]
        (point_estimate, ci_lower, ci_upper)
    """
    if len(data) == 0:
        return (np.nan, np.nan, np.nan)

    rng = np.random.default_rng(seed)
    n = len(data)

    # Compute point estimate
    if statistic == "mean":
        point_est = float(np.mean(data))
        stat_func = np.mean
    elif statistic == "median":
        point_est = float(np.median(data))
        stat_func = np.median
    elif statistic == "std":
        point_est = float(np.std(data, ddof=1))
        stat_func = lambda x: np.std(x, ddof=1)
    else:
        raise ValueError(f"Unknown statistic: {statistic}")

    # Bootstrap resampling
    bootstrap_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        resample = rng.choice(data, size=n, replace=True)
        bootstrap_stats[i] = stat_func(resample)

    # Percentile method for CI
    alpha = 1 - confidence
    ci_lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))

    return (point_est, ci_lower, ci_upper)


def paired_wilcoxon_test(
    x: np.ndarray,
    y: np.ndarray,
    alternative: str = "two-sided",
) -> Dict[str, float]:
    """
    Perform Wilcoxon signed-rank test for paired samples.

    Parameters
    ----------
    x : np.ndarray
        First set of measurements.
    y : np.ndarray
        Second set of measurements (paired with x).
    alternative : str
        Alternative hypothesis: 'two-sided', 'greater', or 'less'.

    Returns
    -------
    Dict[str, float]
        Dictionary with 'statistic', 'p_value', and 'effect_size' (r = Z / sqrt(N)).
    """
    if len(x) != len(y):
        raise ValueError("Arrays must have same length for paired test")

    if len(x) < 5:
        return {"statistic": np.nan, "p_value": np.nan, "effect_size": np.nan}

    # Remove pairs where difference is zero
    diff = x - y
    nonzero_mask = diff != 0
    if nonzero_mask.sum() < 5:
        return {"statistic": np.nan, "p_value": np.nan, "effect_size": np.nan}

    result = stats.wilcoxon(x, y, alternative=alternative, zero_method="wilcox")

    # Effect size: r = Z / sqrt(N) where Z is standardized statistic
    n = nonzero_mask.sum()
    # Approximate Z from p-value for effect size calculation
    if result.pvalue < 1.0:
        z = stats.norm.ppf(1 - result.pvalue / 2) if alternative == "two-sided" else stats.norm.ppf(1 - result.pvalue)
        effect_size = abs(z) / np.sqrt(n)
    else:
        effect_size = 0.0

    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "effect_size": float(effect_size),
    }


def compare_methods_significance(
    benchmark_df: pd.DataFrame,
    method_a: str,
    method_b: str,
    metric: str = "percentile",
) -> Dict[str, float]:
    """
    Compare two benchmarking methods using paired statistical tests.

    Parameters
    ----------
    benchmark_df : pd.DataFrame
        Benchmark results with columns: institution_id, method, percentile, etc.
    method_a : str
        Name of first method (e.g., 'peer').
    method_b : str
        Name of second method (e.g., 'global').
    metric : str
        Metric to compare (e.g., 'percentile', 'zscore').

    Returns
    -------
    Dict[str, float]
        Test results including mean difference, CI, and p-value.
    """
    df_a = benchmark_df[benchmark_df["method"] == method_a].set_index("institution_id")
    df_b = benchmark_df[benchmark_df["method"] == method_b].set_index("institution_id")

    # Align on common institutions
    common_ids = df_a.index.intersection(df_b.index)
    if len(common_ids) < 5:
        return {
            "mean_diff": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "p_value": np.nan,
            "effect_size": np.nan,
            "n_pairs": len(common_ids),
        }

    values_a = df_a.loc[common_ids, metric].to_numpy()
    values_b = df_b.loc[common_ids, metric].to_numpy()

    # Mean difference with bootstrap CI
    diff = values_a - values_b
    mean_diff, ci_lower, ci_upper = bootstrap_ci(diff, statistic="mean", seed=42)

    # Wilcoxon test
    test_result = paired_wilcoxon_test(values_a, values_b)

    return {
        "mean_diff": mean_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": test_result["p_value"],
        "effect_size": test_result["effect_size"],
        "n_pairs": len(common_ids),
    }


def stability_summary_with_ci(
    ari_scores: List[float],
    labels: np.ndarray,
    confidence: np.ndarray,
    X: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    ci_confidence: float = 0.95,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Summarize stability metrics with bootstrap confidence intervals.

    Parameters
    ----------
    ari_scores : List[float]
        ARI scores from bootstrap runs.
    labels : np.ndarray
        Final consensus cluster labels.
    confidence : np.ndarray
        Per-institution confidence scores.
    X : array or tuple of arrays
        Feature matrix for silhouette computation.
    ci_confidence : float
        Confidence level for CIs (default 0.95).
    seed : int, optional
        Random seed for bootstrap CI computation.

    Returns
    -------
    pd.DataFrame
        Summary with point estimates and confidence intervals.
    """
    counts = pd.Series(labels).value_counts()

    # ARI with CI
    ari_array = np.array(ari_scores) if ari_scores else np.array([0.0])
    ari_mean, ari_ci_low, ari_ci_high = bootstrap_ci(
        ari_array, statistic="mean", confidence=ci_confidence, seed=seed
    )

    # Confidence with CI
    conf_mean, conf_ci_low, conf_ci_high = bootstrap_ci(
        confidence, statistic="mean", confidence=ci_confidence, seed=seed
    ) if len(confidence) > 0 else (0.0, 0.0, 0.0)

    summary = {
        "mean_ari": ari_mean,
        "ari_ci_lower": ari_ci_low,
        "ari_ci_upper": ari_ci_high,
        "std_ari": float(np.std(ari_scores)) if ari_scores else 0.0,
        "min_cluster_size": int(counts.min()) if not counts.empty else 0,
        "mean_confidence": conf_mean,
        "confidence_ci_lower": conf_ci_low,
        "confidence_ci_upper": conf_ci_high,
        "silhouette": compute_silhouette(X, labels),
    }
    return pd.DataFrame([summary])
