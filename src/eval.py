from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def stability_summary(
    ari_scores: List[float], labels: np.ndarray, confidence: np.ndarray
) -> pd.DataFrame:
    """Summarize stability metrics for a given consensus run."""
    # Cluster sizes drive minimum size constraints.
    counts = pd.Series(labels).value_counts()
    summary = {
        "mean_ari": float(np.mean(ari_scores)) if ari_scores else 0.0,
        "std_ari": float(np.std(ari_scores)) if ari_scores else 0.0,
        "min_cluster_size": int(counts.min()) if not counts.empty else 0,
        "mean_confidence": float(np.mean(confidence)) if len(confidence) else 0.0,
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
