"""
Publication-quality figures for consensus clustering benchmarking.

All figures are designed for scientific papers with:
- Clear axis labels with units
- Appropriate font sizes (readable at column width)
- Error bars / confidence intervals where applicable
- Consistent color scheme
- High DPI (300) for print quality
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Publication settings
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Consistent color palette (colorblind-friendly)
COLORS = {
    "primary": "#2E86AB",      # Blue
    "secondary": "#A23B72",    # Magenta
    "tertiary": "#F18F01",     # Orange
    "quaternary": "#C73E1D",   # Red
    "success": "#3A7D44",      # Green
    "neutral": "#6C757D",      # Gray
}

METHOD_COLORS = {
    "peer": COLORS["primary"],
    "global": COLORS["neutral"],
    "knn": COLORS["secondary"],
    "rule": COLORS["tertiary"],
}


def plot_coassignment_heatmap(
    C: np.ndarray,
    ids: List[str],
    path: str,
    cluster_labels: Optional[np.ndarray] = None,
    title: Optional[str] = None,
) -> None:
    """
    Plot co-assignment matrix heatmap showing peer cohesion structure.

    The co-assignment matrix C[i,j] shows the proportion of bootstrap
    perturbations where institutions i and j were assigned to the same cluster.
    High values (yellow) indicate stable peer relationships.

    Parameters
    ----------
    C : np.ndarray
        N×N co-assignment matrix with values in [0, 1]
    ids : List[str]
        Institution identifiers for axis labels
    path : str
        Output file path
    cluster_labels : Optional[np.ndarray]
        If provided, reorders matrix by cluster for block-diagonal structure
    title : Optional[str]
        Custom title (default: "Co-assignment Probability Matrix")
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    # Reorder by cluster if labels provided
    if cluster_labels is not None:
        order = np.argsort(cluster_labels)
        C = C[order][:, order]
        ids = [ids[i] for i in order]

    # Custom colormap: white -> blue -> yellow
    cmap = LinearSegmentedColormap.from_list(
        "coassign", ["#FFFFFF", "#2E86AB", "#F4D35E"]
    )

    im = ax.imshow(C, cmap=cmap, vmin=0, vmax=1, aspect="equal")

    # Colorbar with label
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Co-assignment probability", fontsize=10)

    # Axis labels (show subset if too many)
    n = len(ids)
    if n <= 30:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(ids, rotation=90, fontsize=7)
        ax.set_yticklabels(ids, fontsize=7)
    else:
        # Show every nth label
        step = max(1, n // 10)
        ax.set_xticks(range(0, n, step))
        ax.set_yticks(range(0, n, step))
        ax.set_xticklabels([ids[i] for i in range(0, n, step)], rotation=90, fontsize=7)
        ax.set_yticklabels([ids[i] for i in range(0, n, step)], fontsize=7)

    ax.set_xlabel("Institution")
    ax.set_ylabel("Institution")
    ax.set_title(title or "Co-assignment Probability Matrix")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_stability_vs_k(
    k_summary: pd.DataFrame,
    path: str,
    show_confidence: bool = True,
    threshold: Optional[float] = 0.8,
) -> None:
    """
    Plot clustering stability (ARI) across different K values with confidence intervals.

    This figure supports K selection by showing where stability plateaus.
    The dashed line indicates the stability threshold (default: ARI = 0.8).

    Parameters
    ----------
    k_summary : pd.DataFrame
        Must contain columns: k, mean_ari, std_ari (optional), mean_confidence (optional)
    path : str
        Output file path
    show_confidence : bool
        If True, also plots cluster confidence on secondary y-axis
    threshold : Optional[float]
        If provided, draws horizontal threshold line
    """
    fig, ax1 = plt.subplots(figsize=(6, 4))

    k_vals = k_summary["k"].values
    ari_vals = k_summary["mean_ari"].values

    # Plot ARI with error bars if std available
    if "std_ari" in k_summary.columns:
        std_vals = k_summary["std_ari"].values
        ax1.errorbar(
            k_vals, ari_vals, yerr=std_vals,
            marker="o", color=COLORS["primary"],
            capsize=4, capthick=1.5, linewidth=2,
            label="Mean ARI ± SD"
        )
    else:
        ax1.plot(
            k_vals, ari_vals,
            marker="o", color=COLORS["primary"],
            linewidth=2, markersize=8,
            label="Mean ARI"
        )

    ax1.set_xlabel("Number of clusters (K)")
    ax1.set_ylabel("Adjusted Rand Index (ARI)", color=COLORS["primary"])
    ax1.tick_params(axis="y", labelcolor=COLORS["primary"])
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(k_vals)

    # Threshold line
    if threshold:
        ax1.axhline(
            y=threshold, color=COLORS["neutral"],
            linestyle="--", linewidth=1, alpha=0.7,
            label=f"Stability threshold ({threshold})"
        )

    # Secondary axis for cluster confidence
    if show_confidence and "mean_confidence" in k_summary.columns:
        ax2 = ax1.twinx()
        conf_vals = k_summary["mean_confidence"].values
        ax2.plot(
            k_vals, conf_vals,
            marker="s", color=COLORS["secondary"],
            linewidth=2, markersize=7, linestyle="--",
            label="Mean confidence"
        )
        ax2.set_ylabel("Cluster confidence", color=COLORS["secondary"])
        ax2.tick_params(axis="y", labelcolor=COLORS["secondary"])
        ax2.set_ylim(0, 1.05)

    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.set_title("Clustering Stability vs. Number of Clusters")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    if show_confidence and "mean_confidence" in k_summary.columns:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", framealpha=0.9)
    else:
        ax1.legend(loc="lower right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_benchmark_percentiles(
    benchmark_df: pd.DataFrame,
    kpi: str,
    path: str,
    title: Optional[str] = None,
) -> None:
    """
    Compare within-peer vs. global vs. kNN percentile rankings.

    This figure shows how each institution's KPI is ranked under different
    benchmarking methods. Divergence between methods indicates where
    peer-group context matters.

    Parameters
    ----------
    benchmark_df : pd.DataFrame
        Must contain columns: kpi, method, institution_id, percentile
    kpi : str
        KPI name to filter on
    path : str
        Output file path
    """
    subset = benchmark_df[benchmark_df["kpi"] == kpi].copy()
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    methods = subset["method"].unique().tolist()
    n_institutions = subset["institution_id"].nunique()

    # Sort institutions by peer percentile for consistent ordering
    peer_subset = subset[subset["method"] == "peer"] if "peer" in methods else subset
    inst_order = peer_subset.sort_values("percentile")["institution_id"].tolist()

    x_positions = np.arange(n_institutions)
    width = 0.8 / len(methods)

    for idx, method in enumerate(methods):
        method_df = subset[subset["method"] == method].copy()
        # Reorder to match peer ordering
        method_df = method_df.set_index("institution_id").loc[inst_order].reset_index()

        color = METHOD_COLORS.get(method, COLORS["neutral"])
        offset = (idx - len(methods) / 2 + 0.5) * width

        ax.bar(
            x_positions + offset,
            method_df["percentile"],
            width=width * 0.9,
            label=method.replace("_", " ").title(),
            color=color,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_ylabel("Percentile rank")
    ax.set_xlabel("Institution (sorted by peer percentile)")
    ax.set_title(title or f"Benchmark Comparison: {kpi.replace('_', ' ').title()}")
    ax.set_ylim(0, 105)
    ax.set_xlim(-0.5, n_institutions - 0.5)

    # Reference lines
    ax.axhline(y=50, color=COLORS["neutral"], linestyle=":", linewidth=1, alpha=0.5)
    ax.axhline(y=25, color=COLORS["quaternary"], linestyle=":", linewidth=1, alpha=0.3)
    ax.axhline(y=75, color=COLORS["success"], linestyle=":", linewidth=1, alpha=0.3)

    # Simplify x-axis if many institutions
    if n_institutions > 20:
        ax.set_xticks([])
    else:
        ax.set_xticks(x_positions)
        ax.set_xticklabels(range(1, n_institutions + 1), fontsize=7)

    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_perturbation_detection_curve(
    perturb_df: pd.DataFrame,
    kpi: str,
    path: str,
    title: Optional[str] = None,
) -> None:
    """
    Plot sensitivity analysis: detection rate vs. perturbation magnitude.

    This figure shows how well the benchmarking method detects institutions
    with artificially shifted KPIs. Higher recall at lower magnitudes
    indicates better sensitivity.

    Parameters
    ----------
    perturb_df : pd.DataFrame
        Must contain columns: kpi, magnitude, recall, false_positive_rate
    kpi : str
        KPI name to filter on
    path : str
        Output file path
    """
    subset = perturb_df[perturb_df["kpi"] == kpi].copy()
    if subset.empty:
        return

    # Aggregate Monte Carlo runs by magnitude
    grouped = subset.groupby("magnitude").agg(
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        fpr_mean=("false_positive_rate", "mean"),
        fpr_std=("false_positive_rate", "std"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(6, 4.5))

    magnitudes = grouped["magnitude"].values

    # Recall (sensitivity) with error bars
    ax.errorbar(
        magnitudes, grouped["recall_mean"],
        yerr=grouped["recall_std"],
        marker="o", color=COLORS["primary"],
        capsize=4, capthick=1.5, linewidth=2,
        label="Recall (sensitivity)"
    )

    # False positive rate with error bars
    ax.errorbar(
        magnitudes, grouped["fpr_mean"],
        yerr=grouped["fpr_std"],
        marker="s", color=COLORS["quaternary"],
        capsize=4, capthick=1.5, linewidth=2,
        label="False positive rate"
    )

    ax.set_xlabel("Perturbation magnitude (σ)")
    ax.set_ylabel("Rate")
    ax.set_title(title or f"Detection Sensitivity: {kpi.replace('_', ' ').title()}")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(min(magnitudes) - 0.05, max(magnitudes) + 0.05)

    ax.legend(loc="center right", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.3)

    # Annotate ideal region
    ax.fill_between(
        magnitudes, 0, grouped["fpr_mean"],
        alpha=0.1, color=COLORS["quaternary"]
    )
    ax.fill_between(
        magnitudes, grouped["recall_mean"], 1,
        alpha=0.1, color=COLORS["primary"]
    )

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_method_comparison_summary(
    benchmark_df: pd.DataFrame,
    path: str,
    metric: str = "percentile",
) -> None:
    """
    Summary comparison of all benchmarking methods across KPIs.

    Shows distribution statistics (median, IQR) for each method,
    helping identify systematic differences in ranking behavior.

    Parameters
    ----------
    benchmark_df : pd.DataFrame
        Full benchmark results
    path : str
        Output file path
    metric : str
        Column to summarize (default: "percentile")
    """
    if benchmark_df.empty:
        return

    # Compute summary statistics per method
    summary = benchmark_df.groupby("method")[metric].agg(
        ["median", "mean", "std", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
    )
    summary.columns = ["median", "mean", "std", "q25", "q75"]
    summary = summary.reset_index()

    fig, ax = plt.subplots(figsize=(6, 4))

    x = np.arange(len(summary))
    colors = [METHOD_COLORS.get(m, COLORS["neutral"]) for m in summary["method"]]

    # Bar plot with error bars showing IQR
    bars = ax.bar(x, summary["median"], color=colors, alpha=0.8, edgecolor="white")
    ax.errorbar(
        x, summary["median"],
        yerr=[summary["median"] - summary["q25"], summary["q75"] - summary["median"]],
        fmt="none", color="black", capsize=5, capthick=1.5
    )

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in summary["method"]])
    ax.set_ylabel(f"Median {metric}")
    ax.set_title("Benchmarking Method Comparison")
    ax.set_ylim(0, 100 if metric == "percentile" else None)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_cluster_characteristics(
    data_df: pd.DataFrame,
    cluster_col: str,
    feature_cols: List[str],
    path: str,
    title: Optional[str] = None,
) -> None:
    """
    Radar/parallel coordinates showing cluster profiles.

    Helps interpret what each peer group represents in terms
    of institutional characteristics.

    Parameters
    ----------
    data_df : pd.DataFrame
        Institution data with cluster assignments
    cluster_col : str
        Column containing cluster labels
    feature_cols : List[str]
        Features to include in profile
    path : str
        Output file path
    """
    if cluster_col not in data_df.columns:
        return

    # Normalize features to [0, 1] for comparison
    normalized = data_df.copy()
    for col in feature_cols:
        if col in normalized.columns:
            col_min = normalized[col].min()
            col_max = normalized[col].max()
            if col_max > col_min:
                normalized[col] = (normalized[col] - col_min) / (col_max - col_min)

    # Compute cluster centroids
    centroids = normalized.groupby(cluster_col)[feature_cols].mean()

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(feature_cols))
    cluster_colors = [COLORS["primary"], COLORS["secondary"], COLORS["tertiary"],
                      COLORS["quaternary"], COLORS["success"], COLORS["neutral"]]

    for idx, (cluster, row) in enumerate(centroids.iterrows()):
        color = cluster_colors[idx % len(cluster_colors)]
        ax.plot(x, row.values, marker="o", label=f"Cluster {cluster}",
                color=color, linewidth=2, markersize=6)

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in feature_cols], fontsize=8)
    ax.set_ylabel("Normalized value")
    ax.set_title(title or "Cluster Characteristic Profiles")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_ylim(-0.1, 1.1)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def generate_all_figures(
    results: Dict,
    output_dir: str,
    kpis: List[str],
) -> List[str]:
    """
    Generate all publication figures from experiment results.

    Parameters
    ----------
    results : Dict
        Dictionary containing all experiment outputs
    output_dir : str
        Directory to save figures
    kpis : List[str]
        List of KPI names

    Returns
    -------
    List[str]
        Paths to generated figures
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    generated = []

    # Co-assignment heatmap
    if "coassignment" in results and "institution_ids" in results:
        path = str(output_path / "fig_coassignment_heatmap.png")
        plot_coassignment_heatmap(
            results["coassignment"],
            results["institution_ids"],
            path,
            cluster_labels=results.get("cluster_labels"),
        )
        generated.append(path)

    # Stability vs K
    if "k_summary" in results:
        path = str(output_path / "fig_stability_vs_k.png")
        plot_stability_vs_k(results["k_summary"], path)
        generated.append(path)

    # Benchmark percentiles (per KPI)
    if "benchmark_df" in results:
        for kpi in kpis:
            path = str(output_path / f"fig_benchmark_{kpi}.png")
            plot_benchmark_percentiles(results["benchmark_df"], kpi, path)
            generated.append(path)

        # Method comparison summary
        path = str(output_path / "fig_method_comparison.png")
        plot_method_comparison_summary(results["benchmark_df"], path)
        generated.append(path)

    # Perturbation detection (per KPI)
    if "perturb_df" in results:
        for kpi in kpis:
            path = str(output_path / f"fig_detection_{kpi}.png")
            plot_perturbation_detection_curve(results["perturb_df"], kpi, path)
            generated.append(path)

    return generated
