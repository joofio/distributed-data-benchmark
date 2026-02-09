#!/usr/bin/env python3
"""
Multi-KPI agreement analysis for peer group validation.

This script evaluates whether peer groups constructed for one KPI
(e.g., cesarean rate) are also valid for benchmarking other KPIs
(e.g., outcome variability). Uses rank correlation (Kendall's tau)
to measure agreement between KPI rankings within peer groups.

Addresses Major Reviewer Point #2: Validation limited to single KPI.

Outputs:
- results/obscare/multi_kpi_agreement.csv: Correlation metrics per peer group
- results/obscare/figures/fig_multi_kpi_scatter.png: Visualization
"""
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

from src.benchmark import global_multi_kpi_agreement, multi_kpi_peer_agreement
from src.consensus import run_consensus
from src.data import load_dataset
from src.plots import plot_multi_kpi_scatter
from src.preprocess import prepare_representations


def run_multi_kpi_analysis(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    k: int,
    seed: int,
    method: str = "gower_pam",
) -> Dict[str, Any]:
    """Run multi-KPI agreement analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Institution-level data.
    cfg : Dict[str, Any]
        Configuration dictionary.
    k : int
        Number of clusters.
    seed : int
        Random seed.
    method : str
        Clustering method to use.

    Returns
    -------
    Dict[str, Any]
        Results containing peer and global agreement metrics.
    """
    # Get KPIs from config
    kpis = cfg["targets"]["kpis"]
    if len(kpis) < 2:
        raise ValueError("Need at least 2 KPIs for multi-KPI analysis")

    # Prepare representations
    reps = prepare_representations(df, cfg)

    # Select representation based on method
    if method == "mixed_encoded":
        X = reps.mixed_encoded
        representation = "mixed_encoded"
    elif method == "gower_pam":
        X = (reps.mixed_separated_numeric, reps.mixed_separated_categorical)
        representation = "gower_pam"
    elif method == "kprototypes":
        X = (reps.mixed_separated_numeric, reps.mixed_separated_categorical)
        representation = "kprototypes"
    elif method == "hierarchical_gower":
        X = (reps.mixed_separated_numeric, reps.mixed_separated_categorical)
        representation = "hierarchical_gower"
    else:
        raise ValueError(f"Unknown method: {method}")

    # Run consensus clustering
    consensus_result = run_consensus(X, representation, cfg, k, seed)
    labels = consensus_result.labels

    # Compute within-peer KPI agreement
    peer_agreement = multi_kpi_peer_agreement(df, labels, kpis, cfg)

    # Compute global KPI agreement (baseline)
    global_agreement = global_multi_kpi_agreement(df, kpis)

    return {
        "labels": labels,
        "peer_agreement": peer_agreement,
        "global_agreement": global_agreement,
        "kpis": kpis,
        "k": k,
        "method": method,
        "consensus_confidence": consensus_result.confidence,
    }


def summarize_agreement(
    peer_df: pd.DataFrame,
    global_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create summary comparison of peer vs global agreement.

    Parameters
    ----------
    peer_df : pd.DataFrame
        Within-peer agreement metrics.
    global_df : pd.DataFrame
        Global agreement metrics.

    Returns
    -------
    pd.DataFrame
        Summary comparison.
    """
    rows = []

    # Global summary
    for _, row in global_df.iterrows():
        rows.append({
            "method": "global",
            "kpi_pair": f"{row['kpi_1']} vs {row['kpi_2']}",
            "kendall_tau": row["kendall_tau"],
            "kendall_pvalue": row["kendall_pvalue"],
            "spearman_rho": row["spearman_rho"],
            "n_observations": row["n_institutions"],
        })

    # Peer group summary (aggregate across groups)
    if not peer_df.empty:
        for kpi_pair in peer_df.groupby(["kpi_1", "kpi_2"]).groups.keys():
            kpi1, kpi2 = kpi_pair
            subset = peer_df[
                (peer_df["kpi_1"] == kpi1) & (peer_df["kpi_2"] == kpi2)
            ]
            # Weight by peer group size
            valid = subset.dropna(subset=["kendall_tau"])
            if len(valid) > 0:
                weights = valid["peer_size"]
                weighted_tau = np.average(valid["kendall_tau"], weights=weights)
                weighted_rho = np.average(valid["spearman_rho"], weights=weights)
                rows.append({
                    "method": "peer_weighted",
                    "kpi_pair": f"{kpi1} vs {kpi2}",
                    "kendall_tau": weighted_tau,
                    "kendall_pvalue": np.nan,  # Combined p-value is complex
                    "spearman_rho": weighted_rho,
                    "n_observations": valid["peer_size"].sum(),
                })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-KPI agreement analysis for peer groups"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/obscare.yml"),
        help="Configuration file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/obscare/multi_kpi_agreement.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Number of clusters (default: use selection.k_values middle value)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="gower_pam",
        choices=["mixed_encoded", "gower_pam", "kprototypes", "hierarchical_gower"],
        help="Clustering method",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Determine K
    if args.k is not None:
        k = args.k
    else:
        k_values = cfg["selection"]["k_values"]
        k = k_values[len(k_values) // 2]  # Middle value

    print("Multi-KPI Agreement Analysis")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Method: {args.method}")
    print(f"K: {k}")

    # Load data
    df = load_dataset(cfg)
    print(f"Loaded {len(df)} institutions")
    print(f"KPIs: {cfg['targets']['kpis']}")

    # Run analysis
    results = run_multi_kpi_analysis(df, cfg, k, args.seed, args.method)

    # Save detailed peer agreement
    peer_df = results["peer_agreement"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    peer_df.to_csv(args.output, index=False)
    print(f"\nPeer agreement saved to: {args.output}")

    # Create summary
    summary = summarize_agreement(peer_df, results["global_agreement"])

    # Print results
    print("\n" + "=" * 60)
    print("KPI AGREEMENT SUMMARY")
    print("=" * 60)

    kpis = results["kpis"]
    for i, kpi1 in enumerate(kpis):
        for kpi2 in kpis[i + 1:]:
            print(f"\n{kpi1} vs {kpi2}:")

            # Global
            global_row = results["global_agreement"][
                (results["global_agreement"]["kpi_1"] == kpi1) &
                (results["global_agreement"]["kpi_2"] == kpi2)
            ]
            if len(global_row) > 0:
                tau = global_row["kendall_tau"].values[0]
                print(f"  Global Kendall's τ: {tau:.3f}")

            # Per peer group
            peer_subset = peer_df[
                (peer_df["kpi_1"] == kpi1) & (peer_df["kpi_2"] == kpi2)
            ]
            if len(peer_subset) > 0:
                print(f"  Within-peer by group:")
                for _, row in peer_subset.iterrows():
                    if not np.isnan(row["kendall_tau"]):
                        print(f"    Cluster {row['peer_group']} (n={row['peer_size']}): "
                              f"τ={row['kendall_tau']:.3f}")

    # Generate scatter plot if we have 2 KPIs
    if len(kpis) >= 2:
        fig_path = args.output.parent / "figures" / "fig_multi_kpi_scatter.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)

        # Get percentiles for each KPI (using raw values as proxy)
        # Create a fresh DataFrame with only the percentile columns
        kpi_x_name = f"{kpis[0]}_pctl"
        kpi_y_name = f"{kpis[1]}_pctl"
        plot_df = pd.DataFrame({
            kpi_x_name: (df[kpis[0]].rank() / len(df)) * 100,
            kpi_y_name: (df[kpis[1]].rank() / len(df)) * 100,
        })

        plot_multi_kpi_scatter(
            plot_df,
            kpi_x_name,
            kpi_y_name,
            results["labels"],
            str(fig_path),
            institution_ids=df[cfg["features"]["id"]].tolist()
            if len(df) <= 30 else None,
        )
        print(f"\nScatter plot saved to: {fig_path}")

    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    # Check if peer groups generalize across KPIs
    summary_row = summary[summary["method"] == "peer_weighted"]
    if len(summary_row) > 0:
        mean_tau = summary_row["kendall_tau"].mean()
        if mean_tau > 0.5:
            print("Strong agreement: Peer groups valid for multiple KPIs")
        elif mean_tau > 0.3:
            print("Moderate agreement: Peer groups partially generalizable")
        else:
            print("Weak agreement: Peer groups may be KPI-specific")
        print(f"Mean weighted Kendall's τ: {mean_tau:.3f}")


if __name__ == "__main__":
    main()
