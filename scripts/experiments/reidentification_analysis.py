#!/usr/bin/env python3
"""
Re-identification risk analysis for institutional benchmarking data.

This script evaluates the privacy risks associated with releasing peer group
assignments and benchmark percentiles, including:
- K-anonymity analysis of quasi-identifier combinations
- Membership inference attack simulation
- Overall disclosure risk scoring

Addresses Major Reviewer Point #5: Lack of formal privacy guarantees.

Outputs:
- results/privacy/reidentification_risk.csv: Risk metrics per dataset
- results/privacy/privacy_summary.json: Overall risk assessment
"""
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

from src.data import load_dataset
from src.preprocess import prepare_representations
from src.privacy import (
    compute_privacy_risk_summary,
    compute_reidentification_risk,
    simulate_membership_inference_attack,
)


def analyze_dataset_privacy(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    seed: int = 42,
) -> Dict[str, Any]:
    """Analyze privacy risks for a single dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Loaded dataset.
    cfg : Dict[str, Any]
        Configuration dictionary.
    seed : int
        Random seed.

    Returns
    -------
    Dict[str, Any]
        Privacy risk metrics.
    """
    # Identify quasi-identifiers (features that could identify institutions)
    # These are typically size-related and categorical features
    numeric_cols = cfg["features"].get("numeric", [])
    categorical_cols = cfg["features"].get("categorical", [])

    # Common quasi-identifiers: size, region-related features
    quasi_identifiers = []
    for col in numeric_cols:
        if any(kw in col.lower() for kw in ["patient", "size", "n_", "count", "volume"]):
            quasi_identifiers.append(col)
    for col in categorical_cols[:5]:  # Limit to first 5 categorical
        quasi_identifiers.append(col)

    # If no quasi-identifiers found, use first few features
    if not quasi_identifiers:
        quasi_identifiers = (numeric_cols[:3] + categorical_cols[:2])

    # Filter to columns that exist in df
    quasi_identifiers = [c for c in quasi_identifiers if c in df.columns]

    # Get feature columns for membership inference
    feature_cols = [c for c in numeric_cols + categorical_cols if c in df.columns]

    # Run comprehensive privacy analysis
    results = compute_privacy_risk_summary(
        df=df,
        feature_cols=feature_cols,
        quasi_identifiers=quasi_identifiers,
        holdout_fraction=0.2,
        seed=seed,
    )

    # Add dataset info
    results["dataset_info"] = {
        "n_records": len(df),
        "n_features": len(feature_cols),
        "n_quasi_identifiers": len(quasi_identifiers),
        "quasi_identifiers_used": quasi_identifiers,
    }

    return results


def analyze_cluster_disclosure(
    df: pd.DataFrame,
    labels: np.ndarray,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Analyze whether cluster assignments leak information.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with institutions.
    labels : np.ndarray
        Cluster assignments.
    cfg : Dict[str, Any]
        Configuration dictionary.

    Returns
    -------
    Dict[str, Any]
        Cluster-based disclosure metrics.
    """
    results = {}

    # Check cluster sizes
    unique_labels, counts = np.unique(labels, return_counts=True)
    results["n_clusters"] = len(unique_labels)
    results["cluster_sizes"] = dict(zip(map(int, unique_labels), map(int, counts)))
    results["min_cluster_size"] = int(counts.min())
    results["max_cluster_size"] = int(counts.max())
    results["singleton_clusters"] = int(sum(counts == 1))

    # Singleton clusters are high-risk
    results["singleton_institutions"] = int(sum(counts == 1))

    # Calculate cluster-based k-anonymity
    # Each cluster provides some anonymity within the group
    results["mean_cluster_k_anonymity"] = float(counts.mean())

    # Risk score based on small clusters
    risk = 0.0
    for size in counts:
        if size == 1:
            risk += 1.0
        elif size == 2:
            risk += 0.5
        elif size <= 5:
            risk += 0.2
    risk = risk / len(labels) if len(labels) > 0 else 0.0
    results["cluster_disclosure_risk"] = min(risk, 1.0)

    return results


def format_results_table(results: List[Dict]) -> pd.DataFrame:
    """Format results into a summary table."""
    rows = []
    for r in results:
        rows.append({
            "dataset": r.get("dataset", "unknown"),
            "n_records": r["dataset_info"]["n_records"],
            "n_quasi_identifiers": r["dataset_info"]["n_quasi_identifiers"],
            "singleton_count": r["reidentification"]["singleton_count"],
            "singleton_rate": r["reidentification"]["singleton_rate"],
            "min_k_anonymity": r["reidentification"]["min_k_anonymity"],
            "mean_k_anonymity": r["reidentification"]["mean_k_anonymity"],
            "disclosure_risk_score": r["reidentification"]["disclosure_risk_score"],
            "mia_accuracy": r["membership_inference"]["attack_accuracy"],
            "mia_advantage": r["membership_inference"]["attack_advantage"],
            "mia_auc": r["membership_inference"]["auc_roc"],
            "overall_risk_score": r["overall_risk_score"],
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-identification risk analysis for benchmarking data"
    )
    parser.add_argument(
        "--configs",
        type=Path,
        nargs="+",
        default=[Path("configs/obscare.yml")],
        help="Configuration files for datasets to analyze",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("reports/privacy/reidentification_risk.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/privacy/privacy_summary.json"),
        help="Output JSON path for detailed results",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("Re-identification Risk Analysis")
    print("=" * 60)

    all_results = []

    for config_path in args.configs:
        print(f"\nAnalyzing: {config_path}")

        if not config_path.exists():
            print(f"  Config not found, skipping")
            continue

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        # Load dataset
        try:
            df = load_dataset(cfg)
        except Exception as e:
            print(f"  Error loading dataset: {e}")
            continue

        dataset_name = Path(cfg["dataset"]["path"]).stem

        # Analyze privacy risks
        print(f"  Running privacy analysis...")
        results = analyze_dataset_privacy(df, cfg, seed=args.seed)
        results["dataset"] = dataset_name

        # Print summary
        reid = results["reidentification"]
        mia = results["membership_inference"]
        print(f"  K-anonymity: min={reid['min_k_anonymity']}, "
              f"mean={reid['mean_k_anonymity']:.1f}")
        print(f"  Singletons: {reid['singleton_count']} "
              f"({reid['singleton_rate']*100:.1f}%)")
        print(f"  Disclosure risk: {reid['disclosure_risk_score']:.3f}")
        print(f"  MIA accuracy: {mia['attack_accuracy']:.3f} "
              f"(advantage: {mia['attack_advantage']:.3f})")
        print(f"  Overall risk: {results['overall_risk_score']:.3f}")

        all_results.append(results)

    if not all_results:
        print("\nNo datasets analyzed successfully")
        return

    # Save CSV summary
    df_summary = format_results_table(all_results)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(args.output_csv, index=False)
    print(f"\nCSV summary saved to: {args.output_csv}")

    # Save detailed JSON
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Detailed JSON saved to: {args.output_json}")

    # Print final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(df_summary.to_string(index=False))

    # Risk interpretation
    print("\n" + "=" * 60)
    print("RISK INTERPRETATION")
    print("=" * 60)
    for r in all_results:
        dataset = r["dataset"]
        risk = r["overall_risk_score"]
        if risk < 0.1:
            level = "LOW"
            recommendation = "Standard precautions sufficient"
        elif risk < 0.3:
            level = "MODERATE"
            recommendation = "Consider additional privacy measures"
        else:
            level = "HIGH"
            recommendation = "Strong privacy measures recommended (e.g., DP, aggregation)"

        print(f"{dataset}: {level} risk ({risk:.3f})")
        print(f"  Recommendation: {recommendation}")


if __name__ == "__main__":
    main()
