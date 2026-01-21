from __future__ import annotations

import argparse
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.benchmark import (
    combine_benchmarks,
    global_benchmark,
    knn_benchmark,
    rule_based_benchmark,
    within_peer_benchmark,
)
from src.consensus import run_consensus
from src.data import load_dataset
from src.eval import stability_summary, utility_summary
from src.perturbation import run_perturbation_eval
from src.plots import (
    plot_benchmark_percentiles,
    plot_coassignment_heatmap,
    plot_perturbation_detection_curve,
    plot_stability_vs_k,
)
from src.preprocess import prepare_representations
from src.utils import ensure_dir, get_logger, load_config, save_csv, save_json, set_seed


def _select_representation(reps, representation: str):
    """Select the requested representation from the prepared features."""
    if representation == "numeric":
        return reps.numeric
    if representation == "categorical":
        return reps.categorical
    if representation == "mixed_encoded":
        return reps.mixed_encoded
    if representation == "mixed_separated":
        return (reps.mixed_separated_numeric, reps.mixed_separated_categorical)
    raise ValueError(f"Unsupported representation: {representation}")


def _validate_representation(X, representation: str) -> None:
    """Ensure the selected representation has required feature content."""
    if representation == "mixed_separated":
        X_num, X_cat = X
        # mixed_separated requires both numeric and categorical matrices.
        if X_num.size == 0 or X_cat.size == 0:
            raise ValueError("mixed_separated requires both numeric and categorical features")
    else:
        # Other representations must contain at least one feature column.
        if X.size == 0:
            raise ValueError(f"Representation {representation} has no features")


def _knn_representation(reps, representation: str) -> np.ndarray:
    """Choose a representation for kNN baselines."""
    # Use mixed_encoded when the clustering is mixed_separated.
    if representation == "mixed_separated":
        return reps.mixed_encoded
    return _select_representation(reps, representation)


def _k_sweep(
    df: pd.DataFrame,
    reps,
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, Any]]]:
    """Run consensus, benchmarking, and perturbations across K values."""
    representation = cfg["representation"]
    X = _select_representation(reps, representation)
    _validate_representation(X, representation)

    seed = cfg["seed"]
    k_values = cfg["selection"]["k_values"]
    summaries = []
    cache: Dict[int, Dict[str, Any]] = {}

    for k in k_values:
        # Run consensus clustering for this K.
        consensus = run_consensus(X, representation, cfg, k, seed)
        stability_df = stability_summary(consensus.ari_scores, consensus.labels, consensus.confidence)
        stability_row = stability_df.iloc[0].to_dict()

        # Run peer benchmarking and perturbation utility evaluation.
        peer_bench = within_peer_benchmark(df, consensus.labels, cfg)
        perturb_df = run_perturbation_eval(
            df,
            cfg,
            representation,
            k,
            seed + k,
            peer_bench,
        )
        util = utility_summary(perturb_df)

        summaries.append(
            {
                "k": k,
                **stability_row,
                **util,
            }
        )
        # Cache artifacts for later selection and output.
        cache[k] = {
            "consensus": consensus,
            "peer_benchmark": peer_bench,
            "perturbation": perturb_df,
        }

    return pd.DataFrame(summaries), cache


def _choose_k(k_summary: pd.DataFrame, cfg: Dict[str, Any]) -> int:
    """Select K using stability plateau and utility tie-breaker."""
    delta = cfg["selection"]["stability_plateau_delta"]
    min_size = cfg["selection"]["min_cluster_size"]
    max_ari = k_summary["mean_ari"].max()
    # Keep K values within the stability plateau and size constraint.
    plateau = k_summary[k_summary["mean_ari"] >= max_ari - delta]
    plateau = plateau[plateau["min_cluster_size"] >= min_size]
    if plateau.empty:
        plateau = k_summary
    # Tie-break by perturbation recall, then mean ARI.
    best = plateau.sort_values(by=["mean_recall", "mean_ari", "k"], ascending=[False, False, True]).iloc[0]
    return int(best["k"])


def run_pipeline(config_path: str) -> None:
    """Run the full benchmarking pipeline from config."""
    cfg = load_config(config_path)
    logger = get_logger()
    set_seed(cfg["seed"])

    # Prepare output directories.
    outputs = cfg["output"]
    ensure_dir(outputs["tables_dir"])
    ensure_dir(outputs["figures_dir"])

    # Load data and prepare feature representations.
    df = load_dataset(cfg)
    reps = prepare_representations(df, cfg)

    # Sweep K values and select the best K.
    k_summary, cache = _k_sweep(df, reps, cfg)
    selected_k = _choose_k(k_summary, cfg)
    logger.info("Selected K=%s", selected_k)

    representation = cfg["representation"]
    X = _select_representation(reps, representation)
    consensus = cache[selected_k]["consensus"]

    # Assemble benchmarks across methods.
    peer_benchmark = cache[selected_k]["peer_benchmark"]
    global_bench = global_benchmark(df, cfg)
    knn_bench = knn_benchmark(df, _knn_representation(reps, representation), cfg)
    rule_bench = rule_based_benchmark(df, cfg)
    benchmark_df = combine_benchmarks([peer_benchmark, global_bench, knn_bench, rule_bench])

    perturb_df = cache[selected_k]["perturbation"]

    # Save core tables.
    id_col = cfg["features"]["id"]
    peer_assignments = pd.DataFrame(
        {
            "institution_id": df[id_col],
            "peer_group": consensus.labels,
            "confidence": consensus.confidence,
        }
    )

    coassign_df = pd.DataFrame(consensus.coassignment, index=df[id_col], columns=df[id_col])

    tables_dir = outputs["tables_dir"]
    figures_dir = outputs["figures_dir"]

    save_csv(peer_assignments, f"{tables_dir}/peer_assignments.csv")
    coassign_df.to_csv(f"{tables_dir}/coassignment_matrix.csv")
    save_csv(k_summary, f"{tables_dir}/k_sweep_summary.csv")
    save_csv(k_summary[["k", "mean_ari", "std_ari", "min_cluster_size", "mean_confidence"]], f"{tables_dir}/stability_summary.csv")
    save_csv(benchmark_df, f"{tables_dir}/benchmark_results.csv")
    if not perturb_df.empty:
        save_csv(perturb_df, f"{tables_dir}/perturbation_eval.csv")

    # Produce figures.
    plot_coassignment_heatmap(consensus.coassignment, df[id_col].tolist(), f"{figures_dir}/coassignment_heatmap.png")
    plot_stability_vs_k(k_summary, f"{figures_dir}/stability_vs_k.png")
    for kpi in cfg["targets"]["kpis"]:
        plot_benchmark_percentiles(benchmark_df, kpi, f"{figures_dir}/benchmark_percentiles_{kpi}.png")
        if not perturb_df.empty:
            plot_perturbation_detection_curve(
                perturb_df, kpi, f"{figures_dir}/perturbation_detection_curve_{kpi}.png"
            )

    # Save summary JSON for quick inspection.
    summary_json = {
        "selected_k": selected_k,
        "representation": representation,
        "k_sweep": k_summary.to_dict(orient="records"),
    }
    save_json(summary_json, outputs["summary_path"])


def main() -> None:
    """CLI entrypoint for running experiments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
