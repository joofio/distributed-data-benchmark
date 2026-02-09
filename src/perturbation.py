from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.benchmark import within_peer_benchmark
from src.consensus import run_consensus
from src.preprocess import prepare_representations


def apply_kpi_shift(
    df: pd.DataFrame,
    institution_ids: List[str],
    kpis: List[str],
    shift_type: str,
    magnitude: float,
    id_col: str,
) -> pd.DataFrame:
    """Apply additive or multiplicative KPI shifts to selected institutions."""
    shifted = df.copy()
    # Select institutions to perturb.
    mask = shifted[id_col].isin(institution_ids)
    for kpi in kpis:
        if shift_type == "additive":
            shifted.loc[mask, kpi] = shifted.loc[mask, kpi] + magnitude
        elif shift_type == "multiplicative":
            shifted.loc[mask, kpi] = shifted.loc[mask, kpi] * (1.0 + magnitude)
        else:
            raise ValueError(f"Unsupported shift type: {shift_type}")
    return shifted


def _rank_from_percentile(percentile: float, n: int) -> int:
    """Convert percentile to a 1-based rank for a group size."""
    if n <= 1:
        return 1
    return int(round(percentile / 100.0 * (n - 1) + 1))


def _baseline_lookup(benchmark_df: pd.DataFrame) -> Dict[Tuple[str, str], Tuple[float, int]]:
    """Build a lookup for baseline percentiles and peer sizes."""
    lookup: Dict[Tuple[str, str], Tuple[float, int]] = {}
    for _, row in benchmark_df.iterrows():
        key = (row["institution_id"], row["kpi"])
        lookup[key] = (float(row["percentile"]), int(row["peer_size"]))
    return lookup


def run_perturbation_eval(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    representation: str,
    k: int,
    seed: int,
    baseline_peer_benchmark: pd.DataFrame,
) -> pd.DataFrame:
    """Run semi-synthetic perturbations and compute detection metrics."""
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

    # Keep baseline peer benchmarks for shift comparisons.
    baseline_lookup = _baseline_lookup(baseline_peer_benchmark)

    output = []
    for run_id in range(n_runs):
        # Sample institutions and magnitudes per run.
        rng = np.random.default_rng(seed + run_id)
        if pert_cfg.get("institutions"):
            institution_ids = pert_cfg["institutions"]
        else:
            n_inst = min(pert_cfg.get("n_institutions", 1), len(df))
            institution_ids = rng.choice(df[id_col].tolist(), size=n_inst, replace=False).tolist()

        if magnitudes:
            magnitude = float(rng.choice(magnitudes))
        else:
            magnitude = float(rng.uniform(mag_min, mag_max))

        # Apply the KPI perturbation and re-run the pipeline.
        perturbed = apply_kpi_shift(
            df, institution_ids, kpis, shift_type, magnitude, id_col
        )

        reps = prepare_representations(perturbed, cfg)
        if representation == "numeric":
            X = reps.numeric
        elif representation == "categorical":
            X = reps.categorical
        elif representation == "mixed_encoded":
            X = reps.mixed_encoded
        elif representation == "mixed_separated":
            X = (reps.mixed_separated_numeric, reps.mixed_separated_categorical)
        else:
            raise ValueError(f"Unsupported representation: {representation}")

        consensus = run_consensus(X, representation, cfg, k, seed + run_id)
        peer_benchmark = within_peer_benchmark(perturbed, consensus.labels, cfg)

        non_perturbed = ~perturbed[id_col].isin(institution_ids)
        for kpi in kpis:
            peer_kpi = peer_benchmark[peer_benchmark["kpi"] == kpi]
            pert_rows = peer_kpi[peer_kpi["institution_id"].isin(institution_ids)]
            non_pert_rows = peer_kpi[peer_kpi["institution_id"].isin(perturbed.loc[non_perturbed, id_col])]

            # Compute false positives among non-perturbed institutions.
            false_positive_rate = float(non_pert_rows["outlier"].mean()) if not non_pert_rows.empty else 0.0
            for _, row in pert_rows.iterrows():
                base_pct, base_n = baseline_lookup.get(
                    (row["institution_id"], row["kpi"]), (np.nan, 0)
                )
                # Measure percentile and rank shifts versus baseline.
                pct_shift = abs(float(row["percentile"]) - base_pct) if not np.isnan(base_pct) else np.nan
                rank_shift = (
                    abs(
                        _rank_from_percentile(float(row["percentile"]), int(row["peer_size"]))
                        - _rank_from_percentile(float(base_pct), int(base_n))
                    )
                    if base_n > 0 and not np.isnan(base_pct)
                    else np.nan
                )
                output.append(
                    {
                        "run_id": int(run_id),
                        "kpi": kpi,
                        "institution_id": row["institution_id"],
                        "shift_type": shift_type,
                        "magnitude": magnitude,
                        "recall": float(bool(row["outlier"])),
                        "false_positive_rate": false_positive_rate,
                        "percentile_shift": pct_shift,
                        "rank_shift": rank_shift,
                    }
                )

    return pd.DataFrame(output)
