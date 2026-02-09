from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.benchmark import within_peer_benchmark
from src.consensus import run_consensus
from src.perturbation import run_perturbation_eval


def _load_fixture() -> pd.DataFrame:
    path = Path(__file__).parent / "fixtures" / "small_institutions.csv"
    return pd.read_csv(path)


def _config(magnitude: float) -> dict:
    return {
        "seed": 11,
        "features": {"id": "institution_id", "numeric": ["feature_numeric_1"], "categorical": []},
        "targets": {"kpis": ["kpi_quality"]},
        "preprocessing": {"categorical_encoding": "ordinal"},
        "consensus": {
            "n_bootstraps": 3,
            "numeric_jitter_scale": 0.0,
            "feature_bootstrap": False,
            "sample_fraction": 1.0,
        },
        "benchmark": {
            "outlier_percentile_low": 5,
            "outlier_percentile_high": 95,
            "outlier_zscore_abs": 10.0,
        },
        "perturbation": {
            "enabled": True,
            "n_runs": 1,
            "kpis": ["kpi_quality"],
            "institutions": ["C"],
            "shift": {
                "type": "additive",
                "magnitudes": [magnitude],
            },
        },
    }


def _baseline_peer_benchmark(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    X = df[["feature_numeric_1"]].to_numpy()
    consensus = run_consensus(X, "numeric", cfg, k=1, seed=cfg["seed"])
    return within_peer_benchmark(df, consensus.labels, cfg)


def _recall_for_magnitude(df: pd.DataFrame, magnitude: float) -> float:
    cfg = _config(magnitude)
    baseline = _baseline_peer_benchmark(df, cfg)
    out = run_perturbation_eval(df, cfg, "numeric", 1, cfg["seed"], baseline)
    return float(out["recall"].iloc[0])


def test_detection_rate_monotonicity():
    df = _load_fixture()
    magnitudes = [0.0, 5.0, 20.0]
    recalls = [_recall_for_magnitude(df, mag) for mag in magnitudes]
    assert all(a <= b for a, b in zip(recalls, recalls[1:]))
