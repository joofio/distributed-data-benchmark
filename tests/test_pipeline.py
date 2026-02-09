from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from src.benchmark import within_peer_benchmark
from src.consensus import run_consensus
from src.data import load_dataset
from src.preprocess import prepare_representations
from src.perturbation import apply_kpi_shift, run_perturbation_eval
from src.utils import load_config


def _config() -> dict:
    """Load default config and reduce runtime for tests."""
    # Keep tests fast by lowering bootstrap and perturbation counts.
    cfg = load_config("configs/default.yml")
    cfg["consensus"]["n_bootstraps"] = 10
    cfg["perturbation"]["n_runs"] = 2
    return cfg


def test_determinism_consensus():
    """Consensus clustering is deterministic for a fixed seed."""
    cfg = _config()
    df = load_dataset(cfg)
    reps = prepare_representations(df, cfg)
    X = reps.mixed_encoded
    result_a = run_consensus(X, "mixed_encoded", cfg, 2, cfg["seed"])
    result_b = run_consensus(X, "mixed_encoded", cfg, 2, cfg["seed"])
    assert np.array_equal(result_a.labels, result_b.labels)
    assert np.allclose(result_a.coassignment, result_b.coassignment)


def test_coassignment_matrix_properties():
    """Co-assignment matrices are symmetric with valid bounds."""
    cfg = _config()
    df = load_dataset(cfg)
    reps = prepare_representations(df, cfg)
    result = run_consensus(reps.mixed_encoded, "mixed_encoded", cfg, 2, cfg["seed"])
    C = result.coassignment
    assert np.allclose(C, C.T)
    assert np.allclose(np.diag(C), 1.0)
    assert np.all((C >= 0.0) & (C <= 1.0))


def test_preprocess_shapes_and_missing_indicators():
    """Preprocessing adds missing indicators and expected shapes."""
    cfg = _config()
    df = load_dataset(cfg)
    reps = prepare_representations(df, cfg)
    assert reps.numeric.shape[1] == 6
    assert "feature_numeric_1__missing" in reps.feature_names["numeric"]


def test_clustering_reproducible():
    """Clustering results are reproducible with the same seed."""
    cfg = _config()
    df = load_dataset(cfg)
    reps = prepare_representations(df, cfg)
    result_a = run_consensus(reps.numeric, "numeric", cfg, 2, cfg["seed"])
    result_b = run_consensus(reps.numeric, "numeric", cfg, 2, cfg["seed"])
    assert np.array_equal(result_a.labels, result_b.labels)


def test_benchmark_percentiles_and_zscore_guard():
    """Benchmarks keep percentiles bounded and z-scores guarded."""
    df = pd.DataFrame(
        {
            "institution_id": ["a", "b"],
            "kpi_quality": [1.0, 2.0],
        }
    )
    cfg = {
        "features": {"id": "institution_id"},
        "targets": {"kpis": ["kpi_quality"]},
        "benchmark": {
            "outlier_percentile_low": 5,
            "outlier_percentile_high": 95,
            "outlier_zscore_abs": 2.0,
        },
    }
    labels = np.array([0, 0])
    bench = within_peer_benchmark(df, labels, cfg)
    assert bench["percentile"].between(0, 100).all()
    assert bench["zscore"].isna().all()


def test_perturbation_shift():
    """Perturbation shifts apply additive and multiplicative updates."""
    df = pd.DataFrame(
        {
            "institution_id": ["a"],
            "kpi_quality": [1.0],
        }
    )
    shifted_add = apply_kpi_shift(df, ["a"], ["kpi_quality"], "additive", 0.5, "institution_id")
    shifted_mul = apply_kpi_shift(df, ["a"], ["kpi_quality"], "multiplicative", 0.1, "institution_id")
    assert shifted_add.loc[0, "kpi_quality"] == 1.5
    assert np.isclose(shifted_mul.loc[0, "kpi_quality"], 1.1)


def test_ari_sanity():
    """ARI equals 1.0 for identical labelings."""
    labels = np.array([0, 1, 1, 0])
    assert adjusted_rand_score(labels, labels) == 1.0


def test_perturbation_metrics_run():
    """Perturbation evaluation returns detection metrics columns."""
    cfg = _config()
    df = load_dataset(cfg)
    reps = prepare_representations(df, cfg)
    peer_bench = within_peer_benchmark(df, np.zeros(len(df), dtype=int), cfg)
    out = run_perturbation_eval(df, cfg, "mixed_encoded", 2, cfg["seed"], peer_bench)
    assert set(["recall", "false_positive_rate"]).issubset(out.columns)
