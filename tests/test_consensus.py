from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.consensus import run_consensus


def _load_fixture() -> pd.DataFrame:
    path = Path(__file__).parent / "fixtures" / "small_institutions.csv"
    return pd.read_csv(path)


def _config() -> dict:
    return {
        "consensus": {
            "n_bootstraps": 5,
            "numeric_jitter_scale": 0.0,
            "feature_bootstrap": False,
            "sample_fraction": 1.0,
        }
    }


def test_coassignment_symmetry_and_diagonal():
    df = _load_fixture()
    cfg = _config()
    X = df[["feature_numeric_1"]].to_numpy()
    result = run_consensus(X, "numeric", cfg, k=2, seed=7)
    C = result.coassignment
    assert np.allclose(C, C.T)
    assert np.allclose(np.diag(C), 1.0)


def test_ari_scores_within_bounds():
    df = _load_fixture()
    cfg = _config()
    X = df[["feature_numeric_1"]].to_numpy()
    result = run_consensus(X, "numeric", cfg, k=2, seed=7)
    ari_scores = np.array(result.ari_scores)
    assert np.all((ari_scores >= -1.0) & (ari_scores <= 1.0))
