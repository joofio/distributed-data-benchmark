from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.benchmark import within_peer_benchmark


def _load_fixture() -> pd.DataFrame:
    path = Path(__file__).parent / "fixtures" / "small_institutions.csv"
    return pd.read_csv(path)


def _config() -> dict:
    return {
        "features": {"id": "institution_id"},
        "targets": {"kpis": ["kpi_quality"]},
        "benchmark": {
            "outlier_percentile_low": 5,
            "outlier_percentile_high": 95,
            "outlier_zscore_abs": 10.0,
        },
    }


def test_percentiles_within_bounds():
    df = _load_fixture()
    cfg = _config()
    labels = np.zeros(len(df), dtype=int)
    bench = within_peer_benchmark(df, labels, cfg)
    assert bench["percentile"].between(0, 100).all()


def test_zscore_sanity_for_peer_group():
    df = _load_fixture()
    cfg = _config()
    labels = np.zeros(len(df), dtype=int)
    bench = within_peer_benchmark(df, labels, cfg)
    zscores = bench["zscore"].to_numpy()
    assert np.isfinite(zscores).all()
    assert np.isclose(zscores.mean(), 0.0)
    assert np.isclose(np.std(zscores, ddof=1), 1.0)
