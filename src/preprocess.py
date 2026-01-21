from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


@dataclass
class Representations:
    """Container for multiple feature representations and names."""
    numeric: np.ndarray
    categorical: np.ndarray
    mixed_encoded: np.ndarray
    mixed_separated_numeric: np.ndarray
    mixed_separated_categorical: np.ndarray
    feature_names: Dict[str, List[str]]


def _missing_indicator(series: pd.Series) -> pd.Series:
    """Return a boolean missing indicator for a series."""
    # Treat empty strings as missing for categorical fields.
    return series.isna() | (series.astype(str).str.strip() == "")


def _robust_scale(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Scale numeric data with median and IQR."""
    # Use robust statistics to reduce outlier influence.
    medians = df.median()
    q75 = df.quantile(0.75)
    q25 = df.quantile(0.25)
    iqr = (q75 - q25).replace(0, 1.0)
    scaled = (df - medians) / iqr
    return scaled, medians, iqr


def _encode_categorical(df: pd.DataFrame, encoding: str) -> Tuple[np.ndarray, List[str]]:
    """Encode categorical columns as ordinal or one-hot matrices."""
    if df.shape[1] == 0:
        return np.zeros((len(df), 0)), []
    if encoding == "ordinal":
        # Deterministic ordinal mapping per column.
        encoded = np.zeros((len(df), df.shape[1]), dtype=int)
        for idx, col in enumerate(df.columns):
            values = df[col].astype(str)
            uniques = sorted(values.unique())
            mapping = {v: i for i, v in enumerate(uniques)}
            encoded[:, idx] = values.map(mapping).astype(int)
        return encoded.astype(float), list(df.columns)
    if encoding == "onehot":
        # One-hot encode with stable output ordering.
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        encoded = encoder.fit_transform(df)
        names = encoder.get_feature_names_out(df.columns).tolist()
        return encoded.astype(float), names
    raise ValueError(f"Unsupported categorical encoding: {encoding}")


def prepare_representations(df: pd.DataFrame, cfg: Dict[str, Any]) -> Representations:
    """Prepare numeric, categorical, and mixed representations with missing indicators."""
    features = cfg["features"]
    numeric_cols = features.get("numeric", [])
    categorical_cols = features.get("categorical", [])
    encoding = cfg["preprocessing"]["categorical_encoding"]

    # Numeric preprocessing: coerce, impute, scale, and mark missing.
    numeric_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    numeric_missing = {
        f"{col}__missing": _missing_indicator(df[col]).astype(int)
        for col in numeric_cols
    }
    numeric_imputed = numeric_df.copy()
    for col in numeric_cols:
        median = numeric_imputed[col].median()
        numeric_imputed[col] = numeric_imputed[col].fillna(median)
    numeric_scaled, _, _ = _robust_scale(numeric_imputed)

    # Categorical preprocessing: fill missing and encode.
    categorical_df = df[categorical_cols].copy()
    for col in categorical_cols:
        missing_mask = _missing_indicator(categorical_df[col])
        categorical_df.loc[missing_mask, col] = "MISSING"
        categorical_df[col] = categorical_df[col].astype(str)

    categorical_encoded, categorical_feature_names = _encode_categorical(
        categorical_df, encoding
    )

    # Missing indicators for all features.
    missing_indicator_df = pd.DataFrame(
        {**numeric_missing, **{
            f"{col}__missing": _missing_indicator(df[col]).astype(int)
            for col in categorical_cols
        }}
    )

    # Assemble numeric-only and mixed representations.
    numeric_matrix = np.hstack(
        [numeric_scaled.to_numpy(), missing_indicator_df.to_numpy()]
    ) if numeric_cols else missing_indicator_df.to_numpy()

    mixed_encoded = np.hstack(
        [numeric_scaled.to_numpy(), missing_indicator_df.to_numpy(), categorical_encoded]
    )

    # Track feature names for inspection.
    feature_names = {
        "numeric": list(numeric_scaled.columns) + list(missing_indicator_df.columns),
        "categorical": categorical_feature_names,
        "mixed_encoded": list(numeric_scaled.columns)
        + list(missing_indicator_df.columns)
        + categorical_feature_names,
    }

    return Representations(
        numeric=numeric_matrix,
        categorical=categorical_encoded,
        mixed_encoded=mixed_encoded,
        mixed_separated_numeric=numeric_matrix,
        mixed_separated_categorical=categorical_encoded,
        feature_names=feature_names,
    )
