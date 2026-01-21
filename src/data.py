from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    # Keep CSV parsing simple and deterministic.
    return pd.read_csv(path)


def validate_schema(df: pd.DataFrame, cfg: Dict[str, Any]) -> None:
    """Validate required columns and dtypes based on config."""
    features = cfg["features"]
    id_col = features["id"]
    numeric = features.get("numeric", [])
    categorical = features.get("categorical", [])
    kpis = cfg["targets"]["kpis"]
    descriptors = cfg["targets"].get("descriptors", [])

    # Ensure all required columns are present.
    required = [id_col] + numeric + categorical + kpis + descriptors
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Validate numeric and KPI dtypes.
    for col in numeric:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Numeric feature column must be numeric: {col}")

    for col in kpis:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"KPI column must be numeric: {col}")

    # Require non-null institution identifiers.
    if df[id_col].isna().any():
        raise ValueError("institution_id column contains missing values")


def coerce_id(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Coerce institution IDs to string for stable joins."""
    # Keep IDs consistent across outputs.
    df = df.copy()
    df[id_col] = df[id_col].astype(str)
    return df


def split_into_institutions(
    df: pd.DataFrame,
    n_institutions: int,
    seed: int | None = None,
    shuffle: bool = True,
    prefix: str = "INST",
) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    """
    Split a UCI dataset into multiple institutions (silos) for distributed testing.

    Takes a dataset and partitions rows into N institutions, simulating
    distributed data silos. Each row gets an institution_id based on which
    partition it belongs to.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (e.g., loaded from UCI repository).
    n_institutions : int
        Number of institutions/silos to create.
    seed : int | None, default None
        Random seed for reproducibility when shuffling.
    shuffle : bool, default True
        Whether to shuffle rows before splitting. If False, splits sequentially.
    prefix : str, default "INST"
        Prefix for institution IDs (e.g., "INST" -> "INST_1", "INST_2").

    Returns
    -------
    Tuple[List[pd.DataFrame], pd.DataFrame]
        - List of DataFrames, one per institution (each with institution_id column)
        - Combined DataFrame with all rows and institution_id column

    Examples
    --------
    >>> df = pd.DataFrame({"feat": range(100)})
    >>> silos, combined = split_into_institutions(df, n_institutions=3, seed=42)
    >>> len(silos)
    3
    >>> [len(s) for s in silos]
    [34, 33, 33]
    >>> combined["institution_id"].unique().tolist()
    ['INST_1', 'INST_2', 'INST_3']
    """
    if n_institutions < 1:
        raise ValueError("n_institutions must be at least 1")
    if n_institutions > len(df):
        raise ValueError(f"n_institutions ({n_institutions}) > rows ({len(df)})")

    result = df.copy().reset_index(drop=True)

    if shuffle:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(result))
        result = result.iloc[indices].reset_index(drop=True)

    # Split indices as evenly as possible
    splits = np.array_split(range(len(result)), n_institutions)

    # Assign institution IDs
    institution_ids = [""] * len(result)
    for i, split_indices in enumerate(splits):
        inst_id = f"{prefix}_{i + 1}"
        for idx in split_indices:
            institution_ids[idx] = inst_id

    result.insert(0, "institution_id", institution_ids)

    # Create list of separate DataFrames per institution
    silos = []
    for i in range(n_institutions):
        inst_id = f"{prefix}_{i + 1}"
        silo_df = result[result["institution_id"] == inst_id].copy().reset_index(drop=True)
        silos.append(silo_df)

    return silos, result


def aggregate_to_institutions(
    df: pd.DataFrame,
    n_institutions: int,
    numeric_cols: List[str],
    categorical_cols: List[str],
    target_col: str,
    seed: int | None = None,
    shuffle: bool = True,
    prefix: str = "INST",
) -> pd.DataFrame:
    """
    Split patient-level data into institutions and aggregate to institution level.

    This transforms patient-level UCI data into synthetic institutional data suitable
    for benchmarking research. Each institution becomes one row with aggregate
    statistics computed from its patients.

    Parameters
    ----------
    df : pd.DataFrame
        Patient-level DataFrame (e.g., from UCI repository).
    n_institutions : int
        Number of synthetic institutions to create.
    numeric_cols : List[str]
        Columns to aggregate as numeric features (mean, std computed).
    categorical_cols : List[str]
        Columns to aggregate as categorical features (mode computed).
    target_col : str
        Column containing the outcome/diagnosis. Will become a KPI
        (prevalence rate for binary, mean for continuous).
    seed : int | None, default None
        Random seed for reproducibility.
    shuffle : bool, default True
        Whether to shuffle patients before assignment.
    prefix : str, default "INST"
        Prefix for institution IDs.

    Returns
    -------
    pd.DataFrame
        Institution-level DataFrame where each row is one institution with:
        - institution_id: unique identifier
        - {col}_mean: mean of numeric feature across patients
        - {col}_std: std of numeric feature across patients
        - {col}_mode: most common value for categorical features
        - n_patients: number of patients in institution
        - target_rate: prevalence rate (mean of target, works for binary 0/1)
        - target_std: variability of outcomes

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "age": [25, 30, 35, 40, 45, 50],
    ...     "sex": [0, 1, 0, 1, 0, 1],
    ...     "outcome": [0, 0, 1, 0, 1, 1]
    ... })
    >>> result = aggregate_to_institutions(
    ...     df, n_institutions=2,
    ...     numeric_cols=["age"], categorical_cols=["sex"],
    ...     target_col="outcome", seed=42
    ... )
    >>> list(result.columns)
    ['institution_id', 'n_patients', 'age_mean', 'age_std', 'sex_mode', 'target_rate', 'target_std']
    """
    if n_institutions < 1:
        raise ValueError("n_institutions must be at least 1")
    if n_institutions > len(df):
        raise ValueError(f"n_institutions ({n_institutions}) > rows ({len(df)})")

    data = df.copy().reset_index(drop=True)

    if shuffle:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(data))
        data = data.iloc[indices].reset_index(drop=True)

    # Assign institution IDs
    splits = np.array_split(range(len(data)), n_institutions)
    institution_ids = [""] * len(data)
    for i, split_indices in enumerate(splits):
        inst_id = f"{prefix}_{i + 1}"
        for idx in split_indices:
            institution_ids[idx] = inst_id

    data["institution_id"] = institution_ids

    # Build aggregation using groupby with as_index=True for cleaner handling
    grouped = data.groupby("institution_id")

    # Start with patient counts
    result_df = grouped.size().reset_index(name="n_patients")

    # Aggregate numeric columns (mean and std)
    for col in numeric_cols:
        if col in data.columns:
            means = grouped[col].mean().reset_index(name=f"{col}_mean")
            stds = grouped[col].std().reset_index(name=f"{col}_std")
            result_df = result_df.merge(means, on="institution_id")
            result_df = result_df.merge(stds, on="institution_id")

    # Aggregate categorical columns (mode)
    for col in categorical_cols:
        if col in data.columns:
            modes = grouped[col].agg(
                lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan
            ).reset_index(name=f"{col}_mode")
            result_df = result_df.merge(modes, on="institution_id")

    # Aggregate target column as KPI
    if target_col in data.columns:
        target_mean = grouped[target_col].mean().reset_index(name="target_rate")
        target_std = grouped[target_col].std().reset_index(name="target_std")
        result_df = result_df.merge(target_mean, on="institution_id")
        result_df = result_df.merge(target_std, on="institution_id")

    return result_df


def load_dataset(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Load and validate dataset based on config."""
    dataset_cfg = cfg["dataset"]
    dataset_type = dataset_cfg.get("type", "real")
    if dataset_type not in {"real", "uci"}:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    path = dataset_cfg["path"]
    # Local CSV loading only (no network access).
    df = load_csv(path)

    mapping = dataset_cfg.get("mapping", {})
    if mapping:
        # Apply column mapping to match the expected schema.
        df = df.rename(columns=mapping)

    features = cfg["features"]
    id_col = features["id"]
    df = coerce_id(df, id_col)
    validate_schema(df, cfg)
    return df
