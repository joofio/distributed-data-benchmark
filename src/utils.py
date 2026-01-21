from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


def get_logger(name: str = "benchmark") -> logging.Logger:
    """Create or return a configured logger for the pipeline."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Configure a single handler to avoid duplicate logs.
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and hash for deterministic runs."""
    # Keep seeds aligned across common RNGs.
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file from disk."""
    # Use safe_load to avoid executing YAML tags.
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> None:
    """Create a directory if it does not exist."""
    # Ensure nested output paths are created.
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    """Write a dictionary to JSON with stable formatting."""
    # Use sorted keys for deterministic output.
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def save_csv(df, path: str | Path) -> None:
    """Write a DataFrame to CSV without row index."""
    # Keep CSVs clean and deterministic.
    df.to_csv(path, index=False)
