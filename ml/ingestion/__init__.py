"""Data ingestion module for FlixChurn.

Loads raw data from CSV/Parquet files into pandas DataFrames.
Raw data is never modified - a copy is always returned.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from ml.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

# Expected raw columns from the source dataset
EXPECTED_RAW_COLUMNS = [
    "customer_id", "age", "gender", "subscription_type", "watch_hours",
    "last_login_days", "region", "device", "monthly_fee", "churned",
    "payment_method", "number_of_profiles", "avg_watch_time_per_day",
    "favorite_genre",
]


def load_raw_data(
    filepath: Optional[str | Path] = None,
    data_dir: str = "data/raw",
) -> pd.DataFrame:
    """Load raw dataset from file.

    Args:
        filepath: Explicit path to the data file. If None, searches data_dir.
        data_dir: Directory to search for data files (relative to project root).

    Returns:
        DataFrame with raw data (never modified in place).

    Raises:
        FileNotFoundError: If no suitable data file is found.
        ValueError: If the file format is unsupported.
    """
    if filepath is not None:
        path = Path(filepath)
    else:
        search_dir = PROJECT_ROOT / data_dir
        path = _find_data_file(search_dir)

    logger.info(f"Loading raw data from: {path}")

    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Expected .csv or .parquet")

    logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
    logger.info(f"Columns: {df.columns.tolist()}")

    return df.copy()


def _find_data_file(search_dir: Path) -> Path:
    """Find the first suitable data file in a directory."""
    if not search_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {search_dir}")

    # Prefer parquet, then CSV
    for ext in [".parquet", ".csv"]:
        files = list(search_dir.glob(f"*{ext}"))
        if files:
            return files[0]

    raise FileNotFoundError(
        f"No data files (.csv, .parquet) found in {search_dir}"
    )


def save_to_parquet(
    df: pd.DataFrame,
    filename: str,
    output_dir: str = "data/processed",
) -> Path:
    """Save a DataFrame to Parquet format.

    Args:
        df: DataFrame to save.
        filename: Output filename (without extension).
        output_dir: Output directory (relative to project root).

    Returns:
        Path to the saved file.
    """
    out_path = PROJECT_ROOT / output_dir
    out_path.mkdir(parents=True, exist_ok=True)
    filepath = out_path / f"{filename}.parquet"
    df.to_parquet(filepath, index=False, engine="pyarrow")
    logger.info(f"Saved {len(df)} records to {filepath}")
    return filepath


def load_processed_data(
    filename: str,
    data_dir: str = "data/processed",
) -> pd.DataFrame:
    """Load a previously processed Parquet file."""
    filepath = PROJECT_ROOT / data_dir / f"{filename}.parquet"
    if not filepath.exists():
        raise FileNotFoundError(f"Processed data not found: {filepath}")
    return pd.read_parquet(filepath)
