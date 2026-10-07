"""Label generation module for FlixChurn.

DATASET ADAPTATION:
The source dataset already contains a 'churned' column (0/1).
Since the dataset is a flat snapshot without event-level timestamps,
label generation in this context validates and prepares the existing
churn labels rather than computing them from temporal event windows.

The label generation API is designed so that if a temporal dataset
becomes available, the implementation can be swapped without changing
downstream code.

Configuration:
- prediction_horizon_days: loaded from configs/prediction.yaml
- observation_window_days: loaded from configs/prediction.yaml
"""
from __future__ import annotations

import logging

import pandas as pd

from ml.config import get_prediction_config

logger = logging.getLogger(__name__)


def generate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Generate/validate churn labels for the dataset.

    For the current flat-snapshot dataset, this validates the existing
    'churned' column and adds metadata columns.

    For a temporal dataset, this would compute labels based on:
    - observation timestamp
    - prediction horizon window
    - churn events within the window

    Args:
        df: DataFrame with at least 'customer_id' and 'churned' columns.

    Returns:
        DataFrame with validated 'churn_label' column and metadata.

    Raises:
        ValueError: If the 'churned' column is missing or invalid.
    """
    config = get_prediction_config()
    logger.info(
        f"Generating labels with prediction_horizon={config.prediction_horizon_days}d, "
        f"observation_window={config.observation_window_days}d"
    )

    if "churned" not in df.columns:
        raise ValueError("Column 'churned' not found in dataset")

    result = df.copy()

    # Validate binary labels
    invalid = ~result["churned"].isin([0, 1])
    if invalid.any():
        n_invalid = invalid.sum()
        logger.warning(f"Removing {n_invalid} records with non-binary churn values")
        result = result[~invalid].copy()

    # Create standardized label column
    result["churn_label"] = result["churned"].astype(int)

    # Add metadata
    result["prediction_horizon_days"] = config.prediction_horizon_days
    result["observation_window_days"] = config.observation_window_days

    # Log class distribution
    churn_rate = result["churn_label"].mean()
    n_churned = result["churn_label"].sum()
    n_total = len(result)
    logger.info(
        f"Label distribution: {n_churned}/{n_total} churned ({churn_rate:.1%})"
    )

    # Document that this is from a pre-labeled dataset
    result["label_source"] = "dataset_provided"

    return result


def get_label_statistics(df: pd.DataFrame) -> dict:
    """Get statistics about the label distribution.

    Args:
        df: DataFrame with 'churn_label' column.

    Returns:
        Dictionary with label statistics.
    """
    if "churn_label" not in df.columns:
        raise ValueError("Column 'churn_label' not found. Run generate_labels first.")

    total = len(df)
    churned = int(df["churn_label"].sum())
    not_churned = total - churned

    return {
        "total_records": total,
        "churned": churned,
        "not_churned": not_churned,
        "churn_rate": churned / total if total > 0 else 0,
        "class_ratio": churned / not_churned if not_churned > 0 else float("inf"),
        "is_imbalanced": abs(churned / total - 0.5) > 0.15 if total > 0 else False,
    }
