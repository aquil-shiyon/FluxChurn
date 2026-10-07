"""Feature engineering module for FlixChurn.

Generates predictive features from normalized customer data.

DATASET ADAPTATION NOTE:
The source dataset is a flat snapshot (no event-level timestamps).
Feature engineering is adapted to work with the available columns:
- watch_hours, avg_watch_time_per_day → engagement intensity
- last_login_days → recency
- number_of_profiles → account engagement
- subscription_type, monthly_fee → subscription characteristics

Derived features are deterministic and documented.
All feature generation uses only data available at the conceptual
prediction point (no future information leakage).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ml.config import get_feature_config

logger = logging.getLogger(__name__)


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate the complete feature set from normalized data.

    Args:
        df: Normalized DataFrame with validated raw columns.

    Returns:
        DataFrame with original columns plus derived features.
        customer_id is preserved but not used as a model feature.
    """
    logger.info(f"Generating features for {len(df)} records")
    features = df.copy()

    # --- Derived numerical features ---

    # Watch hours per profile: higher = more engaged per account member
    features["watch_hours_per_profile"] = (
        features["watch_hours"] / features["number_of_profiles"]
    )

    # Fee per watch hour: higher = lower value perception
    # Add 1 to avoid division by zero
    features["fee_per_watch_hour"] = (
        features["monthly_fee"] / (features["watch_hours"] + 1)
    )

    # Login recency score: normalized to [0, 1]
    # 0 = logged in today, 1 = maximum recency (most concerning)
    max_login_days = features["last_login_days"].max()
    if max_login_days > 0:
        features["login_recency_score"] = features["last_login_days"] / max_login_days
    else:
        features["login_recency_score"] = 0.0

    # --- Validate features ---
    _validate_features(features)

    logger.info(f"Generated {len(features.columns)} total columns")
    return features


def get_model_feature_columns() -> tuple[list[str], list[str]]:
    """Return the lists of numerical and categorical feature names for model input.

    Returns:
        Tuple of (numerical_features, categorical_features)
    """
    config = get_feature_config()
    return config.numerical_features, config.categorical_features


def get_all_feature_columns() -> list[str]:
    """Return all feature column names used by the model."""
    numerical, categorical = get_model_feature_columns()
    return numerical + categorical


def prepare_model_input(df: pd.DataFrame) -> pd.DataFrame:
    """Extract only the model feature columns from a full DataFrame.

    This ensures the model sees exactly the features it expects,
    in the correct order, without customer_id or target.
    """
    all_features = get_all_feature_columns()
    missing = set(all_features) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required features: {sorted(missing)}")
    return df[all_features].copy()


def get_feature_statistics(df: pd.DataFrame) -> dict:
    """Compute feature-level statistics for monitoring and comparison.

    Returns:
        Dictionary of feature name → statistics dict.
    """
    config = get_feature_config()
    stats = {}

    for col in config.numerical_features:
        if col in df.columns:
            stats[col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "median": float(df[col].median()),
                "q25": float(df[col].quantile(0.25)),
                "q75": float(df[col].quantile(0.75)),
                "missing_pct": float(df[col].isnull().mean() * 100),
            }

    for col in config.categorical_features:
        if col in df.columns:
            value_counts = df[col].value_counts(normalize=True).to_dict()
            stats[col] = {
                "unique_count": int(df[col].nunique()),
                "mode": str(df[col].mode().iloc[0]) if len(df[col].mode()) > 0 else None,
                "distribution": {str(k): float(v) for k, v in value_counts.items()},
                "missing_pct": float(df[col].isnull().mean() * 100),
            }

    return stats


def _validate_features(df: pd.DataFrame) -> None:
    """Validate that derived features are in expected ranges."""
    checks = {
        "watch_hours_per_profile": (0, None),
        "fee_per_watch_hour": (0, None),
        "login_recency_score": (0, 1),
    }

    for col, (min_val, max_val) in checks.items():
        if col not in df.columns:
            logger.warning(f"Feature '{col}' not found during validation")
            continue

        if min_val is not None and (df[col] < min_val).any():
            logger.warning(f"Feature '{col}' has values below {min_val}")

        if max_val is not None and (df[col] > max_val).any():
            logger.warning(f"Feature '{col}' has values above {max_val}")

        if df[col].isnull().any():
            nan_count = df[col].isnull().sum()
            logger.warning(f"Feature '{col}' has {nan_count} NaN values")
