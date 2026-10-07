"""Model monitoring module for FlixChurn.

Implements:
- Data quality monitoring
- Feature drift detection (PSI, KS statistic)
- Prediction drift monitoring
- Performance monitoring (when actual labels available)

Methods and thresholds are documented (R-037 drift rule).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ml.config import get_feature_config

logger = logging.getLogger(__name__)


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Population Stability Index (PSI).

    PSI < 0.1: no significant change
    0.1 <= PSI < 0.2: moderate change
    PSI >= 0.2: significant change (review required)

    Args:
        reference: Reference distribution (e.g., training data).
        current: Current distribution (e.g., new prediction data).
        n_bins: Number of bins for histogram.

    Returns:
        PSI value.
    """
    # Create bins from reference distribution
    ref_clean = reference[~np.isnan(reference)]
    cur_clean = current[~np.isnan(current)]

    if len(ref_clean) == 0 or len(cur_clean) == 0:
        return 0.0

    # Use reference data to define bins
    bins = np.linspace(
        min(ref_clean.min(), cur_clean.min()),
        max(ref_clean.max(), cur_clean.max()),
        n_bins + 1,
    )

    ref_hist, _ = np.histogram(ref_clean, bins=bins)
    cur_hist, _ = np.histogram(cur_clean, bins=bins)

    # Add small epsilon to avoid division by zero
    eps = 1e-6
    ref_pct = ref_hist / ref_hist.sum() + eps
    cur_pct = cur_hist / cur_hist.sum() + eps

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def compute_ks_statistic(
    reference: np.ndarray,
    current: np.ndarray,
) -> dict[str, float]:
    """Compute Kolmogorov-Smirnov statistic for two distributions.

    Args:
        reference: Reference distribution.
        current: Current distribution.

    Returns:
        Dictionary with KS statistic and p-value.
    """
    ref_clean = reference[~np.isnan(reference)]
    cur_clean = current[~np.isnan(current)]

    if len(ref_clean) == 0 or len(cur_clean) == 0:
        return {"ks_statistic": 0.0, "p_value": 1.0}

    ks_stat, p_value = stats.ks_2samp(ref_clean, cur_clean)
    return {
        "ks_statistic": float(ks_stat),
        "p_value": float(p_value),
    }


def compute_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    psi_threshold: float = 0.2,
    ks_threshold: float = 0.1,
) -> dict[str, Any]:
    """Compute drift metrics for all features.

    Args:
        reference_df: Reference DataFrame (training data).
        current_df: Current DataFrame (prediction data).
        psi_threshold: PSI threshold for drift alert.
        ks_threshold: KS threshold for drift alert.

    Returns:
        Dictionary with drift results per feature.
    """
    config = get_feature_config()
    drift_results = {}
    drifted_features = []

    for col in config.numerical_features:
        if col not in reference_df.columns or col not in current_df.columns:
            continue

        ref_vals = reference_df[col].values.astype(float)
        cur_vals = current_df[col].values.astype(float)

        psi = compute_psi(ref_vals, cur_vals)
        ks = compute_ks_statistic(ref_vals, cur_vals)

        is_drifted = psi >= psi_threshold or ks["ks_statistic"] >= ks_threshold

        status = "DRIFT" if is_drifted else "OK"
        if psi >= psi_threshold * 0.8 or ks["ks_statistic"] >= ks_threshold * 0.8:
            status = "REVIEW" if not is_drifted else status

        drift_results[col] = {
            "psi": round(psi, 4),
            "ks_statistic": round(ks["ks_statistic"], 4),
            "ks_p_value": round(ks["p_value"], 4),
            "psi_threshold": psi_threshold,
            "ks_threshold": ks_threshold,
            "status": status,
            "is_drifted": is_drifted,
        }

        if is_drifted:
            drifted_features.append(col)

    for col in config.categorical_features:
        if col not in reference_df.columns or col not in current_df.columns:
            continue

        ref_dist = reference_df[col].value_counts(normalize=True).to_dict()
        cur_dist = current_df[col].value_counts(normalize=True).to_dict()

        all_categories = set(ref_dist.keys()) | set(cur_dist.keys())
        max_diff = max(
            abs(ref_dist.get(cat, 0) - cur_dist.get(cat, 0))
            for cat in all_categories
        ) if all_categories else 0

        is_drifted = max_diff > 0.15
        status = "DRIFT" if is_drifted else ("REVIEW" if max_diff > 0.10 else "OK")

        drift_results[col] = {
            "max_category_diff": round(max_diff, 4),
            "reference_distribution": {str(k): round(v, 4) for k, v in ref_dist.items()},
            "current_distribution": {str(k): round(v, 4) for k, v in cur_dist.items()},
            "status": status,
            "is_drifted": is_drifted,
        }

        if is_drifted:
            drifted_features.append(col)

    return {
        "total_features": len(drift_results),
        "drifted_features": drifted_features,
        "drift_count": len(drifted_features),
        "features": drift_results,
    }


def compute_prediction_drift(
    reference_probabilities: np.ndarray,
    current_probabilities: np.ndarray,
) -> dict[str, Any]:
    """Compare prediction distribution between reference and current.

    Args:
        reference_probabilities: Reference prediction probabilities.
        current_probabilities: Current prediction probabilities.

    Returns:
        Dictionary with prediction drift analysis.
    """
    psi = compute_psi(reference_probabilities, current_probabilities)
    ks = compute_ks_statistic(reference_probabilities, current_probabilities)

    return {
        "psi": round(psi, 4),
        "ks_statistic": round(ks["ks_statistic"], 4),
        "ks_p_value": round(ks["p_value"], 4),
        "reference_mean": round(float(np.mean(reference_probabilities)), 4),
        "current_mean": round(float(np.mean(current_probabilities)), 4),
        "reference_std": round(float(np.std(reference_probabilities)), 4),
        "current_std": round(float(np.std(current_probabilities)), 4),
        "is_drifted": psi >= 0.2 or ks["ks_statistic"] >= 0.1,
        "reference_histogram": np.histogram(
            reference_probabilities, bins=20, range=(0, 1)
        )[0].tolist(),
        "current_histogram": np.histogram(
            current_probabilities, bins=20, range=(0, 1)
        )[0].tolist(),
    }
