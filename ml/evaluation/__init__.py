"""Model evaluation module for FlixChurn.

Computes comprehensive evaluation metrics:
- ROC-AUC, PR-AUC
- Precision, Recall, F1
- Brier Score
- Recall@K, Precision@K, Lift@K
- Confusion matrix
- Calibration curve data

Model selection is based on documented criteria (not just highest AUC).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from ml.config import get_model_config

logger = logging.getLogger(__name__)


def evaluate_model(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "model",
    k_percent: float = 10.0,
) -> dict[str, Any]:
    """Compute all evaluation metrics for a model.

    Args:
        pipeline: Fitted model pipeline.
        X: Feature DataFrame.
        y: True labels.
        model_name: Name for reporting.
        k_percent: Top-K percentage for Recall@K, Precision@K, Lift@K.

    Returns:
        Dictionary of metric name → value.
    """
    y_pred_proba = pipeline.predict_proba(X)[:, 1]
    y_pred = pipeline.predict(X)

    # Core metrics
    metrics = {
        "model_name": model_name,
        "roc_auc": float(roc_auc_score(y, y_pred_proba)),
        "pr_auc": float(average_precision_score(y, y_pred_proba)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y, y_pred_proba)),
        "log_loss": float(log_loss(y, y_pred_proba)),
    }

    # Top-K metrics
    k_metrics = _compute_topk_metrics(y.values, y_pred_proba, k_percent)
    metrics.update(k_metrics)

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["true_negatives"] = int(cm[0, 0])
    metrics["false_positives"] = int(cm[0, 1])
    metrics["false_negatives"] = int(cm[1, 0])
    metrics["true_positives"] = int(cm[1, 1])

    # Calibration data
    cal_data = compute_calibration_curve(y.values, y_pred_proba)
    metrics["calibration"] = cal_data

    logger.info(
        f"{model_name}: ROC-AUC={metrics['roc_auc']:.4f}, "
        f"PR-AUC={metrics['pr_auc']:.4f}, "
        f"Brier={metrics['brier_score']:.4f}, "
        f"Recall@{k_percent:.0f}%={metrics.get('recall_at_k', 'N/A')}"
    )

    return metrics


def _compute_topk_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    k_percent: float = 10.0,
) -> dict[str, float]:
    """Compute Recall@K, Precision@K, and Lift@K.

    K is defined as the top k_percent of predictions by probability.
    """
    n = len(y_true)
    k = max(1, int(n * k_percent / 100))

    # Sort by predicted probability descending
    sorted_indices = np.argsort(-y_pred_proba)
    top_k_indices = sorted_indices[:k]

    # Actual positives in top-K
    tp_at_k = y_true[top_k_indices].sum()

    # Total actual positives
    total_positives = y_true.sum()

    # Baseline: expected positives in a random selection of k items
    baseline_rate = total_positives / n if n > 0 else 0

    recall_at_k = tp_at_k / total_positives if total_positives > 0 else 0
    precision_at_k = tp_at_k / k if k > 0 else 0
    lift_at_k = precision_at_k / baseline_rate if baseline_rate > 0 else 0

    return {
        "k_percent": k_percent,
        "k_count": k,
        f"recall_at_{int(k_percent)}": float(recall_at_k),
        f"precision_at_{int(k_percent)}": float(precision_at_k),
        f"lift_at_{int(k_percent)}": float(lift_at_k),
    }


def compute_calibration_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute calibration curve data.

    Returns:
        Dictionary with calibration curve points and calibration metrics.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)

    # Calibration slope and intercept (linear fit)
    if len(prob_true) >= 2:
        coeffs = np.polyfit(prob_pred, prob_true, 1)
        slope = float(coeffs[0])
        intercept = float(coeffs[1])
    else:
        slope = 1.0
        intercept = 0.0

    return {
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
        "n_bins": n_bins,
        "slope": slope,
        "intercept": intercept,
    }


def compute_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
) -> dict[str, list[float]]:
    """Compute ROC curve data points."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
    }


def compute_pr_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
) -> dict[str, list[float]]:
    """Compute Precision-Recall curve data points."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
    }


def evaluate_all_models(
    trained_models: dict[str, Pipeline],
    splits: dict[str, tuple[pd.DataFrame, pd.Series]],
    eval_split: str = "validation",
) -> dict[str, dict[str, Any]]:
    """Evaluate all trained models on the specified split.

    Args:
        trained_models: Dictionary of model_name → fitted Pipeline.
        splits: Output of split_data().
        eval_split: Which split to evaluate on ("validation" or "test").

    Returns:
        Dictionary of model_name → metrics dict.
    """
    X_eval, y_eval = splits[eval_split]
    results = {}

    for model_key, pipeline in trained_models.items():
        config = get_model_config()
        model_name = config.models[model_key].name
        metrics = evaluate_model(pipeline, X_eval, y_eval, model_name)
        results[model_key] = metrics

    return results


def select_best_model(
    evaluation_results: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """Select the best model based on documented criteria.

    Selection logic (from config):
    1. Primary metric: PR-AUC (because churn prediction is often imbalanced)
    2. Secondary considerations: Recall@10, Brier score, calibration
    3. Minimum thresholds must be met

    The selection logic is documented so it can be audited.

    Returns:
        Tuple of (model_key, selection_rationale).
    """
    config = get_model_config()
    criteria = config.selection_criteria

    # Filter models meeting minimum thresholds
    eligible = {}
    for model_key, metrics in evaluation_results.items():
        meets_thresholds = True
        for metric_name, threshold in criteria.minimum_thresholds.items():
            actual = metrics.get(metric_name, 0)
            if actual < threshold:
                logger.warning(
                    f"Model {model_key} does not meet minimum {metric_name}: "
                    f"{actual:.4f} < {threshold}"
                )
                meets_thresholds = False

        if meets_thresholds:
            eligible[model_key] = metrics

    if not eligible:
        logger.warning("No model meets minimum thresholds. Selecting best available.")
        eligible = evaluation_results

    # Score by primary metric
    primary_metric = criteria.primary
    best_key = max(
        eligible.keys(),
        key=lambda k: eligible[k].get(primary_metric, 0)
    )

    # Build rationale
    rationale = {
        "selected_model": best_key,
        "selection_criteria": {
            "primary_metric": primary_metric,
            "primary_value": eligible[best_key].get(primary_metric, 0),
        },
        "all_models": {},
    }

    for model_key, metrics in evaluation_results.items():
        rationale["all_models"][model_key] = {
            "pr_auc": metrics.get("pr_auc", 0),
            "roc_auc": metrics.get("roc_auc", 0),
            "brier_score": metrics.get("brier_score", 1),
            "recall_at_10": metrics.get("recall_at_10", 0),
            "precision_at_10": metrics.get("precision_at_10", 0),
            "calibration_slope": metrics.get("calibration", {}).get("slope", 0),
        }

    logger.info(f"Selected model: {best_key} (PR-AUC={rationale['selection_criteria']['primary_value']:.4f})")

    return best_key, rationale


def evaluate_calibration(
    pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_name: str = "model",
) -> dict[str, Any]:
    """Evaluate whether calibration improves probability reliability.

    Returns:
        Dictionary with before/after Brier scores and recommendation.
    """
    config = get_model_config()
    y_pred_proba = pipeline.predict_proba(X_val)[:, 1]

    brier_before = brier_score_loss(y_val, y_pred_proba)
    cal_before = compute_calibration_curve(y_val.values, y_pred_proba)

    # Try calibration
    calibrator = CalibratedClassifierCV(
        pipeline,
        method="isotonic" if config.calibration.method == "isotonic" else "sigmoid",
        cv=3,
    )

    try:
        calibrator.fit(X_val, y_val)
        y_cal_proba = calibrator.predict_proba(X_val)[:, 1]
        brier_after = brier_score_loss(y_val, y_cal_proba)
        cal_after = compute_calibration_curve(y_val.values, y_cal_proba)

        improved = brier_after < brier_before
        recommendation = "apply" if improved else "skip"

        logger.info(
            f"Calibration for {model_name}: "
            f"Brier before={brier_before:.4f}, after={brier_after:.4f}, "
            f"recommendation={recommendation}"
        )

        return {
            "model_name": model_name,
            "method": config.calibration.method,
            "brier_before": float(brier_before),
            "brier_after": float(brier_after),
            "calibration_before": cal_before,
            "calibration_after": cal_after,
            "improved": improved,
            "recommendation": recommendation,
            "calibrator": calibrator if improved else None,
        }
    except Exception as e:
        logger.warning(f"Calibration failed for {model_name}: {e}")
        return {
            "model_name": model_name,
            "method": config.calibration.method,
            "brier_before": float(brier_before),
            "brier_after": None,
            "improved": False,
            "recommendation": "skip",
            "error": str(e),
            "calibrator": None,
        }
