"""Inference module for FlixChurn.

Supports:
- Single customer prediction (online inference)
- Batch prediction for the full population with financial risk calculations
- What-if simulation using the same production model with ROI estimation

Risk bands are derived from configurable probability thresholds.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ml.config import get_prediction_config
from ml.data_contract import PredictionResult, RiskBand, SimulationResult

logger = logging.getLogger(__name__)


def assign_risk_band(probability: float) -> RiskBand:
    """Assign a risk band based on configurable probability thresholds.

    Thresholds from configs/prediction.yaml:
    [0, low) → Low, [low, moderate) → Moderate,
    [moderate, high) → High, [high, 1.0] → Critical
    """
    config = get_prediction_config()
    thresholds = config.risk_thresholds

    if probability >= thresholds.high:
        return RiskBand.CRITICAL
    elif probability >= thresholds.moderate:
        return RiskBand.HIGH
    elif probability >= thresholds.low:
        return RiskBand.MODERATE
    else:
        return RiskBand.LOW


def classify_churn_type(row: pd.Series | dict[str, Any]) -> str:
    """Classify primary risk mechanism: voluntary engagement vs involuntary billing friction."""
    payment_method = str(row.get("payment_method", "")).lower()
    last_login_days = float(row.get("last_login_days", 0))
    watch_hours = float(row.get("watch_hours", 0))

    # Involuntary billing indicators: payment issues, or account active but failed renewal
    if "gift" in payment_method or "crypto" in payment_method:
        return "involuntary_billing"
    if last_login_days <= 5 and watch_hours > 20:
        return "involuntary_billing"
    return "voluntary_engagement"


def predict_single(
    pipeline: Pipeline,
    features: pd.DataFrame,
    customer_id: str,
    model_version: str = "1.0.0",
) -> PredictionResult:
    """Generate a prediction for a single customer with financial risk metrics."""
    config = get_prediction_config()

    proba = float(pipeline.predict_proba(features)[:, 1][0])
    risk_band = assign_risk_band(proba)

    monthly_fee = float(features.iloc[0].get("monthly_fee", 12.99)) if "monthly_fee" in features.columns else 12.99
    mrr_at_risk = round(proba * monthly_fee, 2)
    arr_at_risk = round(mrr_at_risk * 12, 2)
    churn_type = classify_churn_type(features.iloc[0].to_dict())

    return PredictionResult(
        customer_id=customer_id,
        churn_probability=round(proba, 4),
        risk_band=risk_band,
        prediction_horizon_days=config.prediction_horizon_days,
        model_version=model_version,
        prediction_timestamp=datetime.now(timezone.utc).isoformat(),
        monthly_fee=monthly_fee,
        mrr_at_risk=mrr_at_risk,
        arr_at_risk=arr_at_risk,
        churn_type_risk=churn_type,
    )


def predict_batch(
    pipeline: Pipeline,
    df: pd.DataFrame,
    feature_columns: list[str],
    model_version: str = "1.0.0",
) -> pd.DataFrame:
    """Generate predictions for a batch of customers with financial metrics."""
    config = get_prediction_config()
    logger.info(f"Running batch prediction for {len(df)} customers")

    X = df[feature_columns].copy()
    probabilities = pipeline.predict_proba(X)[:, 1]

    monthly_fees = df["monthly_fee"].values if "monthly_fee" in df.columns else np.full(len(df), 12.99)
    mrr_at_risk = np.round(probabilities * monthly_fees, 2)
    arr_at_risk = np.round(mrr_at_risk * 12, 2)

    churn_types = [
        classify_churn_type(df.iloc[i].to_dict()) for i in range(len(df))
    ]

    results = pd.DataFrame({
        "customer_id": df["customer_id"].values,
        "churn_probability": np.round(probabilities, 4),
        "risk_band": [assign_risk_band(p).value for p in probabilities],
        "prediction_horizon_days": config.prediction_horizon_days,
        "model_version": model_version,
        "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
        "monthly_fee": np.round(monthly_fees, 2),
        "mrr_at_risk": mrr_at_risk,
        "arr_at_risk": arr_at_risk,
        "churn_type_risk": churn_types,
    })

    # Log summary
    risk_dist = results["risk_band"].value_counts().to_dict()
    total_mrr_risk = float(results["mrr_at_risk"].sum())
    logger.info(f"Batch prediction complete. Risk distribution: {risk_dist}")
    logger.info(f"Total Portfolio MRR at Risk: ${total_mrr_risk:,.2f}")

    return results


def simulate_scenario(
    pipeline: Pipeline,
    baseline_features: pd.DataFrame,
    scenario_features: pd.DataFrame,
    customer_id: str,
    model_version: str = "1.0.0",
) -> SimulationResult:
    """Run a what-if simulation with financial ROI modeling."""
    baseline_proba = float(pipeline.predict_proba(baseline_features)[:, 1][0])
    scenario_proba = float(pipeline.predict_proba(scenario_features)[:, 1][0])

    monthly_fee = float(baseline_features.iloc[0].get("monthly_fee", 12.99)) if "monthly_fee" in baseline_features.columns else 12.99

    # Calculate change in percentage points
    change_pp = round(scenario_proba - baseline_proba, 4) * 100
    mrr_saved = round((baseline_proba - scenario_proba) * monthly_fee, 2)
    annual_value = round(mrr_saved * 12, 2)

    return SimulationResult(
        customer_id=customer_id,
        baseline_probability=round(baseline_proba, 4),
        scenario_probability=round(scenario_proba, 4),
        probability_change_pp=round(change_pp, 1),
        risk_band_baseline=assign_risk_band(baseline_proba),
        risk_band_scenario=assign_risk_band(scenario_proba),
        model_version=model_version,
        monthly_fee=monthly_fee,
        mrr_saved=mrr_saved,
        annual_value_preserved=annual_value,
    )


def get_risk_distribution(predictions: pd.DataFrame) -> dict[str, Any]:
    """Compute risk distribution and financial exposure from batch predictions."""
    total = len(predictions)
    if total == 0:
        return {"total": 0, "bands": {}, "total_mrr_at_risk": 0.0}

    band_counts = predictions["risk_band"].value_counts().to_dict()

    # Ensure all bands are represented
    for band in RiskBand:
        if band.value not in band_counts:
            band_counts[band.value] = 0

    # Calculate financial exposures
    monthly_fees = predictions["monthly_fee"] if "monthly_fee" in predictions.columns else 12.99
    mrr_at_risk = predictions["mrr_at_risk"] if "mrr_at_risk" in predictions.columns else (predictions["churn_probability"] * monthly_fees)
    total_mrr_risk = float(round(mrr_at_risk.sum(), 2))
    total_arr_risk = float(round(total_mrr_risk * 12, 2))

    # High / Critical risk concentration
    high_mask = predictions["risk_band"].isin(["high", "critical"])
    high_risk_mrr = float(round(mrr_at_risk[high_mask].sum(), 2))

    # Churn type breakdown
    voluntary_count = int((predictions["churn_type_risk"] == "voluntary_engagement").sum()) if "churn_type_risk" in predictions.columns else int(total * 0.8)
    involuntary_count = int((predictions["churn_type_risk"] == "involuntary_billing").sum()) if "churn_type_risk" in predictions.columns else int(total * 0.2)

    return {
        "total_eligible": total,
        "high_risk_count": band_counts.get("high", 0) + band_counts.get("critical", 0),
        "avg_churn_probability": round(float(predictions["churn_probability"].mean()), 4),
        "predicted_churn_volume": int(
            (predictions["churn_probability"] >= 0.5).sum()
        ),
        "total_mrr_at_risk": total_mrr_risk,
        "total_arr_at_risk": total_arr_risk,
        "high_risk_mrr_at_risk": high_risk_mrr,
        "voluntary_risk_count": voluntary_count,
        "involuntary_risk_count": involuntary_count,
        "bands": {
            band.value: {
                "count": band_counts.get(band.value, 0),
                "percentage": round(
                    band_counts.get(band.value, 0) / total * 100, 1
                ),
            }
            for band in RiskBand
        },
    }
