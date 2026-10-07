"""Prediction routes: online predict and what-if simulation."""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from apps.api.services import get_model_service

router = APIRouter()


class PredictRequest(BaseModel):
    """Request schema for online prediction."""
    customer_id: Optional[str] = "unknown"
    age: float = Field(ge=18, le=120)
    watch_hours: float = Field(ge=0)
    last_login_days: float = Field(ge=0)
    monthly_fee: float = Field(gt=0)
    number_of_profiles: float = Field(ge=1, le=10)
    avg_watch_time_per_day: float = Field(ge=0)
    subscription_type: str
    region: str
    device: str
    payment_method: str
    gender: str
    favorite_genre: str

    # Derived features will be computed
    watch_hours_per_profile: Optional[float] = None
    fee_per_watch_hour: Optional[float] = None
    login_recency_score: Optional[float] = None


class SimulateRequest(BaseModel):
    """Request schema for what-if simulation."""
    customer_id: str
    scenario: dict[str, Any] = Field(
        description="Modified feature values for the scenario",
    )


@router.post("/predict")
async def predict(request: PredictRequest):
    """Generate a churn prediction for provided features.

    Returns:
        Prediction with probability, risk band, model version.
    """
    service = get_model_service()
    if not service.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Run the training pipeline first.",
        )

    # Compute derived features if not provided
    features = request.model_dump()
    if features["watch_hours_per_profile"] is None:
        features["watch_hours_per_profile"] = (
            features["watch_hours"] / max(features["number_of_profiles"], 1)
        )
    if features["fee_per_watch_hour"] is None:
        features["fee_per_watch_hour"] = (
            features["monthly_fee"] / (features["watch_hours"] + 1)
        )
    if features["login_recency_score"] is None:
        features["login_recency_score"] = min(features["last_login_days"] / 60.0, 1.0)

    result = service.predict(features)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@router.post("/simulate")
async def simulate(request: SimulateRequest):
    """Run a what-if simulation for a customer.

    Modifies selected behavioral features and generates a new prediction
    using the SAME production model. This is a model sensitivity analysis,
    not a causal prediction.

    Returns:
        Baseline vs scenario prediction comparison.
    """
    service = get_model_service()
    if not service.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Run the training pipeline first.",
        )

    # Validate scenario features
    from ml.config import get_feature_config
    config = get_feature_config()

    for key in request.scenario:
        if key not in config.simulatable_features:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Feature '{key}' is not simulatable. "
                    f"Valid features: {config.simulatable_features}"
                ),
            )

    result = service.simulate(request.customer_id, request.scenario)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result
