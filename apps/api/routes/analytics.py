"""Analytics routes: overview, risk distribution, cohorts, drivers."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from apps.api.services import get_model_service

router = APIRouter()


@router.get("/overview")
async def get_overview():
    """Risk overview: eligible customers, high-risk count, avg probability."""
    service = get_model_service()
    result = service.get_overview()
    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])
    return result


@router.get("/risk-distribution")
async def get_risk_distribution():
    """Risk band distribution across the customer population."""
    service = get_model_service()
    result = service.get_risk_distribution()
    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])
    return result


@router.get("/cohorts")
async def get_cohorts(
    group_by: str = Query(
        default="subscription_type",
        description="Dimension to group cohorts by",
    ),
):
    """Cohort analysis: risk distribution by grouping dimension."""
    valid_dimensions = [
        "subscription_type", "region", "device",
        "payment_method", "gender", "favorite_genre",
    ]
    if group_by not in valid_dimensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid group_by. Must be one of: {valid_dimensions}",
        )

    service = get_model_service()
    result = service.get_cohorts(group_by=group_by)
    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])
    return result


@router.get("/drivers")
async def get_drivers():
    """Top predictive features (SHAP-based global explanation)."""
    service = get_model_service()
    result = service.get_drivers()
    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])
    return result
