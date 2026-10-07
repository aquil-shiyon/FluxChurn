"""Customer routes: risk lookup, explanation, customer list."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from apps.api.services import get_model_service

router = APIRouter()


@router.get("/customers")
async def list_customers(
    risk_band: Optional[str] = Query(None, description="Filter by risk band"),
    churn_type: Optional[str] = Query(None, description="Filter by churn type: voluntary_engagement or involuntary_billing"),
    sort_by: str = Query("churn_probability", description="Sort field: churn_probability or mrr_at_risk"),
    order: str = Query("desc", description="Sort order: asc or desc"),
    search: Optional[str] = Query(None, description="Search by customer ID"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List customers with optional financial sorting and mechanism filtering."""
    service = get_model_service()
    return service.get_customer_list(
        risk_band=risk_band,
        churn_type=churn_type,
        sort_by=sort_by,
        order=order,
        limit=limit,
        offset=offset,
        search=search,
    )


@router.get("/customers/{customer_id}/risk")
async def get_customer_risk(customer_id: str):
    """Get risk details for a specific customer."""
    service = get_model_service()
    result = service.get_customer_risk(customer_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.get("/customers/{customer_id}/explanation")
async def get_customer_explanation(customer_id: str):
    """Get SHAP-based explanation for a customer's prediction.

    Note: Feature contributions indicate predictive associations,
    not causal relationships.
    """
    service = get_model_service()
    result = service.get_customer_explanation(customer_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result
