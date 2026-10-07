"""Monitoring routes: data quality, drift, performance."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from apps.api.services import get_model_service

router = APIRouter()


@router.get("/monitoring/data-quality")
async def get_data_quality():
    """Data quality monitoring: schema, missingness, ranges, duplicates."""
    service = get_model_service()
    result = service.get_data_quality()
    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])
    return result


@router.get("/monitoring/drift")
async def get_drift():
    """Feature drift monitoring: PSI, KS statistic per feature."""
    service = get_model_service()
    result = service.get_drift()
    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])
    return result


@router.get("/monitoring/performance")
async def get_performance():
    """Model performance monitoring: metrics trends, calibration."""
    service = get_model_service()
    result = service.get_performance()
    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])
    return result
