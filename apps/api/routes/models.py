"""Model routes: model listing and metrics."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from apps.api.services import get_model_service

router = APIRouter()


@router.get("/models")
async def list_models():
    """List all trained models with comparison data."""
    service = get_model_service()
    result = service.get_models()
    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])
    return result


@router.get("/models/{model_version}/metrics")
async def get_model_metrics(model_version: str):
    """Get detailed metrics for a specific model version."""
    service = get_model_service()
    result = service.get_model_metrics(model_version)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result
