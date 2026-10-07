"""Health check endpoint."""
from fastapi import APIRouter

router = APIRouter()


@router.get("/health", tags=["Health"])
async def health_check():
    """System health check."""
    return {
        "status": "healthy",
        "service": "flixchurn-api",
        "version": "1.0.0",
    }
