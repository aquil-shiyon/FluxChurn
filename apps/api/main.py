"""FastAPI application entry point for FlixChurn.

API-first architecture: all frontend data flows through these endpoints.
No business logic in route handlers (R-022).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from apps.api.routes import analytics, customers, health, models, monitoring, prediction
from ml.config import get_app_settings

logger = logging.getLogger(__name__)

app = FastAPI(
    title="FlixChurn API",
    description="AI Customer Churn Prediction & Retention Intelligence Platform API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
settings = get_app_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(health.router)
app.include_router(analytics.router, prefix="/api/v1", tags=["Analytics"])
app.include_router(customers.router, prefix="/api/v1", tags=["Customers"])
app.include_router(prediction.router, prefix="/api/v1", tags=["Prediction"])
app.include_router(models.router, prefix="/api/v1", tags=["Models"])
app.include_router(monitoring.router, prefix="/api/v1", tags=["Monitoring"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "apps.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
