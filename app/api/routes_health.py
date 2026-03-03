"""
app/api/routes_health.py

Health-check endpoint: GET /health
Returns service status and dependency availability.
"""

from __future__ import annotations

import time

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.logging import log

router = APIRouter(tags=["Health"])

_START_TIME = time.time()


class HealthResponse(BaseModel):
    """Schema for the health-check response."""

    status: str
    uptime_seconds: float
    gemini_model: str
    embed_model: str
    chroma_persist_dir: str


@router.get("/health", response_model=HealthResponse, summary="Service health check")
def health_check() -> HealthResponse:
    """Return service status and configuration summary.

    Returns:
        HealthResponse with uptime and model configuration.
    """
    settings = get_settings()
    log.debug("Health check requested")
    return HealthResponse(
        status="ok",
        uptime_seconds=round(time.time() - _START_TIME, 2),
        gemini_model=settings.gemini_model,
        embed_model=settings.ollama_embed_model,
        chroma_persist_dir=str(settings.chroma_persist_dir),
    )
