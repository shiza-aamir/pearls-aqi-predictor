from __future__ import annotations

from fastapi import APIRouter

from api.core.config import get_api_settings
from api.schemas.common import HealthResponse


router = APIRouter(
    prefix="/health",
    tags=["Health"],
)


@router.get(
    "",
    response_model=HealthResponse,
    summary="API health check",
)
def health_check() -> HealthResponse:
    settings = get_api_settings()

    return HealthResponse(
        status="healthy",
        service=settings.app_name,
        version=settings.app_version,
    )