from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import (
    CORSMiddleware,
)

from api.core.config import (
    get_api_settings,
)
from api.routes.cities import (
    router as cities_router,
)
from api.routes.forecast import (
    router as forecast_router,
)
from api.routes.health import (
    router as health_router,
)
from api.routes.history import (
    router as history_router,
)
from api.routes.models import (
    router as models_router,
)
from api.routes.performance import (
    router as performance_router,
)
from api.schemas.common import (
    APIInfoResponse,
)

settings = (
    get_api_settings()
)


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Production API for the Pearls AQI Predictor. "
        "Provides real-time air-quality observations, "
        "24/48/72-hour AQI forecasts, explainability, "
        "historical observations, model performance, "
        "and production-model information."
    ),
    docs_url=settings.docs_url,
    redoc_url=settings.redoc_url,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=(
        settings.cors_origins()
    ),
    allow_credentials=True,
    allow_methods=[
        "GET",
    ],
    allow_headers=[
        "*",
    ],
)


app.include_router(
    health_router,
    prefix=settings.api_prefix,
)

app.include_router(
    cities_router,
    prefix=settings.api_prefix,
)

app.include_router(
    forecast_router,
    prefix=settings.api_prefix,
)

app.include_router(
    history_router,
    prefix=settings.api_prefix,
)

app.include_router(
    performance_router,
    prefix=settings.api_prefix,
)

app.include_router(
    models_router,
    prefix=settings.api_prefix,
)


@app.get(
    "/",
    response_model=APIInfoResponse,
    tags=[
        "System",
    ],
    summary="API information",
)
def root(
) -> APIInfoResponse:
    return APIInfoResponse(
        name=settings.app_name,
        version=settings.app_version,
        environment=(
            settings.environment
        ),
        status="running",
    )