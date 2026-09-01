from __future__ import annotations

import logging

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from api.dependencies import (
    get_performance_service,
)
from api.routes.forecast import (
    normalize_city,
)
from api.schemas.performance import (
    HoldoutPerformanceResponse,
    LivePerformanceResponse,
    PerformanceResponse,
)
from src.services.performance_service import (
    AQIPerformanceService,
)

logger = logging.getLogger(
    __name__
)


router = APIRouter(
    prefix="/performance",
    tags=[
        "Performance",
    ],
)


@router.get(
    "/{city}",
    response_model=PerformanceResponse,
    summary="Get model performance",
    description=(
        "Returns the frozen final-holdout "
        "evaluation together with genuine "
        "post-deployment live forecast "
        "performance for the selected city."
    ),
)
def get_performance(
    city: str,
    service: AQIPerformanceService = Depends(
        get_performance_service
    ),
) -> PerformanceResponse:
    canonical_city = (
        normalize_city(
            city
        )
    )

    try:
        result = (
            service.get_performance(
                canonical_city
            )
        )

        holdout = [
            HoldoutPerformanceResponse(
                horizon_hours=(
                    item.horizon_hours
                ),
                rows=item.rows,
                mae=item.mae,
                rmse=item.rmse,
                r2=item.r2,
                median_absolute_error=(
                    item.median_absolute_error
                ),
                within_10_aqi_pct=(
                    item.within_10_aqi_pct
                ),
                within_20_aqi_pct=(
                    item.within_20_aqi_pct
                ),
                within_30_aqi_pct=(
                    item.within_30_aqi_pct
                ),
                category_accuracy_pct=(
                    item.category_accuracy_pct
                ),
                persistence_mae=(
                    item.persistence_mae
                ),
                mae_improvement_percent=(
                    item.mae_improvement_percent
                ),
            )
            for item in result.holdout
        ]

        live = [
            LivePerformanceResponse(
                horizon_hours=(
                    item.horizon_hours
                ),
                evaluated_forecasts=(
                    item.evaluated_forecasts
                ),
                mae=item.mae,
                rmse=item.rmse,
                within_10_aqi_pct=(
                    item.within_10_aqi_pct
                ),
                within_20_aqi_pct=(
                    item.within_20_aqi_pct
                ),
                within_30_aqi_pct=(
                    item.within_30_aqi_pct
                ),
                category_accuracy_pct=(
                    item.category_accuracy_pct
                ),
                adjacent_category_accuracy_pct=(
                    item
                    .adjacent_category_accuracy_pct
                ),
            )
            for item in result.live
        ]

        return PerformanceResponse(
            city=result.city,
            holdout_evaluation_label=(
                "Frozen final holdout"
            ),
            holdout=holdout,
            live_status=(
                result.live_status
            ),
            live_evaluated_forecasts=(
                result.live_evaluated_forecasts
            ),
            live=live,
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Performance request failed "
            "for city=%s",
            canonical_city,
        )

        raise HTTPException(
            status_code=(
                status
                .HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail={
                "code": (
                    "PERFORMANCE_UNAVAILABLE"
                ),
                "message": (
                    "Performance information "
                    "is temporarily unavailable."
                ),
            },
        ) from exc