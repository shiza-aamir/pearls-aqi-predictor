from __future__ import annotations

import logging

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    status,
)

from api.dependencies import (
    get_history_service,
)
from api.routes.forecast import (
    normalize_city,
)
from api.schemas.history import (
    HistoryObservationResponse,
    HistoryResponse,
    HistoryStatisticsResponse,
)
from src.services.history_service import (
    AQIHistoryService,
)


logger = logging.getLogger(
    __name__
)


router = APIRouter(
    prefix="/history",
    tags=[
        "History",
    ],
)


@router.get(
    "/{city}",
    response_model=HistoryResponse,
    summary="Get recent AQI history",
    description=(
        "Returns recent derived AQI, pollutant, "
        "and weather observations from the "
        "persisted Pearls live-history store."
    ),
)
def get_history(
    city: str,
    hours: int = Query(
        default=168,
        description=(
            "History window in hours. "
            "Supported values are "
            "24, 48, 72, and 168."
        ),
    ),
    service: AQIHistoryService = Depends(
        get_history_service
    ),
) -> HistoryResponse:
    canonical_city = (
        normalize_city(
            city
        )
    )

    if hours not in {
        24,
        48,
        72,
        168,
    }:
        raise HTTPException(
            status_code=(
                status
                .HTTP_422_UNPROCESSABLE_ENTITY
            ),
            detail={
                "code": (
                    "INVALID_HISTORY_RANGE"
                ),
                "message": (
                    "History range must be "
                    "one of 24, 48, 72, "
                    "or 168 hours."
                ),
            },
        )

    try:
        result = (
            service.get_history(
                city=canonical_city,
                hours=hours,
            )
        )

        observations = [
            HistoryObservationResponse(
                timestamp=(
                    row.timestamp
                ),
                aqi=float(
                    row.aqi_current
                ),
                category=str(
                    row.aqi_category_derived
                ),
                pm2_5=float(
                    row.pm2_5
                ),
                pm10=float(
                    row.pm10
                ),
                ozone=float(
                    row.ozone
                ),
                nitrogen_dioxide=float(
                    row.nitrogen_dioxide
                ),
                sulphur_dioxide=float(
                    row.sulphur_dioxide
                ),
                carbon_monoxide=float(
                    row.carbon_monoxide
                ),
                temperature_c=float(
                    row.temperature
                ),
                humidity_percent=float(
                    row.humidity
                ),
                wind_speed_ms=float(
                    row.wind_speed
                ),
                source=str(
                    row.source
                ),
            )
            for row
            in result.observations.itertuples(
                index=False
            )
        ]

        return HistoryResponse(
            city=result.city,
            start_time=(
                result.start_time
            ),
            end_time=(
                result.end_time
            ),
            requested_hours=(
                result.requested_hours
            ),
            available_hours=(
                result.available_hours
            ),
            statistics=(
                HistoryStatisticsResponse(
                    minimum=(
                        result
                        .aqi_statistics
                        .minimum
                    ),
                    maximum=(
                        result
                        .aqi_statistics
                        .maximum
                    ),
                    average=(
                        result
                        .aqi_statistics
                        .average
                    ),
                    standard_deviation=(
                        result
                        .aqi_statistics
                        .standard_deviation
                    ),
                )
            ),
            observations=observations,
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "History request failed "
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
                    "HISTORY_UNAVAILABLE"
                ),
                "message": (
                    "Recent history for "
                    f"{canonical_city} is "
                    "temporarily unavailable."
                ),
            },
        ) from exc