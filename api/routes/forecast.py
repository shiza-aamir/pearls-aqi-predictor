from __future__ import annotations

import logging
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_production_service
from api.schemas.forecast import (
    AlertResponse,
    CurrentAirQualityResponse,
    ExplanationResponse,
    FeatureContributionResponse,
    ForecastItemResponse,
    ForecastMetadataResponse,
    ForecastResponse,
    ModelResponse,
    PollutantsResponse,
    WeatherResponse,
)
from src.services.production_service import AQIProductionService


logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/forecast",
    tags=["Forecast"],
)


SUPPORTED_CITIES = {
    "faisalabad": "Faisalabad",
    "islamabad": "Islamabad",
    "karachi": "Karachi",
    "lahore": "Lahore",
    "multan": "Multan",
    "peshawar": "Peshawar",
    "quetta": "Quetta",
    "rahim yar khan": "Rahim Yar Khan",
    "sialkot": "Sialkot",
}


FEATURE_DISPLAY_NAMES = {
    "pm2_5": "PM2.5",
    "pm10": "PM10",
    "carbon_monoxide": "Carbon monoxide",
    "nitrogen_dioxide": "Nitrogen dioxide",
    "sulphur_dioxide": "Sulphur dioxide",
    "ozone": "Ozone",
    "temperature": "Temperature",
    "humidity": "Humidity",
    "precipitation": "Precipitation",
    "wind_speed": "Wind speed",
    "pressure": "Air pressure",
    "pm2_5_rolling_mean_6h": "PM2.5 average (6 hours)",
    "pm2_5_rolling_mean_24h": "PM2.5 average (24 hours)",
    "pm10_rolling_mean_6h": "PM10 average (6 hours)",
    "pm10_rolling_mean_24h": "PM10 average (24 hours)",
    "aqi_rolling_mean_3h": "AQI average (3 hours)",
    "aqi_rolling_mean_6h": "AQI average (6 hours)",
    "aqi_rolling_mean_12h": "AQI average (12 hours)",
    "aqi_rolling_mean_24h": "AQI average (24 hours)",
    "aqi_rolling_std_3h": "AQI variability (3 hours)",
    "aqi_rolling_std_6h": "AQI variability (6 hours)",
    "aqi_rolling_std_12h": "AQI variability (12 hours)",
    "aqi_rolling_std_24h": "AQI variability (24 hours)",
    "aqi_change_1h": "AQI change (1 hour)",
    "aqi_change_3h": "AQI change (3 hours)",
    "aqi_change_24h": "AQI change (24 hours)",
    "pm2_5_change_1h": "PM2.5 change (1 hour)",
    "pm10_change_1h": "PM10 change (1 hour)",
    "aqi_lag_1h": "AQI 1 hour ago",
    "aqi_lag_3h": "AQI 3 hours ago",
    "aqi_lag_6h": "AQI 6 hours ago",
    "aqi_lag_12h": "AQI 12 hours ago",
    "aqi_lag_24h": "AQI 24 hours ago",
    "aqi_lag_48h": "AQI 48 hours ago",
    "aqi_lag_72h": "AQI 72 hours ago",
    "month_cos": "Seasonal pattern",
    "month_sin": "Seasonal pattern",
    "hour_cos": "Time-of-day pattern",
    "hour_sin": "Time-of-day pattern",
    "day_of_week_cos": "Weekly pattern",
    "day_of_week_sin": "Weekly pattern",
    "latitude": "Location latitude",
    "longitude": "Location longitude",
    "stagnation_index": "Atmospheric stagnation",
    "temp_humidity_interaction": "Temperature-humidity interaction",
}


def normalize_city(city: str) -> str:
    normalized = " ".join(city.strip().lower().split())

    canonical = SUPPORTED_CITIES.get(normalized)

    if canonical is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "CITY_NOT_SUPPORTED",
                "message": f"Pearls does not currently support '{city}'.",
                "supported_cities": list(SUPPORTED_CITIES.values()),
            },
        )

    return canonical


def parse_horizon_hours(horizon: str) -> int:
    value = horizon.strip().lower()

    if value.endswith("h"):
        value = value[:-1]

    try:
        hours = int(value)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported forecast horizon returned by production service: {horizon}"
        ) from exc

    if hours not in {24, 48, 72}:
        raise ValueError(
            f"Unexpected forecast horizon returned by production service: {hours}"
        )

    return hours


def display_feature_name(feature: str) -> str:
    if feature in FEATURE_DISPLAY_NAMES:
        return FEATURE_DISPLAY_NAMES[feature]

    return feature.replace("_", " ").strip().title()


@router.get(
    "/{city}",
    response_model=ForecastResponse,
    summary="Get current AQI and 72-hour forecast",
    description=(
        "Runs the Pearls production inference pipeline for a supported city and "
        "returns the current AQI, environmental conditions, 24/48/72-hour "
        "forecasts, health alerts, and SHAP explanations."
    ),
)
def get_forecast(
    city: str,
    service: AQIProductionService = Depends(get_production_service),
) -> ForecastResponse:
    canonical_city = normalize_city(city)

    try:
        result = service.run(canonical_city)

        forecasts: list[ForecastItemResponse] = []

        for forecast in result.forecasts:
            horizon_hours = parse_horizon_hours(forecast.horizon)

            contributions = [
                FeatureContributionResponse(
                    feature=item.feature,
                    display_name=display_feature_name(item.feature),
                    feature_value=float(item.feature_value),
                    contribution=float(item.shap_value),
                    direction=item.direction,
                )
                for item in forecast.explanation.contributions
            ]

            forecasts.append(
                ForecastItemResponse(
                    horizon_hours=horizon_hours,
                    target_at=result.timestamp + timedelta(hours=horizon_hours),
                    aqi=float(forecast.predicted_aqi),
                    category=forecast.predicted_category,
                    alert=AlertResponse(
                        level=forecast.alert.level,
                        severity=int(forecast.alert.severity),
                        message=forecast.alert.message,
                    ),
                    model=ModelResponse(
                        name=forecast.model_name,
                        alias=forecast.model_alias,
                    ),
                    explanation=ExplanationResponse(
                        base_value=float(forecast.explanation.base_value),
                        top_features=contributions,
                    ),
                )
            )

        forecasts.sort(key=lambda item: item.horizon_hours)

        return ForecastResponse(
            city=result.city,
            observed_at=result.timestamp,
            timezone="UTC",
            current=CurrentAirQualityResponse(
                aqi=float(result.current_aqi),
                category=result.current_category,
                alert=AlertResponse(
                    level=result.current_alert.level,
                    severity=int(result.current_alert.severity),
                    message=result.current_alert.message,
                ),
                pollutants=PollutantsResponse(
                    pm2_5=float(result.pm2_5),
                    pm10=float(result.pm10),
                ),
                weather=WeatherResponse(
                    temperature_c=float(result.temperature),
                    humidity_percent=float(result.humidity),
                    wind_speed_ms=float(result.wind_speed),
                ),
            ),
            forecasts=forecasts,
            metadata=ForecastMetadataResponse(
                data_source=result.data_source,
                feature_count=int(result.feature_count),
                feature_store="Feast",
                model_registry="MLflow",
                history_rows=int(result.history_rows),
            ),
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Forecast request failed for city=%s",
            canonical_city,
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "FORECAST_UNAVAILABLE",
                "message": (
                    f"The latest forecast for {canonical_city} "
                    "is temporarily unavailable."
                ),
            },
        ) from exc