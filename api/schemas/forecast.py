from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class AlertResponse(BaseModel):
    level: str
    severity: int = Field(ge=0)
    message: str


class PollutantsResponse(BaseModel):
    pm2_5: float = Field(ge=0)
    pm10: float = Field(ge=0)


class WeatherResponse(BaseModel):
    temperature_c: float
    humidity_percent: float = Field(ge=0, le=100)
    wind_speed_ms: float = Field(ge=0)


class CurrentAirQualityResponse(BaseModel):
    aqi: float = Field(ge=0, le=500)
    category: str
    alert: AlertResponse
    pollutants: PollutantsResponse
    weather: WeatherResponse


class ModelResponse(BaseModel):
    name: str
    alias: str


class FeatureContributionResponse(BaseModel):
    feature: str
    display_name: str
    feature_value: float
    contribution: float
    direction: str


class ExplanationResponse(BaseModel):
    base_value: float
    top_features: list[FeatureContributionResponse]


class ForecastItemResponse(BaseModel):
    horizon_hours: int = Field(gt=0)
    target_at: datetime
    aqi: float = Field(ge=0, le=500)
    category: str
    alert: AlertResponse
    model: ModelResponse
    explanation: ExplanationResponse


class ForecastMetadataResponse(BaseModel):
    data_source: str
    feature_count: int
    feature_store: str
    model_registry: str
    history_rows: int


class ForecastResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "city": "Islamabad",
                "observed_at": "2026-08-30T08:00:00Z",
                "timezone": "UTC",
                "current": {
                    "aqi": 126.0,
                    "category": "Unhealthy for Sensitive Groups",
                    "alert": {
                        "level": "Sensitive Group Advisory",
                        "severity": 2,
                        "message": "Sensitive groups should reduce prolonged outdoor exertion.",
                    },
                    "pollutants": {
                        "pm2_5": 84.84,
                        "pm10": 193.3,
                    },
                    "weather": {
                        "temperature_c": 35.46,
                        "humidity_percent": 62.0,
                        "wind_speed_ms": 2.24,
                    },
                },
                "forecasts": [],
                "metadata": {
                    "data_source": "openweather_live",
                    "feature_count": 56,
                    "feature_store": "Feast",
                    "model_registry": "MLflow",
                    "history_rows": 121,
                },
            }
        }
    )

    city: str
    observed_at: datetime
    timezone: str
    current: CurrentAirQualityResponse
    forecasts: list[ForecastItemResponse]
    metadata: ForecastMetadataResponse