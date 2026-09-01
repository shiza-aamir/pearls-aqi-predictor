from __future__ import annotations

from datetime import datetime

from pydantic import (
    BaseModel,
    Field,
)


class HistoryObservationResponse(
    BaseModel
):
    timestamp: datetime

    aqi: float = Field(
        ge=0,
        le=500,
    )

    category: str

    pm2_5: float = Field(
        ge=0
    )

    pm10: float = Field(
        ge=0
    )

    ozone: float = Field(
        ge=0
    )

    nitrogen_dioxide: float = Field(
        ge=0
    )

    sulphur_dioxide: float = Field(
        ge=0
    )

    carbon_monoxide: float = Field(
        ge=0
    )

    temperature_c: float

    humidity_percent: float = Field(
        ge=0,
        le=100,
    )

    wind_speed_ms: float = Field(
        ge=0
    )

    source: str


class HistoryStatisticsResponse(
    BaseModel
):
    minimum: float
    maximum: float
    average: float
    standard_deviation: float


class HistoryResponse(
    BaseModel
):
    city: str

    start_time: datetime
    end_time: datetime

    requested_hours: int
    available_hours: int

    statistics: (
        HistoryStatisticsResponse
    )

    observations: list[
        HistoryObservationResponse
    ]