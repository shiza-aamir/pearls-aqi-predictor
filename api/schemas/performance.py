from __future__ import annotations

from pydantic import (
    BaseModel,
    Field,
)


class HoldoutPerformanceResponse(
    BaseModel
):
    horizon_hours: int

    rows: int

    mae: float
    rmse: float
    r2: float

    median_absolute_error: float

    within_10_aqi_pct: float = Field(
        ge=0,
        le=100,
    )

    within_20_aqi_pct: float = Field(
        ge=0,
        le=100,
    )

    within_30_aqi_pct: float = Field(
        ge=0,
        le=100,
    )

    category_accuracy_pct: float = Field(
        ge=0,
        le=100,
    )

    persistence_mae: float

    mae_improvement_percent: float


class LivePerformanceResponse(
    BaseModel
):
    horizon_hours: int

    evaluated_forecasts: int = Field(
        ge=0
    )

    status: str

    next_maturity_at: (
        str | None
    ) = None

    mae: float | None = None
    rmse: float | None = None

    within_10_aqi_pct: (
        float | None
    ) = None

    within_20_aqi_pct: (
        float | None
    ) = None

    within_30_aqi_pct: (
        float | None
    ) = None

    category_accuracy_pct: (
        float | None
    ) = None

    adjacent_category_accuracy_pct: (
        float | None
    ) = None


class PerformanceResponse(
    BaseModel
):
    city: str

    holdout_evaluation_label: str

    holdout: list[
        HoldoutPerformanceResponse
    ]

    live_status: str

    live_evaluated_forecasts: int

    live: list[
        LivePerformanceResponse
    ]