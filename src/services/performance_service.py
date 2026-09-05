from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.services.forecast_monitoring_service import (
    ForecastMonitoringService,
)


@dataclass(frozen=True)
class HoldoutPerformance:
    horizon_hours: int
    rows: int
    mae: float
    rmse: float
    r2: float
    median_absolute_error: float
    within_10_aqi_pct: float
    within_20_aqi_pct: float
    within_30_aqi_pct: float
    category_accuracy_pct: float
    persistence_mae: float
    mae_improvement_percent: float


@dataclass(frozen=True)
class LivePerformance:
    horizon_hours: int
    evaluated_forecasts: int
    mae: float | None
    rmse: float | None
    within_10_aqi_pct: float | None
    within_20_aqi_pct: float | None
    within_30_aqi_pct: float | None
    category_accuracy_pct: float | None
    adjacent_category_accuracy_pct: float | None


@dataclass(frozen=True)
class PerformanceResult:
    city: str
    holdout: tuple[
        HoldoutPerformance,
        ...,
    ]
    live: tuple[
        LivePerformance,
        ...,
    ]
    live_evaluated_forecasts: int
    live_status: str


class AQIPerformanceService:
    MANIFEST_PATH = Path(
        "artifacts/deployment/"
        "performance_manifest.json"
    )

    HORIZONS = (
        24,
        48,
        72,
    )

    def __init__(
        self,
        monitoring_service: (
            ForecastMonitoringService
            | None
        ) = None,
    ) -> None:
        self.monitoring_service = (
            monitoring_service
            or ForecastMonitoringService()
        )

    @classmethod
    def _load_manifest(
        cls,
    ) -> dict:
        if not cls.MANIFEST_PATH.exists():
            raise FileNotFoundError(
                "Deployment performance "
                "manifest is unavailable: "
                f"{cls.MANIFEST_PATH}"
            )

        with cls.MANIFEST_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            manifest = json.load(
                file
            )

        if not isinstance(
            manifest,
            dict,
        ):
            raise ValueError(
                "Performance deployment "
                "manifest must be an object."
            )

        if (
            manifest.get(
                "schema_version"
            )
            != 1
        ):
            raise ValueError(
                "Unsupported performance "
                "manifest schema."
            )

        return manifest

    def _load_holdout(
        self,
    ) -> tuple[
        HoldoutPerformance,
        ...,
    ]:
        manifest = (
            self._load_manifest()
        )

        rows = manifest.get(
            "horizons"
        )

        if not isinstance(
            rows,
            list,
        ):
            raise ValueError(
                "Performance manifest "
                "does not contain horizons."
            )

        results = []

        for row in rows:
            results.append(
                HoldoutPerformance(
                    horizon_hours=int(
                        row[
                            "horizon_hours"
                        ]
                    ),
                    rows=int(
                        row[
                            "rows"
                        ]
                    ),
                    mae=float(
                        row[
                            "mae"
                        ]
                    ),
                    rmse=float(
                        row[
                            "rmse"
                        ]
                    ),
                    r2=float(
                        row[
                            "r2"
                        ]
                    ),
                    median_absolute_error=float(
                        row[
                            "median_absolute_error"
                        ]
                    ),
                    within_10_aqi_pct=float(
                        row[
                            "within_10_aqi_pct"
                        ]
                    ),
                    within_20_aqi_pct=float(
                        row[
                            "within_20_aqi_pct"
                        ]
                    ),
                    within_30_aqi_pct=float(
                        row[
                            "within_30_aqi_pct"
                        ]
                    ),
                    category_accuracy_pct=float(
                        row[
                            "category_accuracy_pct"
                        ]
                    ),
                    persistence_mae=float(
                        row[
                            "persistence_mae"
                        ]
                    ),
                    mae_improvement_percent=float(
                        row[
                            "mae_improvement_percent"
                        ]
                    ),
                )
            )

        results = sorted(
            results,
            key=lambda item: (
                item.horizon_hours
            ),
        )

        if tuple(
            item.horizon_hours
            for item in results
        ) != self.HORIZONS:
            raise ValueError(
                "Performance manifest must "
                "contain exactly 24h, 48h, "
                "and 72h."
            )

        return tuple(
            results
        )

    def _load_live(
        self,
        city: str,
    ) -> tuple[
        LivePerformance,
        ...,
    ]:
        summary = (
            self.monitoring_service
            .performance_summary(
                city=city
            )
        )

        if summary.empty:
            return tuple(
                LivePerformance(
                    horizon_hours=horizon,
                    evaluated_forecasts=0,
                    mae=None,
                    rmse=None,
                    within_10_aqi_pct=None,
                    within_20_aqi_pct=None,
                    within_30_aqi_pct=None,
                    category_accuracy_pct=None,
                    adjacent_category_accuracy_pct=None,
                )
                for horizon
                in self.HORIZONS
            )

        lookup = {
            int(
                row.horizon_hours
            ): row
            for row
            in summary.itertuples(
                index=False
            )
        }

        results = []

        for horizon in self.HORIZONS:
            row = lookup.get(
                horizon
            )

            if row is None:
                results.append(
                    LivePerformance(
                        horizon_hours=horizon,
                        evaluated_forecasts=0,
                        mae=None,
                        rmse=None,
                        within_10_aqi_pct=None,
                        within_20_aqi_pct=None,
                        within_30_aqi_pct=None,
                        category_accuracy_pct=None,
                        adjacent_category_accuracy_pct=None,
                    )
                )

                continue

            results.append(
                LivePerformance(
                    horizon_hours=horizon,
                    evaluated_forecasts=int(
                        row.evaluated_forecasts
                    ),
                    mae=float(
                        row.mae
                    ),
                    rmse=float(
                        row.rmse
                    ),
                    within_10_aqi_pct=float(
                        row.within_10_aqi_pct
                    ),
                    within_20_aqi_pct=float(
                        row.within_20_aqi_pct
                    ),
                    within_30_aqi_pct=float(
                        row.within_30_aqi_pct
                    ),
                    category_accuracy_pct=float(
                        row.category_accuracy_pct
                    ),
                    adjacent_category_accuracy_pct=float(
                        row.adjacent_category_accuracy_pct
                    ),
                )
            )

        return tuple(
            results
        )

    def get_performance(
        self,
        city: str,
    ) -> PerformanceResult:
        holdout = (
            self._load_holdout()
        )

        live = (
            self._load_live(
                city
            )
        )

        evaluated = sum(
            item.evaluated_forecasts
            for item in live
        )

        if evaluated == 0:
            status = (
                "awaiting_matured_forecasts"
            )
        else:
            status = "available"

        return PerformanceResult(
            city=city,
            holdout=holdout,
            live=live,
            live_evaluated_forecasts=(
                evaluated
            ),
            live_status=status,
        )