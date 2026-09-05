from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

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
    live_status: str
    live_evaluated_forecasts: int
    live: tuple[
        LivePerformance,
        ...,
    ]


class AQIPerformanceService:
    MANIFEST_PATH = Path(
        "artifacts/deployment/"
        "performance_manifest.json"
    )

    EXPECTED_HORIZONS = {
        24,
        48,
        72,
    }

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
    ) -> dict[str, Any]:
        if not cls.MANIFEST_PATH.exists():
            raise FileNotFoundError(
                "Performance deployment "
                "manifest does not exist: "
                f"{cls.MANIFEST_PATH}"
            )

        with cls.MANIFEST_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(
                file
            )

        if not isinstance(
            data,
            dict,
        ):
            raise TypeError(
                "Performance deployment "
                "manifest must be an object."
            )

        if (
            data.get(
                "schema_version"
            )
            != 1
        ):
            raise ValueError(
                "Unsupported performance "
                "manifest schema version."
            )

        return data

    def _load_holdout(
        self,
    ) -> tuple[
        HoldoutPerformance,
        ...,
    ]:
        manifest = (
            self._load_manifest()
        )

        raw_horizons = (
            manifest.get(
                "horizons"
            )
        )

        if not isinstance(
            raw_horizons,
            list,
        ):
            raise TypeError(
                "Performance manifest "
                "does not contain horizons."
            )

        results: list[
            HoldoutPerformance
        ] = []

        found_horizons: set[
            int
        ] = set()

        for item in raw_horizons:
            if not isinstance(
                item,
                dict,
            ):
                raise TypeError(
                    "Performance horizon "
                    "entry must be an object."
                )

            horizon_hours = int(
                item[
                    "horizon_hours"
                ]
            )

            if (
                horizon_hours
                in found_horizons
            ):
                raise ValueError(
                    "Duplicate performance "
                    f"horizon: {horizon_hours}"
                )

            found_horizons.add(
                horizon_hours
            )

            results.append(
                HoldoutPerformance(
                    horizon_hours=(
                        horizon_hours
                    ),
                    rows=int(
                        item[
                            "rows"
                        ]
                    ),
                    mae=float(
                        item[
                            "mae"
                        ]
                    ),
                    rmse=float(
                        item[
                            "rmse"
                        ]
                    ),
                    r2=float(
                        item[
                            "r2"
                        ]
                    ),
                    median_absolute_error=float(
                        item[
                            "median_absolute_error"
                        ]
                    ),
                    within_10_aqi_pct=float(
                        item[
                            "within_10_aqi_pct"
                        ]
                    ),
                    within_20_aqi_pct=float(
                        item[
                            "within_20_aqi_pct"
                        ]
                    ),
                    within_30_aqi_pct=float(
                        item[
                            "within_30_aqi_pct"
                        ]
                    ),
                    category_accuracy_pct=float(
                        item[
                            "category_accuracy_pct"
                        ]
                    ),
                    persistence_mae=float(
                        item[
                            "persistence_mae"
                        ]
                    ),
                    mae_improvement_percent=float(
                        item[
                            "mae_improvement_percent"
                        ]
                    ),
                )
            )

        if (
            found_horizons
            != self.EXPECTED_HORIZONS
        ):
            raise ValueError(
                "Performance manifest must "
                "contain exactly 24h, 48h, "
                "and 72h. "
                f"Found: {sorted(found_horizons)}"
            )

        return tuple(
            sorted(
                results,
                key=lambda result: (
                    result.horizon_hours
                ),
            )
        )

    @staticmethod
    def _empty_live_result(
        horizon_hours: int,
    ) -> LivePerformance:
        return LivePerformance(
            horizon_hours=(
                horizon_hours
            ),
            evaluated_forecasts=0,
            mae=None,
            rmse=None,
            within_10_aqi_pct=None,
            within_20_aqi_pct=None,
            within_30_aqi_pct=None,
            category_accuracy_pct=None,
            adjacent_category_accuracy_pct=None,
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
                city
            )
        )

        if not isinstance(
            summary,
            pd.DataFrame,
        ):
            raise TypeError(
                "Forecast monitoring performance "
                "summary must be a pandas DataFrame."
            )

        if summary.empty:
            return tuple(
                self._empty_live_result(
                    horizon_hours
                )
                for horizon_hours
                in (
                    24,
                    48,
                    72,
                )
            )

        required_columns = {
            "horizon_hours",
            "evaluated_forecasts",
            "mae",
            "rmse",
            "within_10_aqi_pct",
            "within_20_aqi_pct",
            "within_30_aqi_pct",
            "category_accuracy_pct",
            "adjacent_category_accuracy_pct",
        }

        missing_columns = (
            required_columns
            - set(
                summary.columns
            )
        )

        if missing_columns:
            raise ValueError(
                "Live performance summary "
                "is missing columns: "
                f"{sorted(missing_columns)}"
            )

        rows_by_horizon = {
            int(
                row.horizon_hours
            ): row
            for row
            in summary.itertuples(
                index=False
            )
        }

        results: list[
            LivePerformance
        ] = []

        for horizon_hours in (
            24,
            48,
            72,
        ):
            row = rows_by_horizon.get(
                horizon_hours
            )

            if row is None:
                results.append(
                    self._empty_live_result(
                        horizon_hours
                    )
                )

                continue

            results.append(
                LivePerformance(
                    horizon_hours=(
                        horizon_hours
                    ),
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

        live_evaluated_forecasts = sum(
            item.evaluated_forecasts
            for item in live
        )

        if (
            live_evaluated_forecasts
            == 0
        ):
            live_status = (
                "awaiting_matured_forecasts"
            )
        else:
            live_status = (
                "live_metrics_available"
            )

        return PerformanceResult(
            city=(
                city
            ),
            holdout=(
                holdout
            ),
            live_status=(
                live_status
            ),
            live_evaluated_forecasts=(
                live_evaluated_forecasts
            ),
            live=(
                live
            ),
        )