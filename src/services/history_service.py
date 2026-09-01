from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.features.aqi.target_builder import (
    AQITargetBuilder,
)
from src.services.live_history_service import (
    LiveHistoryService,
)


@dataclass(frozen=True)
class HistoryStatistics:
    minimum: float
    maximum: float
    average: float
    standard_deviation: float


@dataclass(frozen=True)
class HistoryResult:
    city: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp

    requested_hours: int
    available_hours: int

    observations: pd.DataFrame
    aqi_statistics: HistoryStatistics


class AQIHistoryService:
    """
    Dashboard-facing recent-history service.

    Reads the persisted live observation history and derives
    hourly AQI values using the same AQITargetBuilder used by
    the rest of the Pearls pipeline.

    No external API request is performed by this service.
    """

    ALLOWED_HOURS = {
        24,
        48,
        72,
        168,
    }

    def __init__(
        self,
        history_service: (
            LiveHistoryService | None
        ) = None,
        target_builder: (
            AQITargetBuilder | None
        ) = None,
    ) -> None:
        self.history_service = (
            history_service
            or LiveHistoryService()
        )

        self.target_builder = (
            target_builder
            or AQITargetBuilder()
        )

    def get_history(
        self,
        city: str,
        hours: int = 168,
    ) -> HistoryResult:
        if hours not in self.ALLOWED_HOURS:
            raise ValueError(
                "History hours must be one of: "
                "24, 48, 72, 168."
            )

        history = (
            self.history_service.load(
                city
            )
        )

        if history.empty:
            raise ValueError(
                f"No live history is available "
                f"for {city}."
            )

        history = (
            history.copy()
        )

        history[
            "timestamp"
        ] = pd.to_datetime(
            history[
                "timestamp"
            ],
            utc=True,
            errors="raise",
        )

        (
            enriched,
            _,
        ) = self.target_builder.build(
            history,
            drop_incomplete_targets=False,
        )

        enriched[
            "timestamp"
        ] = pd.to_datetime(
            enriched[
                "timestamp"
            ],
            utc=True,
            errors="raise",
        )

        usable = (
            enriched.loc[
                enriched[
                    "aqi_current"
                ].notna()
            ]
            .copy()
        )

        if usable.empty:
            raise ValueError(
                f"No derived AQI history is "
                f"available for {city}."
            )

        selected = (
            usable
            .sort_values(
                "timestamp"
            )
            .tail(
                hours
            )
            .reset_index(
                drop=True
            )
        )

        aqi = pd.to_numeric(
            selected[
                "aqi_current"
            ],
            errors="raise",
        )

        statistics = (
            HistoryStatistics(
                minimum=float(
                    aqi.min()
                ),
                maximum=float(
                    aqi.max()
                ),
                average=float(
                    aqi.mean()
                ),
                standard_deviation=float(
                    aqi.std(
                        ddof=0
                    )
                ),
            )
        )

        return HistoryResult(
            city=city,
            start_time=pd.Timestamp(
                selected[
                    "timestamp"
                ].iloc[0]
            ),
            end_time=pd.Timestamp(
                selected[
                    "timestamp"
                ].iloc[-1]
            ),
            requested_hours=hours,
            available_hours=len(selected),
            observations=selected,
            aqi_statistics=statistics,
        )