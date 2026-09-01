from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.aqi.target_builder import (
    AQITargetBuilder,
)
from src.features.engineer import (
    AQIFeatureEngineer,
)


class LiveFeaturePipeline:
    def __init__(self) -> None:
        self.target_builder = (
            AQITargetBuilder()
        )

        self.feature_engineer = (
            AQIFeatureEngineer()
        )

    def build_latest_features(
        self,
        history: pd.DataFrame,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
    ]:
        working = history.copy()

        working["timestamp"] = pd.to_datetime(
            working["timestamp"],
            utc=True,
            errors="raise",
        )

        working = (
            working
            .sort_values(
                ["city", "timestamp"]
            )
            .reset_index(drop=True)
        )

        target_input = working[
            [
                "city",
                "timestamp",
                "latitude",
                "longitude",
                "temperature",
                "humidity",
                "precipitation",
                "wind_speed",
                "wind_direction",
                "pressure",
                "pm2_5",
                "pm10",
                "carbon_monoxide",
                "nitrogen_dioxide",
                "sulphur_dioxide",
                "ozone",
            ]
        ].copy()

        with_aqi, _ = (
            self.target_builder.build(
                target_input,
                drop_incomplete_targets=False,
            )
        )

        # AQI is unavailable for the first 23 raw hours.
        # Feature engineering requires non-null AQI history,
        # so begin from the first valid AQI observation.
        with_aqi = (
            with_aqi.loc[
                with_aqi[
                    "aqi_current"
                ].notna()
            ]
            .reset_index(drop=True)
        )

        if len(with_aqi) < 73:
            raise RuntimeError(
                "Insufficient valid AQI history after "
                "24-hour AQI construction. "
                f"Only {len(with_aqi)} rows available."
            )

        engineered, _ = (
            self.feature_engineer.transform(
                with_aqi
            )
        )

        feature_columns = (
            self.feature_engineer
            .get_model_feature_columns()
        )

        latest = (
            engineered
            .tail(1)
            .copy()
        )

        latest_features = latest[
            feature_columns
        ].copy()

        if latest_features.shape != (
            1,
            56,
        ):
            raise RuntimeError(
                "Expected latest feature shape "
                f"(1, 56), got {latest_features.shape}."
            )

        if (
            latest_features
            .isnull()
            .any()
            .any()
        ):
            missing = (
                latest_features
                .columns[
                    latest_features
                    .isnull()
                    .any()
                ]
                .tolist()
            )

            raise RuntimeError(
                "Latest live feature row contains "
                f"nulls: {missing}"
            )

        values = (
            latest_features
            .astype(float)
            .to_numpy()
        )

        if not np.isfinite(
            values
        ).all():
            raise RuntimeError(
                "Latest live feature row contains "
                "non-finite values."
            )

        return latest, latest_features