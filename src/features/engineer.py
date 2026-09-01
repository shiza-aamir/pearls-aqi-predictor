from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureEngineeringSummary:
    input_rows: int
    output_rows: int
    cities: int
    feature_count: int


class AQIFeatureEngineer:
    REQUIRED_COLUMNS = {
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
        "aqi_current",
    }

    AQI_LAGS = (1, 3, 6, 12, 24, 48, 72)
    PM_LAGS = (1, 3, 6, 24)

    AQI_ROLLING_WINDOWS = (3, 6, 12, 24)
    PM_ROLLING_WINDOWS = (6, 24)

    def transform(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, FeatureEngineeringSummary]:
        self._validate_input(df)

        result = df.copy()

        result["timestamp"] = pd.to_datetime(
            result["timestamp"],
            errors="raise",
        )

        result = result.sort_values(
            ["city", "timestamp"]
        ).reset_index(drop=True)

        self._validate_hourly_continuity(result)

        self._add_time_features(result)
        self._add_lag_features(result)
        self._add_rolling_features(result)
        self._add_change_features(result)
        self._add_weather_features(result)

        feature_columns = self.get_model_feature_columns()

        summary = FeatureEngineeringSummary(
            input_rows=len(df),
            output_rows=len(result),
            cities=int(result["city"].nunique()),
            feature_count=len(feature_columns),
        )

        return result, summary

    @staticmethod
    def _add_time_features(
        df: pd.DataFrame,
    ) -> None:
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["month"] = df["timestamp"].dt.month

        df["is_weekend"] = (
            df["day_of_week"] >= 5
        ).astype(int)

        df["hour_sin"] = np.sin(
            2 * np.pi * df["hour"] / 24
        )

        df["hour_cos"] = np.cos(
            2 * np.pi * df["hour"] / 24
        )

        df["day_of_week_sin"] = np.sin(
            2 * np.pi * df["day_of_week"] / 7
        )

        df["day_of_week_cos"] = np.cos(
            2 * np.pi * df["day_of_week"] / 7
        )

        df["month_sin"] = np.sin(
            2 * np.pi * df["month"] / 12
        )

        df["month_cos"] = np.cos(
            2 * np.pi * df["month"] / 12
        )

    def _add_lag_features(
        self,
        df: pd.DataFrame,
    ) -> None:
        grouped = df.groupby("city")

        for lag in self.AQI_LAGS:
            df[f"aqi_lag_{lag}h"] = (
                grouped["aqi_current"]
                .shift(lag)
            )

        for pollutant in ("pm2_5", "pm10"):
            for lag in self.PM_LAGS:
                df[f"{pollutant}_lag_{lag}h"] = (
                    grouped[pollutant]
                    .shift(lag)
                )

    def _add_rolling_features(
        self,
        df: pd.DataFrame,
    ) -> None:
        for window in self.AQI_ROLLING_WINDOWS:
            df[f"aqi_rolling_mean_{window}h"] = (
                df.groupby("city")["aqi_current"]
                .transform(
                    lambda series, window=window: series.rolling(
                        window=window,
                        min_periods=window,
                    ).mean()
                )
            )

            df[f"aqi_rolling_std_{window}h"] = (
                df.groupby("city")["aqi_current"]
                .transform(
                    lambda series, window=window: series.rolling(
                        window=window,
                        min_periods=window,
                    ).std()
                )
            )

        for pollutant in ("pm2_5", "pm10"):
            for window in self.PM_ROLLING_WINDOWS:
                df[
                    f"{pollutant}_rolling_mean_{window}h"
                ] = (
                    df.groupby("city")[pollutant]
                    .transform(
                        lambda series, window=window: series.rolling(
                            window=window,
                            min_periods=window,
                        ).mean()
                    )
                )

    @staticmethod
    def _add_change_features(
        df: pd.DataFrame,
    ) -> None:
        grouped = df.groupby("city")

        df["aqi_change_1h"] = (
            grouped["aqi_current"]
            .diff(1)
        )

        df["aqi_change_3h"] = (
            grouped["aqi_current"]
            .diff(3)
        )

        df["aqi_change_24h"] = (
            grouped["aqi_current"]
            .diff(24)
        )

        df["pm2_5_change_1h"] = (
            grouped["pm2_5"]
            .diff(1)
        )

        df["pm10_change_1h"] = (
            grouped["pm10"]
            .diff(1)
        )

    @staticmethod
    def _add_weather_features(
        df: pd.DataFrame,
    ) -> None:
        radians = np.deg2rad(
            df["wind_direction"]
        )

        df["wind_direction_sin"] = np.sin(
            radians
        )

        df["wind_direction_cos"] = np.cos(
            radians
        )

        df["temp_humidity_interaction"] = (
            df["temperature"]
            * df["humidity"]
        )

        df["stagnation_index"] = (
            df["humidity"]
            / (
                df["wind_speed"]
                + 1.0
            )
        )

    @classmethod
    def get_model_feature_columns(
        cls,
    ) -> list[str]:
        columns = [
            "latitude",
            "longitude",
            "temperature",
            "humidity",
            "precipitation",
            "wind_speed",
            "pressure",
            "pm2_5",
            "pm10",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
            "hour_sin",
            "hour_cos",
            "day_of_week_sin",
            "day_of_week_cos",
            "month_sin",
            "month_cos",
            "is_weekend",
            "wind_direction_sin",
            "wind_direction_cos",
            "temp_humidity_interaction",
            "stagnation_index",
        ]

        columns.extend(
            f"aqi_lag_{lag}h"
            for lag in cls.AQI_LAGS
        )

        columns.extend(
            f"{pollutant}_lag_{lag}h"
            for pollutant in ("pm2_5", "pm10")
            for lag in cls.PM_LAGS
        )

        columns.extend(
            f"aqi_rolling_mean_{window}h"
            for window in cls.AQI_ROLLING_WINDOWS
        )

        columns.extend(
            f"aqi_rolling_std_{window}h"
            for window in cls.AQI_ROLLING_WINDOWS
        )

        columns.extend(
            f"{pollutant}_rolling_mean_{window}h"
            for pollutant in ("pm2_5", "pm10")
            for window in cls.PM_ROLLING_WINDOWS
        )

        columns.extend(
            [
                "aqi_change_1h",
                "aqi_change_3h",
                "aqi_change_24h",
                "pm2_5_change_1h",
                "pm10_change_1h",
            ]
        )

        return columns

    def _validate_input(
        self,
        df: pd.DataFrame,
    ) -> None:
        missing = (
            self.REQUIRED_COLUMNS
            - set(df.columns)
        )

        if missing:
            raise ValueError(
                "Missing required columns: "
                + ", ".join(
                    sorted(missing)
                )
            )

        if df.empty:
            raise ValueError(
                "Input dataframe cannot be empty."
            )

        required_values = [
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

        if df[
            required_values
        ].isna().any().any():
            raise ValueError(
                "Required feature columns contain "
                "missing values."
            )

    @staticmethod
    def _validate_hourly_continuity(
        df: pd.DataFrame,
    ) -> None:
        gaps = (
            df.groupby("city")["timestamp"]
            .diff()
            .dropna()
        )

        invalid = gaps[
            gaps != pd.Timedelta(hours=1)
        ]

        if not invalid.empty:
            raise ValueError(
                "Input data must contain continuous "
                "hourly observations for every city."
            )