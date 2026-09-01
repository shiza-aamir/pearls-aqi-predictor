from dataclasses import dataclass

import pandas as pd

from src.features.aqi.calculator import AQICalculator


@dataclass(frozen=True)
class TargetBuildSummary:
    input_rows: int
    output_rows: int
    cities: int
    rows_with_current_aqi: int
    rows_with_all_targets: int


class AQITargetBuilder:
    REQUIRED_COLUMNS = {
        "city",
        "timestamp",
        "pm2_5",
        "pm10",
    }

    ROLLING_WINDOW_HOURS = 24

    TARGET_HORIZONS = {
        "target_aqi_24h": 24,
        "target_aqi_48h": 48,
        "target_aqi_72h": 72,
    }

    def build(
        self,
        df: pd.DataFrame,
        drop_incomplete_targets: bool = False,
    ) -> tuple[pd.DataFrame, TargetBuildSummary]:
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

        result["pm2_5_24h_mean"] = (
            result.groupby("city")["pm2_5"]
            .transform(
                lambda series: series.rolling(
                    window=self.ROLLING_WINDOW_HOURS,
                    min_periods=self.ROLLING_WINDOW_HOURS,
                ).mean()
            )
        )

        result["pm10_24h_mean"] = (
            result.groupby("city")["pm10"]
            .transform(
                lambda series: series.rolling(
                    window=self.ROLLING_WINDOW_HOURS,
                    min_periods=self.ROLLING_WINDOW_HOURS,
                ).mean()
            )
        )

        valid_aqi_mask = (
            result["pm2_5_24h_mean"].notna()
            & result["pm10_24h_mean"].notna()
        )

        result["aqi_current"] = pd.NA
        result["aqi_category_derived"] = pd.NA
        result["dominant_pollutant"] = pd.NA
        result["pm2_5_aqi"] = pd.NA
        result["pm10_aqi"] = pd.NA

        for index in result.index[valid_aqi_mask]:
            aqi_result = AQICalculator.calculate_aqi(
                pm25=float(
                    result.at[index, "pm2_5_24h_mean"]
                ),
                pm10=float(
                    result.at[index, "pm10_24h_mean"]
                ),
            )

            result.at[index, "aqi_current"] = aqi_result.aqi
            result.at[
                index,
                "aqi_category_derived",
            ] = aqi_result.category
            result.at[
                index,
                "dominant_pollutant",
            ] = aqi_result.dominant_pollutant
            result.at[
                index,
                "pm2_5_aqi",
            ] = aqi_result.pm25_aqi
            result.at[
                index,
                "pm10_aqi",
            ] = aqi_result.pm10_aqi

        result["aqi_current"] = pd.to_numeric(
            result["aqi_current"],
            errors="coerce",
        )

        result["pm2_5_aqi"] = pd.to_numeric(
            result["pm2_5_aqi"],
            errors="coerce",
        )

        result["pm10_aqi"] = pd.to_numeric(
            result["pm10_aqi"],
            errors="coerce",
        )

        for target_name, horizon in self.TARGET_HORIZONS.items():
            result[target_name] = (
                result.groupby("city")["aqi_current"]
                .shift(-horizon)
            )

        rows_with_current_aqi = int(
            result["aqi_current"].notna().sum()
        )

        target_columns = list(
            self.TARGET_HORIZONS.keys()
        )

        complete_target_mask = result[
            target_columns
        ].notna().all(axis=1)

        rows_with_all_targets = int(
            complete_target_mask.sum()
        )

        if drop_incomplete_targets:
            result = result.loc[
                valid_aqi_mask & complete_target_mask
            ].reset_index(drop=True)

        summary = TargetBuildSummary(
            input_rows=len(df),
            output_rows=len(result),
            cities=int(result["city"].nunique()),
            rows_with_current_aqi=rows_with_current_aqi,
            rows_with_all_targets=rows_with_all_targets,
        )

        return result, summary

    def _validate_input(
        self,
        df: pd.DataFrame,
    ) -> None:
        missing_columns = (
            self.REQUIRED_COLUMNS
            - set(df.columns)
        )

        if missing_columns:
            missing = ", ".join(
                sorted(missing_columns)
            )

            raise ValueError(
                f"Missing required columns: {missing}"
            )

        if df.empty:
            raise ValueError(
                "Input dataframe cannot be empty."
            )

        if df[
            ["city", "timestamp", "pm2_5", "pm10"]
        ].isna().any().any():
            raise ValueError(
                "Required AQI target columns contain missing values."
            )

        if (df["pm2_5"] < 0).any():
            raise ValueError(
                "PM2.5 contains negative concentrations."
            )

        if (df["pm10"] < 0).any():
            raise ValueError(
                "PM10 contains negative concentrations."
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

        invalid_gaps = gaps[
            gaps != pd.Timedelta(hours=1)
        ]

        if not invalid_gaps.empty:
            raise ValueError(
                "Input data must contain continuous hourly "
                "observations for every city."
            )