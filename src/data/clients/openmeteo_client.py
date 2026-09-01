from __future__ import annotations

import pandas as pd
import requests

from src.core.settings import CityConfig


class OpenMeteoClient:
    WEATHER_URL = (
        "https://historical-forecast-api.open-meteo.com/v1/forecast"
    )

    AIR_QUALITY_URL = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
    )

    REQUEST_TIMEOUT_SECONDS = 30

    WEATHER_VARIABLES = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "wind_speed_10m",
        "wind_direction_10m",
        "surface_pressure",
    ]

    POLLUTION_VARIABLES = [
        "pm2_5",
        "pm10",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
    ]

    def get_recent_history(
        self,
        city: CityConfig,
        hours: int = 120,
    ) -> pd.DataFrame:
        if hours < 97:
            raise ValueError(
                "At least 97 raw hourly observations are "
                "required for the current feature pipeline."
            )

        weather = self._get_weather(
            city=city,
            hours=hours,
        )

        pollution = self._get_pollution(
            city=city,
            hours=hours,
        )

        merged = weather.merge(
            pollution,
            on="timestamp",
            how="inner",
            validate="one_to_one",
        )

        merged["city"] = city.name
        merged["latitude"] = city.latitude
        merged["longitude"] = city.longitude
        merged["source"] = "open_meteo_bootstrap"

        required = [
            "timestamp",
            "city",
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
            "source",
        ]

        merged = (
            merged[required]
            .sort_values("timestamp")
            .drop_duplicates(
                subset=["timestamp"],
                keep="last",
            )
            .reset_index(drop=True)
        )

        # Keep only fully completed UTC hours.
        now_hour = pd.Timestamp.now(
            tz="UTC"
        ).floor("h")

        merged = merged.loc[
            merged["timestamp"] < now_hour
        ].reset_index(drop=True)

        if len(merged) < hours:
            raise RuntimeError(
                f"Only {len(merged)} completed common hourly "
                f"observations were returned; expected at least {hours}."
            )

        # Retain the latest requested completed hours.
        merged = (
            merged.tail(hours)
            .reset_index(drop=True)
        )

        self._validate_history(
            merged,
            expected_hours=hours,
        )

        return merged

    def _get_weather(
        self,
        city: CityConfig,
        hours: int,
    ) -> pd.DataFrame:
        response = requests.get(
            self.WEATHER_URL,
            params={
                "latitude": city.latitude,
                "longitude": city.longitude,
                "hourly": ",".join(
                    self.WEATHER_VARIABLES
                ),
                "past_hours": hours + 6,
                "forecast_hours": 0,
                "timezone": "UTC",
                "wind_speed_unit": "ms",
            },
            timeout=self.REQUEST_TIMEOUT_SECONDS,
        )

        response.raise_for_status()
        payload = response.json()

        hourly = payload.get("hourly")

        if not hourly:
            raise RuntimeError(
                "Open-Meteo returned no historical weather data."
            )

        frame = pd.DataFrame(hourly)

        frame["timestamp"] = pd.to_datetime(
            frame["time"],
            utc=True,
            errors="raise",
        )

        frame = frame.rename(
            columns={
                "temperature_2m": "temperature",
                "relative_humidity_2m": "humidity",
                "wind_speed_10m": "wind_speed",
                "wind_direction_10m": "wind_direction",
                "surface_pressure": "pressure",
            }
        )

        return frame[
            [
                "timestamp",
                "temperature",
                "humidity",
                "precipitation",
                "wind_speed",
                "wind_direction",
                "pressure",
            ]
        ]

    def _get_pollution(
        self,
        city: CityConfig,
        hours: int,
    ) -> pd.DataFrame:
        response = requests.get(
            self.AIR_QUALITY_URL,
            params={
                "latitude": city.latitude,
                "longitude": city.longitude,
                "hourly": ",".join(
                    self.POLLUTION_VARIABLES
                ),
                "past_hours": hours + 6,
                "forecast_hours": 0,
                "timezone": "UTC",
            },
            timeout=self.REQUEST_TIMEOUT_SECONDS,
        )

        response.raise_for_status()
        payload = response.json()

        hourly = payload.get("hourly")

        if not hourly:
            raise RuntimeError(
                "Open-Meteo returned no air-quality history."
            )

        frame = pd.DataFrame(hourly)

        frame["timestamp"] = pd.to_datetime(
            frame["time"],
            utc=True,
            errors="raise",
        )

        return frame[
            [
                "timestamp",
                "pm2_5",
                "pm10",
                "carbon_monoxide",
                "nitrogen_dioxide",
                "sulphur_dioxide",
                "ozone",
            ]
        ]

    @staticmethod
    def _validate_history(
        df: pd.DataFrame,
        expected_hours: int,
    ) -> None:
        if len(df) != expected_hours:
            raise RuntimeError(
                f"Expected {expected_hours} rows, got {len(df)}."
            )

        if df.isnull().any().any():
            missing = (
                df.columns[
                    df.isnull().any()
                ]
                .tolist()
            )

            raise RuntimeError(
                f"Bootstrap contains missing values: {missing}"
            )

        differences = (
            df["timestamp"]
            .diff()
            .dropna()
        )

        invalid = differences[
            differences != pd.Timedelta(hours=1)
        ]

        if not invalid.empty:
            raise RuntimeError(
                "Bootstrap history is not continuous hourly data."
            )

        pollutant_columns = [
            "pm2_5",
            "pm10",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
        ]

        if (
            df[pollutant_columns] < 0
        ).any().any():
            raise RuntimeError(
                "Bootstrap contains negative pollutant values."
            )