from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.core.settings import get_city
from src.data.clients.openmeteo_client import (
    OpenMeteoClient,
)
from src.data.clients.openweather_client import (
    OpenWeatherClient,
)


class LiveHistoryService:
    HISTORY_ROOT = Path(
        "data/live/history"
    )

    BOOTSTRAP_HOURS = 120
    MAX_HISTORY_HOURS = 168

    def __init__(
        self,
        openmeteo_client: OpenMeteoClient | None = None,
        openweather_client: OpenWeatherClient | None = None,
    ) -> None:
        self.openmeteo_client = (
            openmeteo_client
            or OpenMeteoClient()
        )

        self.openweather_client = (
            openweather_client
            or OpenWeatherClient()
        )

        self.HISTORY_ROOT.mkdir(
            parents=True,
            exist_ok=True,
        )

    def bootstrap(
        self,
        city_name: str,
        force: bool = False,
    ) -> pd.DataFrame:
        path = self.get_history_path(
            city_name
        )

        if path.exists() and not force:
            return self.load(
                city_name
            )

        city = get_city(
            city_name
        )

        history = (
            self.openmeteo_client
            .get_recent_history(
                city=city,
                hours=self.BOOTSTRAP_HOURS,
            )
        )

        self.validate_history(
            history
        )

        self.save(
            city_name,
            history,
        )

        return history

    def ensure_current_history(
        self,
        city_name: str,
    ) -> pd.DataFrame:
        path = self.get_history_path(
            city_name
        )

        if not path.exists():
            self.bootstrap(
                city_name,
                force=True,
            )

        history = self.load(
            city_name
        )

        city = get_city(
            city_name
        )

        weather = (
            self.openweather_client
            .get_current_weather(
                city
            )
        )

        pollution = (
            self.openweather_client
            .get_current_pollution(
                city
            )
        )

        weather_hour = self._hour_bucket(
            weather.timestamp
        )

        pollution_hour = self._hour_bucket(
            pollution.timestamp
        )

        if weather_hour != pollution_hour:
            raise RuntimeError(
                "OpenWeather weather and pollution "
                "belong to different UTC hours: "
                f"{weather_hour} vs {pollution_hour}."
            )

        latest_hour = (
            history["timestamp"].max()
        )

        gap_hours = int(
            (
                weather_hour
                - latest_hour
            ).total_seconds()
            // 3600
        )

        if gap_hours > 1:
            history = self._recover_gap(
                city_name=city_name,
                existing_history=history,
            )

        return self._upsert_openweather_observation(
            city_name=city_name,
            history=history,
            weather=weather,
            pollution=pollution,
        )

    def update_from_openweather(
        self,
        city_name: str,
    ) -> pd.DataFrame:
        return self.ensure_current_history(
            city_name
        )

    def _recover_gap(
        self,
        city_name: str,
        existing_history: pd.DataFrame,
    ) -> pd.DataFrame:
        city = get_city(
            city_name
        )

        recovered = (
            self.openmeteo_client
            .get_recent_history(
                city=city,
                hours=self.BOOTSTRAP_HOURS,
            )
        )

        existing = (
            existing_history.copy()
        )

        existing["timestamp"] = pd.to_datetime(
            existing["timestamp"],
            utc=True,
            errors="raise",
        )

        recovered["timestamp"] = pd.to_datetime(
            recovered["timestamp"],
            utc=True,
            errors="raise",
        )

        # Existing OpenWeather observations should win over
        # bootstrap observations for the same hour.
        live_rows = existing.loc[
            existing["source"]
            == "openweather_live"
        ].copy()

        combined = pd.concat(
            [
                recovered,
                live_rows,
            ],
            ignore_index=True,
        )

        source_priority = {
            "open_meteo_bootstrap": 0,
            "openweather_live": 1,
        }

        combined["_source_priority"] = (
            combined["source"]
            .map(source_priority)
            .fillna(0)
        )

        combined = (
            combined
            .sort_values(
                [
                    "timestamp",
                    "_source_priority",
                ]
            )
            .drop_duplicates(
                subset=["timestamp"],
                keep="last",
            )
            .drop(
                columns=[
                    "_source_priority"
                ]
            )
            .sort_values("timestamp")
            .tail(self.MAX_HISTORY_HOURS)
            .reset_index(drop=True)
        )

        self.validate_history(
            combined
        )

        self.save(
            city_name,
            combined,
        )

        return combined

    def _upsert_openweather_observation(
        self,
        city_name: str,
        history: pd.DataFrame,
        weather,
        pollution,
    ) -> pd.DataFrame:
        city = get_city(
            city_name
        )

        weather_hour = self._hour_bucket(
            weather.timestamp
        )

        pollution_hour = self._hour_bucket(
            pollution.timestamp
        )

        if weather_hour != pollution_hour:
            raise RuntimeError(
                "Weather/pollution hour mismatch."
            )

        history = history.copy()

        history["timestamp"] = pd.to_datetime(
            history["timestamp"],
            utc=True,
            errors="raise",
        )

        latest_hour = (
            history["timestamp"].max()
        )

        if (
            weather_hour
            > latest_hour
            + pd.Timedelta(hours=1)
        ):
            raise RuntimeError(
                "Gap recovery failed. "
                f"Latest history={latest_hour}, "
                f"OpenWeather={weather_hour}."
            )

        observation = pd.DataFrame(
            [
                {
                    "timestamp": weather_hour,
                    "city": city.name,
                    "latitude": city.latitude,
                    "longitude": city.longitude,
                    "temperature": weather.temperature,
                    "humidity": weather.humidity,
                    "precipitation": weather.precipitation,
                    "wind_speed": weather.wind_speed,
                    "wind_direction": weather.wind_direction,
                    "pressure": weather.pressure,
                    "pm2_5": pollution.pm2_5,
                    "pm10": pollution.pm10,
                    "carbon_monoxide": (
                        pollution.carbon_monoxide
                    ),
                    "nitrogen_dioxide": (
                        pollution.nitrogen_dioxide
                    ),
                    "sulphur_dioxide": (
                        pollution.sulphur_dioxide
                    ),
                    "ozone": pollution.ozone,
                    "source": "openweather_live",
                }
            ]
        )

        history = history.loc[
            history["timestamp"]
            != weather_hour
        ].copy()

        history = pd.concat(
            [
                history,
                observation,
            ],
            ignore_index=True,
        )

        history = (
            history
            .sort_values("timestamp")
            .drop_duplicates(
                subset=["timestamp"],
                keep="last",
            )
            .tail(self.MAX_HISTORY_HOURS)
            .reset_index(drop=True)
        )

        self.validate_history(
            history
        )

        self.save(
            city_name,
            history,
        )

        return history

    def load(
        self,
        city_name: str,
    ) -> pd.DataFrame:
        path = self.get_history_path(
            city_name
        )

        if not path.exists():
            raise FileNotFoundError(
                f"No live history exists for "
                f"{city_name}. Bootstrap it first."
            )

        history = pd.read_parquet(
            path
        )

        history["timestamp"] = pd.to_datetime(
            history["timestamp"],
            utc=True,
            errors="raise",
        )

        self.validate_history(
            history
        )

        return history

    def save(
        self,
        city_name: str,
        history: pd.DataFrame,
    ) -> None:
        self.validate_history(
            history
        )

        history.to_parquet(
            self.get_history_path(
                city_name
            ),
            index=False,
        )

    @classmethod
    def get_history_path(
        cls,
        city_name: str,
    ) -> Path:
        safe_name = (
            city_name
            .lower()
            .replace(" ", "_")
        )

        return (
            cls.HISTORY_ROOT
            / f"{safe_name}.parquet"
        )

    @staticmethod
    def _hour_bucket(
        timestamp,
    ) -> pd.Timestamp:
        value = pd.Timestamp(
            timestamp
        )

        if value.tzinfo is None:
            value = value.tz_localize(
                "UTC"
            )
        else:
            value = value.tz_convert(
                "UTC"
            )

        return value.floor(
            "h"
        )

    @staticmethod
    def validate_history(
        history: pd.DataFrame,
    ) -> None:
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

        missing = [
            column
            for column in required
            if column not in history.columns
        ]

        if missing:
            raise ValueError(
                f"History missing columns: {missing}"
            )

        if len(history) < 97:
            raise ValueError(
                "History requires at least 97 "
                "continuous raw hourly observations."
            )

        if (
            history[required]
            .isnull()
            .any()
            .any()
        ):
            raise ValueError(
                "History contains missing "
                "required values."
            )

        timestamps = pd.to_datetime(
            history["timestamp"],
            utc=True,
            errors="raise",
        )

        if timestamps.duplicated().any():
            raise ValueError(
                "History contains duplicate timestamps."
            )

        timestamps = (
            timestamps.sort_values()
        )

        gaps = (
            timestamps
            .diff()
            .dropna()
        )

        if (
            gaps
            != pd.Timedelta(hours=1)
        ).any():
            raise ValueError(
                "History is not continuous hourly data."
            )

        non_negative_columns = [
            "precipitation",
            "wind_speed",
            "pm2_5",
            "pm10",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
        ]

        if (
            history[
                non_negative_columns
            ] < 0
        ).any().any():
            raise ValueError(
                "History contains negative values "
                "in non-negative fields."
            )