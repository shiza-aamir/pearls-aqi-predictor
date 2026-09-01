from __future__ import annotations

from datetime import datetime, timezone

import requests

from src.core.settings import (
    CityConfig,
    get_openweather_api_key,
)
from src.data.schemas.live_observation import (
    PollutionObservation,
    WeatherObservation,
)


class OpenWeatherClient:
    WEATHER_URL = (
        "https://api.openweathermap.org/"
        "data/2.5/weather"
    )

    AIR_POLLUTION_URL = (
        "https://api.openweathermap.org/"
        "data/2.5/air_pollution"
    )

    REQUEST_TIMEOUT_SECONDS = 15

    def __init__(
        self,
        api_key: str | None = None,
    ) -> None:
        self.api_key = (
            api_key
            or get_openweather_api_key()
        )

    def get_current_weather(
        self,
        city: CityConfig,
    ) -> WeatherObservation:
        response = requests.get(
            self.WEATHER_URL,
            params={
                "lat": city.latitude,
                "lon": city.longitude,
                "appid": self.api_key,
                "units": "metric",
            },
            timeout=self.REQUEST_TIMEOUT_SECONDS,
        )

        response.raise_for_status()

        payload = response.json()

        timestamp = datetime.fromtimestamp(
            int(payload["dt"]),
            tz=timezone.utc,
        )

        main = payload["main"]
        wind = payload.get("wind", {})

        precipitation = self._extract_precipitation(
            payload
        )

        return WeatherObservation(
            timestamp=timestamp,
            latitude=float(
                payload["coord"]["lat"]
            ),
            longitude=float(
                payload["coord"]["lon"]
            ),
            temperature=float(
                main["temp"]
            ),
            humidity=float(
                main["humidity"]
            ),
            precipitation=float(
                precipitation
            ),
            wind_speed=float(
                wind.get("speed", 0.0)
            ),
            wind_direction=float(
                wind.get("deg", 0.0)
            ),
            pressure=float(
                main["pressure"]
            ),
        )

    def get_current_pollution(
        self,
        city: CityConfig,
    ) -> PollutionObservation:
        response = requests.get(
            self.AIR_POLLUTION_URL,
            params={
                "lat": city.latitude,
                "lon": city.longitude,
                "appid": self.api_key,
            },
            timeout=self.REQUEST_TIMEOUT_SECONDS,
        )

        response.raise_for_status()

        payload = response.json()

        rows = payload.get(
            "list",
            [],
        )

        if not rows:
            raise RuntimeError(
                "OpenWeather returned no "
                "air-pollution observations."
            )

        row = rows[0]

        components = row["components"]

        timestamp = datetime.fromtimestamp(
            int(row["dt"]),
            tz=timezone.utc,
        )

        return PollutionObservation(
            timestamp=timestamp,
            latitude=float(
                payload["coord"]["lat"]
            ),
            longitude=float(
                payload["coord"]["lon"]
            ),
            pm2_5=self._non_negative(
                components["pm2_5"],
                "pm2_5",
            ),
            pm10=self._non_negative(
                components["pm10"],
                "pm10",
            ),
            carbon_monoxide=self._non_negative(
                components["co"],
                "carbon_monoxide",
            ),
            nitrogen_dioxide=self._non_negative(
                components["no2"],
                "nitrogen_dioxide",
            ),
            sulphur_dioxide=self._non_negative(
                components["so2"],
                "sulphur_dioxide",
            ),
            ozone=self._non_negative(
                components["o3"],
                "ozone",
            ),
        )

    @staticmethod
    def _extract_precipitation(
        payload: dict,
    ) -> float:
        rain = payload.get(
            "rain",
            {},
        )

        snow = payload.get(
            "snow",
            {},
        )

        rain_1h = float(
            rain.get("1h", 0.0)
        )

        snow_1h = float(
            snow.get("1h", 0.0)
        )

        value = (
            rain_1h
            + snow_1h
        )

        return max(
            value,
            0.0,
        )

    @staticmethod
    def _non_negative(
        value: float,
        field: str,
    ) -> float:
        number = float(
            value
        )

        if number < 0:
            raise ValueError(
                f"OpenWeather returned a "
                f"negative {field}: {number}"
            )

        return number