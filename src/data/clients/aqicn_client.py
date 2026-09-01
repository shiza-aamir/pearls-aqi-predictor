from __future__ import annotations

from datetime import datetime

import requests

from src.core.settings import (
    CityConfig,
    get_aqicn_api_token,
)
from src.data.schemas.live_observation import (
    AQICNObservation,
)


class AQICNClient:
    BASE_URL = (
        "https://api.waqi.info/feed/"
    )

    REQUEST_TIMEOUT_SECONDS = 15

    def __init__(
        self,
        token: str | None = None,
    ) -> None:
        self.token = (
            token
            or get_aqicn_api_token()
        )

    def get_current(
        self,
        city: CityConfig,
    ) -> AQICNObservation:
        location = (
            f"geo:"
            f"{city.latitude};"
            f"{city.longitude}"
        )

        url = (
            f"{self.BASE_URL}"
            f"{location}/"
        )

        response = requests.get(
            url,
            params={
                "token": self.token,
            },
            timeout=self.REQUEST_TIMEOUT_SECONDS,
        )

        response.raise_for_status()

        payload = response.json()

        if payload.get("status") != "ok":
            raise RuntimeError(
                "AQICN request failed: "
                f"{payload}"
            )

        data = payload["data"]

        reported_aqi = self._parse_aqi(
            data.get("aqi")
        )

        station = data.get(
            "city",
            {},
        )

        station_name = station.get(
            "name"
        )

        geo = station.get(
            "geo"
        )

        latitude = None
        longitude = None

        if (
            isinstance(geo, list)
            and len(geo) >= 2
        ):
            latitude = float(
                geo[0]
            )
            longitude = float(
                geo[1]
            )

        timestamp = self._parse_time(
            data.get("time")
        )

        return AQICNObservation(
            timestamp=timestamp,
            reported_aqi=reported_aqi,
            station_name=station_name,
            latitude=latitude,
            longitude=longitude,
        )

    @staticmethod
    def _parse_aqi(
        value,
    ) -> float | None:
        if value in (
            None,
            "-",
            "",
        ):
            return None

        try:
            return float(
                value
            )
        except (
            TypeError,
            ValueError,
        ):
            return None

    @staticmethod
    def _parse_time(
        value,
    ) -> datetime | None:
        if not isinstance(
            value,
            dict,
        ):
            return None

        iso = value.get(
            "iso"
        )

        if not iso:
            return None

        try:
            return datetime.fromisoformat(
                iso
            )
        except ValueError:
            return None