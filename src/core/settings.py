from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(
    PROJECT_ROOT / ".env"
)


@dataclass(frozen=True)
class CityConfig:
    name: str
    latitude: float
    longitude: float


CITIES: dict[str, CityConfig] = {
    "Faisalabad": CityConfig(
        name="Faisalabad",
        latitude=31.4504,
        longitude=73.1350,
    ),
    "Islamabad": CityConfig(
        name="Islamabad",
        latitude=33.6844,
        longitude=73.0479,
    ),
    "Karachi": CityConfig(
        name="Karachi",
        latitude=24.8607,
        longitude=67.0011,
    ),
    "Lahore": CityConfig(
        name="Lahore",
        latitude=31.5204,
        longitude=74.3587,
    ),
    "Multan": CityConfig(
        name="Multan",
        latitude=30.1575,
        longitude=71.5249,
    ),
    "Peshawar": CityConfig(
        name="Peshawar",
        latitude=34.0151,
        longitude=71.5249,
    ),
    "Quetta": CityConfig(
        name="Quetta",
        latitude=30.1798,
        longitude=66.9750,
    ),
    "Rahim Yar Khan": CityConfig(
        name="Rahim Yar Khan",
        latitude=28.4212,
        longitude=70.2989,
    ),
    "Sialkot": CityConfig(
        name="Sialkot",
        latitude=32.4945,
        longitude=74.5229,
    ),
}


def get_openweather_api_key() -> str:
    value = os.getenv(
        "OPENWEATHER_API_KEY",
        "",
    ).strip()

    if not value:
        raise RuntimeError(
            "OPENWEATHER_API_KEY is not configured."
        )

    return value


def get_aqicn_api_token() -> str:
    value = os.getenv(
        "AQICN_API_TOKEN",
        "",
    ).strip()

    if not value:
        raise RuntimeError(
            "AQICN_API_TOKEN is not configured."
        )

    return value


def get_city(
    city_name: str,
) -> CityConfig:
    try:
        return CITIES[city_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported city: {city_name}. "
            f"Supported cities: {list(CITIES)}"
        ) from exc