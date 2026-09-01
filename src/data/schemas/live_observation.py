from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class WeatherObservation:
    timestamp: datetime
    latitude: float
    longitude: float
    temperature: float
    humidity: float
    precipitation: float
    wind_speed: float
    wind_direction: float
    pressure: float


@dataclass(frozen=True)
class PollutionObservation:
    timestamp: datetime
    latitude: float
    longitude: float
    pm2_5: float
    pm10: float
    carbon_monoxide: float
    nitrogen_dioxide: float
    sulphur_dioxide: float
    ozone: float


@dataclass(frozen=True)
class AQICNObservation:
    timestamp: datetime | None
    reported_aqi: float | None
    station_name: str | None
    latitude: float | None
    longitude: float | None