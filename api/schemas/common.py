from __future__ import annotations

from pydantic import BaseModel


class APIInfoResponse(BaseModel):
    name: str
    version: str
    environment: str
    status: str


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class CityResponse(BaseModel):
    name: str
    latitude: float
    longitude: float


class CitiesResponse(BaseModel):
    cities: list[CityResponse]
    count: int