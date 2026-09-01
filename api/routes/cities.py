from __future__ import annotations

from fastapi import APIRouter

from api.schemas.common import (
    CitiesResponse,
    CityResponse,
)
from src.core.settings import get_city

router = APIRouter(
    prefix="/cities",
    tags=["Cities"],
)


SUPPORTED_CITIES = (
    "Faisalabad",
    "Islamabad",
    "Karachi",
    "Lahore",
    "Multan",
    "Peshawar",
    "Quetta",
    "Rahim Yar Khan",
    "Sialkot",
)


@router.get(
    "",
    response_model=CitiesResponse,
    summary="List supported cities",
)
def list_cities() -> CitiesResponse:
    cities: list[CityResponse] = []

    for city_name in SUPPORTED_CITIES:
        city = get_city(city_name)

        cities.append(
            CityResponse(
                name=city_name,
                latitude=float(city.latitude),
                longitude=float(city.longitude),
            )
        )

    return CitiesResponse(
        cities=cities,
        count=len(cities),
    )