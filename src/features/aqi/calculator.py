from dataclasses import dataclass
from math import floor

from src.features.aqi.breakpoints import (
    AQIBreakpoint,
    PM10_BREAKPOINTS,
    PM25_BREAKPOINTS,
)


@dataclass(frozen=True)
class AQIResult:
    aqi: int
    category: str
    dominant_pollutant: str
    pm25_aqi: int
    pm10_aqi: int


class AQICalculator:
    @classmethod
    def calculate_pm25_aqi(cls, concentration: float) -> int:
        cls._validate_concentration(concentration, "PM2.5")

        truncated = floor(concentration * 10) / 10

        return cls._calculate_sub_index(
            concentration=truncated,
            breakpoints=PM25_BREAKPOINTS,
        )

    @classmethod
    def calculate_pm10_aqi(cls, concentration: float) -> int:
        cls._validate_concentration(concentration, "PM10")

        truncated = float(floor(concentration))

        return cls._calculate_sub_index(
            concentration=truncated,
            breakpoints=PM10_BREAKPOINTS,
        )

    @classmethod
    def calculate_aqi(
        cls,
        pm25: float,
        pm10: float,
    ) -> AQIResult:
        pm25_aqi = cls.calculate_pm25_aqi(pm25)
        pm10_aqi = cls.calculate_pm10_aqi(pm10)

        if pm25_aqi >= pm10_aqi:
            final_aqi = pm25_aqi
            dominant_pollutant = "pm2_5"
        else:
            final_aqi = pm10_aqi
            dominant_pollutant = "pm10"

        return AQIResult(
            aqi=final_aqi,
            category=cls.category_from_aqi(final_aqi),
            dominant_pollutant=dominant_pollutant,
            pm25_aqi=pm25_aqi,
            pm10_aqi=pm10_aqi,
        )

    @staticmethod
    def category_from_aqi(aqi: int) -> str:
        if aqi < 0:
            raise ValueError("AQI cannot be negative.")

        if aqi <= 50:
            return "Good"
        if aqi <= 100:
            return "Moderate"
        if aqi <= 150:
            return "Unhealthy for Sensitive Groups"
        if aqi <= 200:
            return "Unhealthy"
        if aqi <= 300:
            return "Very Unhealthy"

        return "Hazardous"

    @staticmethod
    def _calculate_sub_index(
        concentration: float,
        breakpoints: tuple[AQIBreakpoint, ...],
    ) -> int:
        for breakpoint in breakpoints:
            if (
                breakpoint.concentration_low
                <= concentration
                <= breakpoint.concentration_high
            ):
                aqi = (
                    (
                        breakpoint.aqi_high
                        - breakpoint.aqi_low
                    )
                    / (
                        breakpoint.concentration_high
                        - breakpoint.concentration_low
                    )
                    * (
                        concentration
                        - breakpoint.concentration_low
                    )
                    + breakpoint.aqi_low
                )

                return round(aqi)

        if concentration > breakpoints[-1].concentration_high:
            return 500

        raise ValueError(
            f"No AQI breakpoint found for concentration {concentration}."
        )

    @staticmethod
    def _validate_concentration(
        concentration: float,
        pollutant: str,
    ) -> None:
        if concentration < 0:
            raise ValueError(
                f"{pollutant} concentration cannot be negative."
            )