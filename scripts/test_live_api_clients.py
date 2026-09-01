from __future__ import annotations

from datetime import timezone

from src.core.settings import (
    CITIES,
)
from src.data.clients.aqicn_client import (
    AQICNClient,
)
from src.data.clients.openweather_client import (
    OpenWeatherClient,
)


TEST_CITY = "Islamabad"


def main() -> None:
    print("=" * 80)
    print(
        "PEARLS AQI - LIVE API CLIENT TEST"
    )
    print("=" * 80)

    city = CITIES[
        TEST_CITY
    ]

    openweather = (
        OpenWeatherClient()
    )

    aqicn = (
        AQICNClient()
    )

    print(
        f"\nCity: {city.name}"
    )

    print(
        "\nFetching OpenWeather weather..."
    )

    weather = (
        openweather
        .get_current_weather(
            city
        )
    )

    print(
        f"Timestamp:       "
        f"{weather.timestamp.astimezone(timezone.utc)}"
    )

    print(
        f"Coordinates:     "
        f"{weather.latitude:.4f}, "
        f"{weather.longitude:.4f}"
    )

    print(
        f"Temperature:     "
        f"{weather.temperature:.2f} C"
    )

    print(
        f"Humidity:        "
        f"{weather.humidity:.2f} %"
    )

    print(
        f"Precipitation:   "
        f"{weather.precipitation:.2f} mm"
    )

    print(
        f"Wind speed:      "
        f"{weather.wind_speed:.2f} m/s"
    )

    print(
        f"Wind direction:  "
        f"{weather.wind_direction:.2f} deg"
    )

    print(
        f"Pressure:        "
        f"{weather.pressure:.2f} hPa"
    )

    print(
        "\nFetching OpenWeather pollution..."
    )

    pollution = (
        openweather
        .get_current_pollution(
            city
        )
    )

    print(
        f"Timestamp:       "
        f"{pollution.timestamp.astimezone(timezone.utc)}"
    )

    print(
        f"PM2.5:           "
        f"{pollution.pm2_5:.2f} ug/m3"
    )

    print(
        f"PM10:            "
        f"{pollution.pm10:.2f} ug/m3"
    )

    print(
        f"CO:              "
        f"{pollution.carbon_monoxide:.2f} ug/m3"
    )

    print(
        f"NO2:             "
        f"{pollution.nitrogen_dioxide:.2f} ug/m3"
    )

    print(
        f"SO2:             "
        f"{pollution.sulphur_dioxide:.2f} ug/m3"
    )

    print(
        f"O3:              "
        f"{pollution.ozone:.2f} ug/m3"
    )

    print(
        "\nFetching AQICN reported AQI..."
    )

    aqicn_observation = (
        aqicn.get_current(
            city
        )
    )

    print(
        f"Reported AQI:     "
        f"{aqicn_observation.reported_aqi}"
    )

    print(
        f"Station:          "
        f"{aqicn_observation.station_name}"
    )

    print(
        f"Timestamp:        "
        f"{aqicn_observation.timestamp}"
    )

    print(
        f"Station coords:   "
        f"{aqicn_observation.latitude}, "
        f"{aqicn_observation.longitude}"
    )

    print("\n" + "=" * 80)
    print(
        "LIVE API CLIENT TEST: PASS"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()