from __future__ import annotations

import argparse
import sys
import time
import traceback
from dataclasses import dataclass

from src.services.production_service import (
    AQIProductionService,
)

CITIES = (
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


@dataclass(frozen=True)
class CityRunSummary:
    city: str
    success: bool
    runtime_seconds: float
    timestamp: str | None = None
    current_aqi: float | None = None
    forecast_24h: float | None = None
    forecast_48h: float | None = None
    forecast_72h: float | None = None
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the PEARLS AQI hourly production "
            "forecast pipeline."
        )
    )

    parser.add_argument(
        "--city",
        choices=CITIES,
        help=(
            "Run one city only. "
            "If omitted, all production cities run."
        ),
    )

    return parser.parse_args()


def run_city(
    service: AQIProductionService,
    city: str,
) -> CityRunSummary:
    started_at = time.perf_counter()

    try:
        result = service.run(city)

        forecasts = {
            forecast.horizon: forecast
            for forecast in result.forecasts
        }

        required_horizons = {
            "24h",
            "48h",
            "72h",
        }

        if set(forecasts) != required_horizons:
            raise RuntimeError(
                "Production service returned unexpected "
                f"forecast horizons: {set(forecasts)}"
            )

        runtime = (
            time.perf_counter()
            - started_at
        )

        print()
        print(
            f"[PASS] {city}"
        )
        print(
            f"       Observation: "
            f"{result.timestamp}"
        )
        print(
            f"       Source:      "
            f"{result.data_source}"
        )
        print(
            f"       History:     "
            f"{result.history_rows} rows"
        )
        print(
            f"       Features:    "
            f"{result.feature_count}"
        )
        print(
            f"       Current AQI: "
            f"{result.current_aqi:.2f} "
            f"({result.current_category})"
        )
        print(
            f"       24h:         "
            f"{forecasts['24h'].predicted_aqi:.2f}"
        )
        print(
            f"       48h:         "
            f"{forecasts['48h'].predicted_aqi:.2f}"
        )
        print(
            f"       72h:         "
            f"{forecasts['72h'].predicted_aqi:.2f}"
        )
        print(
            f"       Runtime:     "
            f"{runtime:.2f}s"
        )

        return CityRunSummary(
            city=city,
            success=True,
            runtime_seconds=runtime,
            timestamp=str(
                result.timestamp
            ),
            current_aqi=float(
                result.current_aqi
            ),
            forecast_24h=float(
                forecasts[
                    "24h"
                ].predicted_aqi
            ),
            forecast_48h=float(
                forecasts[
                    "48h"
                ].predicted_aqi
            ),
            forecast_72h=float(
                forecasts[
                    "72h"
                ].predicted_aqi
            ),
        )

    except Exception as exc:
        runtime = (
            time.perf_counter()
            - started_at
        )

        print()
        print(
            f"[FAIL] {city}"
        )
        print(
            f"       {type(exc).__name__}: "
            f"{exc}"
        )

        traceback.print_exc()

        return CityRunSummary(
            city=city,
            success=False,
            runtime_seconds=runtime,
            error=(
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        )


def print_summary(
    results: list[CityRunSummary],
) -> None:
    print()
    print("=" * 88)
    print(
        "PEARLS AQI HOURLY PRODUCTION SUMMARY"
    )
    print("=" * 88)

    for result in results:
        status = (
            "PASS"
            if result.success
            else "FAIL"
        )

        if result.success:
            print(
                f"{result.city:<20} "
                f"{status:<5} "
                f"AQI={result.current_aqi:>7.2f} "
                f"24h={result.forecast_24h:>7.2f} "
                f"48h={result.forecast_48h:>7.2f} "
                f"72h={result.forecast_72h:>7.2f} "
                f"{result.runtime_seconds:>7.2f}s"
            )
        else:
            print(
                f"{result.city:<20} "
                f"{status:<5} "
                f"{result.runtime_seconds:>7.2f}s "
                f"{result.error}"
            )

    passed = sum(
        result.success
        for result in results
    )

    failed = (
        len(results)
        - passed
    )

    print("-" * 88)

    print(
        f"Passed: {passed}/{len(results)}"
    )

    print(
        f"Failed: {failed}/{len(results)}"
    )

    print("=" * 88)


def main() -> None:
    args = parse_args()

    cities = (
        (args.city,)
        if args.city
        else CITIES
    )

    print("=" * 88)
    print(
        "PEARLS AQI HOURLY PRODUCTION PIPELINE"
    )
    print("=" * 88)

    print(
        "Cities: "
        + ", ".join(cities)
    )

    service = AQIProductionService()

    results = [
        run_city(
            service=service,
            city=city,
        )
        for city in cities
    ]

    print_summary(
        results
    )

    if any(
        not result.success
        for result in results
    ):
        sys.exit(1)


if __name__ == "__main__":
    main()