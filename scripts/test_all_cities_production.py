from __future__ import annotations

import time
import traceback

import numpy as np
import pandas as pd

from src.services.production_service import (
    AQIProductionService,
)


CITIES = [
    "Faisalabad",
    "Islamabad",
    "Karachi",
    "Lahore",
    "Multan",
    "Peshawar",
    "Quetta",
    "Rahim Yar Khan",
    "Sialkot",
]


def main() -> None:
    print("=" * 100)
    print(
        "PEARLS AQI - ALL CITIES PRODUCTION SMOKE TEST"
    )
    print("=" * 100)

    service = AQIProductionService()

    results = []
    failures = []

    for index, city in enumerate(
        CITIES,
        start=1,
    ):
        print(
            "\n"
            + "-" * 100
        )

        print(
            f"[{index}/{len(CITIES)}] "
            f"Running production pipeline "
            f"for {city}..."
        )

        print(
            "-" * 100
        )

        start_time = time.perf_counter()

        try:
            result = service.run(
                city
            )

            elapsed = (
                time.perf_counter()
                - start_time
            )

            # =============================================
            # VALIDATIONS
            # =============================================

            if result.city != city:
                raise AssertionError(
                    "Returned city does not match "
                    f"requested city: {city}"
                )

            if not isinstance(
                result.timestamp,
                pd.Timestamp,
            ):
                raise AssertionError(
                    "Timestamp is not a "
                    "pandas Timestamp."
                )

            if (
                result.timestamp.tzinfo
                is None
            ):
                raise AssertionError(
                    "Timestamp is not "
                    "timezone-aware."
                )

            if result.feature_count != 56:
                raise AssertionError(
                    "Expected exactly 56 "
                    "production features."
                )

            if not np.isfinite(
                result.current_aqi
            ):
                raise AssertionError(
                    "Current AQI is not finite."
                )

            if not (
                0.0
                <= result.current_aqi
                <= 500.0
            ):
                raise AssertionError(
                    "Current AQI is outside "
                    "0-500."
                )

            if len(
                result.forecasts
            ) != 3:
                raise AssertionError(
                    "Expected exactly three "
                    "production forecasts."
                )

            expected_horizons = {
                "24h",
                "48h",
                "72h",
            }

            received_horizons = {
                forecast.horizon
                for forecast
                in result.forecasts
            }

            if (
                received_horizons
                != expected_horizons
            ):
                raise AssertionError(
                    "Unexpected forecast "
                    f"horizons: "
                    f"{received_horizons}"
                )

            for forecast in (
                result.forecasts
            ):
                if not np.isfinite(
                    forecast.predicted_aqi
                ):
                    raise AssertionError(
                        f"{forecast.horizon} "
                        "prediction is not finite."
                    )

                if not (
                    0.0
                    <= forecast.predicted_aqi
                    <= 500.0
                ):
                    raise AssertionError(
                        f"{forecast.horizon} "
                        "prediction is outside "
                        "0-500."
                    )

                if not (
                    forecast.explanation
                    .contributions
                ):
                    raise AssertionError(
                        f"{forecast.horizon} "
                        "has no SHAP "
                        "contributions."
                    )

            forecast_lookup = {
                forecast.horizon:
                    forecast
                for forecast
                in result.forecasts
            }

            # =============================================
            # VERIFY REAL FORECAST LEDGER
            # =============================================

            ledger = (
                service
                .monitoring_service
                .load_ledger()
            )

            city_ledger = ledger[
                ledger["city"]
                .astype(str)
                .str.casefold()
                == city.casefold()
            ].copy()

            current_run_ledger = (
                city_ledger[
                    city_ledger[
                        "forecast_created_at"
                    ]
                    == result.timestamp
                ]
            )

            if (
                len(current_run_ledger)
                != 3
            ):
                raise AssertionError(
                    "Expected exactly three "
                    "forecast-ledger entries "
                    "for this city and "
                    "timestamp."
                )

            # =============================================
            # PRINT CITY RESULT
            # =============================================

            print(
                f"Timestamp:       "
                f"{result.timestamp}"
            )

            print(
                f"Source:          "
                f"{result.data_source}"
            )

            print(
                f"History rows:    "
                f"{result.history_rows}"
            )

            print(
                f"Features:        "
                f"{result.feature_count}"
            )

            print(
                f"Current AQI:     "
                f"{result.current_aqi:.2f}"
            )

            print(
                f"Category:        "
                f"{result.current_category}"
            )

            print(
                f"Alert:           "
                f"{result.current_alert.level}"
            )

            print(
                "\nForecasts:"
            )

            for horizon in [
                "24h",
                "48h",
                "72h",
            ]:
                forecast = (
                    forecast_lookup[
                        horizon
                    ]
                )

                print(
                    f"  {horizon:>3}: "
                    f"{forecast.predicted_aqi:7.2f} "
                    f"| "
                    f"{forecast.predicted_category}"
                    f" | "
                    f"{forecast.model_alias}"
                )

            print(
                f"\nRuntime: "
                f"{elapsed:.2f} seconds"
            )

            print(
                f"{city}: PASS"
            )

            results.append(
                {
                    "city": city,
                    "timestamp": (
                        result.timestamp
                    ),
                    "source": (
                        result.data_source
                    ),
                    "history_rows": (
                        result.history_rows
                    ),
                    "current_aqi": (
                        result.current_aqi
                    ),
                    "current_category": (
                        result.current_category
                    ),
                    "alert": (
                        result
                        .current_alert
                        .level
                    ),
                    "aqi_24h": (
                        forecast_lookup[
                            "24h"
                        ]
                        .predicted_aqi
                    ),
                    "aqi_48h": (
                        forecast_lookup[
                            "48h"
                        ]
                        .predicted_aqi
                    ),
                    "aqi_72h": (
                        forecast_lookup[
                            "72h"
                        ]
                        .predicted_aqi
                    ),
                    "runtime_seconds": (
                        elapsed
                    ),
                    "status": "PASS",
                }
            )

        except Exception as exc:
            elapsed = (
                time.perf_counter()
                - start_time
            )

            print(
                f"\n{city}: FAIL"
            )

            print(
                f"Error type: "
                f"{type(exc).__name__}"
            )

            print(
                f"Error: {exc}"
            )

            print(
                "\nTraceback:"
            )

            traceback.print_exc()

            failures.append(
                {
                    "city": city,
                    "error_type": (
                        type(exc).__name__
                    ),
                    "error": str(
                        exc
                    ),
                    "runtime_seconds": (
                        elapsed
                    ),
                }
            )

    # =====================================================
    # SUMMARY
    # =====================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "ALL CITIES SUMMARY"
    )

    print(
        "=" * 100
    )

    if results:
        summary = pd.DataFrame(
            results
        )

        print(
            "\nSuccessful cities:"
        )

        print(
            summary[
                [
                    "city",
                    "current_aqi",
                    "aqi_24h",
                    "aqi_48h",
                    "aqi_72h",
                    "source",
                    "runtime_seconds",
                    "status",
                ]
            ].to_string(
                index=False,
                float_format=lambda value: (
                    f"{value:.2f}"
                ),
            )
        )

    print(
        "\n"
        f"Passed: "
        f"{len(results)}/{len(CITIES)}"
    )

    print(
        f"Failed: "
        f"{len(failures)}/{len(CITIES)}"
    )

    if failures:
        failure_df = pd.DataFrame(
            failures
        )

        print(
            "\nFailures:"
        )

        print(
            failure_df.to_string(
                index=False
            )
        )

        print(
            "\n"
            + "=" * 100
        )

        print(
            "ALL CITIES PRODUCTION "
            "SMOKE TEST: FAIL"
        )

        print(
            "=" * 100
        )

        raise SystemExit(
            1
        )

    # =====================================================
    # LEDGER VALIDATION
    # =====================================================

    ledger = (
        service
        .monitoring_service
        .load_ledger()
    )

    ledger_cities = set(
        ledger["city"]
        .astype(str)
        .tolist()
    )

    missing_ledger_cities = [
        city
        for city in CITIES
        if city not in ledger_cities
    ]

    if missing_ledger_cities:
        raise AssertionError(
            "Production ledger is missing "
            "cities: "
            f"{missing_ledger_cities}"
        )

    print(
        "\nForecast ledger coverage:"
    )

    for city in CITIES:
        count = int(
            (
                ledger["city"]
                .astype(str)
                .str.casefold()
                == city.casefold()
            )
            .sum()
        )

        print(
            f"  {city:<20} "
            f"{count:>4} rows"
        )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "ALL CITIES PRODUCTION "
        "SMOKE TEST: PASS"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()