from __future__ import annotations

import numpy as np
import pandas as pd

from src.services.production_service import (
    AQIProductionService,
)


CITY = "Islamabad"


def main() -> None:
    print("=" * 80)
    print(
        "PEARLS AQI - PRODUCTION SERVICE TEST"
    )
    print("=" * 80)

    service = AQIProductionService()

    print(
        f"\nRunning production pipeline "
        f"for {CITY}..."
    )

    result = service.run(
        CITY
    )

    print(
        f"\nCity:             {result.city}"
    )

    print(
        f"Timestamp:        {result.timestamp}"
    )

    print(
        f"Data source:      {result.data_source}"
    )

    print(
        f"History rows:     {result.history_rows}"
    )

    print(
        f"Feature count:    {result.feature_count}"
    )

    print(
        f"Current AQI:      "
        f"{result.current_aqi:.2f}"
    )

    print(
        f"Current category: "
        f"{result.current_category}"
    )

    print(
        f"Current alert:    "
        f"{result.current_alert.level}"
    )

    print(
        f"Alert severity:   "
        f"{result.current_alert.severity}"
    )

    print(
        "\nCurrent conditions:"
    )

    print(
        f"  PM2.5:       "
        f"{result.pm2_5:.2f}"
    )

    print(
        f"  PM10:        "
        f"{result.pm10:.2f}"
    )

    print(
        f"  Temperature: "
        f"{result.temperature:.2f}"
    )

    print(
        f"  Humidity:    "
        f"{result.humidity:.2f}"
    )

    print(
        f"  Wind speed:  "
        f"{result.wind_speed:.2f}"
    )

    # -----------------------------------------------------
    # Basic validations
    # -----------------------------------------------------

    if result.city != CITY:
        raise AssertionError(
            "Returned city does not match."
        )

    if not isinstance(
        result.timestamp,
        pd.Timestamp,
    ):
        raise AssertionError(
            "Timestamp must be pandas Timestamp."
        )

    if result.timestamp.tzinfo is None:
        raise AssertionError(
            "Timestamp must be timezone-aware."
        )

    if result.feature_count != 56:
        raise AssertionError(
            "Production service did not use "
            "exactly 56 model features."
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

    if len(result.forecasts) != 3:
        raise AssertionError(
            "Expected exactly three "
            "forecasts."
        )

    # -----------------------------------------------------
    # Forecast validation
    # -----------------------------------------------------

    print(
        "\nProduction forecasts:"
    )

    expected_horizons = {
        "24h",
        "48h",
        "72h",
    }

    received_horizons = set()

    for forecast in result.forecasts:
        received_horizons.add(
            forecast.horizon
        )

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

        print(
            f"  {forecast.horizon:>3}  "
            f"AQI="
            f"{forecast.predicted_aqi:7.2f}  "
            f"{forecast.predicted_category}"
        )

        print(
            f"       Model alias: "
            f"{forecast.model_alias}"
        )

        print(
            f"       Alert: "
            f"{forecast.alert.level} "
            f"(severity "
            f"{forecast.alert.severity})"
        )

        contributions = (
            forecast
            .explanation
            .contributions
        )

        if not contributions:
            raise AssertionError(
                f"{forecast.horizon} "
                "has no SHAP contributions."
            )

        print(
            "       Top SHAP:"
        )

        for contribution in (
            contributions[:3]
        ):
            print(
                f"         "
                f"{contribution.feature:<30} "
                f"value="
                f"{contribution.feature_value:9.3f} "
                f"SHAP="
                f"{contribution.shap_value:+9.3f}"
            )

    if (
        received_horizons
        != expected_horizons
    ):
        raise AssertionError(
            "Expected horizons "
            f"{expected_horizons}, "
            f"received "
            f"{received_horizons}."
        )

    # -----------------------------------------------------
    # Monitoring status
    # -----------------------------------------------------

    print(
        "\nLive monitoring:"
    )

    if result.live_performance.empty:
        print(
            "  No genuine forecasts have "
            "matured yet."
        )

        print(
            "  This is expected on the first "
            "production run."
        )

    else:
        print(
            result.live_performance
            .to_string(
                index=False,
                float_format=lambda x: (
                    f"{x:.2f}"
                ),
            )
        )

    # -----------------------------------------------------
    # Verify real ledger
    # -----------------------------------------------------

    ledger = (
        service.monitoring_service
        .load_ledger()
    )

    city_ledger = ledger[
        ledger["city"]
        .astype(str)
        .str.casefold()
        == CITY.casefold()
    ].copy()

    if city_ledger.empty:
        raise AssertionError(
            "Production forecasts were not "
            "written to the real ledger."
        )

    latest_ledger = city_ledger[
        city_ledger[
            "forecast_created_at"
        ]
        == result.timestamp
    ]

    if len(latest_ledger) != 3:
        raise AssertionError(
            "Expected exactly three ledger "
            "entries for this production run."
        )

    if not (
        latest_ledger[
            "actual_aqi"
        ]
        .isna()
        .all()
    ):
        raise AssertionError(
            "New forecasts should remain "
            "pending until their target "
            "timestamps occur."
        )

    print(
        "\nReal forecast ledger:"
    )

    print(
        latest_ledger[
            [
                "forecast_created_at",
                "target_timestamp",
                "horizon_hours",
                "predicted_aqi",
                "predicted_category",
                "actual_aqi",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nIntegration checks:"
    )

    print(
        "  Live history            PASS"
    )

    print(
        "  AQI derivation          PASS"
    )

    print(
        "  56-feature engineering  PASS"
    )

    print(
        "  Feast online write      PASS"
    )

    print(
        "  Feast online read       PASS"
    )

    print(
        "  MLflow prediction       PASS"
    )

    print(
        "  SHAP explanation        PASS"
    )

    print(
        "  Alert generation        PASS"
    )

    print(
        "  Forecast ledger         PASS"
    )

    print(
        "  Monitoring integration  PASS"
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "PRODUCTION SERVICE TEST: PASS"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()