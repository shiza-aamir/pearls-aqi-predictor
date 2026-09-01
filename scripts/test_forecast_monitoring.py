from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.services.forecast_monitoring_service import (
    ForecastMonitoringService,
)
from src.services.live_feature_pipeline import (
    LiveFeaturePipeline,
)
from src.services.live_history_service import (
    LiveHistoryService,
)
from src.services.prediction_service import (
    AQIPredictionService,
)


CITY = "Islamabad"

TEST_LEDGER = Path(
    "artifacts/test_forecast_monitoring/"
    "forecast_ledger.parquet"
)


def main() -> None:
    print("=" * 80)
    print(
        "PEARLS AQI - FORECAST MONITORING TEST"
    )
    print("=" * 80)

    TEST_LEDGER.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if TEST_LEDGER.exists():
        TEST_LEDGER.unlink()

    history_service = LiveHistoryService()

    feature_pipeline = LiveFeaturePipeline()

    prediction_service = AQIPredictionService()

    monitoring_service = ForecastMonitoringService(
        ledger_path=TEST_LEDGER
    )

    # =========================================================
    # 1. PREPARE LIVE HISTORY
    # =========================================================

    print(
        f"\nPreparing live data for {CITY}..."
    )

    history = (
        history_service
        .ensure_current_history(
            CITY
        )
    )

    if history.empty:
        raise ValueError(
            "Live history is empty."
        )

    if "timestamp" not in history.columns:
        raise ValueError(
            "Live history does not contain "
            "'timestamp'."
        )

    latest_timestamp = (
        pd.to_datetime(
            history["timestamp"],
            utc=True,
            errors="raise",
        )
        .max()
    )

    print(
        f"Latest timestamp: {latest_timestamp}"
    )

    # =========================================================
    # 2. BUILD LATEST FEATURES
    #
    # LiveFeaturePipeline returns:
    #
    #   enriched_row -> 72 columns
    #   feature_row  -> exact 56 model features
    # =========================================================

    (
        enriched_row,
        feature_row,
    ) = (
        feature_pipeline
        .build_latest_features(
            history
        )
    )

    if not isinstance(
        enriched_row,
        pd.DataFrame,
    ):
        raise TypeError(
            "Expected enriched_row to be "
            "a pandas DataFrame."
        )

    if not isinstance(
        feature_row,
        pd.DataFrame,
    ):
        raise TypeError(
            "Expected feature_row to be "
            "a pandas DataFrame."
        )

    if enriched_row.shape[0] != 1:
        raise ValueError(
            "Expected exactly one "
            "enriched live row."
        )

    if feature_row.shape != (1, 56):
        raise ValueError(
            "Expected feature_row shape "
            "(1, 56). "
            f"Received: {feature_row.shape}"
        )

    print(
        f"Enriched columns: {enriched_row.shape[1]}"
    )

    print(
        f"Model features:   {feature_row.shape[1]}"
    )

    print(
        "Feature extraction: PASS"
    )

    # =========================================================
    # 3. CURRENT AQI INFORMATION
    # =========================================================

    required_enriched_columns = [
        "timestamp",
        "aqi_current",
        "pm2_5",
        "pm10",
    ]

    missing = [
        column
        for column in required_enriched_columns
        if column not in enriched_row.columns
    ]

    if missing:
        raise ValueError(
            "Enriched row is missing: "
            f"{missing}"
        )

    current_aqi = float(
        enriched_row.iloc[0][
            "aqi_current"
        ]
    )

    current_pm25 = float(
        enriched_row.iloc[0][
            "pm2_5"
        ]
    )

    current_pm10 = float(
        enriched_row.iloc[0][
            "pm10"
        ]
    )

    print(
        f"Current AQI:      {current_aqi:.0f}"
    )

    print(
        f"Current PM2.5:    {current_pm25:.2f}"
    )

    print(
        f"Current PM10:     {current_pm10:.2f}"
    )

    # =========================================================
    # 4. GENERATE REAL MODEL FORECASTS
    # =========================================================

    predictions = (
        prediction_service
        .predict_all(
            feature_row
        )
    )

    if len(predictions) != 3:
        raise ValueError(
            "Expected exactly three "
            "predictions."
        )

    print(
        "\nGenerated forecasts:"
    )

    for prediction in predictions:
        print(
            f"  {prediction.horizon:>3}  "
            f"AQI={prediction.predicted_aqi:7.2f}  "
            f"{prediction.predicted_category}"
        )

    # =========================================================
    # 5. RECORD FORECASTS
    # =========================================================

    print(
        "\nRecording forecasts..."
    )

    ledger = (
        monitoring_service
        .record_forecasts(
            city=CITY,
            forecast_created_at=(
                latest_timestamp
            ),
            predictions=predictions,
        )
    )

    print(
        f"Ledger rows: {len(ledger)}"
    )

    print(
        "\nStored forecasts:"
    )

    print(
        ledger[
            [
                "city",
                "forecast_created_at",
                "target_timestamp",
                "horizon_hours",
                "predicted_aqi",
                "predicted_category",
            ]
        ].to_string(
            index=False
        )
    )

    if len(ledger) != 3:
        raise AssertionError(
            "Expected exactly three "
            "forecast ledger rows."
        )

    horizons = set(
        ledger[
            "horizon_hours"
        ]
        .astype(int)
        .tolist()
    )

    expected_horizons = {
        24,
        48,
        72,
    }

    if horizons != expected_horizons:
        raise AssertionError(
            "Expected horizons "
            "{24, 48, 72}. "
            f"Received: {horizons}"
        )

    if not (
        ledger["actual_aqi"]
        .isna()
        .all()
    ):
        raise AssertionError(
            "New future forecasts must "
            "remain unevaluated."
        )

    print(
        "\nFuture forecasts correctly "
        "remain pending: PASS"
    )

    # =========================================================
    # 6. TEST IDEMPOTENCY
    # =========================================================

    print(
        "\nTesting idempotency..."
    )

    duplicate_ledger = (
        monitoring_service
        .record_forecasts(
            city=CITY,
            forecast_created_at=(
                latest_timestamp
            ),
            predictions=predictions,
        )
    )

    print(
        "Rows after duplicate write: "
        f"{len(duplicate_ledger)}"
    )

    if len(duplicate_ledger) != 3:
        raise AssertionError(
            "Duplicate forecast rows "
            "were inserted."
        )

    print(
        "Duplicate protection: PASS"
    )

    # =========================================================
    # 7. SIMULATE FUTURE ACTUAL VALUES
    #
    # These values exist ONLY to test the monitoring logic.
    # They are NOT real model-performance results.
    # =========================================================

    print(
        "\nSimulating matured forecasts..."
    )

    actual_values = {
        24: 160.0,
        48: 135.0,
        72: 90.0,
    }

    simulated_history = (
        history.copy()
    )

    simulation_rows = []

    base_row = (
        simulated_history
        .iloc[-1]
        .copy()
    )

    for (
        horizon,
        actual_aqi,
    ) in actual_values.items():

        row = base_row.copy()

        row["timestamp"] = (
            latest_timestamp
            + pd.Timedelta(
                hours=horizon
            )
        )

        row["aqi_current"] = (
            actual_aqi
        )

        simulation_rows.append(
            row
        )

    simulated_future = pd.DataFrame(
        simulation_rows
    )

    simulated_history = pd.concat(
        [
            simulated_history,
            simulated_future,
        ],
        ignore_index=True,
    )

    evaluated_at = (
        latest_timestamp
        + pd.Timedelta(
            hours=73
        )
    )

    # =========================================================
    # 8. EVALUATE MATURED FORECASTS
    # =========================================================

    evaluated = (
        monitoring_service
        .evaluate_available_forecasts(
            city=CITY,
            history=simulated_history,
            evaluated_at=evaluated_at,
        )
    )

    print(
        "\nEvaluated forecasts:"
    )

    print(
        evaluated[
            [
                "horizon_hours",
                "predicted_aqi",
                "actual_aqi",
                "absolute_error",
                "predicted_category",
                "actual_category",
                "category_correct",
                "category_distance",
                "adjacent_category_correct",
            ]
        ].to_string(
            index=False
        )
    )

    if not (
        evaluated["actual_aqi"]
        .notna()
        .all()
    ):
        raise AssertionError(
            "Not all matured forecasts "
            "were evaluated."
        )

    if not (
        evaluated["absolute_error"]
        .notna()
        .all()
    ):
        raise AssertionError(
            "Absolute errors were not "
            "calculated for every forecast."
        )

    print(
        "\nForecast maturation: PASS"
    )

    # =========================================================
    # 9. VERIFY ACTUAL VALUES AND ERRORS
    # =========================================================

    for _, row in evaluated.iterrows():

        horizon = int(
            row["horizon_hours"]
        )

        predicted_aqi = float(
            row["predicted_aqi"]
        )

        actual_aqi = float(
            row["actual_aqi"]
        )

        expected_actual = (
            actual_values[horizon]
        )

        if abs(
            actual_aqi
            - expected_actual
        ) > 1e-9:
            raise AssertionError(
                f"{horizon}h actual AQI "
                "does not match simulation."
            )

        expected_error = abs(
            actual_aqi
            - predicted_aqi
        )

        stored_error = float(
            row["absolute_error"]
        )

        if abs(
            expected_error
            - stored_error
        ) > 1e-9:
            raise AssertionError(
                f"{horizon}h absolute "
                "error calculation failed."
            )

    print(
        "Absolute-error calculation: PASS"
    )

    # =========================================================
    # 10. PERFORMANCE SUMMARY
    # =========================================================

    summary = (
        monitoring_service
        .performance_summary(
            city=CITY
        )
    )

    print(
        "\nLive performance summary:"
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.2f}"
            ),
        )
    )

    required_summary_columns = {
        "horizon_hours",
        "evaluated_forecasts",
        "mae",
        "rmse",
        "within_10_aqi_pct",
        "within_20_aqi_pct",
        "within_30_aqi_pct",
        "category_accuracy_pct",
        "adjacent_category_accuracy_pct",
    }

    missing_summary_columns = (
        required_summary_columns
        - set(summary.columns)
    )

    if missing_summary_columns:
        raise AssertionError(
            "Performance summary missing "
            f"columns: "
            f"{sorted(missing_summary_columns)}"
        )

    if len(summary) != 3:
        raise AssertionError(
            "Expected one summary row "
            "per horizon."
        )

    # =========================================================
    # 11. ADJACENT-CATEGORY CHECK
    # =========================================================

    print(
        "\nAdjacent-category evaluation:"
    )

    for _, row in evaluated.iterrows():

        horizon = int(
            row["horizon_hours"]
        )

        print(
            f"  {horizon:>2}h: "
            f"predicted="
            f"'{row['predicted_category']}' | "
            f"actual="
            f"'{row['actual_category']}' | "
            f"distance="
            f"{row['category_distance']} | "
            f"adjacent="
            f"{row['adjacent_category_correct']}"
        )

    # =========================================================
    # 12. VERIFY TEST LEDGER EXISTS
    # =========================================================

    if not TEST_LEDGER.exists():
        raise AssertionError(
            "Test forecast ledger "
            "was not created."
        )

    print(
        "\nTest ledger:"
    )

    print(
        f"  {TEST_LEDGER}"
    )

    print(
        "\nNOTE:"
    )

    print(
        "  Simulated future AQI values "
        "were used only to test forecast "
        "maturation and scoring."
    )

    print(
        "  They must not be reported as "
        "real model performance."
    )

    # =========================================================
    # FINAL RESULT
    # =========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "FORECAST MONITORING TEST: PASS"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()