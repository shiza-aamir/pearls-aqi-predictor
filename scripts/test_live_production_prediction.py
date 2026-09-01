from __future__ import annotations

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


def main() -> None:
    print("=" * 80)
    print(
        "PEARLS AQI - LIVE PRODUCTION "
        "PREDICTION TEST"
    )
    print("=" * 80)

    history_service = (
        LiveHistoryService()
    )

    feature_pipeline = (
        LiveFeaturePipeline()
    )

    prediction_service = (
        AQIPredictionService()
    )

    print(
        f"\nUpdating live history for {CITY}..."
    )

    history = (
        history_service
        .ensure_current_history(
            CITY
        )
    )

    latest, features = (
        feature_pipeline
        .build_latest_features(
            history
        )
    )

    row = latest.iloc[0]

    print(
        f"Timestamp:   {row['timestamp']}"
    )
    print(
        f"Current AQI: {row['aqi_current']:.0f}"
    )
    print(
        f"PM2.5:      {row['pm2_5']:.2f}"
    )
    print(
        f"PM10:       {row['pm10']:.2f}"
    )
    print(
        f"Features:   {features.shape[1]}"
    )

    print(
        "\nLoading MLflow XGBoost champions..."
    )

    forecasts = (
        prediction_service.predict_all(
            features
        )
    )

    if len(forecasts) != 3:
        raise RuntimeError(
            "Expected exactly 3 forecasts, "
            f"got {len(forecasts)}."
        )

    expected_horizons = {
        "24h",
        "48h",
        "72h",
    }

    actual_horizons = {
        forecast.horizon
        for forecast in forecasts
    }

    if actual_horizons != expected_horizons:
        raise RuntimeError(
            "Unexpected forecast horizons: "
            f"{actual_horizons}"
        )

    print(
        "\nForecasts:"
    )

    for forecast in forecasts:
        print(
            f"  +{forecast.horizon:<3}  "
            f"AQI={forecast.predicted_aqi:>7.2f}  "
            f"{forecast.predicted_category}"
        )

        print(
            f"        model="
            f"{forecast.model_name}"
        )

        print(
            f"        alias="
            f"{forecast.model_alias}"
        )

    print("\n" + "=" * 80)
    print(
        "LIVE PRODUCTION PREDICTION TEST: PASS"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()