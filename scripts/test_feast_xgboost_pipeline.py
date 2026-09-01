from __future__ import annotations

from src.services.feature_service import (
    AQIFeatureService,
)
from src.services.prediction_service import (
    AQIPredictionService,
)


CITY = "Islamabad"


def main() -> None:
    print("=" * 80)
    print(
        "PEARLS AQI - FEAST -> MLFLOW "
        "XGBOOST PIPELINE"
    )
    print("=" * 80)

    feature_service = (
        AQIFeatureService()
    )

    prediction_service = (
        AQIPredictionService()
    )

    print(
        f"\nRetrieving online features "
        f"for {CITY}..."
    )

    features = (
        feature_service
        .get_online_features(
            CITY
        )
    )

    print(
        f"Retrieved: {features.shape[1]} features"
    )

    predictions = (
        prediction_service
        .predict_all(
            features
        )
    )

    print("\nForecasts:")

    for prediction in predictions:
        print(
            f"  {prediction.horizon}: "
            f"AQI={prediction.predicted_aqi:.2f} | "
            f"{prediction.predicted_category} | "
            f"{prediction.model_name}"
            f"@{prediction.model_alias}"
        )

    if len(predictions) != 3:
        raise RuntimeError(
            "Expected three forecast horizons."
        )

    print("\n" + "=" * 80)
    print(
        "FEAST -> MLFLOW XGBOOST "
        "PIPELINE: PASS"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()