from src.services.feature_service import (
    AQIFeatureService,
)
from src.services.prediction_service import (
    AQIPredictionService,
)


def main() -> None:
    city = "Islamabad"

    print("=" * 70)
    print(
        "FEAST + MLFLOW END-TO-END "
        "INFERENCE TEST"
    )
    print("=" * 70)

    print(
        f"City: {city}"
    )

    feature_service = AQIFeatureService()

    print(
        "\nRetrieving model features "
        "from Feast..."
    )

    feature_row = (
        feature_service
        .get_online_features(
            city=city
        )
    )

    print(
        "Feature retrieval successful."
    )

    print(
        f"Feature count: "
        f"{len(feature_row.columns)}"
    )

    print(
        f"Rows: "
        f"{len(feature_row)}"
    )

    print(
        f"Missing values: "
        f"{int(feature_row.isnull().sum().sum())}"
    )

    print(
        "\nSelected Feast features:"
    )

    selected_features = [
        "latitude",
        "longitude",
        "temperature",
        "humidity",
        "pm2_5",
        "pm10",
        "aqi_lag_24h",
        "aqi_rolling_mean_24h",
    ]

    for feature in selected_features:
        print(
            f"{feature:<25} "
            f"{feature_row[feature].iloc[0]:.4f}"
        )

    print(
        "\nLoading registered MLflow "
        "models..."
    )

    prediction_service = (
        AQIPredictionService()
    )

    predictions = (
        prediction_service
        .predict_all(
            feature_row=feature_row
        )
    )

    print(
        "\nForecasts:"
    )

    for prediction in predictions:
        print(
            "\n"
            + "-" * 45
        )

        print(
            f"Horizon:   "
            f"{prediction.horizon}"
        )

        print(
            f"AQI:       "
            f"{prediction.predicted_aqi:.2f}"
        )

        print(
            f"Category:  "
            f"{prediction.predicted_category}"
        )

        print(
            f"Model:     "
            f"{prediction.model_name}"
        )

        print(
            f"Alias:     "
            f"@{prediction.model_alias}"
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FEAST + MLFLOW INFERENCE "
        "TEST COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()