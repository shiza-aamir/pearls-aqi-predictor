from __future__ import annotations

import numpy as np

from src.services.feature_service import (
    AQIFeatureService,
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

ABSOLUTE_TOLERANCE = 1e-9


def main() -> None:
    print("=" * 80)
    print(
        "PEARLS AQI - LIVE FEAST "
        "ONLINE PIPELINE TEST"
    )
    print("=" * 80)

    history_service = (
        LiveHistoryService()
    )

    feature_pipeline = (
        LiveFeaturePipeline()
    )

    feast_service = (
        AQIFeatureService()
    )

    prediction_service = (
        AQIPredictionService()
    )

    print(
        f"\nPreparing live data for {CITY}..."
    )

    history = (
        history_service
        .ensure_current_history(
            CITY
        )
    )

    latest, direct_features = (
        feature_pipeline
        .build_latest_features(
            history
        )
    )

    timestamp = (
        latest.iloc[0][
            "timestamp"
        ]
    )

    current_aqi = float(
        latest.iloc[0][
            "aqi_current"
        ]
    )

    print(
        f"Timestamp:       {timestamp}"
    )
    print(
        f"Current AQI:     {current_aqi:.0f}"
    )
    print(
        f"Direct features: "
        f"{direct_features.shape[1]}"
    )

    if direct_features.shape != (1, 56):
        raise RuntimeError(
            "Expected direct feature row "
            "with shape (1, 56)."
        )

    print(
        "\nWriting live row to "
        "Feast online store..."
    )

    feast_service.write_online_features(
        city=CITY,
        event_timestamp=timestamp,
        feature_row=direct_features,
    )

    print(
        "Retrieving row back from "
        "Feast online store..."
    )

    feast_features = (
        feast_service
        .get_online_features(
            CITY
        )
    )

    print(
        f"Feast features:  "
        f"{feast_features.shape[1]}"
    )

    if feast_features.shape != (1, 56):
        raise RuntimeError(
            "Expected Feast feature row "
            "with shape (1, 56)."
        )

    direct = (
        direct_features[
            feast_service.feature_columns
        ]
        .astype(float)
        .to_numpy()
    )

    retrieved = (
        feast_features[
            feast_service.feature_columns
        ]
        .astype(float)
        .to_numpy()
    )

    differences = np.abs(
        direct - retrieved
    )

    maximum_difference = float(
        differences.max()
    )

    mismatched_cells = int(
        (
            differences
            > ABSOLUTE_TOLERANCE
        ).sum()
    )

    print(
        f"Maximum diff:    "
        f"{maximum_difference:.12f}"
    )

    print(
        f"Mismatched:      "
        f"{mismatched_cells}"
    )

    if mismatched_cells != 0:
        flat_index = int(
            np.argmax(
                differences
            )
        )

        feature_index = (
            flat_index
            % len(
                feast_service
                .feature_columns
            )
        )

        feature_name = (
            feast_service
            .feature_columns[
                feature_index
            ]
        )

        raise RuntimeError(
            "Feast live feature parity "
            "check failed. "
            f"Largest mismatch: "
            f"{feature_name}, "
            f"direct="
            f"{direct[0, feature_index]}, "
            f"Feast="
            f"{retrieved[0, feature_index]}, "
            f"difference="
            f"{maximum_difference}."
        )

    print(
        "\n56/56 Feast feature "
        "parity: PASS"
    )

    print(
        "\nPredicting FROM FEAST "
        "retrieved features..."
    )

    forecasts = (
        prediction_service.predict_all(
            feast_features
        )
    )

    if len(forecasts) != 3:
        raise RuntimeError(
            "Expected 3 predictions."
        )

    print(
        "\nForecasts from Feast:"
    )

    for forecast in forecasts:
        print(
            f"  +{forecast.horizon:<3}  "
            f"AQI="
            f"{forecast.predicted_aqi:>7.2f}  "
            f"{forecast.predicted_category}"
        )

        print(
            f"        alias="
            f"{forecast.model_alias}"
        )

    print(
        "\n" + "=" * 80
    )

    print(
        "LIVE FEAST ONLINE "
        "PIPELINE TEST: PASS"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()