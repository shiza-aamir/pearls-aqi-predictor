from __future__ import annotations

from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd

from src.features.engineer import AQIFeatureEngineer


TRACKING_URI = "sqlite:///mlflow.db"

MODEL_NAME = "pearls-aqi-xgboost"

SAMPLE_PATH = Path(
    "data/splits/expanded/final/train.parquet"
)

ALIASES = {
    "24h": "champion-24h",
    "48h": "champion-48h",
    "72h": "champion-72h",
}


def main() -> None:
    print("=" * 80)
    print(
        "PEARLS AQI - XGBOOST REGISTRY "
        "INFERENCE TEST"
    )
    print("=" * 80)

    mlflow.set_tracking_uri(
        TRACKING_URI
    )

    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(
            f"Missing sample dataset: {SAMPLE_PATH}"
        )

    feature_columns = (
        AQIFeatureEngineer
        .get_model_feature_columns()
    )

    if len(feature_columns) != 56:
        raise ValueError(
            f"Expected 56 features, "
            f"got {len(feature_columns)}."
        )

    dataframe = pd.read_parquet(
        SAMPLE_PATH
    )

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"],
        utc=True,
    )

    sample = (
        dataframe[
            dataframe["city"]
            == "Islamabad"
        ]
        .sort_values("timestamp")
        .tail(1)
        .copy()
    )

    if sample.empty:
        raise RuntimeError(
            "No Islamabad sample found."
        )

    missing = [
        column
        for column in feature_columns
        if column not in sample.columns
    ]

    if missing:
        raise ValueError(
            f"Missing model features: {missing}"
        )

    null_count = int(
        sample[
            feature_columns
        ]
        .isna()
        .sum()
        .sum()
    )

    if null_count:
        raise ValueError(
            f"Sample contains {null_count} "
            "missing feature cells."
        )

    x = sample[
        feature_columns
    ]

    print("\nInput sample:")
    print(
        f"  City:      {sample.iloc[0]['city']}"
    )
    print(
        f"  Timestamp: {sample.iloc[0]['timestamp']}"
    )
    print(
        f"  AQI now:   "
        f"{sample.iloc[0]['aqi_current']:.2f}"
    )
    print(
        f"  Features:  {x.shape[1]}"
    )

    predictions = {}

    for horizon, alias in ALIASES.items():
        print("\n" + "-" * 80)
        print(
            f"LOADING {horizon} MODEL"
        )
        print("-" * 80)

        model_uri = (
            f"models:/{MODEL_NAME}@{alias}"
        )

        print(
            f"URI: {model_uri}"
        )

        model = (
            mlflow.xgboost.load_model(
                model_uri
            )
        )

        prediction = model.predict(
            x
        )

        if len(prediction) != 1:
            raise RuntimeError(
                f"{horizon}: expected one "
                "prediction."
            )

        value = float(
            np.clip(
                prediction[0],
                0.0,
                500.0,
            )
        )

        if not np.isfinite(value):
            raise RuntimeError(
                f"{horizon}: non-finite prediction."
            )

        predictions[
            horizon
        ] = value

        print(
            f"Prediction: {value:.2f}"
        )

    print("\n" + "=" * 80)
    print(
        "REGISTRY INFERENCE TEST: PASS"
    )
    print("=" * 80)

    print(
        f"\n24h AQI: {predictions['24h']:.2f}"
    )
    print(
        f"48h AQI: {predictions['48h']:.2f}"
    )
    print(
        f"72h AQI: {predictions['72h']:.2f}"
    )


if __name__ == "__main__":
    main()