from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.services.prediction_service import (
    AQIPredictionService,
)


DATA_PATH = Path(
    "data/splits/expanded/final/train.parquet"
)


def main() -> None:
    print("=" * 80)
    print(
        "PEARLS AQI - PRODUCTION "
        "PREDICTION SERVICE TEST"
    )
    print("=" * 80)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_parquet(
        DATA_PATH
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    sample = (
        df[
            df["city"]
            == "Islamabad"
        ]
        .sort_values(
            "timestamp"
        )
        .tail(1)
        .copy()
    )

    if sample.empty:
        raise RuntimeError(
            "No Islamabad sample found."
        )

    print("\nInput:")
    print(
        f"  City:      "
        f"{sample.iloc[0]['city']}"
    )
    print(
        f"  Timestamp: "
        f"{sample.iloc[0]['timestamp']}"
    )
    print(
        f"  Current AQI: "
        f"{sample.iloc[0]['aqi_current']}"
    )

    service = (
        AQIPredictionService()
    )

    results = (
        service.predict_all(
            sample
        )
    )

    print("\nPredictions:")

    for result in results:
        print(
            f"  {result.horizon}: "
            f"AQI={result.predicted_aqi:.2f} | "
            f"{result.predicted_category} | "
            f"{result.model_name}"
            f"@{result.model_alias}"
        )

    expected_aliases = {
        "24h": "champion-24h",
        "48h": "champion-48h",
        "72h": "champion-72h",
    }

    for result in results:
        if (
            result.model_name
            != "pearls-aqi-xgboost"
        ):
            raise RuntimeError(
                "Prediction service loaded "
                "the wrong registry."
            )

        if (
            result.model_alias
            != expected_aliases[
                result.horizon
            ]
        ):
            raise RuntimeError(
                f"Incorrect alias for "
                f"{result.horizon}."
            )

        if not (
            0.0
            <= result.predicted_aqi
            <= 500.0
        ):
            raise RuntimeError(
                "Prediction outside AQI range."
            )

    print("\n" + "=" * 80)
    print(
        "PRODUCTION PREDICTION "
        "SERVICE TEST: PASS"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()