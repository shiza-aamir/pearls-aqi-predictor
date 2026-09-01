from pathlib import Path

import pandas as pd

from src.services.prediction_service import (
    AQIPredictionService,
)


TEST_PATH = Path(
    "data/processed/splits/test.parquet"
)


def main() -> None:
    df = pd.read_parquet(
        TEST_PATH
    )

    islamabad = (
        df[
            df["city"] == "Islamabad"
        ]
        .sort_values("timestamp")
        .iloc[[-1]]
        .copy()
    )

    print("=" * 70)
    print("REGISTERED MODEL INFERENCE TEST")
    print("=" * 70)

    print(
        f"City:      "
        f"{islamabad['city'].iloc[0]}"
    )

    print(
        f"Timestamp: "
        f"{islamabad['timestamp'].iloc[0]}"
    )

    service = AQIPredictionService()

    predictions = service.predict_all(
        islamabad
    )

    for prediction in predictions:
        print("\n" + "-" * 40)

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

    print("\n" + "=" * 70)
    print("INFERENCE TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()