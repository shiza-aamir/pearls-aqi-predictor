from __future__ import annotations

from pathlib import Path

import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb

from src.features.engineer import AQIFeatureEngineer

TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "pearls-aqi-xgboost"

MODEL_DIR = Path(
    "models/production"
)

HORIZON_ALIASES = {
    "24h": "champion-24h",
    "48h": "champion-48h",
    "72h": "champion-72h",
}

DATA_PATH = Path(
    "data/processed/splits/validation.parquet"
)


def main() -> None:
    mlflow.set_tracking_uri(
        TRACKING_URI
    )

    df = pd.read_parquet(
        DATA_PATH
    )

    feature_columns = (
        AQIFeatureEngineer
        .get_model_feature_columns()
    )

    sample = (
        df[feature_columns]
        .dropna()
        .head(10)
        .astype(float)
    )

    if sample.empty:
        raise RuntimeError(
            "No validation rows available."
        )

    for horizon, alias in HORIZON_ALIASES.items():
        registry_model = (
            mlflow.xgboost.load_model(
                f"models:/"
                f"{MODEL_NAME}"
                f"@{alias}"
            )
        )

        deployed_model = (
            xgb.XGBRegressor()
        )

        deployed_model.load_model(
            MODEL_DIR
            / f"xgboost_{horizon}.ubj"
        )

        registry_predictions = (
            registry_model.predict(
                sample
            )
        )

        deployed_predictions = (
            deployed_model.predict(
                sample
            )
        )

        difference = abs(
            registry_predictions
            - deployed_predictions
        ).max()

        print(
            f"{horizon}: "
            f"max difference = "
            f"{difference:.12f}"
        )

        if difference > 1e-6:
            raise RuntimeError(
                f"{horizon}: exported model "
                "does not match registry model."
            )

    print(
        "\nAll exported production models "
        "match the MLflow champions."
    )


if __name__ == "__main__":
    main()