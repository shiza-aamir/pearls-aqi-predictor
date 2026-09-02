from __future__ import annotations

from pathlib import Path

import mlflow
import mlflow.xgboost

TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "pearls-aqi-xgboost"

OUTPUT_DIR = Path("models/production")

HORIZON_ALIASES = {
    "24h": "champion-24h",
    "48h": "champion-48h",
    "72h": "champion-72h",
}


def main() -> None:
    mlflow.set_tracking_uri(
        TRACKING_URI
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for horizon, alias in HORIZON_ALIASES.items():
        model_uri = (
            f"models:/"
            f"{MODEL_NAME}"
            f"@{alias}"
        )

        print(
            f"Loading {horizon} champion "
            f"from {model_uri}"
        )

        model = mlflow.xgboost.load_model(
            model_uri
        )

        output_path = (
            OUTPUT_DIR
            / f"xgboost_{horizon}.ubj"
        )

        model.save_model(
            output_path
        )

        print(
            f"Saved {horizon} model to "
            f"{output_path}"
        )

    print(
        "\nProduction model export complete."
    )


if __name__ == "__main__":
    main()