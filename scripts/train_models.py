import argparse
from pathlib import Path

import mlflow
import pandas as pd

from src.ml.training.trainer import (
    AQIModelTrainer,
)


TRAIN_PATH = Path(
    "data/processed/splits/train.parquet"
)

VALIDATION_PATH = Path(
    "data/processed/splits/validation.parquet"
)

TEST_PATH = Path(
    "data/processed/splits/test.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one AQI forecasting model."
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "persistence",
            "ridge",
            "random_forest",
            "xgboost",
        ],
    )

    parser.add_argument(
        "--horizon",
        required=True,
        choices=[
            "24h",
            "48h",
            "72h",
        ],
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    mlflow.set_tracking_uri(
        "sqlite:///mlflow.db"
    )

    train_df = pd.read_parquet(
        TRAIN_PATH
    )

    validation_df = pd.read_parquet(
        VALIDATION_PATH
    )

    test_df = pd.read_parquet(
        TEST_PATH
    )

    trainer = AQIModelTrainer()

    result = trainer.train_single(
        model_name=args.model,
        horizon=args.horizon,
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
    )

    val = result.validation_metrics
    test = result.test_metrics

    print("=" * 70)
    print("AQI MODEL RUN COMPLETE")
    print("=" * 70)

    print(
        f"Model:     {result.model_name}"
    )

    print(
        f"Horizon:   {result.horizon}"
    )

    print("\nValidation:")
    print(f"  MAE:  {val.mae:.3f}")
    print(f"  RMSE: {val.rmse:.3f}")
    print(f"  R2:   {val.r2:.3f}")

    print("\nTest:")
    print(f"  MAE:  {test.mae:.3f}")
    print(f"  RMSE: {test.rmse:.3f}")
    print(f"  R2:   {test.r2:.3f}")

    print(
        "\nResults logged to MLflow."
    )


if __name__ == "__main__":
    main()