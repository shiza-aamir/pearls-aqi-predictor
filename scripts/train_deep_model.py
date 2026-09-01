from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.ml.training.deep_trainer import (
    DeepLearningTrainer,
)
from src.ml.training.sequence_builder import (
    AQISequenceBuilder,
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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a deep-learning model "
            "for AQI forecasting."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "cnn",
            "gru",
            "cnn_lstm",
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

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=72,
    )

    return parser.parse_args()


def load_split(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Split file does not exist: {path}"
        )

    dataframe = pd.read_parquet(path)

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"]
    )

    return dataframe


def main() -> None:
    args = parse_arguments()

    print("=" * 72)
    print("PEARLS AQI - DEEP LEARNING TRAINING")
    print("=" * 72)

    print(
        f"Model: {args.model}"
    )

    print(
        f"Horizon: {args.horizon}"
    )

    print(
        f"Sequence length: "
        f"{args.sequence_length} hours"
    )

    print(
        f"Maximum epochs: "
        f"{args.epochs}"
    )

    print(
        f"Batch size: "
        f"{args.batch_size}"
    )

    print("\nLoading chronological splits...")

    train_dataframe = load_split(
        TRAIN_PATH
    )

    validation_dataframe = load_split(
        VALIDATION_PATH
    )

    test_dataframe = load_split(
        TEST_PATH
    )

    print(
        f"Train rows: "
        f"{len(train_dataframe):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation_dataframe):,}"
    )

    print(
        f"Test rows: "
        f"{len(test_dataframe):,}"
    )

    sequence_builder = AQISequenceBuilder(
        sequence_length=(
            args.sequence_length
        )
    )

    print(
        "\nFitting feature scaler "
        "on TRAINING data only..."
    )

    sequence_builder.fit_scaler(
        train_dataframe
    )

    print(
        "Building training sequences..."
    )

    train_data = sequence_builder.build(
        dataframe=train_dataframe,
        horizon=args.horizon,
    )

    print(
        "Building validation sequences..."
    )

    validation_data = sequence_builder.build(
        dataframe=validation_dataframe,
        horizon=args.horizon,
    )

    print(
        "Building test sequences..."
    )

    test_data = sequence_builder.build(
        dataframe=test_dataframe,
        horizon=args.horizon,
    )

    print("\nSequence shapes:")
    print(
        f"Train:      {train_data.X.shape}"
    )
    print(
        f"Validation: "
        f"{validation_data.X.shape}"
    )
    print(
        f"Test:       {test_data.X.shape}"
    )

    print(
        f"\nInput feature count: "
        f"{train_data.X.shape[2]}"
    )

    output_directory = (
        Path("artifacts")
        / "deep_learning"
        / args.model
        / args.horizon
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    scaler_path = (
        output_directory
        / "feature_scaler.joblib"
    )

    joblib.dump(
        sequence_builder.scaler,
        scaler_path,
    )

    print(
        f"Scaler saved: {scaler_path}"
    )

    trainer = DeepLearningTrainer(
        model_name=args.model,
        horizon=args.horizon,
        sequence_length=(
            args.sequence_length
        ),
        feature_count=(
            train_data.X.shape[2]
        ),
    )

    print("\nStarting training...\n")

    (
        _,
        validation_metrics,
        test_metrics,
    ) = trainer.train(
        train_data=train_data,
        validation_data=validation_data,
        test_data=test_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    print("\n" + "=" * 72)
    print("TRAINING COMPLETE")
    print("=" * 72)

    print("\nValidation metrics")
    print(
        f"MAE:  "
        f"{validation_metrics.mae:.3f}"
    )
    print(
        f"RMSE: "
        f"{validation_metrics.rmse:.3f}"
    )
    print(
        f"R2:   "
        f"{validation_metrics.r2:.3f}"
    )

    print("\nTest metrics")
    print(
        f"MAE:  "
        f"{test_metrics.mae:.3f}"
    )
    print(
        f"RMSE: "
        f"{test_metrics.rmse:.3f}"
    )
    print(
        f"R2:   "
        f"{test_metrics.r2:.3f}"
    )

    print(
        "\nArtifacts:"
    )

    print(
        output_directory
    )

    print("=" * 72)


if __name__ == "__main__":
    main()