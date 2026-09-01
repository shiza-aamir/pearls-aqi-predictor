from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

os.environ.setdefault(
    "TF_CPP_MIN_LOG_LEVEL",
    "2",
)

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from src.ml.evaluation.metrics import (
    calculate_regression_metrics,
)
from src.ml.models.cnn import (
    build_cnn_model,
)
from src.ml.models.cnn_lstm import (
    build_cnn_lstm_model,
)
from src.ml.models.gru import (
    build_gru_model,
)
from src.ml.training.sequence_builder import (
    AQISequenceBuilder,
)


SPLIT_ROOT = Path(
    "data/splits/expanded"
)

OUTPUT_ROOT = Path(
    "artifacts/expanded_deep_benchmark"
)

FOLDS = [
    "fold_1",
    "fold_2",
    "fold_3",
]

HORIZONS = [
    "24h",
    "48h",
    "72h",
]

SUPPORTED_MODELS = [
    "cnn",
    "gru",
    "cnn_lstm",
]

SEQUENCE_LENGTH = 72


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Walk-forward deep-learning "
            "benchmark for expanded AQI data."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=SUPPORTED_MODELS,
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
        "--fold",
        choices=[
            "all",
            *FOLDS,
        ],
        default="all",
    )

    parser.add_argument(
        "--horizon",
        choices=[
            "all",
            *HORIZONS,
        ],
        default="all",
    )

    return parser.parse_args()


def configure_tensorflow() -> None:
    try:
        tf.config.threading.set_intra_op_parallelism_threads(
            2
        )

        tf.config.threading.set_inter_op_parallelism_threads(
            1
        )

    except RuntimeError:
        pass


def load_split(
    fold: str,
    split: str,
) -> pd.DataFrame:
    path = (
        SPLIT_ROOT
        / fold
        / f"{split}.parquet"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Split not found: {path}"
        )

    dataframe = pd.read_parquet(
        path
    )

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"],
        utc=True,
    )

    dataframe = (
        dataframe
        .sort_values(
            [
                "city",
                "timestamp",
            ]
        )
        .reset_index(drop=True)
    )

    return dataframe


def build_model(
    model_name: str,
    sequence_length: int,
    feature_count: int,
) -> keras.Model:
    if model_name == "cnn":
        return build_cnn_model(
            sequence_length=sequence_length,
            feature_count=feature_count,
        )

    if model_name == "gru":
        return build_gru_model(
            sequence_length=sequence_length,
            feature_count=feature_count,
        )

    if model_name == "cnn_lstm":
        return build_cnn_lstm_model(
            sequence_length=sequence_length,
            feature_count=feature_count,
        )

    raise ValueError(
        f"Unsupported model: {model_name}"
    )


def validate_sequence_keys(
    cities: np.ndarray,
    timestamps: np.ndarray,
    label: str,
) -> None:
    key_frame = pd.DataFrame(
        {
            "city": cities.astype(str),
            "timestamp": pd.to_datetime(
                timestamps,
                utc=True,
            ),
        }
    )

    duplicates = int(
        key_frame.duplicated(
            subset=[
                "city",
                "timestamp",
            ]
        ).sum()
    )

    if duplicates:
        raise AssertionError(
            f"{label}: found "
            f"{duplicates} duplicate "
            "sequence keys."
        )


def train_one(
    model_name: str,
    fold: str,
    horizon: str,
    epochs: int,
    batch_size: int,
) -> dict:
    print(
        "\n" + "=" * 90
    )

    print(
        f"{model_name.upper()} | "
        f"{fold.upper()} | "
        f"{horizon}"
    )

    print(
        "=" * 90
    )

    train_dataframe = load_split(
        fold,
        "train",
    )

    validation_dataframe = load_split(
        fold,
        "validation",
    )

    print(
        f"Train rows: "
        f"{len(train_dataframe):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation_dataframe):,}"
    )

    sequence_builder = (
        AQISequenceBuilder(
            sequence_length=SEQUENCE_LENGTH
        )
    )

    print(
        "\nFitting scaler on training "
        "features only..."
    )

    sequence_builder.fit_scaler(
        train_dataframe
    )

    print(
        "Building training sequences..."
    )

    train_data = (
        sequence_builder.build(
            dataframe=train_dataframe,
            horizon=horizon,
        )
    )

    print(
        "Building validation sequences..."
    )

    validation_data = (
        sequence_builder.build(
            dataframe=validation_dataframe,
            horizon=horizon,
        )
    )

    validate_sequence_keys(
        train_data.cities,
        train_data.timestamps,
        "train",
    )

    validate_sequence_keys(
        validation_data.cities,
        validation_data.timestamps,
        "validation",
    )

    print(
        "\nSequence shapes:"
    )

    print(
        f"  Train:      "
        f"{train_data.X.shape}"
    )

    print(
        f"  Validation: "
        f"{validation_data.X.shape}"
    )

    print(
        f"  Feature count: "
        f"{train_data.X.shape[2]}"
    )

    expected_validation_sequences = (
        len(validation_dataframe)
        - (
            SEQUENCE_LENGTH - 1
        )
        * validation_dataframe[
            "city"
        ].nunique()
    )

    if (
        len(validation_data.y)
        != expected_validation_sequences
    ):
        raise AssertionError(
            "Unexpected validation sequence "
            "count. "
            f"Expected "
            f"{expected_validation_sequences:,}, "
            f"got {len(validation_data.y):,}."
        )

    print(
        f"  Expected validation sequences: "
        f"{expected_validation_sequences:,}"
    )

    tf.keras.backend.clear_session()
    gc.collect()

    model = build_model(
        model_name=model_name,
        sequence_length=SEQUENCE_LENGTH,
        feature_count=(
            train_data.X.shape[2]
        ),
    )

    output_directory = (
        OUTPUT_ROOT
        / model_name
        / fold
        / horizon
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        output_directory
        / "best_model.keras"
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            min_delta=0.0001,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=0.00001,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    fit_start = (
        time.perf_counter()
    )

    history = model.fit(
        train_data.X,
        train_data.y,
        validation_data=(
            validation_data.X,
            validation_data.y,
        ),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=2,
        callbacks=callbacks,
    )

    fit_seconds = (
        time.perf_counter()
        - fit_start
    )

    prediction_start = (
        time.perf_counter()
    )

    predictions = (
        model.predict(
            validation_data.X,
            batch_size=batch_size,
            verbose=0,
        )
        .reshape(-1)
    )

    predict_seconds = (
        time.perf_counter()
        - prediction_start
    )

    predictions = np.clip(
        predictions,
        0.0,
        500.0,
    )

    metrics = (
        calculate_regression_metrics(
            validation_data.y,
            predictions,
        )
    )

    completed_epochs = len(
        history.history["loss"]
    )

    print(
        "\nValidation metrics:"
    )

    print(
        f"  MAE:  {metrics.mae:.4f}"
    )

    print(
        f"  RMSE: {metrics.rmse:.4f}"
    )

    print(
        f"  R2:   {metrics.r2:.4f}"
    )

    print(
        f"  Epochs: "
        f"{completed_epochs}"
    )

    print(
        f"  Fit time: "
        f"{fit_seconds / 60:.2f} min"
    )

    model.save(
        output_directory
        / "final_model.keras"
    )

    joblib.dump(
        sequence_builder.scaler,
        output_directory
        / "feature_scaler.joblib",
    )

    history_payload = {
        key: [
            float(value)
            for value in values
        ]
        for key, values
        in history.history.items()
    }

    (
        output_directory
        / "history.json"
    ).write_text(
        json.dumps(
            history_payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    prediction_frame = pd.DataFrame(
        {
            "city": (
                validation_data.cities
                .astype(str)
            ),
            "timestamp": pd.to_datetime(
                validation_data.timestamps,
                utc=True,
            ),
            "y_true": (
                validation_data.y
                .astype(float)
            ),
            "y_pred": (
                predictions
                .astype(float)
            ),
        }
    )

    prediction_frame.to_parquet(
        output_directory
        / "validation_predictions.parquet",
        index=False,
    )

    prediction_frame.to_csv(
        output_directory
        / "validation_predictions.csv",
        index=False,
    )

    result = {
        "model": model_name,
        "fold": fold,
        "horizon": horizon,
        "train_rows": int(
            len(train_dataframe)
        ),
        "validation_rows": int(
            len(validation_dataframe)
        ),
        "train_sequences": int(
            len(train_data.y)
        ),
        "validation_sequences": int(
            len(validation_data.y)
        ),
        "sequence_length": (
            SEQUENCE_LENGTH
        ),
        "feature_count": int(
            train_data.X.shape[2]
        ),
        "mae": float(
            metrics.mae
        ),
        "rmse": float(
            metrics.rmse
        ),
        "r2": float(
            metrics.r2
        ),
        "epochs_completed": int(
            completed_epochs
        ),
        "fit_seconds": float(
            fit_seconds
        ),
        "predict_seconds": float(
            predict_seconds
        ),
        "final_2026_test_used": False,
    }

    (
        output_directory
        / "metrics.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    del model
    del train_data
    del validation_data
    del sequence_builder

    tf.keras.backend.clear_session()
    gc.collect()

    return result


def aggregate_results(
    results: pd.DataFrame,
) -> pd.DataFrame:
    aggregate = (
        results
        .groupby(
            [
                "model",
                "horizon",
            ],
            as_index=False,
        )
        .agg(
            folds=(
                "fold",
                "nunique",
            ),
            mean_mae=(
                "mae",
                "mean",
            ),
            std_mae=(
                "mae",
                "std",
            ),
            worst_mae=(
                "mae",
                "max",
            ),
            best_mae=(
                "mae",
                "min",
            ),
            mean_rmse=(
                "rmse",
                "mean",
            ),
            std_rmse=(
                "rmse",
                "std",
            ),
            mean_r2=(
                "r2",
                "mean",
            ),
            worst_r2=(
                "r2",
                "min",
            ),
            mean_epochs=(
                "epochs_completed",
                "mean",
            ),
            total_fit_seconds=(
                "fit_seconds",
                "sum",
            ),
        )
    )

    numeric_columns = [
        "mean_mae",
        "std_mae",
        "worst_mae",
        "best_mae",
        "mean_rmse",
        "std_rmse",
        "mean_r2",
        "worst_r2",
        "mean_epochs",
        "total_fit_seconds",
    ]

    for column in numeric_columns:
        aggregate[column] = (
            aggregate[column]
            .round(4)
        )

    return (
        aggregate
        .sort_values(
            [
                "horizon",
                "mean_mae",
            ]
        )
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_arguments()

    configure_tensorflow()

    folds = (
        FOLDS
        if args.fold == "all"
        else [args.fold]
    )

    horizons = (
        HORIZONS
        if args.horizon == "all"
        else [args.horizon]
    )

    model_output_directory = (
        OUTPUT_ROOT
        / args.model
    )

    model_output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 90
    )

    print(
        "PEARLS AQI - EXPANDED "
        "DEEP WALK-FORWARD BENCHMARK"
    )

    print(
        "=" * 90
    )

    print(
        f"\nModel: {args.model}"
    )

    print(
        f"Folds: {folds}"
    )

    print(
        f"Horizons: {horizons}"
    )

    print(
        f"Sequence length: "
        f"{SEQUENCE_LENGTH}"
    )

    print(
        f"Epochs maximum: "
        f"{args.epochs}"
    )

    print(
        f"Batch size: "
        f"{args.batch_size}"
    )

    print(
        "\nFinal 2026 test will NOT "
        "be loaded."
    )

    total_jobs = (
        len(folds)
        * len(horizons)
    )

    print(
        f"Training jobs: "
        f"{total_jobs}"
    )

    benchmark_start = (
        time.perf_counter()
    )

    results = []

    job_number = 0

    for fold in folds:
        for horizon in horizons:
            job_number += 1

            print(
                f"\nJOB "
                f"{job_number}/{total_jobs}"
            )

            result = train_one(
                model_name=args.model,
                fold=fold,
                horizon=horizon,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )

            results.append(
                result
            )

            partial = pd.DataFrame(
                results
            )

            partial.to_csv(
                model_output_directory
                / "walk_forward_results.csv",
                index=False,
            )

    detailed = pd.DataFrame(
        results
    )

    aggregate = (
        aggregate_results(
            detailed
        )
    )

    aggregate.to_csv(
        model_output_directory
        / "walk_forward_aggregate.csv",
        index=False,
    )

    total_seconds = (
        time.perf_counter()
        - benchmark_start
    )

    report = {
        "model": args.model,
        "folds": folds,
        "horizons": horizons,
        "sequence_length": (
            SEQUENCE_LENGTH
        ),
        "epochs_max": int(
            args.epochs
        ),
        "batch_size": int(
            args.batch_size
        ),
        "training_jobs": int(
            total_jobs
        ),
        "total_seconds": float(
            total_seconds
        ),
        "final_2026_test_used": False,
    }

    (
        model_output_directory
        / "benchmark_report.json"
    ).write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n" + "=" * 90
    )

    print(
        "AGGREGATED RESULTS"
    )

    print(
        "=" * 90
    )

    print(
        aggregate.to_string(
            index=False
        )
    )

    print(
        "\n" + "=" * 90
    )

    print(
        "BENCHMARK COMPLETE"
    )

    print(
        "=" * 90
    )

    print(
        f"\nRuntime: "
        f"{total_seconds / 60:.2f} minutes"
    )

    print(
        "\nFinal 2026 test touched: NO"
    )

    print(
        f"\nResults:\n"
        f"{model_output_directory}"
    )


if __name__ == "__main__":
    main()