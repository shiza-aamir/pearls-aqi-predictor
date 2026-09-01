from __future__ import annotations

import gc
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.features.engineer import AQIFeatureEngineer
from src.ml.models import (
    PersistenceBaseline,
    create_random_forest_model,
    create_ridge_model,
    create_xgboost_model,
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

OUTPUT_DIR = Path(
    "artifacts/model_comparison"
)

HORIZONS = [
    "24h",
    "48h",
    "72h",
]

TARGET_COLUMNS = {
    "24h": "target_aqi_24h",
    "48h": "target_aqi_48h",
    "72h": "target_aqi_72h",
}

CLASSICAL_MODELS = [
    "persistence",
    "ridge",
    "random_forest",
    "xgboost",
]

DEEP_MODELS = [
    "cnn",
    "gru",
    "cnn_lstm",
]


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


def load_dataframe(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required split does not exist: {path}"
        )

    dataframe = pd.read_parquet(path)

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"]
    )

    return dataframe


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(
        y_true,
        dtype=float,
    ).reshape(-1)

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    ).reshape(-1)

    y_pred = np.clip(
        y_pred,
        0.0,
        500.0,
    )

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def create_common_subset(
    dataframe: pd.DataFrame,
    sequence_cities: np.ndarray,
    sequence_timestamps: np.ndarray,
) -> pd.DataFrame:
    keys = pd.DataFrame(
        {
            "city": sequence_cities.astype(str),
            "timestamp": pd.to_datetime(
                sequence_timestamps
            ),
        }
    )

    keys = keys.drop_duplicates(
        subset=[
            "city",
            "timestamp",
        ]
    )

    working = dataframe.copy()

    working["city"] = (
        working["city"]
        .astype(str)
    )

    working["timestamp"] = pd.to_datetime(
        working["timestamp"]
    )

    common = working.merge(
        keys,
        how="inner",
        on=[
            "city",
            "timestamp",
        ],
    )

    common = (
        common
        .sort_values(
            [
                "city",
                "timestamp",
            ]
        )
        .reset_index(drop=True)
    )

    return common


def get_classical_model(
    model_name: str,
):
    if model_name == "ridge":
        return create_ridge_model()

    if model_name == "random_forest":
        return create_random_forest_model()

    if model_name == "xgboost":
        return create_xgboost_model()

    raise ValueError(
        f"Unknown classical model: {model_name}"
    )


def evaluate_classical_models(
    horizon: str,
    train_df: pd.DataFrame,
    validation_common: pd.DataFrame,
    test_common: pd.DataFrame,
) -> list[dict]:
    target_column = TARGET_COLUMNS[
        horizon
    ]

    feature_columns = (
        AQIFeatureEngineer
        .get_model_feature_columns()
    )

    results = []

    print(
        "\nEvaluating classical models "
        "on common timestamps..."
    )

    for model_name in CLASSICAL_MODELS:
        print(
            f"  {model_name}"
        )

        if model_name == "persistence":
            validation_predictions = (
                PersistenceBaseline.predict(
                    validation_common
                )
            )

            test_predictions = (
                PersistenceBaseline.predict(
                    test_common
                )
            )

        else:
            model = get_classical_model(
                model_name
            )

            x_train = (
                train_df[
                    feature_columns
                ]
                .astype(float)
            )

            y_train = (
                train_df[
                    target_column
                ]
                .astype(float)
            )

            model.fit(
                x_train,
                y_train,
            )

            validation_predictions = (
                model.predict(
                    validation_common[
                        feature_columns
                    ].astype(float)
                )
            )

            test_predictions = (
                model.predict(
                    test_common[
                        feature_columns
                    ].astype(float)
                )
            )

        validation_metrics = (
            calculate_metrics(
                validation_common[
                    target_column
                ].to_numpy(),
                validation_predictions,
            )
        )

        test_metrics = calculate_metrics(
            test_common[
                target_column
            ].to_numpy(),
            test_predictions,
        )

        results.append(
            {
                "model": model_name,
                "family": "classical",
                "horizon": horizon,
                "validation_samples": (
                    len(validation_common)
                ),
                "test_samples": (
                    len(test_common)
                ),
                "val_mae": (
                    validation_metrics["mae"]
                ),
                "val_rmse": (
                    validation_metrics["rmse"]
                ),
                "val_r2": (
                    validation_metrics["r2"]
                ),
                "test_mae": (
                    test_metrics["mae"]
                ),
                "test_rmse": (
                    test_metrics["rmse"]
                ),
                "test_r2": (
                    test_metrics["r2"]
                ),
            }
        )

        del validation_predictions
        del test_predictions

        if model_name != "persistence":
            del model

        gc.collect()

    return results


def load_dl_sequence_builder(
    model_name: str,
    horizon: str,
) -> AQISequenceBuilder:
    scaler_path = (
        Path("artifacts")
        / "deep_learning"
        / model_name
        / horizon
        / "feature_scaler.joblib"
    )

    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler not found: {scaler_path}"
        )

    builder = AQISequenceBuilder(
        sequence_length=72
    )

    builder.scaler = joblib.load(
        scaler_path
    )

    builder.is_fitted = True

    return builder


def evaluate_deep_models(
    horizon: str,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> list[dict]:
    results = []

    print(
        "\nEvaluating deep-learning models..."
    )

    for model_name in DEEP_MODELS:
        print(
            f"  {model_name}"
        )

        model_directory = (
            Path("artifacts")
            / "deep_learning"
            / model_name
            / horizon
        )

        model_path = (
            model_directory
            / "best_model.keras"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"DL model not found: {model_path}"
            )

        builder = load_dl_sequence_builder(
            model_name=model_name,
            horizon=horizon,
        )

        validation_sequences = (
            builder.build(
                dataframe=validation_df,
                horizon=horizon,
            )
        )

        test_sequences = builder.build(
            dataframe=test_df,
            horizon=horizon,
        )

        tf.keras.backend.clear_session()

        model = tf.keras.models.load_model(
            model_path
        )

        validation_predictions = (
            model.predict(
                validation_sequences.X,
                batch_size=32,
                verbose=0,
            )
            .reshape(-1)
        )

        test_predictions = (
            model.predict(
                test_sequences.X,
                batch_size=32,
                verbose=0,
            )
            .reshape(-1)
        )

        validation_metrics = (
            calculate_metrics(
                validation_sequences.y,
                validation_predictions,
            )
        )

        test_metrics = calculate_metrics(
            test_sequences.y,
            test_predictions,
        )

        results.append(
            {
                "model": model_name,
                "family": "deep_learning",
                "horizon": horizon,
                "validation_samples": (
                    len(
                        validation_sequences.y
                    )
                ),
                "test_samples": (
                    len(
                        test_sequences.y
                    )
                ),
                "val_mae": (
                    validation_metrics["mae"]
                ),
                "val_rmse": (
                    validation_metrics["rmse"]
                ),
                "val_r2": (
                    validation_metrics["r2"]
                ),
                "test_mae": (
                    test_metrics["mae"]
                ),
                "test_rmse": (
                    test_metrics["rmse"]
                ),
                "test_r2": (
                    test_metrics["r2"]
                ),
            }
        )

        del model
        del builder
        del validation_sequences
        del test_sequences
        del validation_predictions
        del test_predictions

        tf.keras.backend.clear_session()
        gc.collect()

    return results


def build_reference_sequence_keys(
    horizon: str,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    builder = AQISequenceBuilder(
        sequence_length=72
    )

    builder.fit_scaler(
        train_df
    )

    validation_sequences = (
        builder.build(
            dataframe=validation_df,
            horizon=horizon,
        )
    )

    test_sequences = builder.build(
        dataframe=test_df,
        horizon=horizon,
    )

    return (
        validation_sequences,
        test_sequences,
    )


def print_horizon_table(
    dataframe: pd.DataFrame,
    horizon: str,
) -> None:
    horizon_df = dataframe[
        dataframe["horizon"] == horizon
    ].copy()

    horizon_df = (
        horizon_df
        .sort_values(
            [
                "val_mae",
                "val_rmse",
            ]
        )
        .reset_index(drop=True)
    )

    display_columns = [
        "model",
        "family",
        "val_mae",
        "val_rmse",
        "val_r2",
        "test_mae",
        "test_rmse",
        "test_r2",
    ]

    print(
        "\n"
        + "=" * 100
    )

    print(
        f"{horizon.upper()} "
        "COMMON-TIMESTAMP COMPARISON"
    )

    print(
        "=" * 100
    )

    print(
        horizon_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.3f}"
            ),
        )
    )


def select_validation_champions(
    results_df: pd.DataFrame,
) -> dict[str, dict]:
    champions = {}

    for horizon in HORIZONS:
        horizon_results = (
            results_df[
                results_df["horizon"]
                == horizon
            ]
            .sort_values(
                [
                    "val_mae",
                    "val_rmse",
                    "val_r2",
                ],
                ascending=[
                    True,
                    True,
                    False,
                ],
            )
            .reset_index(drop=True)
        )

        winner = horizon_results.iloc[0]

        champions[horizon] = {
            "model": winner["model"],
            "family": winner["family"],
            "val_mae": float(
                winner["val_mae"]
            ),
            "val_rmse": float(
                winner["val_rmse"]
            ),
            "val_r2": float(
                winner["val_r2"]
            ),
            "test_mae": float(
                winner["test_mae"]
            ),
            "test_rmse": float(
                winner["test_rmse"]
            ),
            "test_r2": float(
                winner["test_r2"]
            ),
        }

    return champions


def main() -> None:
    configure_tensorflow()

    print("=" * 100)
    print(
        "PEARLS AQI - COMMON-TIMESTAMP "
        "MODEL COMPARISON"
    )
    print("=" * 100)

    print(
        "\nLoading chronological splits..."
    )

    train_df = load_dataframe(
        TRAIN_PATH
    )

    validation_df = load_dataframe(
        VALIDATION_PATH
    )

    test_df = load_dataframe(
        TEST_PATH
    )

    print(
        f"Train rows:      "
        f"{len(train_df):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation_df):,}"
    )

    print(
        f"Test rows:       "
        f"{len(test_df):,}"
    )

    all_results = []

    for horizon in HORIZONS:
        print(
            "\n"
            + "#" * 100
        )

        print(
            f"HORIZON: {horizon}"
        )

        print(
            "#" * 100
        )

        (
            reference_validation,
            reference_test,
        ) = build_reference_sequence_keys(
            horizon=horizon,
            train_df=train_df,
            validation_df=validation_df,
            test_df=test_df,
        )

        validation_common = (
            create_common_subset(
                dataframe=validation_df,
                sequence_cities=(
                    reference_validation.cities
                ),
                sequence_timestamps=(
                    reference_validation.timestamps
                ),
            )
        )

        test_common = create_common_subset(
            dataframe=test_df,
            sequence_cities=(
                reference_test.cities
            ),
            sequence_timestamps=(
                reference_test.timestamps
            ),
        )

        print(
            "\nCommon evaluation sizes:"
        )

        print(
            f"  Validation: "
            f"{len(validation_common):,}"
        )

        print(
            f"  Test:       "
            f"{len(test_common):,}"
        )

        if len(validation_common) != len(
            reference_validation.y
        ):
            raise RuntimeError(
                "Validation common subset size "
                "does not match DL sequence count."
            )

        if len(test_common) != len(
            reference_test.y
        ):
            raise RuntimeError(
                "Test common subset size "
                "does not match DL sequence count."
            )

        classical_results = (
            evaluate_classical_models(
                horizon=horizon,
                train_df=train_df,
                validation_common=(
                    validation_common
                ),
                test_common=test_common,
            )
        )

        deep_results = (
            evaluate_deep_models(
                horizon=horizon,
                validation_df=validation_df,
                test_df=test_df,
            )
        )

        all_results.extend(
            classical_results
        )

        all_results.extend(
            deep_results
        )

        del reference_validation
        del reference_test
        del validation_common
        del test_common

        gc.collect()

    results_df = pd.DataFrame(
        all_results
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        OUTPUT_DIR
        / "common_timestamp_model_comparison.csv"
    )

    results_df.to_csv(
        csv_path,
        index=False,
    )

    for horizon in HORIZONS:
        print_horizon_table(
            dataframe=results_df,
            horizon=horizon,
        )

    champions = (
        select_validation_champions(
            results_df
        )
    )

    champions_path = (
        OUTPUT_DIR
        / "validation_champions.json"
    )

    champions_path.write_text(
        json.dumps(
            champions,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "VALIDATION-SELECTED CANDIDATES"
    )

    print(
        "=" * 100
    )

    for horizon, result in (
        champions.items()
    ):
        print(
            f"\n{horizon}: "
            f"{result['model']} "
            f"({result['family']})"
        )

        print(
            f"  Val MAE:  "
            f"{result['val_mae']:.3f}"
        )

        print(
            f"  Val RMSE: "
            f"{result['val_rmse']:.3f}"
        )

        print(
            f"  Val R2:   "
            f"{result['val_r2']:.3f}"
        )

        print(
            f"  Test MAE: "
            f"{result['test_mae']:.3f}"
        )

    print(
        "\nSaved comparison:"
    )

    print(
        f"  {csv_path}"
    )

    print(
        "\nSaved candidate summary:"
    )

    print(
        f"  {champions_path}"
    )

    print(
        "\nNOTE: These files identify candidates "
        "using validation performance."
    )

    print(
        "They DO NOT modify MLflow registry aliases."
    )

    print("=" * 100)


if __name__ == "__main__":
    main()