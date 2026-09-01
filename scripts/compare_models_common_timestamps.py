from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.engineer import AQIFeatureEngineer
from src.ml.evaluation.metrics import calculate_regression_metrics
from src.ml.models.xgboost_model import create_xgboost_model


SPLIT_ROOT = Path("data/splits/expanded")
CNN_ROOT = Path("artifacts/expanded_deep_benchmark/cnn")
OUTPUT_DIR = Path("artifacts/model_selection/common_timestamps")

FOLDS = [
    "fold_1",
    "fold_2",
    "fold_3",
]

HORIZONS = {
    24: "target_aqi_24h",
    48: "target_aqi_48h",
    72: "target_aqi_72h",
}

EXPECTED_VALIDATION_SEQUENCES = 38_457
EXPECTED_FEATURE_COUNT = 56


def load_fold(
    fold_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_dir = SPLIT_ROOT / fold_name

    train_path = fold_dir / "train.parquet"
    validation_path = fold_dir / "validation.parquet"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Missing training split: {train_path}"
        )

    if not validation_path.exists():
        raise FileNotFoundError(
            f"Missing validation split: {validation_path}"
        )

    train = pd.read_parquet(train_path)
    validation = pd.read_parquet(validation_path)

    train["timestamp"] = pd.to_datetime(
        train["timestamp"],
        utc=True,
    )

    validation["timestamp"] = pd.to_datetime(
        validation["timestamp"],
        utc=True,
    )

    return train, validation


def load_cnn_predictions(
    fold_name: str,
    horizon: int,
) -> pd.DataFrame:
    path = (
        CNN_ROOT
        / fold_name
        / f"{horizon}h"
        / "validation_predictions.parquet"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing CNN predictions: {path}"
        )

    dataframe = pd.read_parquet(path)

    required = {
        "city",
        "timestamp",
        "y_true",
        "y_pred",
    }

    missing = required - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"{path}: missing columns {sorted(missing)}"
        )

    dataframe = dataframe.copy()

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"],
        utc=True,
    )

    if len(dataframe) != EXPECTED_VALIDATION_SEQUENCES:
        raise ValueError(
            f"{fold_name}/{horizon}h: expected "
            f"{EXPECTED_VALIDATION_SEQUENCES:,} CNN rows, "
            f"got {len(dataframe):,}."
        )

    duplicate_count = int(
        dataframe.duplicated(
            subset=["city", "timestamp"]
        ).sum()
    )

    if duplicate_count:
        raise ValueError(
            f"{fold_name}/{horizon}h: CNN predictions "
            f"contain {duplicate_count} duplicate keys."
        )

    return dataframe


def validate_training_data(
    train: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    fold_name: str,
) -> None:
    required = (
        feature_columns
        + [
            target_column,
            "city",
            "timestamp",
        ]
    )

    missing = [
        column
        for column in required
        if column not in train.columns
    ]

    if missing:
        raise ValueError(
            f"{fold_name} training data missing columns: "
            f"{missing}"
        )

    feature_nulls = int(
        train[feature_columns]
        .isna()
        .sum()
        .sum()
    )

    target_nulls = int(
        train[target_column]
        .isna()
        .sum()
    )

    if feature_nulls:
        raise ValueError(
            f"{fold_name}: training features contain "
            f"{feature_nulls} null cells."
        )

    if target_nulls:
        raise ValueError(
            f"{fold_name}: training target contains "
            f"{target_nulls} null values."
        )


def build_common_validation(
    validation: pd.DataFrame,
    cnn_predictions: pd.DataFrame,
    target_column: str,
    fold_name: str,
    horizon: int,
) -> pd.DataFrame:
    if validation.duplicated(
        subset=["city", "timestamp"]
    ).any():
        raise ValueError(
            f"{fold_name}: validation split contains "
            "duplicate city/timestamp keys."
        )

    cnn_keys = cnn_predictions[
        [
            "city",
            "timestamp",
            "y_true",
            "y_pred",
        ]
    ].rename(
        columns={
            "y_true": "cnn_y_true",
            "y_pred": "cnn_prediction",
        }
    )

    common = validation.merge(
        cnn_keys,
        on=["city", "timestamp"],
        how="inner",
        validate="one_to_one",
    )

    common = common.sort_values(
        ["city", "timestamp"]
    ).reset_index(drop=True)

    if len(common) != EXPECTED_VALIDATION_SEQUENCES:
        raise ValueError(
            f"{fold_name}/{horizon}h: expected "
            f"{EXPECTED_VALIDATION_SEQUENCES:,} common rows, "
            f"got {len(common):,}."
        )

    if common["city"].nunique() != 9:
        raise ValueError(
            f"{fold_name}/{horizon}h: expected 9 cities, "
            f"got {common['city'].nunique()}."
        )

    actual_target = common[
        target_column
    ].to_numpy(dtype=float)

    cnn_target = common[
        "cnn_y_true"
    ].to_numpy(dtype=float)

    if not np.allclose(
        actual_target,
        cnn_target,
        rtol=0.0,
        atol=1e-6,
    ):
        max_difference = float(
            np.max(
                np.abs(
                    actual_target
                    - cnn_target
                )
            )
        )

        raise ValueError(
            f"{fold_name}/{horizon}h: CNN y_true does not "
            f"match validation target. Maximum difference: "
            f"{max_difference}"
        )

    return common


def evaluate_predictions(
    y_true,
    y_pred,
) -> dict:
    metrics = calculate_regression_metrics(
        y_true,
        y_pred,
    )

    return metrics.to_dict()


def run_comparison() -> pd.DataFrame:
    feature_columns = (
        AQIFeatureEngineer
        .get_model_feature_columns()
    )

    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} model features, "
            f"got {len(feature_columns)}."
        )

    rows = []

    total_jobs = len(FOLDS) * len(HORIZONS)
    job_number = 0

    for fold_name in FOLDS:
        train, validation = load_fold(
            fold_name
        )

        for horizon, target_column in HORIZONS.items():
            job_number += 1

            print("\n" + "=" * 90)
            print(
                f"JOB {job_number}/{total_jobs} | "
                f"{fold_name.upper()} | {horizon}h"
            )
            print("=" * 90)

            validate_training_data(
                train=train,
                feature_columns=feature_columns,
                target_column=target_column,
                fold_name=fold_name,
            )

            cnn_predictions = load_cnn_predictions(
                fold_name=fold_name,
                horizon=horizon,
            )

            common = build_common_validation(
                validation=validation,
                cnn_predictions=cnn_predictions,
                target_column=target_column,
                fold_name=fold_name,
                horizon=horizon,
            )

            print(
                f"Training rows: {len(train):,}"
            )
            print(
                f"Original validation rows: "
                f"{len(validation):,}"
            )
            print(
                f"Common validation rows: "
                f"{len(common):,}"
            )
            print(
                f"Cities: {common['city'].nunique()}"
            )

            x_train = train[
                feature_columns
            ]

            y_train = train[
                target_column
            ]

            x_common = common[
                feature_columns
            ]

            y_common = common[
                target_column
            ]

            model = create_xgboost_model()

            fit_start = time.perf_counter()

            model.fit(
                x_train,
                y_train,
            )

            fit_seconds = (
                time.perf_counter()
                - fit_start
            )

            predict_start = time.perf_counter()

            xgb_prediction = model.predict(
                x_common
            )

            predict_seconds = (
                time.perf_counter()
                - predict_start
            )

            xgb_prediction = np.clip(
                xgb_prediction,
                0.0,
                500.0,
            )

            cnn_prediction = np.clip(
                common[
                    "cnn_prediction"
                ].to_numpy(dtype=float),
                0.0,
                500.0,
            )

            persistence_prediction = common[
                "aqi_current"
            ].to_numpy(dtype=float)

            xgb_metrics = evaluate_predictions(
                y_common,
                xgb_prediction,
            )

            cnn_metrics = evaluate_predictions(
                y_common,
                cnn_prediction,
            )

            persistence_metrics = evaluate_predictions(
                y_common,
                persistence_prediction,
            )

            model_results = [
                (
                    "xgboost",
                    xgb_metrics,
                    fit_seconds,
                    predict_seconds,
                ),
                (
                    "cnn",
                    cnn_metrics,
                    np.nan,
                    np.nan,
                ),
                (
                    "persistence",
                    persistence_metrics,
                    0.0,
                    0.0,
                ),
            ]

            for (
                model_name,
                metrics,
                model_fit_seconds,
                model_predict_seconds,
            ) in model_results:
                rows.append(
                    {
                        "fold": fold_name,
                        "horizon_hours": horizon,
                        "model": model_name,
                        "common_rows": int(
                            len(common)
                        ),
                        "cities": int(
                            common[
                                "city"
                            ].nunique()
                        ),
                        "mae": float(
                            metrics["mae"]
                        ),
                        "rmse": float(
                            metrics["rmse"]
                        ),
                        "r2": float(
                            metrics["r2"]
                        ),
                        "fit_seconds": (
                            float(model_fit_seconds)
                            if not pd.isna(
                                model_fit_seconds
                            )
                            else np.nan
                        ),
                        "predict_seconds": (
                            float(model_predict_seconds)
                            if not pd.isna(
                                model_predict_seconds
                            )
                            else np.nan
                        ),
                    }
                )

            prediction_output = pd.DataFrame(
                {
                    "city": common["city"],
                    "timestamp": common["timestamp"],
                    "y_true": y_common,
                    "xgboost_prediction": xgb_prediction,
                    "cnn_prediction": cnn_prediction,
                    "persistence_prediction": (
                        persistence_prediction
                    ),
                }
            )

            prediction_dir = (
                OUTPUT_DIR
                / fold_name
                / f"{horizon}h"
            )

            prediction_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            prediction_output.to_parquet(
                prediction_dir
                / "common_predictions.parquet",
                index=False,
            )

            prediction_output.to_csv(
                prediction_dir
                / "common_predictions.csv",
                index=False,
            )

            print("\nCommon-timestamp metrics:")

            print(
                f"  XGBoost     MAE="
                f"{xgb_metrics['mae']:.4f} "
                f"RMSE={xgb_metrics['rmse']:.4f} "
                f"R2={xgb_metrics['r2']:.4f}"
            )

            print(
                f"  CNN         MAE="
                f"{cnn_metrics['mae']:.4f} "
                f"RMSE={cnn_metrics['rmse']:.4f} "
                f"R2={cnn_metrics['r2']:.4f}"
            )

            print(
                f"  Persistence MAE="
                f"{persistence_metrics['mae']:.4f} "
                f"RMSE={persistence_metrics['rmse']:.4f} "
                f"R2={persistence_metrics['r2']:.4f}"
            )

    return pd.DataFrame(rows)


def build_aggregate(
    detailed: pd.DataFrame,
) -> pd.DataFrame:
    aggregate = (
        detailed
        .groupby(
            [
                "horizon_hours",
                "model",
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
            mean_rmse=(
                "rmse",
                "mean",
            ),
            mean_r2=(
                "r2",
                "mean",
            ),
        )
    )

    numeric_columns = [
        "mean_mae",
        "std_mae",
        "worst_mae",
        "mean_rmse",
        "mean_r2",
    ]

    aggregate[
        numeric_columns
    ] = (
        aggregate[
            numeric_columns
        ].round(4)
    )

    aggregate = aggregate.sort_values(
        [
            "horizon_hours",
            "mean_mae",
        ]
    ).reset_index(drop=True)

    aggregate[
        "rank_on_common_timestamps"
    ] = (
        aggregate
        .groupby(
            "horizon_hours"
        )["mean_mae"]
        .rank(
            method="dense",
            ascending=True,
        )
        .astype(int)
    )

    return aggregate


def main() -> None:
    print("=" * 90)
    print(
        "PEARLS AQI - COMMON-TIMESTAMP "
        "MODEL COMPARISON"
    )
    print("=" * 90)

    print(
        "\nPurpose: reporting fairness only."
    )
    print(
        "Model selection is already frozen."
    )
    print(
        "Final 2026 test will NOT be loaded."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    start = time.perf_counter()

    detailed = run_comparison()

    aggregate = build_aggregate(
        detailed
    )

    runtime = (
        time.perf_counter()
        - start
    )

    detailed_path = (
        OUTPUT_DIR
        / "common_timestamp_results.csv"
    )

    aggregate_path = (
        OUTPUT_DIR
        / "common_timestamp_aggregate.csv"
    )

    report_path = (
        OUTPUT_DIR
        / "common_timestamp_report.json"
    )

    detailed.to_csv(
        detailed_path,
        index=False,
    )

    aggregate.to_csv(
        aggregate_path,
        index=False,
    )

    report = {
        "purpose": (
            "Fair reporting comparison using identical "
            "city/timestamp validation observations."
        ),
        "model_selection_frozen": True,
        "selected_model": {
            "24h": "xgboost",
            "48h": "xgboost",
            "72h": "xgboost",
        },
        "models_compared": [
            "xgboost",
            "cnn",
            "persistence",
        ],
        "folds": FOLDS,
        "horizons_hours": list(
            HORIZONS.keys()
        ),
        "expected_common_rows_per_job": (
            EXPECTED_VALIDATION_SEQUENCES
        ),
        "feature_count_xgboost": 56,
        "cnn_sequence_length": 72,
        "cnn_sequence_feature_count": 24,
        "final_2026_test_used": False,
        "runtime_seconds": float(
            runtime
        ),
    }

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    print("\n" + "=" * 90)
    print("AGGREGATED COMMON-TIMESTAMP RESULTS")
    print("=" * 90)

    print(
        aggregate.to_string(
            index=False
        )
    )

    print("\n" + "=" * 90)
    print("COMPARISON COMPLETE")
    print("=" * 90)

    print(
        f"\nRuntime: {runtime / 60:.2f} minutes"
    )

    print(
        "\nModel selection frozen: YES"
    )

    print(
        "Final 2026 test touched: NO"
    )

    print(
        f"\nDetailed results:\n{detailed_path}"
    )

    print(
        f"\nAggregate results:\n{aggregate_path}"
    )

    print(
        f"\nReport:\n{report_path}"
    )


if __name__ == "__main__":
    main()