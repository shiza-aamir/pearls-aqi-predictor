from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import mlflow.xgboost
import pandas as pd

from src.features.engineer import AQIFeatureEngineer
from src.ml.evaluation.metrics import (
    calculate_regression_metrics,
)
from src.ml.models.xgboost_model import (
    create_xgboost_model,
)

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "pearls-aqi-daily-training"

SPLIT_ROOT = Path(
    "data/splits/expanded/fold_3"
)

TRAIN_PATH = (
    SPLIT_ROOT / "train.parquet"
)

VALIDATION_PATH = (
    SPLIT_ROOT / "validation.parquet"
)

OUTPUT_DIR = Path(
    "artifacts/daily_training"
)

REPORT_PATH = (
    OUTPUT_DIR / "latest_training_report.json"
)

HORIZONS = {
    24: "target_aqi_24h",
    48: "target_aqi_48h",
    72: "target_aqi_72h",
}

EXPECTED_FEATURE_COUNT = 56


def load_dataframe(
    path: Path,
    label: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} dataset not found: {path}"
        )

    dataframe = pd.read_parquet(
        path
    )

    if dataframe.empty:
        raise ValueError(
            f"{label} dataset is empty."
        )

    if "timestamp" not in dataframe.columns:
        raise ValueError(
            f"{label} dataset has no timestamp column."
        )

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"],
        utc=True,
        errors="raise",
    )

    return (
        dataframe
        .sort_values(
            [
                "timestamp",
                "city",
            ]
        )
        .reset_index(drop=True)
    )


def validate_split(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            "Expected "
            f"{EXPECTED_FEATURE_COUNT} features, "
            f"got {len(feature_columns)}."
        )

    required_columns = {
        "timestamp",
        "city",
        *feature_columns,
        *HORIZONS.values(),
    }

    for label, dataframe in (
        ("train", train),
        ("validation", validation),
    ):
        missing_columns = (
            required_columns
            - set(dataframe.columns)
        )

        if missing_columns:
            raise ValueError(
                f"{label} dataset is missing columns: "
                f"{sorted(missing_columns)}"
            )

        missing_features = int(
            dataframe[
                feature_columns
            ]
            .isna()
            .sum()
            .sum()
        )

        if missing_features:
            raise ValueError(
                f"{label} dataset contains "
                f"{missing_features} missing "
                "feature values."
            )

        for target_column in (
            HORIZONS.values()
        ):
            missing_targets = int(
                dataframe[
                    target_column
                ]
                .isna()
                .sum()
            )

            if missing_targets:
                raise ValueError(
                    f"{label} dataset contains "
                    f"{missing_targets} missing "
                    f"{target_column} values."
                )

    train_end = (
        train["timestamp"].max()
    )

    validation_start = (
        validation["timestamp"].min()
    )

    if validation_start <= train_end:
        raise ValueError(
            "Training and validation periods "
            "overlap."
        )

    purge_gap = (
        validation_start
        - train_end
    )

    minimum_gap = pd.Timedelta(
        hours=72
    )

    if purge_gap < minimum_gap:
        raise ValueError(
            "Training/validation purge gap "
            "is smaller than 72 hours."
        )

    train_cities = set(
        train["city"]
        .astype(str)
        .unique()
    )

    validation_cities = set(
        validation["city"]
        .astype(str)
        .unique()
    )

    if train_cities != validation_cities:
        raise ValueError(
            "Training and validation city "
            "sets do not match."
        )

    if len(train_cities) != 9:
        raise ValueError(
            "Expected 9 cities, "
            f"got {len(train_cities)}."
        )


def train_horizon(
    horizon_hours: int,
    target_column: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_columns: list[str],
) -> dict:
    model = create_xgboost_model()

    x_train = (
        train[
            feature_columns
        ]
        .astype(float)
    )

    y_train = (
        train[
            target_column
        ]
        .astype(float)
    )

    x_validation = (
        validation[
            feature_columns
        ]
        .astype(float)
    )

    y_validation = (
        validation[
            target_column
        ]
        .astype(float)
    )

    fit_start = (
        time.perf_counter()
    )

    model.fit(
        x_train,
        y_train,
    )

    fit_seconds = (
        time.perf_counter()
        - fit_start
    )

    prediction_start = (
        time.perf_counter()
    )

    predictions = model.predict(
        x_validation
    )

    prediction_seconds = (
        time.perf_counter()
        - prediction_start
    )

    metrics = (
        calculate_regression_metrics(
            y_validation,
            predictions,
        )
    )

    run_name = (
        f"daily-xgboost-{horizon_hours}h"
    )

    with mlflow.start_run(
        run_name=run_name
    ) as run:
        mlflow.log_param(
            "pipeline",
            "daily_candidate_training",
        )

        mlflow.log_param(
            "model_name",
            "xgboost",
        )

        mlflow.log_param(
            "horizon_hours",
            horizon_hours,
        )

        mlflow.log_param(
            "feature_count",
            len(feature_columns),
        )

        mlflow.log_param(
            "training_rows",
            len(train),
        )

        mlflow.log_param(
            "validation_rows",
            len(validation),
        )

        mlflow.log_param(
            "city_count",
            int(
                train[
                    "city"
                ].nunique()
            ),
        )

        mlflow.log_param(
            "training_start",
            str(
                train[
                    "timestamp"
                ].min()
            ),
        )

        mlflow.log_param(
            "training_end",
            str(
                train[
                    "timestamp"
                ].max()
            ),
        )

        mlflow.log_param(
            "validation_start",
            str(
                validation[
                    "timestamp"
                ].min()
            ),
        )

        mlflow.log_param(
            "validation_end",
            str(
                validation[
                    "timestamp"
                ].max()
            ),
        )

        mlflow.log_param(
            "final_holdout_used",
            False,
        )

        mlflow.log_param(
            "automatic_production_promotion",
            False,
        )

        mlflow.log_metric(
            "validation_mae",
            metrics.mae,
        )

        mlflow.log_metric(
            "validation_rmse",
            metrics.rmse,
        )

        mlflow.log_metric(
            "validation_r2",
            metrics.r2,
        )

        mlflow.log_metric(
            "fit_seconds",
            fit_seconds,
        )

        mlflow.log_metric(
            "prediction_seconds",
            prediction_seconds,
        )

        mlflow.xgboost.log_model(
            xgb_model=model,
            name="candidate_model",
            input_example=(
                x_train
                .head(5)
            ),
            model_format="json",
        )

        run_id = (
            run.info.run_id
        )

    return {
        "horizon_hours": (
            horizon_hours
        ),
        "model": "xgboost",
        "run_id": run_id,
        "validation_mae": float(
            metrics.mae
        ),
        "validation_rmse": float(
            metrics.rmse
        ),
        "validation_r2": float(
            metrics.r2
        ),
        "fit_seconds": float(
            fit_seconds
        ),
        "prediction_seconds": float(
            prediction_seconds
        ),
        "training_rows": len(train),
        "validation_rows": len(
            validation
        ),
    }


def main() -> None:
    print(
        "=" * 80
    )

    print(
        "PEARLS AQI - DAILY "
        "CANDIDATE TRAINING"
    )

    print(
        "=" * 80
    )

    mlflow.set_tracking_uri(
        TRACKING_URI
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    feature_columns = (
        AQIFeatureEngineer
        .get_model_feature_columns()
    )

    train = load_dataframe(
        TRAIN_PATH,
        "Training",
    )

    validation = load_dataframe(
        VALIDATION_PATH,
        "Validation",
    )

    validate_split(
        train,
        validation,
        feature_columns,
    )

    print(
        f"Training rows: "
        f"{len(train):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation):,}"
    )

    print(
        "Final 2026 holdout used: NO"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pipeline_start = (
        time.perf_counter()
    )

    results = []

    for (
        horizon_hours,
        target_column,
    ) in HORIZONS.items():
        print(
            "\n"
            + "-" * 80
        )

        print(
            f"Training XGBoost "
            f"{horizon_hours}h candidate"
        )

        print(
            "-" * 80
        )

        result = train_horizon(
            horizon_hours=(
                horizon_hours
            ),
            target_column=(
                target_column
            ),
            train=train,
            validation=validation,
            feature_columns=(
                feature_columns
            ),
        )

        results.append(
            result
        )

        print(
            "Validation MAE: "
            f"{result['validation_mae']:.4f}"
        )

        print(
            "Validation RMSE: "
            f"{result['validation_rmse']:.4f}"
        )

        print(
            "Validation R²: "
            f"{result['validation_r2']:.4f}"
        )

    total_seconds = (
        time.perf_counter()
        - pipeline_start
    )

    report = {
        "schema_version": 1,
        "generated_at": (
            datetime.now(
                UTC
            ).isoformat()
        ),
        "pipeline": (
            "daily_candidate_training"
        ),
        "tracking_backend": "MLflow",
        "tracking_uri": TRACKING_URI,
        "experiment_name": (
            EXPERIMENT_NAME
        ),
        "model_type": "xgboost",
        "feature_count": (
            len(feature_columns)
        ),
        "training_rows": len(train),
        "validation_rows": len(
            validation
        ),
        "cities": int(
            train[
                "city"
            ].nunique()
        ),
        "training_start": str(
            train[
                "timestamp"
            ].min()
        ),
        "training_end": str(
            train[
                "timestamp"
            ].max()
        ),
        "validation_start": str(
            validation[
                "timestamp"
            ].min()
        ),
        "validation_end": str(
            validation[
                "timestamp"
            ].max()
        ),
        "purge_hours": 72,
        "final_2026_holdout_used": (
            False
        ),
        "automatic_production_promotion": (
            False
        ),
        "results": results,
        "total_seconds": float(
            total_seconds
        ),
    }

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "DAILY TRAINING COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        "Final 2026 holdout touched: NO"
    )

    print(
        "Production aliases changed: NO"
    )

    print(
        f"Report: {REPORT_PATH}"
    )


if __name__ == "__main__":
    main()