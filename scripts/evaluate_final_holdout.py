from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.engineer import AQIFeatureEngineer
from src.ml.evaluation.metrics import calculate_regression_metrics
from src.ml.models.xgboost_model import create_xgboost_model


SPLIT_ROOT = Path("data/splits/expanded/final")

TRAIN_PATH = SPLIT_ROOT / "train.parquet"
TEST_PATH = SPLIT_ROOT / "test.parquet"

FREEZE_PATH = Path(
    "artifacts/model_selection/model_selection_freeze.json"
)

OUTPUT_DIR = Path(
    "artifacts/final_holdout"
)

HORIZONS = {
    24: "target_aqi_24h",
    48: "target_aqi_48h",
    72: "target_aqi_72h",
}

EXPECTED_MODEL = "xgboost"
EXPECTED_FEATURE_COUNT = 56
EXPECTED_TRAIN_ROWS = 268_281
EXPECTED_TEST_ROWS = 50_544
EXPECTED_CITIES = 9


def verify_model_selection_freeze() -> dict:
    if not FREEZE_PATH.exists():
        raise FileNotFoundError(
            "Model-selection freeze artifact is missing. "
            "Final holdout evaluation is not allowed."
        )

    with FREEZE_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        freeze = json.load(file)

    if (
        freeze.get("decision_status")
        != "FROZEN_BEFORE_FINAL_TEST"
    ):
        raise ValueError(
            "Model selection is not marked as frozen."
        )

    if (
        freeze.get(
            "final_2026_test_used_for_selection"
        )
        is not False
    ):
        raise ValueError(
            "Freeze artifact does not confirm that "
            "the final test was untouched."
        )

    selected = freeze.get(
        "selected_models",
        {}
    )

    for horizon in ["24h", "48h", "72h"]:
        if selected.get(horizon) != EXPECTED_MODEL:
            raise ValueError(
                f"{horizon}: frozen model is "
                f"{selected.get(horizon)}, "
                f"expected {EXPECTED_MODEL}."
            )

    return freeze


def load_final_splits() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Missing final training split: {TRAIN_PATH}"
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Missing final test split: {TEST_PATH}"
        )

    train = pd.read_parquet(
        TRAIN_PATH
    )

    test = pd.read_parquet(
        TEST_PATH
    )

    train["timestamp"] = pd.to_datetime(
        train["timestamp"],
        utc=True,
    )

    test["timestamp"] = pd.to_datetime(
        test["timestamp"],
        utc=True,
    )

    return train, test


def validate_final_splits(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    if len(train) != EXPECTED_TRAIN_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_TRAIN_ROWS:,} "
            f"training rows, got {len(train):,}."
        )

    if len(test) != EXPECTED_TEST_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_TEST_ROWS:,} "
            f"test rows, got {len(test):,}."
        )

    if train["city"].nunique() != EXPECTED_CITIES:
        raise ValueError(
            "Unexpected number of training cities."
        )

    if test["city"].nunique() != EXPECTED_CITIES:
        raise ValueError(
            "Unexpected number of test cities."
        )

    required = (
        feature_columns
        + list(HORIZONS.values())
        + [
            "aqi_current",
            "city",
            "timestamp",
        ]
    )

    for label, dataframe in [
        ("train", train),
        ("test", test),
    ]:
        missing = [
            column
            for column in required
            if column not in dataframe.columns
        ]

        if missing:
            raise ValueError(
                f"{label}: missing columns: {missing}"
            )

        feature_nulls = int(
            dataframe[
                feature_columns
            ]
            .isna()
            .sum()
            .sum()
        )

        if feature_nulls:
            raise ValueError(
                f"{label}: {feature_nulls} "
                "missing feature cells."
            )

        for target_column in HORIZONS.values():
            target_nulls = int(
                dataframe[
                    target_column
                ]
                .isna()
                .sum()
            )

            if target_nulls:
                raise ValueError(
                    f"{label}: target "
                    f"{target_column} contains "
                    f"{target_nulls} missing values."
                )

        duplicates = int(
            dataframe.duplicated(
                subset=[
                    "city",
                    "timestamp",
                ]
            ).sum()
        )

        if duplicates:
            raise ValueError(
                f"{label}: found {duplicates} "
                "duplicate city/timestamp keys."
            )

    train_end = train[
        "timestamp"
    ].max()

    test_start = test[
        "timestamp"
    ].min()

    if test_start <= train_end:
        raise ValueError(
            "Final test overlaps training data."
        )

    gap_hours = (
        test_start - train_end
    ).total_seconds() / 3600.0

    if gap_hours <= 72:
        raise ValueError(
            f"Expected purge gap >72h, "
            f"got {gap_hours:.2f}h."
        )

    print("\nFinal split integrity:")
    print(
        f"  Train rows: {len(train):,}"
    )
    print(
        f"  Test rows:  {len(test):,}"
    )
    print(
        f"  Cities:     {test['city'].nunique()}"
    )
    print(
        f"  Train end:  {train_end}"
    )
    print(
        f"  Test start: {test_start}"
    )
    print(
        f"  Purge gap:  {gap_hours:.1f} hours"
    )


def metrics_dict(
    y_true,
    y_pred,
) -> dict:
    return (
        calculate_regression_metrics(
            y_true,
            y_pred,
        ).to_dict()
    )


def calculate_city_metrics(
    predictions: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    rows = []

    for city, group in predictions.groupby(
        "city",
        sort=True,
    ):
        xgb_metrics = metrics_dict(
            group["y_true"],
            group["xgboost_prediction"],
        )

        persistence_metrics = metrics_dict(
            group["y_true"],
            group["persistence_prediction"],
        )

        rows.append(
            {
                "city": city,
                "horizon_hours": horizon,
                "rows": int(len(group)),
                "xgboost_mae": float(
                    xgb_metrics["mae"]
                ),
                "xgboost_rmse": float(
                    xgb_metrics["rmse"]
                ),
                "xgboost_r2": float(
                    xgb_metrics["r2"]
                ),
                "persistence_mae": float(
                    persistence_metrics["mae"]
                ),
                "persistence_rmse": float(
                    persistence_metrics["rmse"]
                ),
                "persistence_r2": float(
                    persistence_metrics["r2"]
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 90)
    print(
        "PEARLS AQI - FINAL 2026 HOLDOUT EVALUATION"
    )
    print("=" * 90)

    print(
        "\nWARNING: This script evaluates the "
        "previously untouched final 2026 holdout."
    )

    freeze = verify_model_selection_freeze()

    print("\nModel-selection freeze verified:")
    print(
        f"  Frozen at: "
        f"{freeze['frozen_at_utc']}"
    )
    print(
        "  24h -> XGBoost"
    )
    print(
        "  48h -> XGBoost"
    )
    print(
        "  72h -> XGBoost"
    )

    feature_columns = (
        AQIFeatureEngineer
        .get_model_feature_columns()
    )

    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} "
            f"features, got {len(feature_columns)}."
        )

    train, test = load_final_splits()

    validate_final_splits(
        train=train,
        test=test,
        feature_columns=feature_columns,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluation_started = datetime.now(
        timezone.utc
    ).isoformat()

    overall_rows = []
    city_frames = []

    total_start = time.perf_counter()

    for horizon, target_column in HORIZONS.items():
        print("\n" + "=" * 90)
        print(
            f"FINAL EVALUATION | {horizon}h"
        )
        print("=" * 90)

        x_train = train[
            feature_columns
        ]

        y_train = train[
            target_column
        ]

        x_test = test[
            feature_columns
        ]

        y_test = test[
            target_column
        ]

        model = create_xgboost_model()

        print(
            f"Training XGBoost on "
            f"{len(train):,} rows..."
        )

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
            x_test
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

        persistence_prediction = (
            test[
                "aqi_current"
            ]
            .to_numpy(dtype=float)
        )

        xgb_metrics = metrics_dict(
            y_test,
            xgb_prediction,
        )

        persistence_metrics = metrics_dict(
            y_test,
            persistence_prediction,
        )

        improvement = (
            (
                persistence_metrics["mae"]
                - xgb_metrics["mae"]
            )
            / persistence_metrics["mae"]
            * 100.0
        )

        print("\nFinal test metrics:")
        print(
            f"  XGBoost:"
            f"     MAE={xgb_metrics['mae']:.4f} "
            f"RMSE={xgb_metrics['rmse']:.4f} "
            f"R2={xgb_metrics['r2']:.4f}"
        )

        print(
            f"  Persistence:"
            f" MAE={persistence_metrics['mae']:.4f} "
            f"RMSE={persistence_metrics['rmse']:.4f} "
            f"R2={persistence_metrics['r2']:.4f}"
        )

        print(
            f"  MAE improvement over persistence: "
            f"{improvement:.2f}%"
        )

        print(
            f"  Fit time: {fit_seconds:.2f}s"
        )

        predictions = pd.DataFrame(
            {
                "city": test["city"],
                "timestamp": test["timestamp"],
                "horizon_hours": horizon,
                "y_true": y_test.to_numpy(),
                "xgboost_prediction": (
                    xgb_prediction
                ),
                "persistence_prediction": (
                    persistence_prediction
                ),
            }
        )

        horizon_dir = (
            OUTPUT_DIR
            / f"{horizon}h"
        )

        horizon_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        predictions.to_parquet(
            horizon_dir
            / "final_test_predictions.parquet",
            index=False,
        )

        predictions.to_csv(
            horizon_dir
            / "final_test_predictions.csv",
            index=False,
        )

        model_path = (
            horizon_dir
            / "xgboost_final.joblib"
        )

        joblib.dump(
            model,
            model_path,
        )

        horizon_metrics = {
            "horizon_hours": horizon,
            "selected_model": "xgboost",
            "selection_frozen_before_test": True,
            "test_rows": int(len(test)),
            "xgboost": {
                "mae": float(
                    xgb_metrics["mae"]
                ),
                "rmse": float(
                    xgb_metrics["rmse"]
                ),
                "r2": float(
                    xgb_metrics["r2"]
                ),
            },
            "persistence_reference": {
                "mae": float(
                    persistence_metrics["mae"]
                ),
                "rmse": float(
                    persistence_metrics["rmse"]
                ),
                "r2": float(
                    persistence_metrics["r2"]
                ),
            },
            "mae_improvement_over_persistence_percent": (
                float(improvement)
            ),
            "fit_seconds": float(
                fit_seconds
            ),
            "predict_seconds": float(
                predict_seconds
            ),
        }

        with (
            horizon_dir / "metrics.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                horizon_metrics,
                file,
                indent=2,
            )

        overall_rows.append(
            {
                "horizon_hours": horizon,
                "model": "xgboost",
                "test_rows": int(
                    len(test)
                ),
                "mae": float(
                    xgb_metrics["mae"]
                ),
                "rmse": float(
                    xgb_metrics["rmse"]
                ),
                "r2": float(
                    xgb_metrics["r2"]
                ),
                "persistence_mae": float(
                    persistence_metrics["mae"]
                ),
                "mae_improvement_percent": float(
                    improvement
                ),
                "fit_seconds": float(
                    fit_seconds
                ),
                "predict_seconds": float(
                    predict_seconds
                ),
            }
        )

        city_metrics = calculate_city_metrics(
            predictions=predictions,
            horizon=horizon,
        )

        city_frames.append(
            city_metrics
        )

        city_metrics.to_csv(
            horizon_dir
            / "city_metrics.csv",
            index=False,
        )

    total_seconds = (
        time.perf_counter()
        - total_start
    )

    overall = pd.DataFrame(
        overall_rows
    )

    overall.to_csv(
        OUTPUT_DIR
        / "final_test_results.csv",
        index=False,
    )

    all_city_metrics = pd.concat(
        city_frames,
        ignore_index=True,
    )

    all_city_metrics.to_csv(
        OUTPUT_DIR
        / "all_city_metrics.csv",
        index=False,
    )

    final_report = {
        "evaluation_type": (
            "ONE_TIME_FINAL_HOLDOUT_EVALUATION"
        ),
        "evaluation_started_utc": (
            evaluation_started
        ),
        "model_selection_freeze_utc": (
            freeze["frozen_at_utc"]
        ),
        "selection_was_frozen_before_test": True,
        "selected_models": {
            "24h": "xgboost",
            "48h": "xgboost",
            "72h": "xgboost",
        },
        "selection_metric": (
            "mean walk-forward validation MAE"
        ),
        "training_rows": int(
            len(train)
        ),
        "test_rows": int(
            len(test)
        ),
        "cities": int(
            test["city"].nunique()
        ),
        "feature_count": int(
            len(feature_columns)
        ),
        "train_start": str(
            train["timestamp"].min()
        ),
        "train_end": str(
            train["timestamp"].max()
        ),
        "test_start": str(
            test["timestamp"].min()
        ),
        "test_end": str(
            test["timestamp"].max()
        ),
        "test_metrics_are_for_model_assessment_only": True,
        "test_metrics_must_not_be_used_for_model_reselection": True,
        "total_runtime_seconds": float(
            total_seconds
        ),
    }

    with (
        OUTPUT_DIR
        / "final_holdout_report.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            final_report,
            file,
            indent=2,
        )

    print("\n" + "=" * 90)
    print(
        "FINAL HOLDOUT RESULTS"
    )
    print("=" * 90)

    display = overall[
        [
            "horizon_hours",
            "mae",
            "rmse",
            "r2",
            "persistence_mae",
            "mae_improvement_percent",
        ]
    ].copy()

    print(
        display.to_string(
            index=False
        )
    )

    print("\n" + "=" * 90)
    print(
        "FINAL EVALUATION COMPLETE"
    )
    print("=" * 90)

    print(
        f"\nRuntime: "
        f"{total_seconds / 60:.2f} minutes"
    )

    print(
        "\nIMPORTANT:"
    )
    print(
        "The 2026 holdout has now been consumed."
    )
    print(
        "Do NOT use these metrics to select "
        "a different model."
    )

    print(
        "\nResults:"
    )
    print(
        OUTPUT_DIR
        / "final_test_results.csv"
    )

    print(
        OUTPUT_DIR
        / "all_city_metrics.csv"
    )

    print(
        OUTPUT_DIR
        / "final_holdout_report.json"
    )


if __name__ == "__main__":
    main()