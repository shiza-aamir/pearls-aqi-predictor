from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.engineer import AQIFeatureEngineer
from src.ml.evaluation.metrics import (
    calculate_regression_metrics,
)
from src.ml.models.baseline import PersistenceBaseline
from src.ml.models.random_forest import (
    create_random_forest_model,
)
from src.ml.models.ridge import create_ridge_model
from src.ml.models.xgboost_model import (
    create_xgboost_model,
)


SPLIT_ROOT = Path(
    "data/splits/expanded"
)

OUTPUT_DIR = Path(
    "artifacts/expanded_classical_benchmark"
)

DETAILED_RESULTS_PATH = (
    OUTPUT_DIR
    / "walk_forward_results.csv"
)

AGGREGATE_RESULTS_PATH = (
    OUTPUT_DIR
    / "walk_forward_aggregate.csv"
)

CHAMPIONS_PATH = (
    OUTPUT_DIR
    / "validation_champions.json"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "benchmark_report.json"
)

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

MODEL_NAMES = [
    "persistence",
    "ridge",
    "random_forest",
    "xgboost",
]


def load_fold(
    fold_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_dir = (
        SPLIT_ROOT / fold_name
    )

    train_path = (
        fold_dir / "train.parquet"
    )

    validation_path = (
        fold_dir
        / "validation.parquet"
    )

    if not train_path.exists():
        raise FileNotFoundError(
            f"Missing training split: "
            f"{train_path}"
        )

    if not validation_path.exists():
        raise FileNotFoundError(
            f"Missing validation split: "
            f"{validation_path}"
        )

    train = pd.read_parquet(
        train_path
    )

    validation = pd.read_parquet(
        validation_path
    )

    train["timestamp"] = pd.to_datetime(
        train["timestamp"],
        utc=True,
    )

    validation["timestamp"] = pd.to_datetime(
        validation["timestamp"],
        utc=True,
    )

    return train, validation


def validate_dataset(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    label: str,
) -> None:
    required = (
        feature_columns
        + [
            target_column,
            "aqi_current",
            "city",
            "timestamp",
        ]
    )

    missing_columns = [
        column
        for column in required
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{label}: missing columns: "
            f"{missing_columns}"
        )

    null_features = int(
        dataframe[
            feature_columns
        ]
        .isna()
        .sum()
        .sum()
    )

    null_targets = int(
        dataframe[
            target_column
        ]
        .isna()
        .sum()
    )

    if null_features:
        raise ValueError(
            f"{label}: found "
            f"{null_features} missing "
            "feature cells."
        )

    if null_targets:
        raise ValueError(
            f"{label}: found "
            f"{null_targets} missing "
            "target values."
        )


def create_model(
    model_name: str,
):
    if model_name == "ridge":
        return create_ridge_model()

    if model_name == "random_forest":
        return create_random_forest_model()

    if model_name == "xgboost":
        return create_xgboost_model()

    raise ValueError(
        f"Unknown trainable model: "
        f"{model_name}"
    )


def evaluate_persistence(
    validation: pd.DataFrame,
    target_column: str,
) -> dict:
    start = time.perf_counter()

    predictions = (
        PersistenceBaseline.predict(
            validation
        )
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    metrics = (
        calculate_regression_metrics(
            validation[target_column],
            predictions,
        )
    )

    return {
        **metrics.to_dict(),
        "fit_seconds": 0.0,
        "predict_seconds": float(
            elapsed
        ),
    }


def evaluate_model(
    model_name: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> dict:
    model = create_model(
        model_name
    )

    x_train = train[
        feature_columns
    ]

    y_train = train[
        target_column
    ]

    x_validation = validation[
        feature_columns
    ]

    y_validation = validation[
        target_column
    ]

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

    predict_start = (
        time.perf_counter()
    )

    predictions = model.predict(
        x_validation
    )

    predict_seconds = (
        time.perf_counter()
        - predict_start
    )

    metrics = (
        calculate_regression_metrics(
            y_validation,
            predictions,
        )
    )

    return {
        **metrics.to_dict(),
        "fit_seconds": float(
            fit_seconds
        ),
        "predict_seconds": float(
            predict_seconds
        ),
    }


def run_benchmark() -> pd.DataFrame:
    feature_columns = (
        AQIFeatureEngineer
        .get_model_feature_columns()
    )

    if len(feature_columns) != 56:
        raise ValueError(
            f"Expected 56 model features, "
            f"got {len(feature_columns)}."
        )

    rows = []

    total_jobs = (
        len(FOLDS)
        * len(HORIZONS)
        * len(MODEL_NAMES)
    )

    job_number = 0

    for fold_name in FOLDS:
        print(
            "\n" + "=" * 90
        )
        print(
            fold_name.upper()
        )
        print(
            "=" * 90
        )

        train, validation = (
            load_fold(
                fold_name
            )
        )

        print(
            f"Train rows: "
            f"{len(train):,}"
        )

        print(
            f"Validation rows: "
            f"{len(validation):,}"
        )

        for horizon, target_column in (
            HORIZONS.items()
        ):
            print(
                "\n" + "-" * 90
            )

            print(
                f"HORIZON: {horizon}h"
            )

            print(
                "-" * 90
            )

            validate_dataset(
                train,
                feature_columns,
                target_column,
                f"{fold_name} train",
            )

            validate_dataset(
                validation,
                feature_columns,
                target_column,
                f"{fold_name} validation",
            )

            for model_name in (
                MODEL_NAMES
            ):
                job_number += 1

                print(
                    f"\n[{job_number}/{total_jobs}] "
                    f"{model_name}"
                )

                if (
                    model_name
                    == "persistence"
                ):
                    result = (
                        evaluate_persistence(
                            validation,
                            target_column,
                        )
                    )

                else:
                    result = (
                        evaluate_model(
                            model_name,
                            train,
                            validation,
                            feature_columns,
                            target_column,
                        )
                    )

                row = {
                    "fold": fold_name,
                    "horizon_hours": (
                        horizon
                    ),
                    "target_column": (
                        target_column
                    ),
                    "model": model_name,
                    "train_rows": int(
                        len(train)
                    ),
                    "validation_rows": int(
                        len(validation)
                    ),
                    **result,
                }

                rows.append(row)

                print(
                    f"  MAE:  "
                    f"{result['mae']:.4f}"
                )

                print(
                    f"  RMSE: "
                    f"{result['rmse']:.4f}"
                )

                print(
                    f"  R2:   "
                    f"{result['r2']:.4f}"
                )

                print(
                    f"  Fit:  "
                    f"{result['fit_seconds']:.2f}s"
                )

                print(
                    f"  Pred: "
                    f"{result['predict_seconds']:.2f}s"
                )

    return pd.DataFrame(
        rows
    )


def build_aggregate_results(
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
            total_fit_seconds=(
                "fit_seconds",
                "sum",
            ),
        )
    )

    aggregate["mean_mae"] = (
        aggregate[
            "mean_mae"
        ].round(4)
    )

    aggregate["std_mae"] = (
        aggregate[
            "std_mae"
        ].round(4)
    )

    aggregate["worst_mae"] = (
        aggregate[
            "worst_mae"
        ].round(4)
    )

    aggregate["best_mae"] = (
        aggregate[
            "best_mae"
        ].round(4)
    )

    aggregate["mean_rmse"] = (
        aggregate[
            "mean_rmse"
        ].round(4)
    )

    aggregate["std_rmse"] = (
        aggregate[
            "std_rmse"
        ].round(4)
    )

    aggregate["mean_r2"] = (
        aggregate[
            "mean_r2"
        ].round(4)
    )

    aggregate["worst_r2"] = (
        aggregate[
            "worst_r2"
        ].round(4)
    )

    aggregate[
        "total_fit_seconds"
    ] = (
        aggregate[
            "total_fit_seconds"
        ].round(2)
    )

    aggregate = (
        aggregate
        .sort_values(
            [
                "horizon_hours",
                "mean_mae",
                "worst_mae",
            ]
        )
        .reset_index(drop=True)
    )

    aggregate["rank_by_mean_mae"] = (
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


def select_champions(
    aggregate: pd.DataFrame,
) -> dict:
    champions = {}

    for horizon in HORIZONS:
        subset = (
            aggregate[
                aggregate[
                    "horizon_hours"
                ]
                == horizon
            ]
            .sort_values(
                [
                    "mean_mae",
                    "worst_mae",
                    "std_mae",
                ]
            )
        )

        winner = (
            subset.iloc[0]
        )

        champions[
            f"{horizon}h"
        ] = {
            "model": str(
                winner["model"]
            ),
            "mean_mae": float(
                winner[
                    "mean_mae"
                ]
            ),
            "std_mae": float(
                winner[
                    "std_mae"
                ]
            ),
            "worst_mae": float(
                winner[
                    "worst_mae"
                ]
            ),
            "mean_rmse": float(
                winner[
                    "mean_rmse"
                ]
            ),
            "mean_r2": float(
                winner[
                    "mean_r2"
                ]
            ),
            "selection_basis": (
                "Lowest mean validation "
                "MAE across the three "
                "walk-forward folds. "
                "Worst-fold MAE and MAE "
                "standard deviation used "
                "as tie-breakers."
            ),
        }

    return champions


def print_aggregate(
    aggregate: pd.DataFrame,
) -> None:
    print(
        "\n" + "=" * 90
    )

    print(
        "AGGREGATED WALK-FORWARD RESULTS"
    )

    print(
        "=" * 90
    )

    display_columns = [
        "horizon_hours",
        "model",
        "mean_mae",
        "std_mae",
        "worst_mae",
        "mean_rmse",
        "mean_r2",
        "rank_by_mean_mae",
    ]

    print(
        aggregate[
            display_columns
        ].to_string(
            index=False
        )
    )


def print_champions(
    champions: dict,
) -> None:
    print(
        "\n" + "=" * 90
    )

    print(
        "VALIDATION CHAMPIONS"
    )

    print(
        "=" * 90
    )

    for horizon, result in (
        champions.items()
    ):
        print(
            f"{horizon}: "
            f"{result['model']} "
            f"| mean MAE="
            f"{result['mean_mae']:.4f} "
            f"| worst MAE="
            f"{result['worst_mae']:.4f}"
        )


def main() -> None:
    print(
        "=" * 90
    )

    print(
        "PEARLS AQI - EXPANDED "
        "CLASSICAL WALK-FORWARD BENCHMARK"
    )

    print(
        "=" * 90
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    benchmark_start = (
        time.perf_counter()
    )

    detailed = run_benchmark()

    aggregate = (
        build_aggregate_results(
            detailed
        )
    )

    champions = (
        select_champions(
            aggregate
        )
    )

    total_seconds = (
        time.perf_counter()
        - benchmark_start
    )

    detailed.to_csv(
        DETAILED_RESULTS_PATH,
        index=False,
    )

    aggregate.to_csv(
        AGGREGATE_RESULTS_PATH,
        index=False,
    )

    with CHAMPIONS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            champions,
            file,
            indent=2,
        )

    report = {
        "folds": FOLDS,
        "horizons": list(
            HORIZONS.keys()
        ),
        "models": MODEL_NAMES,
        "feature_count": 56,
        "total_evaluations": int(
            len(detailed)
        ),
        "total_seconds": float(
            total_seconds
        ),
        "final_2026_test_used": False,
        "champions": champions,
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

    print_aggregate(
        aggregate
    )

    print_champions(
        champions
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
        f"\nEvaluations: "
        f"{len(detailed)}"
    )

    print(
        f"Total runtime: "
        f"{total_seconds / 60:.2f} minutes"
    )

    print(
        "\nFinal 2026 test touched: NO"
    )

    print(
        f"\nDetailed results:\n"
        f"{DETAILED_RESULTS_PATH}"
    )

    print(
        f"\nAggregate results:\n"
        f"{AGGREGATE_RESULTS_PATH}"
    )

    print(
        f"\nChampions:\n"
        f"{CHAMPIONS_PATH}"
    )


if __name__ == "__main__":
    main()