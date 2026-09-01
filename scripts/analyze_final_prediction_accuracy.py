from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.features.aqi.calculator import AQICalculator


ARTIFACT_ROOT = Path(
    "artifacts/final_holdout"
)

OUTPUT_ROOT = Path(
    "artifacts/prediction_accuracy"
)

HORIZONS = [
    "24h",
    "48h",
    "72h",
]

TOLERANCES = [
    5,
    10,
    20,
    30,
]


def category_from_value(
    value: float,
) -> str:
    value = float(
        np.clip(
            value,
            0.0,
            500.0,
        )
    )

    return (
        AQICalculator
        .category_from_aqi(
            int(
                round(value)
            )
        )
    )


def calculate_metrics(
    df: pd.DataFrame,
) -> dict:
    y_true = (
        df["y_true"]
        .astype(float)
        .to_numpy()
    )

    y_pred = (
        df["xgboost_prediction"]
        .astype(float)
        .to_numpy()
    )

    absolute_error = np.abs(
        y_true - y_pred
    )

    metrics = {
        "rows": int(
            len(df)
        ),
        "mae": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),
        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            )
        ),
        "r2": float(
            r2_score(
                y_true,
                y_pred,
            )
        ),
        "median_absolute_error": float(
            np.median(
                absolute_error
            )
        ),
        "mean_actual_aqi": float(
            np.mean(
                y_true
            )
        ),
        "mean_predicted_aqi": float(
            np.mean(
                y_pred
            )
        ),
    }

    for tolerance in TOLERANCES:
        metrics[
            f"within_{tolerance}_aqi_pct"
        ] = float(
            np.mean(
                absolute_error
                <= tolerance
            )
            * 100.0
        )

    true_categories = [
        category_from_value(
            value
        )
        for value in y_true
    ]

    predicted_categories = [
        category_from_value(
            value
        )
        for value in y_pred
    ]

    category_correct = (
        np.asarray(
            true_categories
        )
        == np.asarray(
            predicted_categories
        )
    )

    metrics[
        "category_accuracy_pct"
    ] = float(
        category_correct.mean()
        * 100.0
    )

    return metrics


def main() -> None:
    print("=" * 80)
    print(
        "PEARLS AQI - FINAL HOLDOUT "
        "PREDICTION ACCURACY ANALYSIS"
    )
    print("=" * 80)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    horizon_results = []

    city_results = []

    all_enriched = []

    for horizon in HORIZONS:
        input_path = (
            ARTIFACT_ROOT
            / horizon
            / "final_test_predictions.csv"
        )

        if not input_path.exists():
            raise FileNotFoundError(
                f"Missing prediction file: "
                f"{input_path}"
            )

        df = pd.read_csv(
            input_path
        )

        required = [
            "city",
            "timestamp",
            "horizon_hours",
            "y_true",
            "xgboost_prediction",
            "persistence_prediction",
        ]

        missing = [
            column
            for column in required
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{horizon}: missing columns: "
                f"{missing}"
            )

        if df.empty:
            raise ValueError(
                f"{horizon}: prediction "
                "file is empty."
            )

        df["timestamp"] = (
            pd.to_datetime(
                df["timestamp"],
                utc=True,
                errors="raise",
            )
        )

        numeric_columns = [
            "y_true",
            "xgboost_prediction",
            "persistence_prediction",
        ]

        for column in numeric_columns:
            df[column] = (
                pd.to_numeric(
                    df[column],
                    errors="raise",
                )
            )

        if (
            df[numeric_columns]
            .isnull()
            .any()
            .any()
        ):
            raise ValueError(
                f"{horizon}: null prediction "
                "values detected."
            )

        if not np.isfinite(
            df[numeric_columns]
            .to_numpy()
        ).all():
            raise ValueError(
                f"{horizon}: non-finite "
                "prediction values detected."
            )

        df["absolute_error"] = (
            (
                df["y_true"]
                - df[
                    "xgboost_prediction"
                ]
            )
            .abs()
        )

        for tolerance in TOLERANCES:
            df[
                f"within_{tolerance}"
            ] = (
                df["absolute_error"]
                <= tolerance
            )

        df["actual_category"] = (
            df["y_true"]
            .apply(
                category_from_value
            )
        )

        df["predicted_category"] = (
            df[
                "xgboost_prediction"
            ]
            .apply(
                category_from_value
            )
        )

        df["category_correct"] = (
            df["actual_category"]
            == df["predicted_category"]
        )

        metrics = (
            calculate_metrics(
                df
            )
        )

        metrics["horizon"] = horizon

        horizon_results.append(
            metrics
        )

        print(
            f"\n{horizon} FORECAST"
        )

        print("-" * 80)

        print(
            f"Predictions:       "
            f"{metrics['rows']:,}"
        )

        print(
            f"MAE:               "
            f"{metrics['mae']:.2f}"
        )

        print(
            f"RMSE:              "
            f"{metrics['rmse']:.2f}"
        )

        print(
            f"R²:                 "
            f"{metrics['r2']:.4f}"
        )

        print(
            f"Median abs error:  "
            f"{metrics['median_absolute_error']:.2f}"
        )

        for tolerance in TOLERANCES:
            value = metrics[
                f"within_{tolerance}_aqi_pct"
            ]

            print(
                f"Within ±{tolerance:<2} AQI:    "
                f"{value:6.2f}%"
            )

        print(
            f"Category accuracy: "
            f"{metrics['category_accuracy_pct']:6.2f}%"
        )

        for city, city_df in (
            df.groupby(
                "city"
            )
        ):
            city_metrics = (
                calculate_metrics(
                    city_df
                )
            )

            city_metrics[
                "horizon"
            ] = horizon

            city_metrics[
                "city"
            ] = str(
                city
            )

            city_results.append(
                city_metrics
            )

        confusion = pd.crosstab(
            df["actual_category"],
            df["predicted_category"],
            rownames=[
                "actual_category"
            ],
            colnames=[
                "predicted_category"
            ],
            dropna=False,
        )

        confusion.to_csv(
            OUTPUT_ROOT
            / (
                f"{horizon}_"
                "category_confusion_matrix.csv"
            )
        )

        df["horizon"] = horizon

        all_enriched.append(
            df
        )

    horizon_df = pd.DataFrame(
        horizon_results
    )

    city_df = pd.DataFrame(
        city_results
    )

    enriched_df = pd.concat(
        all_enriched,
        ignore_index=True,
    )

    horizon_columns = [
        "horizon",
        "rows",
        "mae",
        "rmse",
        "r2",
        "median_absolute_error",
        "within_5_aqi_pct",
        "within_10_aqi_pct",
        "within_20_aqi_pct",
        "within_30_aqi_pct",
        "category_accuracy_pct",
        "mean_actual_aqi",
        "mean_predicted_aqi",
    ]

    horizon_df = horizon_df[
        horizon_columns
    ]

    city_columns = [
        "horizon",
        "city",
        "rows",
        "mae",
        "rmse",
        "r2",
        "median_absolute_error",
        "within_5_aqi_pct",
        "within_10_aqi_pct",
        "within_20_aqi_pct",
        "within_30_aqi_pct",
        "category_accuracy_pct",
    ]

    city_df = city_df[
        city_columns
    ]

    horizon_df.to_csv(
        OUTPUT_ROOT
        / "horizon_accuracy_summary.csv",
        index=False,
    )

    city_df.to_csv(
        OUTPUT_ROOT
        / "city_accuracy_summary.csv",
        index=False,
    )

    enriched_df.to_parquet(
        OUTPUT_ROOT
        / "enriched_final_predictions.parquet",
        index=False,
    )

    report = {
        "evaluation_type": (
            "FINAL_HOLDOUT_DESCRIPTIVE_ANALYSIS"
        ),
        "model": (
            "pearls-aqi-xgboost"
        ),
        "note": (
            "Additional descriptive metrics "
            "computed from the already-consumed "
            "final holdout predictions. "
            "No model selection or tuning "
            "performed using these results."
        ),
        "tolerances_aqi": (
            TOLERANCES
        ),
        "horizons": (
            horizon_results
        ),
    }

    with (
        OUTPUT_ROOT
        / "prediction_accuracy_report.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    print(
        "\n" + "=" * 80
    )

    print(
        "SUMMARY"
    )

    print("=" * 80)

    display_columns = [
        "horizon",
        "mae",
        "rmse",
        "r2",
        "within_10_aqi_pct",
        "within_20_aqi_pct",
        "within_30_aqi_pct",
        "category_accuracy_pct",
    ]

    print(
        horizon_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.2f}"
            ),
        )
    )

    print(
        "\nSaved:"
    )

    print(
        "  artifacts/prediction_accuracy/"
        "horizon_accuracy_summary.csv"
    )

    print(
        "  artifacts/prediction_accuracy/"
        "city_accuracy_summary.csv"
    )

    print(
        "  artifacts/prediction_accuracy/"
        "prediction_accuracy_report.json"
    )

    print(
        "  artifacts/prediction_accuracy/"
        "enriched_final_predictions.parquet"
    )

    print(
        "  category confusion matrices"
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "FINAL PREDICTION ACCURACY "
        "ANALYSIS: PASS"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()