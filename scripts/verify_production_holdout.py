from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.features.engineer import AQIFeatureEngineer
from src.ml.evaluation.metrics import calculate_regression_metrics

TEST_PATH = Path("data/splits/expanded/final/test.parquet")

MODEL_PATHS = {
    24: Path("models/production/xgboost_24h.ubj"),
    48: Path("models/production/xgboost_48h.ubj"),
    72: Path("models/production/xgboost_72h.ubj"),
}

TARGETS = {
    24: "target_aqi_24h",
    48: "target_aqi_48h",
    72: "target_aqi_72h",
}

FROZEN_METRICS = {
    24: {
        "mae": 14.90946710468107,
        "rmse": 21.16419577930405,
        "r2": 0.7907557377067282,
    },
    48: {
        "mae": 19.34609640715021,
        "rmse": 26.89485792484945,
        "r2": 0.6543274585103032,
    },
    72: {
        "mae": 20.467895556346942,
        "rmse": 28.24855829239133,
        "r2": 0.6127984773598563,
    },
}


def main() -> None:
    print("=" * 90)
    print("PEARLS AQI - PRODUCTION ARTIFACT HOLDOUT VERIFICATION")
    print("=" * 90)

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Missing frozen test split: {TEST_PATH}"
        )

    feature_columns = (
        AQIFeatureEngineer.get_model_feature_columns()
    )

    if len(feature_columns) != 56:
        raise ValueError(
            f"Expected 56 features, got {len(feature_columns)}."
        )

    test = pd.read_parquet(TEST_PATH)

    missing_features = [
        column
        for column in feature_columns
        if column not in test.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing feature columns: {missing_features}"
        )

    x_test = test[feature_columns]

    results = []

    for horizon, model_path in MODEL_PATHS.items():
        print("\n" + "-" * 90)
        print(f"Checking deployed {horizon}h model")
        print("-" * 90)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing production model: {model_path}"
            )

        target_column = TARGETS[horizon]

        if target_column not in test.columns:
            raise ValueError(
                f"Missing target column: {target_column}"
            )

        model = XGBRegressor()
        model.load_model(model_path)

        predictions = model.predict(x_test)

        predictions = np.clip(
            predictions,
            0.0,
            500.0,
        )

        y_true = test[target_column]

        metrics = calculate_regression_metrics(
            y_true,
            predictions,
        ).to_dict()

        frozen = FROZEN_METRICS[horizon]

        mae_delta = (
            float(metrics["mae"])
            - frozen["mae"]
        )

        rmse_delta = (
            float(metrics["rmse"])
            - frozen["rmse"]
        )

        r2_delta = (
            float(metrics["r2"])
            - frozen["r2"]
        )

        same = (
            abs(mae_delta) < 1e-8
            and abs(rmse_delta) < 1e-8
            and abs(r2_delta) < 1e-8
        )

        print(
            f"Current production MAE: "
            f"{metrics['mae']:.12f}"
        )

        print(
            f"Frozen evaluation MAE:  "
            f"{frozen['mae']:.12f}"
        )

        print(
            f"MAE delta:              "
            f"{mae_delta:+.12f}"
        )

        print(
            f"Current production RMSE: "
            f"{metrics['rmse']:.12f}"
        )

        print(
            f"Frozen evaluation RMSE:  "
            f"{frozen['rmse']:.12f}"
        )

        print(
            f"Current production R²: "
            f"{metrics['r2']:.12f}"
        )

        print(
            f"Frozen evaluation R²:  "
            f"{frozen['r2']:.12f}"
        )

        print(
            "Exact frozen-metric match: "
            f"{'YES' if same else 'NO'}"
        )

        results.append(
            {
                "horizon_hours": horizon,
                "production_model": str(model_path),
                "production_metrics": {
                    "mae": float(metrics["mae"]),
                    "rmse": float(metrics["rmse"]),
                    "r2": float(metrics["r2"]),
                },
                "frozen_metrics": frozen,
                "delta": {
                    "mae": float(mae_delta),
                    "rmse": float(rmse_delta),
                    "r2": float(r2_delta),
                },
                "exact_match": same,
            }
        )

    output_path = Path(
        "artifacts/verification/"
        "production_holdout_verification.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "verification_type": (
                    "READ_ONLY_PRODUCTION_ARTIFACT_EVALUATION"
                ),
                "training_performed": False,
                "model_selection_performed": False,
                "production_promotion_performed": False,
                "test_rows": len(test),
                "feature_count": len(feature_columns),
                "results": results,
            },
            file,
            indent=2,
        )

    print("\n" + "=" * 90)
    print("VERIFICATION COMPLETE")
    print("=" * 90)

    all_match = all(
        result["exact_match"]
        for result in results
    )

    print(
        "All production artifacts match frozen metrics: "
        f"{'YES' if all_match else 'NO'}"
    )

    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()