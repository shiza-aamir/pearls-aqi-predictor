from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path("artifacts/model_selection")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


CLASSICAL_RESULTS = {
    "24h": {
        "persistence": 15.4005,
        "ridge": 13.6634,
        "random_forest": 13.6623,
        "xgboost": 13.3346,
    },
    "48h": {
        "persistence": 20.7931,
        "ridge": 19.4529,
        "random_forest": 19.3003,
        "xgboost": 19.1634,
    },
    "72h": {
        "persistence": 22.9612,
        "ridge": 21.2856,
        "random_forest": 21.1477,
        "xgboost": 20.9650,
    },
}

CNN_RESULTS = {
    "24h": {
        "mean_mae": 16.9662,
        "std_mae": 2.8545,
        "worst_mae": 20.0695,
    },
    "48h": {
        "mean_mae": 21.1976,
        "std_mae": 1.0813,
        "worst_mae": 21.8965,
    },
    "72h": {
        "mean_mae": 22.2317,
        "std_mae": 1.7184,
        "worst_mae": 23.5477,
    },
}

TARGETED_DEEP_RESULTS = [
    {
        "model": "gru",
        "fold": "fold_3",
        "horizon": "24h",
        "mae": 13.6320,
        "rmse": 18.5925,
        "r2": 0.8469,
        "scope": "targeted_experiment",
    },
    {
        "model": "gru",
        "fold": "fold_3",
        "horizon": "72h",
        "mae": 19.9820,
        "rmse": 26.2276,
        "r2": 0.7009,
        "scope": "targeted_experiment",
    },
    {
        "model": "cnn_lstm",
        "fold": "fold_3",
        "horizon": "24h",
        "mae": 13.6362,
        "rmse": 18.7707,
        "r2": 0.8440,
        "scope": "targeted_experiment",
    },
]

SELECTED_MODELS = {
    "24h": "xgboost",
    "48h": "xgboost",
    "72h": "xgboost",
}


def main() -> None:
    rows = []

    for horizon, models in CLASSICAL_RESULTS.items():
        for model, mae in models.items():
            rows.append(
                {
                    "model": model,
                    "horizon": horizon,
                    "evaluation": "3_fold_walk_forward",
                    "mean_validation_mae": mae,
                }
            )

    for horizon, metrics in CNN_RESULTS.items():
        rows.append(
            {
                "model": "cnn",
                "horizon": horizon,
                "evaluation": "3_fold_walk_forward",
                "mean_validation_mae": metrics["mean_mae"],
            }
        )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(
        OUTPUT_DIR / "validation_model_comparison.csv",
        index=False,
    )

    targeted_df = pd.DataFrame(TARGETED_DEEP_RESULTS)
    targeted_df.to_csv(
        OUTPUT_DIR / "targeted_deep_experiments.csv",
        index=False,
    )

    freeze_time = datetime.now(timezone.utc).isoformat()

    decision = {
        "decision_status": "FROZEN_BEFORE_FINAL_TEST",
        "frozen_at_utc": freeze_time,
        "selection_metric": "mean walk-forward validation MAE",
        "selection_period": "2023-2025 validation folds",
        "number_of_validation_folds": 3,
        "purge_gap_hours": 73,
        "selected_models": SELECTED_MODELS,
        "selection_reason": (
            "XGBoost achieved the lowest mean walk-forward validation MAE "
            "for all three forecast horizons among models evaluated across "
            "all three validation folds."
        ),
        "cnn_evaluation": (
            "CNN completed full 3-fold walk-forward evaluation for all "
            "three forecast horizons."
        ),
        "gru_evaluation": (
            "GRU was evaluated only as targeted experiments on fold_3 "
            "for 24h and 72h because of high CPU training cost."
        ),
        "cnn_lstm_evaluation": (
            "CNN-LSTM was evaluated only as a targeted experiment on "
            "fold_3 for 24h because of high CPU training cost."
        ),
        "final_2026_test_used_for_selection": False,
        "final_2026_test_status": "UNTOUCHED_AT_SELECTION_FREEZE",
    }

    with open(
        OUTPUT_DIR / "model_selection_freeze.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(decision, f, indent=2)

    print("=" * 80)
    print("PEARLS AQI - MODEL SELECTION FREEZE")
    print("=" * 80)

    print("\nSelected models:")
    for horizon, model in SELECTED_MODELS.items():
        print(f"  {horizon}: {model}")

    print("\nSelection basis:")
    print("  Metric: Mean walk-forward validation MAE")
    print("  Validation folds: 3")
    print("  Validation years: 2023-2025")
    print("  Final 2026 test used: NO")

    print("\nArtifacts:")
    print(
        OUTPUT_DIR / "validation_model_comparison.csv"
    )
    print(
        OUTPUT_DIR / "targeted_deep_experiments.csv"
    )
    print(
        OUTPUT_DIR / "model_selection_freeze.json"
    )

    print("\nMODEL SELECTION IS NOW FROZEN.")
    print("24h -> XGBoost")
    print("48h -> XGBoost")
    print("72h -> XGBoost")


if __name__ == "__main__":
    main()