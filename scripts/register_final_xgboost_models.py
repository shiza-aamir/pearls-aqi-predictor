from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost
from mlflow import MlflowClient


TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "pearls-aqi-forecasting"

MODEL_NAME = "pearls-aqi-xgboost"

FINAL_ROOT = Path("artifacts/final_holdout")

FREEZE_PATH = Path(
    "artifacts/model_selection/model_selection_freeze.json"
)

FINAL_REPORT_PATH = (
    FINAL_ROOT / "final_holdout_report.json"
)

HORIZONS = {
    "24h": 24,
    "48h": 48,
    "72h": 72,
}

ALIASES = {
    "24h": "champion-24h",
    "48h": "champion-48h",
    "72h": "champion-72h",
}


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def verify_freeze(freeze: dict) -> None:
    if (
        freeze.get("decision_status")
        != "FROZEN_BEFORE_FINAL_TEST"
    ):
        raise ValueError(
            "Model-selection freeze is invalid."
        )

    if (
        freeze.get(
            "final_2026_test_used_for_selection"
        )
        is not False
    ):
        raise ValueError(
            "Freeze artifact does not confirm "
            "an untouched final test."
        )

    selected = freeze.get(
        "selected_models",
        {}
    )

    for horizon in HORIZONS:
        if selected.get(horizon) != "xgboost":
            raise ValueError(
                f"{horizon}: frozen model is not XGBoost."
            )


def verify_final_report(report: dict) -> None:
    if (
        report.get("evaluation_type")
        != "ONE_TIME_FINAL_HOLDOUT_EVALUATION"
    ):
        raise ValueError(
            "Unexpected final evaluation type."
        )

    if (
        report.get(
            "selection_was_frozen_before_test"
        )
        is not True
    ):
        raise ValueError(
            "Final report does not confirm "
            "pre-test model selection."
        )

    if (
        report.get(
            "test_metrics_must_not_be_used_for_model_reselection"
        )
        is not True
    ):
        raise ValueError(
            "Final report does not contain "
            "the holdout protection flag."
        )


def get_or_create_experiment() -> str:
    experiment = mlflow.get_experiment_by_name(
        EXPERIMENT_NAME
    )

    if experiment is not None:
        return experiment.experiment_id

    return mlflow.create_experiment(
        EXPERIMENT_NAME
    )


def register_horizon(
    client: MlflowClient,
    experiment_id: str,
    horizon_name: str,
    horizon_hours: int,
    freeze: dict,
    final_report: dict,
) -> str:
    horizon_dir = (
        FINAL_ROOT / horizon_name
    )

    model_path = (
        horizon_dir / "xgboost_final.joblib"
    )

    metrics_path = (
        horizon_dir / "metrics.json"
    )

    city_metrics_path = (
        horizon_dir / "city_metrics.csv"
    )

    required_paths = [
        model_path,
        metrics_path,
        city_metrics_path,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing final artifact: {path}"
            )

    metrics = load_json(
        metrics_path
    )

    if (
        metrics.get("selected_model")
        != "xgboost"
    ):
        raise ValueError(
            f"{horizon_name}: final artifact "
            "is not marked as XGBoost."
        )

    if (
        metrics.get("horizon_hours")
        != horizon_hours
    ):
        raise ValueError(
            f"{horizon_name}: horizon mismatch."
        )

    model = joblib.load(
        model_path
    )

    model_class = (
        f"{model.__class__.__module__}."
        f"{model.__class__.__name__}"
    )

    if model.__class__.__name__ != "XGBRegressor":
        raise TypeError(
            f"{horizon_name}: expected XGBRegressor, "
            f"got {model_class}."
        )

    print(
        f"Loaded model: {model_class}"
    )

    run_name = (
        f"final-xgboost-{horizon_name}"
    )

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=run_name,
    ) as run:
        run_id = run.info.run_id

        mlflow.log_params(
            {
                "model_name": "xgboost",
                "horizon": horizon_name,
                "horizon_hours": horizon_hours,
                "feature_count": (
                    final_report["feature_count"]
                ),
                "training_rows": (
                    final_report["training_rows"]
                ),
                "test_rows": (
                    final_report["test_rows"]
                ),
                "cities": (
                    final_report["cities"]
                ),
                "train_start": (
                    final_report["train_start"]
                ),
                "train_end": (
                    final_report["train_end"]
                ),
                "test_start": (
                    final_report["test_start"]
                ),
                "test_end": (
                    final_report["test_end"]
                ),
                "selection_metric": (
                    final_report["selection_metric"]
                ),
                "selection_frozen_before_test": True,
                "final_holdout_consumed": True,
            }
        )

        xgb_metrics = metrics[
            "xgboost"
        ]

        persistence = metrics[
            "persistence_reference"
        ]

        mlflow.log_metrics(
            {
                "final_test_mae": (
                    xgb_metrics["mae"]
                ),
                "final_test_rmse": (
                    xgb_metrics["rmse"]
                ),
                "final_test_r2": (
                    xgb_metrics["r2"]
                ),
                "persistence_test_mae": (
                    persistence["mae"]
                ),
                "persistence_test_rmse": (
                    persistence["rmse"]
                ),
                "persistence_test_r2": (
                    persistence["r2"]
                ),
                "mae_improvement_over_persistence_percent": (
                    metrics[
                        "mae_improvement_over_persistence_percent"
                    ]
                ),
                "fit_seconds": (
                    metrics["fit_seconds"]
                ),
                "predict_seconds": (
                    metrics["predict_seconds"]
                ),
            }
        )

        mlflow.set_tags(
            {
                "project": "pearls-aqi-predictor",
                "stage": "final_holdout",
                "model_family": "xgboost",
                "forecast_horizon": horizon_name,
                "selection_status": (
                    "frozen_before_final_test"
                ),
                "holdout_policy": (
                    "assessment_only_no_reselection"
                ),
            }
        )

        mlflow.log_artifact(
            str(metrics_path),
            artifact_path="evaluation",
        )

        mlflow.log_artifact(
            str(city_metrics_path),
            artifact_path="evaluation",
        )

        mlflow.log_artifact(
            str(FREEZE_PATH),
            artifact_path="model_selection",
        )

        mlflow.log_artifact(
            str(FINAL_REPORT_PATH),
            artifact_path="evaluation",
        )

        print(
            "Logging model with native "
            "MLflow XGBoost flavor..."
        )

        model_info = mlflow.xgboost.log_model(
            xgb_model=model,
            name="model",
        )

        print(
            f"MLflow model logged: "
            f"{model_info.model_uri}"
        )

        model_uri = (
            f"runs:/{run_id}/model"
        )

        registered = mlflow.register_model(
            model_uri=model_uri,
            name=MODEL_NAME,
        )

        version = str(
            registered.version
        )

    client.set_model_version_tag(
        name=MODEL_NAME,
        version=version,
        key="forecast_horizon",
        value=horizon_name,
    )

    client.set_model_version_tag(
        name=MODEL_NAME,
        version=version,
        key="model_type",
        value="xgboost",
    )

    client.set_model_version_tag(
        name=MODEL_NAME,
        version=version,
        key="deployment_status",
        value="champion",
    )

    client.set_model_version_tag(
        name=MODEL_NAME,
        version=version,
        key="selection_frozen_before_test",
        value="true",
    )

    client.set_model_version_tag(
        name=MODEL_NAME,
        version=version,
        key="training_data_end",
        value=final_report["train_end"],
    )

    client.set_model_version_tag(
        name=MODEL_NAME,
        version=version,
        key="final_test_year",
        value="2026",
    )

    alias = ALIASES[
        horizon_name
    ]

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=alias,
        version=version,
    )

    return version


def main() -> None:
    print("=" * 80)
    print(
        "PEARLS AQI - REGISTER FINAL XGBOOST MODELS"
    )
    print("=" * 80)

    mlflow.set_tracking_uri(
        TRACKING_URI
    )

    freeze = load_json(
        FREEZE_PATH
    )

    final_report = load_json(
        FINAL_REPORT_PATH
    )

    verify_freeze(
        freeze
    )

    verify_final_report(
        final_report
    )

    print(
        "\nModel-selection freeze verified."
    )

    print(
        "Final holdout report verified."
    )

    experiment_id = (
        get_or_create_experiment()
    )

    client = MlflowClient()

    registered_versions = {}

    for (
        horizon_name,
        horizon_hours,
    ) in HORIZONS.items():
        print("\n" + "-" * 80)
        print(
            f"REGISTERING {horizon_name}"
        )
        print("-" * 80)

        version = register_horizon(
            client=client,
            experiment_id=experiment_id,
            horizon_name=horizon_name,
            horizon_hours=horizon_hours,
            freeze=freeze,
            final_report=final_report,
        )

        registered_versions[
            horizon_name
        ] = version

        print(
            f"Registered {MODEL_NAME} "
            f"version {version}"
        )

        print(
            f"Alias: @{ALIASES[horizon_name]}"
        )

    print("\n" + "=" * 80)
    print(
        "FINAL XGBOOST REGISTRATION COMPLETE"
    )
    print("=" * 80)

    for horizon in HORIZONS:
        print(
            f"{horizon}: "
            f"{MODEL_NAME} "
            f"v{registered_versions[horizon]} "
            f"@{ALIASES[horizon]}"
        )

    print(
        "\nOld Random Forest registry "
        "has NOT been modified."
    )


if __name__ == "__main__":
    main()