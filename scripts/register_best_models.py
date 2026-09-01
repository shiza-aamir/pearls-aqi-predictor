from __future__ import annotations

import mlflow
from mlflow import MlflowClient


TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "pearls-aqi-forecasting"

MODEL_NAME = "pearls-aqi-random-forest"

HORIZONS = ("24h", "48h", "72h")


def find_latest_successful_run(
    client: MlflowClient,
    experiment_id: str,
    horizon: str,
):
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=(
            "attributes.status = 'FINISHED' "
            "AND params.model_name = 'random_forest' "
            f"AND params.horizon = '{horizon}'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )

    if not runs:
        raise RuntimeError(
            f"No successful Random Forest run found for {horizon}."
        )

    return runs[0]


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)

    client = MlflowClient()

    experiment = client.get_experiment_by_name(
        EXPERIMENT_NAME
    )

    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment '{EXPERIMENT_NAME}' was not found."
        )

    print("=" * 70)
    print("REGISTERING AQI FORECAST MODELS")
    print("=" * 70)

    for horizon in HORIZONS:
        run = find_latest_successful_run(
            client=client,
            experiment_id=experiment.experiment_id,
            horizon=horizon,
        )

        run_id = run.info.run_id

        model_uri = f"runs:/{run_id}/model"

        print(f"\nHorizon: {horizon}")
        print(f"Run ID:  {run_id}")
        print(
            f"Test MAE: "
            f"{run.data.metrics.get('test_mae', float('nan')):.3f}"
        )
        print(
            f"Test RMSE: "
            f"{run.data.metrics.get('test_rmse', float('nan')):.3f}"
        )
        print(
            f"Test R2: "
            f"{run.data.metrics.get('test_r2', float('nan')):.3f}"
        )

        registered = mlflow.register_model(
            model_uri=model_uri,
            name=MODEL_NAME,
        )

        version = registered.version

        client.set_model_version_tag(
            name=MODEL_NAME,
            version=version,
            key="forecast_horizon",
            value=horizon,
        )

        client.set_model_version_tag(
            name=MODEL_NAME,
            version=version,
            key="model_type",
            value="random_forest",
        )

        client.set_model_version_tag(
            name=MODEL_NAME,
            version=version,
            key="deployment_status",
            value="candidate",
        )

        print(
            f"Registered: {MODEL_NAME} "
            f"version {version}"
        )

    print("\n" + "=" * 70)
    print("REGISTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()