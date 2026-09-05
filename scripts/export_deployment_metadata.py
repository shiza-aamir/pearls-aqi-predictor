# scripts/export_deployment_metadata.py

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import yaml
from mlflow import MlflowClient

TRACKING_URI = "sqlite:///mlflow.db"

MODEL_NAME = "pearls-aqi-xgboost"

FINAL_REPORT_PATH = Path(
    "artifacts/final_holdout/final_holdout_report.json"
)

FINAL_RESULTS_PATH = Path(
    "artifacts/final_holdout/final_test_results.csv"
)

ACCURACY_PATH = Path(
    "artifacts/prediction_accuracy/"
    "horizon_accuracy_summary.csv"
)

FEAST_CONFIG_PATH = Path(
    "feature_repo/feature_repo/feature_store.yaml"
)

FEAST_DEFINITION_PATH = Path(
    "feature_repo/feature_repo/feature_definitions.py"
)

FEAST_REPORT_PATH = Path(
    "artifacts/feature_store/"
    "expanded_feast_dataset_report.json"
)

DEPLOYMENT_ROOT = Path(
    "artifacts/deployment"
)

MODEL_MANIFEST_PATH = (
    DEPLOYMENT_ROOT
    / "model_registry_manifest.json"
)

PERFORMANCE_MANIFEST_PATH = (
    DEPLOYMENT_ROOT
    / "performance_manifest.json"
)

FEATURE_STORE_MANIFEST_PATH = (
    DEPLOYMENT_ROOT
    / "feature_store_manifest.json"
)

ALIASES = {
    24: "champion-24h",
    48: "champion-48h",
    72: "champion-72h",
}

EXPECTED_HORIZONS = {
    24,
    48,
    72,
}


def require_file(
    path: Path,
) -> None:
    if not path.exists():
        raise FileNotFoundError(
            "Required source artifact "
            f"does not exist: {path}"
        )


def load_json(
    path: Path,
) -> dict[str, Any]:
    require_file(
        path
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(
            file
        )

    if not isinstance(
        data,
        dict,
    ):
        raise TypeError(
            f"Expected JSON object in {path}."
        )

    return data


def write_json(
    path: Path,
    data: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )

        file.write(
            "\n"
        )


def exported_at() -> str:
    return (
        datetime.now(
            UTC
        )
        .isoformat()
    )


def export_performance_manifest() -> None:
    print(
        "Exporting final holdout performance..."
    )

    report = load_json(
        FINAL_REPORT_PATH
    )

    require_file(
        FINAL_RESULTS_PATH
    )

    require_file(
        ACCURACY_PATH
    )

    final_results = pd.read_csv(
        FINAL_RESULTS_PATH
    )

    accuracy = pd.read_csv(
        ACCURACY_PATH
    )

    required_final_columns = {
        "horizon_hours",
        "model",
        "test_rows",
        "mae",
        "rmse",
        "r2",
        "persistence_mae",
        "mae_improvement_percent",
    }

    missing_final = (
        required_final_columns
        - set(
            final_results.columns
        )
    )

    if missing_final:
        raise ValueError(
            "Final holdout results are "
            "missing columns: "
            f"{sorted(missing_final)}"
        )

    required_accuracy_columns = {
        "horizon",
        "rows",
        "mae",
        "rmse",
        "r2",
        "median_absolute_error",
        "within_10_aqi_pct",
        "within_20_aqi_pct",
        "within_30_aqi_pct",
        "category_accuracy_pct",
    }

    missing_accuracy = (
        required_accuracy_columns
        - set(
            accuracy.columns
        )
    )

    if missing_accuracy:
        raise ValueError(
            "Accuracy summary is missing "
            f"columns: {sorted(missing_accuracy)}"
        )

    accuracy = (
        accuracy.copy()
    )

    accuracy[
        "horizon_hours"
    ] = (
        accuracy[
            "horizon"
        ]
        .astype(
            str
        )
        .str.lower()
        .str.replace(
            "h",
            "",
            regex=False,
        )
        .astype(
            int
        )
    )

    merged = accuracy.merge(
        final_results[
            [
                "horizon_hours",
                "model",
                "test_rows",
                "persistence_mae",
                "mae_improvement_percent",
            ]
        ],
        on="horizon_hours",
        how="inner",
        validate="one_to_one",
    )

    found_horizons = set(
        merged[
            "horizon_hours"
        ]
        .astype(
            int
        )
        .tolist()
    )

    if (
        found_horizons
        != EXPECTED_HORIZONS
    ):
        raise ValueError(
            "Performance artifacts must "
            "contain exactly 24h, 48h, "
            "and 72h. "
            f"Found: {sorted(found_horizons)}"
        )

    expected_report_fields = {
        "evaluation_type",
        "selection_was_frozen_before_test",
        "selection_metric",
        "training_rows",
        "test_rows",
        "cities",
        "feature_count",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
    }

    missing_report = (
        expected_report_fields
        - set(
            report
        )
    )

    if missing_report:
        raise ValueError(
            "Final holdout report is missing "
            f"fields: {sorted(missing_report)}"
        )

    if (
        report[
            "selection_was_frozen_before_test"
        ]
        is not True
    ):
        raise ValueError(
            "Final model selection was not "
            "confirmed frozen before testing."
        )

    horizons: list[
        dict[str, Any]
    ] = []

    for row in (
        merged
        .sort_values(
            "horizon_hours"
        )
        .itertuples(
            index=False
        )
    ):
        horizons.append(
            {
                "horizon_hours": int(
                    row.horizon_hours
                ),
                "model": str(
                    row.model
                ),
                "rows": int(
                    row.rows
                ),
                "mae": float(
                    row.mae
                ),
                "rmse": float(
                    row.rmse
                ),
                "r2": float(
                    row.r2
                ),
                "median_absolute_error": float(
                    row.median_absolute_error
                ),
                "within_10_aqi_pct": float(
                    row.within_10_aqi_pct
                ),
                "within_20_aqi_pct": float(
                    row.within_20_aqi_pct
                ),
                "within_30_aqi_pct": float(
                    row.within_30_aqi_pct
                ),
                "category_accuracy_pct": float(
                    row.category_accuracy_pct
                ),
                "persistence_mae": float(
                    row.persistence_mae
                ),
                "mae_improvement_percent": float(
                    row.mae_improvement_percent
                ),
            }
        )

    manifest = {
        "schema_version": 1,
        "generated_at_utc": (
            exported_at()
        ),
        "source": {
            "accuracy_artifact": str(
                ACCURACY_PATH
            ),
            "final_results_artifact": str(
                FINAL_RESULTS_PATH
            ),
            "final_report_artifact": str(
                FINAL_REPORT_PATH
            ),
        },
        "evaluation": {
            "evaluation_type": str(
                report[
                    "evaluation_type"
                ]
            ),
            "selection_metric": str(
                report[
                    "selection_metric"
                ]
            ),
            "selection_frozen_before_test": bool(
                report[
                    "selection_was_frozen_before_test"
                ]
            ),
            "training_rows": int(
                report[
                    "training_rows"
                ]
            ),
            "test_rows": int(
                report[
                    "test_rows"
                ]
            ),
            "cities": int(
                report[
                    "cities"
                ]
            ),
            "feature_count": int(
                report[
                    "feature_count"
                ]
            ),
            "train_start": str(
                report[
                    "train_start"
                ]
            ),
            "train_end": str(
                report[
                    "train_end"
                ]
            ),
            "test_start": str(
                report[
                    "test_start"
                ]
            ),
            "test_end": str(
                report[
                    "test_end"
                ]
            ),
        },
        "horizons": horizons,
    }

    write_json(
        PERFORMANCE_MANIFEST_PATH,
        manifest,
    )

    print(
        f"Saved: {PERFORMANCE_MANIFEST_PATH}"
    )


def export_model_registry_manifest() -> None:
    print(
        "Exporting MLflow Model Registry..."
    )

    require_file(
        Path(
            "mlflow.db"
        )
    )

    report = load_json(
        FINAL_REPORT_PATH
    )

    mlflow.set_tracking_uri(
        TRACKING_URI
    )

    client = (
        MlflowClient()
    )

    production_models: list[
        dict[str, Any]
    ] = []

    for (
        horizon_hours,
        alias,
    ) in ALIASES.items():
        version = (
            client
            .get_model_version_by_alias(
                name=MODEL_NAME,
                alias=alias,
            )
        )

        if version is None:
            raise RuntimeError(
                "No MLflow version found for "
                f"{MODEL_NAME}@{alias}."
            )

        details = (
            client
            .get_model_version(
                name=MODEL_NAME,
                version=version.version,
            )
        )

        run_id = str(
            details.run_id
        )

        if not run_id:
            raise RuntimeError(
                f"{alias} does not contain "
                "a source MLflow run."
            )

        run = client.get_run(
            run_id
        )

        tags = dict(
            details.tags
            or {}
        )

        registered_horizon = (
            tags.get(
                "forecast_horizon"
            )
        )

        expected_horizon = (
            f"{horizon_hours}h"
        )

        if (
            registered_horizon
            != expected_horizon
        ):
            raise ValueError(
                f"{alias} points to horizon "
                f"{registered_horizon!r}; "
                f"expected {expected_horizon!r}."
            )

        if (
            tags.get(
                "deployment_status"
            )
            != "champion"
        ):
            raise ValueError(
                f"{alias} is not tagged as "
                "deployment_status=champion."
            )

        if (
            tags.get(
                "selection_frozen_before_test"
            )
            != "true"
        ):
            raise ValueError(
                f"{alias} does not confirm "
                "selection_frozen_before_test."
            )

        artifact_location = (
            Path(
                "mlruns"
            )
            / str(
                run.info.experiment_id
            )
            / run_id
            / "artifacts"
        )

        production_models.append(
            {
                "horizon_hours": (
                    horizon_hours
                ),
                "algorithm": (
                    "xgboost"
                ),
                "registry_name": (
                    MODEL_NAME
                ),
                "registry_alias": (
                    alias
                ),
                "registry_version": int(
                    details.version
                ),
                "run_id": (
                    run_id
                ),
                "source": str(
                    details.source
                ),
                "status": str(
                    details.status
                ),
                "created_at_ms": int(
                    details.creation_timestamp
                ),
                "tags": (
                    tags
                ),
                "run": {
                    "run_name": (
                        run.data.tags.get(
                            "mlflow.runName"
                        )
                    ),
                    "experiment_id": str(
                        run.info.experiment_id
                    ),
                    "artifact_location": (
                        artifact_location
                        .as_posix()
                    ),
                    "lifecycle_stage": str(
                        run.info.lifecycle_stage
                    ),
                    "metrics": {
                        key: float(
                            value
                        )
                        for (
                            key,
                            value,
                        )
                        in run.data.metrics.items()
                    },
                    "params": {
                        key: str(
                            value
                        )
                        for (
                            key,
                            value,
                        )
                        in run.data.params.items()
                    },
                },
            }
        )

    selected = report.get(
        "selected_models",
        {},
    )

    if not isinstance(
        selected,
        dict,
    ):
        raise TypeError(
            "Final holdout report field "
            "'selected_models' must be an object."
        )

    for horizon in (
        "24h",
        "48h",
        "72h",
    ):
        if (
            selected.get(
                horizon
            )
            != "xgboost"
        ):
            raise ValueError(
                "Final report does not select "
                f"XGBoost for {horizon}."
            )

    manifest = {
        "schema_version": 1,
        "generated_at_utc": (
            exported_at()
        ),
        "tracking_backend": (
            "MLflow"
        ),
        "tracking_uri": (
            TRACKING_URI
        ),
        "registered_model": (
            MODEL_NAME
        ),
        "production_models": (
            sorted(
                production_models,
                key=lambda item: (
                    item[
                        "horizon_hours"
                    ]
                ),
            )
        ),
        "evaluated_candidates": [
            "Persistence",
            "Ridge",
            "Random Forest",
            "XGBoost",
            "CNN",
            "GRU",
            "CNN-LSTM",
        ],
    }

    write_json(
        MODEL_MANIFEST_PATH,
        manifest,
    )

    print(
        f"Saved: {MODEL_MANIFEST_PATH}"
    )


def export_feature_store_manifest() -> None:
    print(
        "Exporting Feast feature-store metadata..."
    )

    require_file(
        FEAST_CONFIG_PATH
    )

    require_file(
        FEAST_DEFINITION_PATH
    )

    config = yaml.safe_load(
        FEAST_CONFIG_PATH.read_text(
            encoding="utf-8",
        )
    )

    if not isinstance(
        config,
        dict,
    ):
        raise TypeError(
            "Feast configuration is invalid."
        )

    project = str(
        config.get(
            "project",
            "",
        )
    )

    if (
        project
        != "pearls_aqi_predictor"
    ):
        raise ValueError(
            "Unexpected Feast project name: "
            f"{project!r}"
        )

    feature_report: dict[
        str,
        Any,
    ] = {}

    if (
        FEAST_REPORT_PATH.exists()
    ):
        feature_report = load_json(
            FEAST_REPORT_PATH
        )

    definitions_text = (
        FEAST_DEFINITION_PATH
        .read_text(
            encoding="utf-8",
        )
    )

    required_definition_markers = [
        'name="aqi_features"',
        'name="aqi_prediction_service"',
        '"feature_count": "56"',
    ]

    missing_markers = [
        marker
        for marker
        in required_definition_markers
        if (
            marker
            not in definitions_text
        )
    ]

    if missing_markers:
        raise ValueError(
            "Feast feature definitions do not "
            "contain expected production "
            f"definitions: {missing_markers}"
        )

    online_store = (
        config.get(
            "online_store",
            {},
        )
    )

    if not isinstance(
        online_store,
        dict,
    ):
        raise TypeError(
            "Feast online_store "
            "configuration must be an object."
        )

    manifest = {
        "schema_version": 1,
        "generated_at_utc": (
            exported_at()
        ),
        "feature_store": (
            "Feast"
        ),
        "project": (
            project
        ),
        "provider": str(
            config.get(
                "provider",
                "",
            )
        ),
        "registry": str(
            config.get(
                "registry",
                "",
            )
        ),
        "online_store": (
            online_store
        ),
        "entity_key_serialization_version": (
            config.get(
                "entity_key_serialization_version"
            )
        ),
        "feature_view": {
            "name": (
                "aqi_features"
            ),
            "feature_count": (
                56
            ),
            "online": (
                True
            ),
        },
        "feature_service": {
            "name": (
                "aqi_prediction_service"
            ),
        },
        "validation": {
            "definition_markers_verified": (
                True
            ),
            "expanded_dataset_report": (
                feature_report
            ),
        },
        "source": {
            "feature_store_config": str(
                FEAST_CONFIG_PATH
            ),
            "feature_definitions": str(
                FEAST_DEFINITION_PATH
            ),
            "dataset_report": str(
                FEAST_REPORT_PATH
            ),
        },
    }

    write_json(
        FEATURE_STORE_MANIFEST_PATH,
        manifest,
    )

    print(
        f"Saved: {FEATURE_STORE_MANIFEST_PATH}"
    )


def main() -> None:
    print(
        "=" * 80
    )

    print(
        "PEARLS AQI - EXPORT DEPLOYMENT METADATA"
    )

    print(
        "=" * 80
    )

    DEPLOYMENT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    export_performance_manifest()

    export_model_registry_manifest()

    export_feature_store_manifest()

    print(
        "\n"
        + "=" * 80
    )

    print(
        "DEPLOYMENT METADATA EXPORT COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"\n{PERFORMANCE_MANIFEST_PATH}"
    )

    print(
        MODEL_MANIFEST_PATH
    )

    print(
        FEATURE_STORE_MANIFEST_PATH
    )


if __name__ == "__main__":
    main()