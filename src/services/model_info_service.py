from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProductionModel:
    horizon_hours: int
    algorithm: str
    registry_name: str
    registry_alias: str


@dataclass(frozen=True)
class ModelEvaluation:
    evaluation_type: str
    selection_metric: str
    selection_frozen_before_test: bool
    training_rows: int
    test_rows: int
    cities: int
    feature_count: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str


@dataclass(frozen=True)
class ModelInfoResult:
    production_models: tuple[
        ProductionModel,
        ...,
    ]
    evaluated_candidates: tuple[
        str,
        ...,
    ]
    evaluation: ModelEvaluation


class AQIModelInfoService:
    MODEL_MANIFEST_PATH = Path(
        "artifacts/deployment/"
        "model_registry_manifest.json"
    )

    PERFORMANCE_MANIFEST_PATH = Path(
        "artifacts/deployment/"
        "performance_manifest.json"
    )

    EXPECTED_HORIZONS = {
        24,
        48,
        72,
    }

    @staticmethod
    def _load_manifest(
        path: Path,
    ) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(
                "Deployment manifest does "
                f"not exist: {path}"
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
                "Deployment manifest "
                f"must be an object: {path}"
            )

        if (
            data.get(
                "schema_version"
            )
            != 1
        ):
            raise ValueError(
                "Unsupported deployment "
                f"manifest schema: {path}"
            )

        return data

    def get_model_info(
        self,
    ) -> ModelInfoResult:
        model_manifest = (
            self._load_manifest(
                self.MODEL_MANIFEST_PATH
            )
        )

        performance_manifest = (
            self._load_manifest(
                self.PERFORMANCE_MANIFEST_PATH
            )
        )

        raw_models = (
            model_manifest.get(
                "production_models"
            )
        )

        if not isinstance(
            raw_models,
            list,
        ):
            raise TypeError(
                "MLflow registry manifest "
                "does not contain production models."
            )

        production_models: list[
            ProductionModel
        ] = []

        found_horizons: set[
            int
        ] = set()

        for item in raw_models:
            if not isinstance(
                item,
                dict,
            ):
                raise TypeError(
                    "Production model entry "
                    "must be an object."
                )

            horizon_hours = int(
                item[
                    "horizon_hours"
                ]
            )

            algorithm = str(
                item[
                    "algorithm"
                ]
            )

            registry_name = str(
                item[
                    "registry_name"
                ]
            )

            registry_alias = str(
                item[
                    "registry_alias"
                ]
            )

            if (
                horizon_hours
                in found_horizons
            ):
                raise ValueError(
                    "Duplicate production "
                    f"horizon: {horizon_hours}"
                )

            found_horizons.add(
                horizon_hours
            )

            production_models.append(
                ProductionModel(
                    horizon_hours=(
                        horizon_hours
                    ),
                    algorithm=(
                        algorithm
                    ),
                    registry_name=(
                        registry_name
                    ),
                    registry_alias=(
                        registry_alias
                    ),
                )
            )

        if (
            found_horizons
            != self.EXPECTED_HORIZONS
        ):
            raise ValueError(
                "Production registry must "
                "contain exactly 24h, 48h, "
                "and 72h models. "
                f"Found: {sorted(found_horizons)}"
            )

        raw_candidates = (
            model_manifest.get(
                "evaluated_candidates"
            )
        )

        if not isinstance(
            raw_candidates,
            list,
        ):
            raise TypeError(
                "MLflow registry manifest "
                "does not contain evaluated "
                "model candidates."
            )

        evaluated_candidates = tuple(
            str(
                candidate
            )
            for candidate
            in raw_candidates
        )

        evaluation_data = (
            performance_manifest.get(
                "evaluation"
            )
        )

        if not isinstance(
            evaluation_data,
            dict,
        ):
            raise TypeError(
                "Performance manifest "
                "does not contain evaluation "
                "metadata."
            )

        required_fields = {
            "evaluation_type",
            "selection_metric",
            "selection_frozen_before_test",
            "training_rows",
            "test_rows",
            "cities",
            "feature_count",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
        }

        missing_fields = (
            required_fields
            - set(
                evaluation_data
            )
        )

        if missing_fields:
            raise ValueError(
                "Performance manifest "
                "evaluation metadata is "
                "missing fields: "
                f"{sorted(missing_fields)}"
            )

        evaluation = (
            ModelEvaluation(
                evaluation_type=str(
                    evaluation_data[
                        "evaluation_type"
                    ]
                ),
                selection_metric=str(
                    evaluation_data[
                        "selection_metric"
                    ]
                ),
                selection_frozen_before_test=bool(
                    evaluation_data[
                        "selection_frozen_before_test"
                    ]
                ),
                training_rows=int(
                    evaluation_data[
                        "training_rows"
                    ]
                ),
                test_rows=int(
                    evaluation_data[
                        "test_rows"
                    ]
                ),
                cities=int(
                    evaluation_data[
                        "cities"
                    ]
                ),
                feature_count=int(
                    evaluation_data[
                        "feature_count"
                    ]
                ),
                train_start=str(
                    evaluation_data[
                        "train_start"
                    ]
                ),
                train_end=str(
                    evaluation_data[
                        "train_end"
                    ]
                ),
                test_start=str(
                    evaluation_data[
                        "test_start"
                    ]
                ),
                test_end=str(
                    evaluation_data[
                        "test_end"
                    ]
                ),
            )
        )

        return ModelInfoResult(
            production_models=tuple(
                sorted(
                    production_models,
                    key=lambda model: (
                        model.horizon_hours
                    ),
                )
            ),
            evaluated_candidates=(
                evaluated_candidates
            ),
            evaluation=(
                evaluation
            ),
        )