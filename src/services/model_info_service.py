from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


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
    REGISTRY_MANIFEST_PATH = Path(
        "artifacts/deployment/"
        "model_registry_manifest.json"
    )

    PERFORMANCE_MANIFEST_PATH = Path(
        "artifacts/deployment/"
        "performance_manifest.json"
    )

    HORIZONS = (
        24,
        48,
        72,
    )

    @staticmethod
    def _load_json(
        path: Path,
    ) -> dict:
        if not path.exists():
            raise FileNotFoundError(
                "Required deployment "
                f"manifest does not exist: {path}"
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
            raise ValueError(
                f"Deployment manifest "
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
        registry = self._load_json(
            self.REGISTRY_MANIFEST_PATH
        )

        performance = self._load_json(
            self.PERFORMANCE_MANIFEST_PATH
        )

        registered_model = str(
            registry.get(
                "registered_model",
                "",
            )
        )

        if not registered_model:
            raise ValueError(
                "MLflow registry manifest "
                "does not contain a registered model."
            )

        model_rows = registry.get(
            "production_models"
        )

        if not isinstance(
            model_rows,
            list,
        ):
            raise ValueError(
                "MLflow registry manifest "
                "does not contain production models."
            )

        production_models = []

        found_horizons = []

        for row in model_rows:
            horizon = int(
                row[
                    "horizon_hours"
                ]
            )

            found_horizons.append(
                horizon
            )

            registry_name = str(
                row[
                    "registry_name"
                ]
            )

            if (
                registry_name
                != registered_model
            ):
                raise ValueError(
                    "Inconsistent registered "
                    "model name in MLflow manifest."
                )

            production_models.append(
                ProductionModel(
                    horizon_hours=horizon,
                    algorithm=str(
                        row[
                            "algorithm"
                        ]
                    ),
                    registry_name=(
                        registry_name
                    ),
                    registry_alias=str(
                        row[
                            "registry_alias"
                        ]
                    ),
                )
            )

        production_models.sort(
            key=lambda item: (
                item.horizon_hours
            )
        )

        if tuple(
            sorted(
                found_horizons
            )
        ) != self.HORIZONS:
            raise ValueError(
                "MLflow registry manifest "
                "must contain exactly the "
                "24h, 48h, and 72h champions."
            )

        candidates = registry.get(
            "evaluated_candidates"
        )

        if not isinstance(
            candidates,
            list,
        ):
            raise ValueError(
                "MLflow registry manifest "
                "does not contain evaluated "
                "model candidates."
            )

        evaluation_data = (
            performance.get(
                "evaluation"
            )
        )

        if not isinstance(
            evaluation_data,
            dict,
        ):
            raise ValueError(
                "Performance manifest "
                "does not contain evaluation "
                "metadata."
            )

        evaluation = ModelEvaluation(
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

        return ModelInfoResult(
            production_models=tuple(
                production_models
            ),
            evaluated_candidates=tuple(
                str(item)
                for item in candidates
            ),
            evaluation=evaluation,
        )