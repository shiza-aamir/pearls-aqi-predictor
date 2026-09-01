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
    REPORT_PATH = Path(
        "artifacts/final_holdout/"
        "final_holdout_report.json"
    )

    REGISTRY_NAME = (
        "pearls-aqi-xgboost"
    )

    CANDIDATES = (
        "Persistence",
        "Ridge",
        "Random Forest",
        "XGBoost",
        "CNN",
        "GRU",
        "CNN-LSTM",
    )

    def _load_report(
        self,
    ) -> dict:
        if not self.REPORT_PATH.exists():
            raise FileNotFoundError(
                "Final holdout report does "
                "not exist: "
                f"{self.REPORT_PATH}"
            )

        with self.REPORT_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            report = json.load(
                file
            )

        required = {
            "evaluation_type",
            "selection_was_frozen_before_test",
            "selected_models",
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

        missing = (
            required
            - set(report)
        )

        if missing:
            raise ValueError(
                "Final holdout report is "
                "missing fields: "
                f"{sorted(missing)}"
            )

        return report

    def get_model_info(
        self,
    ) -> ModelInfoResult:
        report = (
            self._load_report()
        )

        selected = report[
            "selected_models"
        ]

        production_models = []

        for horizon in (
            24,
            48,
            72,
        ):
            key = (
                f"{horizon}h"
            )

            if key not in selected:
                raise ValueError(
                    "Final holdout report "
                    f"is missing {key}."
                )

            algorithm = str(
                selected[key]
            )

            production_models.append(
                ProductionModel(
                    horizon_hours=horizon,
                    algorithm=algorithm,
                    registry_name=(
                        self.REGISTRY_NAME
                    ),
                    registry_alias=(
                        f"champion-{horizon}h"
                    ),
                )
            )

        evaluation = (
            ModelEvaluation(
                evaluation_type=str(
                    report[
                        "evaluation_type"
                    ]
                ),
                selection_metric=str(
                    report[
                        "selection_metric"
                    ]
                ),
                selection_frozen_before_test=bool(
                    report[
                        "selection_was_frozen_before_test"
                    ]
                ),
                training_rows=int(
                    report[
                        "training_rows"
                    ]
                ),
                test_rows=int(
                    report[
                        "test_rows"
                    ]
                ),
                cities=int(
                    report[
                        "cities"
                    ]
                ),
                feature_count=int(
                    report[
                        "feature_count"
                    ]
                ),
                train_start=str(
                    report[
                        "train_start"
                    ]
                ),
                train_end=str(
                    report[
                        "train_end"
                    ]
                ),
                test_start=str(
                    report[
                        "test_start"
                    ]
                ),
                test_end=str(
                    report[
                        "test_end"
                    ]
                ),
            )
        )

        return ModelInfoResult(
            production_models=tuple(
                production_models
            ),
            evaluated_candidates=(
                self.CANDIDATES
            ),
            evaluation=evaluation,
        )