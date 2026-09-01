from __future__ import annotations

from pydantic import BaseModel


class ProductionModelResponse(
    BaseModel
):
    horizon_hours: int
    algorithm: str
    registry_name: str
    registry_alias: str


class ModelEvaluationResponse(
    BaseModel
):
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


class ModelsResponse(
    BaseModel
):
    production_models: list[
        ProductionModelResponse
    ]

    evaluated_candidates: list[
        str
    ]

    evaluation: ModelEvaluationResponse