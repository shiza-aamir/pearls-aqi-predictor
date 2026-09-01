from __future__ import annotations

import logging

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from api.dependencies import (
    get_model_info_service,
)
from api.schemas.models import (
    ModelEvaluationResponse,
    ModelsResponse,
    ProductionModelResponse,
)
from src.services.model_info_service import (
    AQIModelInfoService,
)

logger = logging.getLogger(
    __name__
)


router = APIRouter(
    prefix="/models",
    tags=[
        "Models",
    ],
)


@router.get(
    "",
    response_model=ModelsResponse,
    summary="Get production model information",
    description=(
        "Returns production model aliases, "
        "evaluated model families, and the "
        "frozen final-holdout evaluation setup."
    ),
)
def get_models(
    service: AQIModelInfoService = Depends(
        get_model_info_service
    ),
) -> ModelsResponse:
    try:
        result = (
            service.get_model_info()
        )

        models = [
            ProductionModelResponse(
                horizon_hours=(
                    item.horizon_hours
                ),
                algorithm=(
                    item.algorithm
                ),
                registry_name=(
                    item.registry_name
                ),
                registry_alias=(
                    item.registry_alias
                ),
            )
            for item
            in result.production_models
        ]

        evaluation = (
            ModelEvaluationResponse(
                evaluation_type=(
                    result
                    .evaluation
                    .evaluation_type
                ),
                selection_metric=(
                    result
                    .evaluation
                    .selection_metric
                ),
                selection_frozen_before_test=(
                    result
                    .evaluation
                    .selection_frozen_before_test
                ),
                training_rows=(
                    result
                    .evaluation
                    .training_rows
                ),
                test_rows=(
                    result
                    .evaluation
                    .test_rows
                ),
                cities=(
                    result
                    .evaluation
                    .cities
                ),
                feature_count=(
                    result
                    .evaluation
                    .feature_count
                ),
                train_start=(
                    result
                    .evaluation
                    .train_start
                ),
                train_end=(
                    result
                    .evaluation
                    .train_end
                ),
                test_start=(
                    result
                    .evaluation
                    .test_start
                ),
                test_end=(
                    result
                    .evaluation
                    .test_end
                ),
            )
        )

        return ModelsResponse(
            production_models=models,
            evaluated_candidates=list(
                result.evaluated_candidates
            ),
            evaluation=evaluation,
        )

    except Exception as exc:
        logger.exception(
            "Model information request failed."
        )

        raise HTTPException(
            status_code=(
                status
                .HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail={
                "code": (
                    "MODEL_INFO_UNAVAILABLE"
                ),
                "message": (
                    "Model information is "
                    "temporarily unavailable."
                ),
            },
        ) from exc