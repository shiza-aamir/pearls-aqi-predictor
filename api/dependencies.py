from __future__ import annotations

from functools import lru_cache

from src.services.history_service import (
    AQIHistoryService,
)
from src.services.model_info_service import (
    AQIModelInfoService,
)
from src.services.performance_service import (
    AQIPerformanceService,
)
from src.services.production_service import (
    AQIProductionService,
)


@lru_cache(maxsize=1)
def get_production_service(
) -> AQIProductionService:
    return AQIProductionService()


@lru_cache(maxsize=1)
def get_history_service(
) -> AQIHistoryService:
    return AQIHistoryService()


@lru_cache(maxsize=1)
def get_performance_service(
) -> AQIPerformanceService:
    return AQIPerformanceService()


@lru_cache(maxsize=1)
def get_model_info_service(
) -> AQIModelInfoService:
    return AQIModelInfoService()