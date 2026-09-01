from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


@dataclass(frozen=True)
class RegressionMetrics:
    mae: float
    rmse: float
    r2: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def calculate_regression_metrics(
    y_true,
    y_pred,
) -> RegressionMetrics:
    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    return RegressionMetrics(
        mae=float(mae),
        rmse=float(rmse),
        r2=float(r2),
    )