from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_ridge_model(
    alpha: float = 1.0,
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                Ridge(
                    alpha=alpha,
                ),
            ),
        ]
    )