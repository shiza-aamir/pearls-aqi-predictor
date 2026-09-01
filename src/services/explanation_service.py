from __future__ import annotations

from dataclasses import dataclass

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import shap

from src.features.engineer import AQIFeatureEngineer


@dataclass(frozen=True)
class FeatureContribution:
    feature: str
    feature_value: float
    shap_value: float
    direction: str


@dataclass(frozen=True)
class AQIExplanation:
    horizon: str
    predicted_aqi: float
    base_value: float
    contributions: list[FeatureContribution]


class AQIExplanationService:
    TRACKING_URI = "sqlite:///mlflow.db"

    MODEL_NAME = "pearls-aqi-xgboost"

    HORIZON_ALIASES = {
        "24h": "champion-24h",
        "48h": "champion-48h",
        "72h": "champion-72h",
    }

    def __init__(self) -> None:
        mlflow.set_tracking_uri(
            self.TRACKING_URI
        )

        self.feature_columns = (
            AQIFeatureEngineer
            .get_model_feature_columns()
        )

        if len(self.feature_columns) != 56:
            raise RuntimeError(
                "Expected exactly 56 model features."
            )

        self._models = {}
        self._explainers = {}

    def _validate_horizon(
        self,
        horizon: str,
    ) -> str:
        if horizon not in self.HORIZON_ALIASES:
            raise ValueError(
                f"Unsupported horizon: {horizon}. "
                f"Expected one of "
                f"{list(self.HORIZON_ALIASES)}."
            )

        return self.HORIZON_ALIASES[horizon]

    def get_model(
        self,
        horizon: str,
    ):
        if horizon in self._models:
            return self._models[horizon]

        alias = self._validate_horizon(
            horizon
        )

        model_uri = (
            f"models:/"
            f"{self.MODEL_NAME}"
            f"@{alias}"
        )

        model = mlflow.xgboost.load_model(
            model_uri
        )

        self._models[horizon] = model

        return model

    def get_explainer(
        self,
        horizon: str,
    ) -> shap.TreeExplainer:
        if horizon in self._explainers:
            return self._explainers[horizon]

        model = self.get_model(
            horizon
        )

        explainer = shap.TreeExplainer(
            model
        )

        self._explainers[horizon] = (
            explainer
        )

        return explainer

    def prepare_features(
        self,
        feature_data: pd.DataFrame,
    ) -> pd.DataFrame:
        if feature_data.empty:
            raise ValueError(
                "Feature dataframe is empty."
            )

        missing = [
            column
            for column in self.feature_columns
            if column not in feature_data.columns
        ]

        if missing:
            raise ValueError(
                f"Missing model features: {missing}"
            )

        prepared = (
            feature_data[
                self.feature_columns
            ]
            .copy()
            .astype(float)
        )

        if (
            prepared
            .isnull()
            .any()
            .any()
        ):
            raise ValueError(
                "Feature dataframe contains nulls."
            )

        if not np.isfinite(
            prepared.to_numpy()
        ).all():
            raise ValueError(
                "Feature dataframe contains "
                "non-finite values."
            )

        return prepared

    def shap_values(
        self,
        feature_data: pd.DataFrame,
        horizon: str,
    ) -> tuple[
        pd.DataFrame,
        np.ndarray,
        float,
    ]:
        prepared = self.prepare_features(
            feature_data
        )

        explainer = self.get_explainer(
            horizon
        )

        values = explainer.shap_values(
            prepared
        )

        values = np.asarray(
            values,
            dtype=float,
        )

        if values.shape != prepared.shape:
            raise RuntimeError(
                "Unexpected SHAP output shape. "
                f"Features={prepared.shape}, "
                f"SHAP={values.shape}."
            )

        expected_value = np.asarray(
            explainer.expected_value
        )

        base_value = float(
            expected_value.reshape(-1)[0]
        )

        return (
            prepared,
            values,
            base_value,
        )

    def explain_single(
        self,
        feature_row: pd.DataFrame,
        horizon: str,
        top_n: int = 10,
    ) -> AQIExplanation:
        if len(feature_row) != 1:
            raise ValueError(
                "Individual explanation requires "
                "exactly one feature row."
            )

        if top_n < 1:
            raise ValueError(
                "top_n must be at least 1."
            )

        (
            prepared,
            values,
            base_value,
        ) = self.shap_values(
            feature_row,
            horizon,
        )

        model = self.get_model(
            horizon
        )

        prediction = float(
            model.predict(
                prepared
            )[0]
        )

        row_values = values[0]

        order = np.argsort(
            np.abs(row_values)
        )[::-1]

        contributions = []

        for index in order[:top_n]:
            shap_value = float(
                row_values[index]
            )

            contributions.append(
                FeatureContribution(
                    feature=(
                        self.feature_columns[
                            index
                        ]
                    ),
                    feature_value=float(
                        prepared.iloc[
                            0,
                            index,
                        ]
                    ),
                    shap_value=shap_value,
                    direction=(
                        "increase"
                        if shap_value > 0
                        else (
                            "decrease"
                            if shap_value < 0
                            else "neutral"
                        )
                    ),
                )
            )

        return AQIExplanation(
            horizon=horizon,
            predicted_aqi=prediction,
            base_value=base_value,
            contributions=contributions,
        )