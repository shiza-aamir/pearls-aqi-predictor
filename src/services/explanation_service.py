from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb

from src.services.prediction_service import (
    AQIPredictionService,
)


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
    def __init__(self) -> None:
        self.prediction_service = (
            AQIPredictionService()
        )

        self.feature_columns = (
            self.prediction_service
            .feature_columns
        )

        if len(self.feature_columns) != 56:
            raise RuntimeError(
                "Expected exactly 56 model features."
            )

    def prepare_features(
        self,
        feature_data: pd.DataFrame,
    ) -> pd.DataFrame:
        return (
            self.prediction_service
            .get_prepared_features(
                feature_data
            )
        )

    def _get_booster(
        self,
        horizon: str,
    ) -> xgb.Booster:
        model = (
            self.prediction_service
            .get_model(
                horizon
            )
        )

        if not hasattr(
            model,
            "get_booster",
        ):
            raise RuntimeError(
                "Production explanation requires "
                "an XGBoost model."
            )

        return model.get_booster()

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

        booster = self._get_booster(
            horizon
        )

        dmatrix = xgb.DMatrix(
            prepared,
            feature_names=self.feature_columns,
        )

        contribution_values = (
            booster.predict(
                dmatrix,
                pred_contribs=True,
            )
        )

        contribution_values = np.asarray(
            contribution_values,
            dtype=float,
        )

        expected_columns = (
            len(self.feature_columns) + 1
        )

        if (
            contribution_values.ndim != 2
            or contribution_values.shape[1]
            != expected_columns
        ):
            raise RuntimeError(
                "Unexpected XGBoost contribution "
                "output shape: "
                f"{contribution_values.shape}"
            )

        values = contribution_values[
            :,
            :-1,
        ]

        base_values = contribution_values[
            :,
            -1,
        ]

        if values.shape != prepared.shape:
            raise RuntimeError(
                "Unexpected explanation shape. "
                f"Features={prepared.shape}, "
                f"Contributions={values.shape}."
            )

        base_value = float(
            base_values[0]
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

        model = (
            self.prediction_service
            .get_model(
                horizon
            )
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
            contribution_value = float(
                row_values[index]
            )

            contributions.append(
                FeatureContribution(
                    feature=self.feature_columns[
                        index
                    ],
                    feature_value=float(
                        prepared.iloc[
                            0,
                            index,
                        ]
                    ),
                    shap_value=(
                        contribution_value
                    ),
                    direction=(
                        "increase"
                        if contribution_value > 0
                        else (
                            "decrease"
                            if contribution_value < 0
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