from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from src.features.aqi.calculator import AQICalculator
from src.features.engineer import AQIFeatureEngineer


@dataclass(frozen=True)
class AQIPrediction:
    horizon: str
    predicted_aqi: float
    predicted_category: str
    model_name: str
    model_alias: str


class AQIPredictionService:
    MODEL_NAME = "pearls-aqi-xgboost"

    HORIZON_ALIASES = {
        "24h": "champion-24h",
        "48h": "champion-48h",
        "72h": "champion-72h",
    }

    BUNDLED_MODEL_PATHS = {
        "24h": Path("models/production/xgboost_24h.ubj"),
        "48h": Path("models/production/xgboost_48h.ubj"),
        "72h": Path("models/production/xgboost_72h.ubj"),
    }

    def __init__(self) -> None:
        self.model_source = os.getenv(
            "MODEL_SOURCE",
            "bundled",
        ).strip().lower()

        if self.model_source not in {
            "bundled",
            "mlflow",
        }:
            raise ValueError(
                "MODEL_SOURCE must be either "
                "'bundled' or 'mlflow'."
            )

        self.feature_columns = (
            AQIFeatureEngineer.get_model_feature_columns()
        )

        if len(self.feature_columns) != 56:
            raise RuntimeError(
                "Production model expects exactly "
                f"56 features, but "
                f"{len(self.feature_columns)} were found."
            )

        self._models: dict[str, Any] = {}

    def _get_alias(
        self,
        horizon: str,
    ) -> str:
        if horizon not in self.HORIZON_ALIASES:
            raise ValueError(
                f"Unsupported forecast horizon: {horizon}. "
                f"Supported values: "
                f"{list(self.HORIZON_ALIASES)}"
            )

        return self.HORIZON_ALIASES[horizon]

    def _load_mlflow_model(
        self,
        horizon: str,
    ) -> Any:
        try:
            import mlflow
            import mlflow.xgboost
        except ImportError as exc:
            raise RuntimeError(
                "MLflow model loading was requested, "
                "but MLflow is not installed. "
                "Use MODEL_SOURCE=bundled for deployment."
            ) from exc

        tracking_uri = os.getenv(
            "MLFLOW_TRACKING_URI",
            "sqlite:///mlflow.db",
        ).strip()

        mlflow.set_tracking_uri(tracking_uri)

        alias = self._get_alias(horizon)

        model_uri = (
            f"models:/"
            f"{self.MODEL_NAME}"
            f"@{alias}"
        )

        return mlflow.xgboost.load_model(
            model_uri
        )

    def _load_bundled_model(
        self,
        horizon: str,
    ) -> xgb.XGBRegressor:
        if horizon not in self.BUNDLED_MODEL_PATHS:
            raise ValueError(
                f"Unsupported forecast horizon: {horizon}."
            )

        model_path = self.BUNDLED_MODEL_PATHS[
            horizon
        ]

        if not model_path.exists():
            raise FileNotFoundError(
                "Bundled production model was "
                f"not found: {model_path}"
            )

        model = xgb.XGBRegressor()

        model.load_model(model_path)

        return model

    def _get_model(
        self,
        horizon: str,
    ) -> Any:
        if horizon in self._models:
            return self._models[horizon]

        if self.model_source == "mlflow":
            model = self._load_mlflow_model(
                horizon
            )
        else:
            model = self._load_bundled_model(
                horizon
            )

        self._models[horizon] = model

        return model

    def _prepare_features(
        self,
        feature_row: pd.DataFrame,
    ) -> pd.DataFrame:
        if feature_row.empty:
            raise ValueError(
                "Feature dataframe is empty."
            )

        if len(feature_row) != 1:
            raise ValueError(
                "Prediction service expects "
                "exactly one feature row."
            )

        missing_columns = [
            column
            for column in self.feature_columns
            if column not in feature_row.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing model features: "
                f"{missing_columns}"
            )

        prepared = (
            feature_row[
                self.feature_columns
            ]
            .copy()
            .astype(float)
        )

        null_columns = (
            prepared.columns[
                prepared.isnull().any()
            ]
            .tolist()
        )

        if null_columns:
            raise ValueError(
                "Missing values detected in "
                f"features: {null_columns}"
            )

        if not np.isfinite(
            prepared.to_numpy()
        ).all():
            raise ValueError(
                "Non-finite values detected "
                "in model features."
            )

        return prepared

    def get_prepared_features(
        self,
        feature_row: pd.DataFrame,
    ) -> pd.DataFrame:
        return self._prepare_features(
            feature_row
        )

    def get_model(
        self,
        horizon: str,
    ) -> Any:
        self._get_alias(horizon)

        return self._get_model(
            horizon
        )

    def predict(
        self,
        feature_row: pd.DataFrame,
        horizon: str,
    ) -> AQIPrediction:
        prepared = self._prepare_features(
            feature_row
        )

        model = self._get_model(
            horizon
        )

        predictions = model.predict(
            prepared
        )

        if len(predictions) != 1:
            raise RuntimeError(
                f"{horizon}: model returned "
                f"{len(predictions)} predictions "
                "for one input row."
            )

        predicted_aqi = float(
            predictions[0]
        )

        if not np.isfinite(
            predicted_aqi
        ):
            raise RuntimeError(
                f"{horizon}: model produced "
                "a non-finite AQI prediction."
            )

        predicted_aqi = float(
            np.clip(
                predicted_aqi,
                0.0,
                500.0,
            )
        )

        predicted_category = (
            AQICalculator.category_from_aqi(
                round(predicted_aqi)
            )
        )

        alias = self._get_alias(
            horizon
        )

        return AQIPrediction(
            horizon=horizon,
            predicted_aqi=predicted_aqi,
            predicted_category=predicted_category,
            model_name=self.MODEL_NAME,
            model_alias=alias,
        )

    def predict_all(
        self,
        feature_row: pd.DataFrame,
    ) -> list[AQIPrediction]:
        return [
            self.predict(
                feature_row=feature_row,
                horizon=horizon,
            )
            for horizon in (
                "24h",
                "48h",
                "72h",
            )
        ]