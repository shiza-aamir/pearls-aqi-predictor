from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.engineer import AQIFeatureEngineer


class AQIFeatureService:
    def __init__(self) -> None:
        self.feature_columns = (
            AQIFeatureEngineer
            .get_model_feature_columns()
        )

        if len(self.feature_columns) != 56:
            raise RuntimeError(
                "Feature service expects exactly "
                f"56 features, got "
                f"{len(self.feature_columns)}."
            )

        self._online_features: dict[
            str,
            pd.DataFrame,
        ] = {}

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
                "Feature service requires "
                "exactly one feature row."
            )

        missing_columns = [
            column
            for column in self.feature_columns
            if column not in feature_row.columns
        ]

        if missing_columns:
            raise ValueError(
                "Live feature row is missing "
                f"features: {missing_columns}"
            )

        prepared = (
            feature_row[
                self.feature_columns
            ]
            .copy()
            .astype("float64")
        )

        null_columns = (
            prepared.columns[
                prepared.isnull().any()
            ]
            .tolist()
        )

        if null_columns:
            raise ValueError(
                "Cannot store null features: "
                f"{null_columns}"
            )

        if not np.isfinite(
            prepared.to_numpy()
        ).all():
            raise ValueError(
                "Cannot store non-finite "
                "feature values."
            )

        return prepared

    def write_online_features(
        self,
        city: str,
        event_timestamp,
        feature_row: pd.DataFrame,
    ) -> None:
        if not city:
            raise ValueError(
                "City cannot be empty."
            )

        prepared = self._prepare_features(
            feature_row
        )

        self._online_features[
            city
        ] = prepared

    def get_online_features(
        self,
        city: str,
    ) -> pd.DataFrame:
        if not city:
            raise ValueError(
                "City cannot be empty."
            )

        if city not in self._online_features:
            raise ValueError(
                f"No online features found "
                f"for city: {city}"
            )

        return (
            self._online_features[
                city
            ]
            .copy()
        )