from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from feast import FeatureStore

from src.features.engineer import AQIFeatureEngineer


class AQIFeatureService:
    FEATURE_REPO_PATH = Path(
        "feature_repo/feature_repo"
    )

    FEATURE_VIEW_NAME = "aqi_features"

    def __init__(self) -> None:
        self.store = FeatureStore(
            repo_path=str(
                self.FEATURE_REPO_PATH
            )
        )

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

        self.feature_refs = [
            f"{self.FEATURE_VIEW_NAME}:{column}"
            for column in self.feature_columns
        ]

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

        if feature_row.empty:
            raise ValueError(
                "Feature dataframe is empty."
            )

        if len(feature_row) != 1:
            raise ValueError(
                "Online Feast write requires "
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

        if (
            prepared
            .isnull()
            .any()
            .any()
        ):
            null_columns = (
                prepared.columns[
                    prepared
                    .isnull()
                    .any()
                ]
                .tolist()
            )

            raise ValueError(
                "Cannot write null features "
                f"to Feast: {null_columns}"
            )

        if not np.isfinite(
            prepared.to_numpy()
        ).all():
            raise ValueError(
                "Cannot write non-finite "
                "features to Feast."
            )

        timestamp = pd.Timestamp(
            event_timestamp
        )

        if timestamp.tzinfo is None:
            timestamp = (
                timestamp.tz_localize(
                    "UTC"
                )
            )
        else:
            timestamp = (
                timestamp.tz_convert(
                    "UTC"
                )
            )

        feast_row = prepared.copy()

        feast_row.insert(
            0,
            "city_id",
            str(city),
        )

        feast_row.insert(
            1,
            "event_timestamp",
            timestamp,
        )

        feast_row.insert(
            2,
            "created",
            timestamp,
        )

        self.store.write_to_online_store(
            feature_view_name=(
                self.FEATURE_VIEW_NAME
            ),
            df=feast_row,
        )

    def get_online_features(
        self,
        city: str,
    ) -> pd.DataFrame:
        if not city:
            raise ValueError(
                "City cannot be empty."
            )

        response = (
            self.store.get_online_features(
                features=self.feature_refs,
                entity_rows=[
                    {
                        "city_id": city,
                    }
                ],
            )
        )

        result = response.to_df()

        if result.empty:
            raise ValueError(
                f"No online features found "
                f"for city: {city}"
            )

        missing_columns = [
            column
            for column in self.feature_columns
            if column not in result.columns
        ]

        if missing_columns:
            raise ValueError(
                "Feast response is missing "
                f"model features: "
                f"{missing_columns}"
            )

        model_features = (
            result[
                self.feature_columns
            ]
            .copy()
            .astype(float)
        )

        if (
            model_features
            .isnull()
            .any()
            .any()
        ):
            null_columns = (
                model_features
                .columns[
                    model_features
                    .isnull()
                    .any()
                ]
                .tolist()
            )

            raise ValueError(
                "Feast returned null values "
                f"for features: "
                f"{null_columns}"
            )

        if not np.isfinite(
            model_features.to_numpy()
        ).all():
            raise ValueError(
                "Feast returned non-finite "
                "feature values."
            )

        return model_features