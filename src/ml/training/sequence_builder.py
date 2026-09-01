from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DEFAULT_SEQUENCE_LENGTH = 72


SEQUENCE_FEATURES = [
    "latitude",
    "longitude",
    "temperature",
    "humidity",
    "precipitation",
    "wind_speed",
    "pressure",
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "wind_direction_sin",
    "wind_direction_cos",
    "temp_humidity_interaction",
    "stagnation_index",
]


TARGET_COLUMNS = {
    "24h": "target_aqi_24h",
    "48h": "target_aqi_48h",
    "72h": "target_aqi_72h",
}


@dataclass(frozen=True)
class SequenceDataset:
    X: np.ndarray
    y: np.ndarray
    cities: np.ndarray
    timestamps: np.ndarray


class AQISequenceBuilder:
    def __init__(
        self,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError(
                "Sequence length must be greater than zero."
            )

        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.is_fitted = False

    @staticmethod
    def get_sequence_features() -> list[str]:
        return SEQUENCE_FEATURES.copy()

    @staticmethod
    def get_target_column(horizon: str) -> str:
        if horizon not in TARGET_COLUMNS:
            raise ValueError(
                f"Unsupported horizon '{horizon}'. "
                f"Expected one of: {list(TARGET_COLUMNS)}"
            )

        return TARGET_COLUMNS[horizon]

    def validate_dataframe(
        self,
        dataframe: pd.DataFrame,
        horizon: str,
    ) -> None:
        target_column = self.get_target_column(horizon)

        required_columns = [
            "city",
            "timestamp",
            *SEQUENCE_FEATURES,
            target_column,
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing required sequence columns: "
                f"{missing_columns}"
            )

    def fit_scaler(
        self,
        train_dataframe: pd.DataFrame,
    ) -> None:
        missing_columns = [
            column
            for column in SEQUENCE_FEATURES
            if column not in train_dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                "Training dataframe is missing features: "
                f"{missing_columns}"
            )

        values = (
            train_dataframe[SEQUENCE_FEATURES]
            .astype(float)
            .to_numpy()
        )

        if not np.isfinite(values).all():
            raise ValueError(
                "Training features contain NaN or infinite values."
            )

        self.scaler.fit(values)
        self.is_fitted = True

    def build(
        self,
        dataframe: pd.DataFrame,
        horizon: str,
    ) -> SequenceDataset:
        if not self.is_fitted:
            raise RuntimeError(
                "Scaler has not been fitted. "
                "Call fit_scaler() using training data first."
            )

        self.validate_dataframe(
            dataframe=dataframe,
            horizon=horizon,
        )

        target_column = self.get_target_column(horizon)

        working_dataframe = dataframe.copy()

        working_dataframe["timestamp"] = pd.to_datetime(
            working_dataframe["timestamp"]
        )

        X_sequences: list[np.ndarray] = []
        y_values: list[float] = []
        cities: list[str] = []
        timestamps: list[np.datetime64] = []

        for city, city_dataframe in working_dataframe.groupby(
            "city",
            sort=False,
        ):
            city_dataframe = (
                city_dataframe
                .sort_values("timestamp")
                .reset_index(drop=True)
            )

            raw_features = (
                city_dataframe[SEQUENCE_FEATURES]
                .astype(float)
                .to_numpy()
            )

            scaled_features = self.scaler.transform(
                raw_features
            )

            targets = (
                city_dataframe[target_column]
                .astype(float)
                .to_numpy()
            )

            city_timestamps = (
                city_dataframe["timestamp"]
                .to_numpy()
            )

            for end_index in range(
                self.sequence_length - 1,
                len(city_dataframe),
            ):
                start_index = (
                    end_index
                    - self.sequence_length
                    + 1
                )

                target_value = targets[end_index]

                if not np.isfinite(target_value):
                    continue

                sequence = scaled_features[
                    start_index:end_index + 1
                ]

                if not np.isfinite(sequence).all():
                    continue

                sequence_timestamps = pd.to_datetime(
                    city_timestamps[
                        start_index:end_index + 1
                    ]
                )

                timestamp_differences = (
                    pd.Series(sequence_timestamps)
                    .diff()
                    .dropna()
                )

                expected_difference = pd.Timedelta(
                    hours=1
                )

                if not (
                    timestamp_differences
                    == expected_difference
                ).all():
                    continue

                X_sequences.append(
                    sequence.astype(np.float32)
                )

                y_values.append(
                    float(target_value)
                )

                cities.append(
                    str(city)
                )

                timestamps.append(
                    city_timestamps[end_index]
                )

        if not X_sequences:
            raise ValueError(
                "No valid 72-hour sequences could be generated."
            )

        return SequenceDataset(
            X=np.asarray(
                X_sequences,
                dtype=np.float32,
            ),
            y=np.asarray(
                y_values,
                dtype=np.float32,
            ),
            cities=np.asarray(cities),
            timestamps=np.asarray(timestamps),
        )