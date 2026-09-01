from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SplitSummary:
    train_rows: int
    validation_rows: int
    test_rows: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    excluded_cities: tuple[str, ...]
    purge_hours: int


class PurgedTimeSeriesSplitter:
    def __init__(
        self,
        train_ratio: float = 0.70,
        validation_ratio: float = 0.15,
        purge_hours: int = 72,
        excluded_cities: tuple[str, ...] = (),
    ) -> None:
        if not 0 < train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1.")

        if not 0 < validation_ratio < 1:
            raise ValueError(
                "validation_ratio must be between 0 and 1."
            )

        if train_ratio + validation_ratio >= 1:
            raise ValueError(
                "train_ratio + validation_ratio must be less than 1."
            )

        if purge_hours < 0:
            raise ValueError(
                "purge_hours cannot be negative."
            )

        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.purge_hours = purge_hours
        self.excluded_cities = excluded_cities

    def split(
        self,
        df: pd.DataFrame,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        SplitSummary,
    ]:
        self._validate_input(df)

        working = df.copy()

        working["timestamp"] = pd.to_datetime(
            working["timestamp"],
            errors="raise",
        )

        if self.excluded_cities:
            working = working[
                ~working["city"].isin(
                    self.excluded_cities
                )
            ].copy()

        working = working.sort_values(
            ["timestamp", "city"]
        ).reset_index(drop=True)

        timestamps = (
            working["timestamp"]
            .drop_duplicates()
            .sort_values()
            .reset_index(drop=True)
        )

        timestamp_count = len(timestamps)

        train_cut_index = int(
            timestamp_count * self.train_ratio
        )

        validation_cut_index = int(
            timestamp_count
            * (
                self.train_ratio
                + self.validation_ratio
            )
        )

        train_boundary = timestamps.iloc[
            train_cut_index
        ]

        validation_boundary = timestamps.iloc[
            validation_cut_index
        ]

        purge_delta = pd.Timedelta(
            hours=self.purge_hours
        )

        train_end = (
            train_boundary - purge_delta
        )

        validation_start = train_boundary

        validation_end = (
            validation_boundary - purge_delta
        )

        test_start = validation_boundary

        train = working[
            working["timestamp"] < train_end
        ].copy()

        validation = working[
            (working["timestamp"] >= validation_start)
            & (
                working["timestamp"]
                < validation_end
            )
        ].copy()

        test = working[
            working["timestamp"] >= test_start
        ].copy()

        if train.empty:
            raise ValueError(
                "Training split is empty."
            )

        if validation.empty:
            raise ValueError(
                "Validation split is empty."
            )

        if test.empty:
            raise ValueError(
                "Test split is empty."
            )

        summary = SplitSummary(
            train_rows=len(train),
            validation_rows=len(validation),
            test_rows=len(test),
            train_start=train["timestamp"].min(),
            train_end=train["timestamp"].max(),
            validation_start=validation[
                "timestamp"
            ].min(),
            validation_end=validation[
                "timestamp"
            ].max(),
            test_start=test["timestamp"].min(),
            test_end=test["timestamp"].max(),
            excluded_cities=self.excluded_cities,
            purge_hours=self.purge_hours,
        )

        return (
            train.reset_index(drop=True),
            validation.reset_index(drop=True),
            test.reset_index(drop=True),
            summary,
        )

    @staticmethod
    def _validate_input(
        df: pd.DataFrame,
    ) -> None:
        required = {
            "timestamp",
            "city",
            "target_aqi_24h",
            "target_aqi_48h",
            "target_aqi_72h",
        }

        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                "Missing required columns: "
                + ", ".join(sorted(missing))
            )

        if df.empty:
            raise ValueError(
                "Input dataframe cannot be empty."
            )