from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.aqi.calculator import AQICalculator


@dataclass(frozen=True)
class ForecastRecord:
    city: str
    forecast_created_at: datetime
    target_timestamp: datetime
    horizon_hours: int
    predicted_aqi: float
    predicted_category: str
    model_name: str
    model_alias: str


class ForecastMonitoringService:
    CATEGORY_ORDER = [
        "Good",
        "Moderate",
        "Unhealthy for Sensitive Groups",
        "Unhealthy",
        "Very Unhealthy",
        "Hazardous",
    ]

    REQUIRED_LEDGER_COLUMNS = [
        "city",
        "forecast_created_at",
        "target_timestamp",
        "horizon_hours",
        "predicted_aqi",
        "predicted_category",
        "model_name",
        "model_alias",
        "actual_aqi",
        "actual_category",
        "absolute_error",
        "category_correct",
        "category_distance",
        "adjacent_category_correct",
        "evaluated_at",
    ]

    def __init__(
        self,
        ledger_path: Path | None = None,
    ) -> None:
        self.ledger_path = (
            Path(ledger_path)
            if ledger_path is not None
            else self._resolve_ledger_path()
        )

        self.ledger_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    @staticmethod
    def _is_vercel() -> bool:
        return bool(
            os.getenv("VERCEL")
        )

    @classmethod
    def _resolve_ledger_path(
        cls,
    ) -> Path:
        if cls._is_vercel():
            return Path(
                "/tmp/pearls-aqi/"
                "forecast_ledger.csv"
            )

        return Path(
            "data/live/"
            "forecast_ledger.parquet"
        )

    @staticmethod
    def _to_utc_timestamp(
        value,
    ) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)

        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(
                "UTC"
            )
        else:
            timestamp = timestamp.tz_convert(
                "UTC"
            )

        return timestamp

    @staticmethod
    def _category_from_aqi(
        value: float,
    ) -> str:
        clipped = float(
            np.clip(
                value,
                0.0,
                500.0,
            )
        )

        return (
            AQICalculator
            .category_from_aqi(
                round(clipped)
            )
        )

    def _empty_ledger(
        self,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            columns=self.REQUIRED_LEDGER_COLUMNS
        )

    def load_ledger(
        self,
    ) -> pd.DataFrame:
        if not self.ledger_path.exists():
            return self._empty_ledger()

        if (
            self.ledger_path
            .suffix
            .lower()
            == ".csv"
        ):
            df = pd.read_csv(
                self.ledger_path
            )
        else:
            df = pd.read_parquet(
                self.ledger_path
            )

        for column in self.REQUIRED_LEDGER_COLUMNS:
            if column not in df.columns:
                df[column] = pd.NA

        df = df[
            self.REQUIRED_LEDGER_COLUMNS
        ].copy()

        for column in [
            "forecast_created_at",
            "target_timestamp",
            "evaluated_at",
        ]:
            df[column] = pd.to_datetime(
                df[column],
                utc=True,
                errors="coerce",
            )

        return df

    def _save_ledger(
        self,
        df: pd.DataFrame,
    ) -> None:
        df = df.copy()

        df = df.sort_values(
            [
                "city",
                "target_timestamp",
                "horizon_hours",
                "forecast_created_at",
            ]
        ).reset_index(
            drop=True
        )

        if (
            self.ledger_path
            .suffix
            .lower()
            == ".csv"
        ):
            df.to_csv(
                self.ledger_path,
                index=False,
            )
        else:
            df.to_parquet(
                self.ledger_path,
                index=False,
            )

    def record_forecasts(
        self,
        city: str,
        forecast_created_at,
        predictions: Iterable,
    ) -> pd.DataFrame:
        created_at = (
            self._to_utc_timestamp(
                forecast_created_at
            )
        )

        records = []

        for prediction in predictions:
            horizon_text = str(
                prediction.horizon
            ).lower()

            horizon_hours = int(
                horizon_text.replace(
                    "h",
                    "",
                )
            )

            target_timestamp = (
                created_at
                + pd.Timedelta(
                    hours=horizon_hours
                )
            )

            record = ForecastRecord(
                city=city,
                forecast_created_at=(
                    created_at.to_pydatetime()
                ),
                target_timestamp=(
                    target_timestamp
                    .to_pydatetime()
                ),
                horizon_hours=horizon_hours,
                predicted_aqi=float(
                    prediction.predicted_aqi
                ),
                predicted_category=str(
                    prediction.predicted_category
                ),
                model_name=str(
                    prediction.model_name
                ),
                model_alias=str(
                    prediction.model_alias
                ),
            )

            row = asdict(
                record
            )

            row.update(
                {
                    "actual_aqi": np.nan,
                    "actual_category": None,
                    "absolute_error": np.nan,
                    "category_correct": pd.NA,
                    "category_distance": np.nan,
                    "adjacent_category_correct": pd.NA,
                    "evaluated_at": pd.NaT,
                }
            )

            records.append(
                row
            )

        if not records:
            raise ValueError(
                "No forecasts were provided."
            )

        new_df = pd.DataFrame(
            records
        )

        ledger = self.load_ledger()

        if not ledger.empty:
            duplicate_keys = {
                (
                    str(
                        row["city"]
                    ),
                    self._to_utc_timestamp(
                        row[
                            "forecast_created_at"
                        ]
                    ),
                    int(
                        row["horizon_hours"]
                    ),
                )
                for _, row in ledger.iterrows()
            }

            new_df = new_df[
                new_df.apply(
                    lambda row: (
                        str(
                            row["city"]
                        ),
                        self._to_utc_timestamp(
                            row[
                                "forecast_created_at"
                            ]
                        ),
                        int(
                            row[
                                "horizon_hours"
                            ]
                        ),
                    )
                    not in duplicate_keys,
                    axis=1,
                )
            ]

        if new_df.empty:
            return ledger

        if ledger.empty:
            combined = new_df.copy()
        else:
            combined = pd.concat(
                [
                    ledger,
                    new_df,
                ],
                ignore_index=True,
            )

        self._save_ledger(
            combined
        )

        return combined

    def evaluate_available_forecasts(
        self,
        city: str,
        history: pd.DataFrame,
        evaluated_at=None,
    ) -> pd.DataFrame:
        ledger = self.load_ledger()

        if ledger.empty:
            return ledger

        required_history = [
            "timestamp",
            "aqi_current",
        ]

        missing = [
            column
            for column in required_history
            if column not in history.columns
        ]

        if missing:
            raise ValueError(
                "History is missing required "
                f"columns: {missing}"
            )

        history = history.copy()

        history["timestamp"] = pd.to_datetime(
            history["timestamp"],
            utc=True,
            errors="raise",
        )

        history = (
            history
            .sort_values(
                "timestamp"
            )
            .drop_duplicates(
                subset=[
                    "timestamp"
                ],
                keep="last",
            )
        )

        actual_lookup = (
            history
            .set_index(
                "timestamp"
            )[
                "aqi_current"
            ]
            .to_dict()
        )

        now = (
            self._to_utc_timestamp(
                evaluated_at
            )
            if evaluated_at is not None
            else pd.Timestamp.now(
                tz="UTC"
            )
        )

        city_mask = (
            ledger["city"]
            .astype(str)
            .str.casefold()
            == str(city).casefold()
        )

        pending_mask = (
            ledger[
                "actual_aqi"
            ]
            .isna()
        )

        due_mask = (
            ledger[
                "target_timestamp"
            ]
            <= now
        )

        indices = ledger.index[
            city_mask
            & pending_mask
            & due_mask
        ]

        for index in indices:
            target_timestamp = (
                self._to_utc_timestamp(
                    ledger.at[
                        index,
                        "target_timestamp",
                    ]
                )
            )

            if (
                target_timestamp
                not in actual_lookup
            ):
                continue

            actual_aqi = float(
                actual_lookup[
                    target_timestamp
                ]
            )

            if not np.isfinite(
                actual_aqi
            ):
                continue

            predicted_aqi = float(
                ledger.at[
                    index,
                    "predicted_aqi",
                ]
            )

            actual_category = (
                self._category_from_aqi(
                    actual_aqi
                )
            )

            predicted_category = str(
                ledger.at[
                    index,
                    "predicted_category",
                ]
            )

            try:
                actual_index = (
                    self.CATEGORY_ORDER.index(
                        actual_category
                    )
                )

                predicted_index = (
                    self.CATEGORY_ORDER.index(
                        predicted_category
                    )
                )

                category_distance = abs(
                    actual_index
                    - predicted_index
                )

            except ValueError:
                category_distance = (
                    np.nan
                )

            ledger.at[
                index,
                "actual_aqi",
            ] = actual_aqi

            ledger.at[
                index,
                "actual_category",
            ] = actual_category

            ledger.at[
                index,
                "absolute_error",
            ] = abs(
                actual_aqi
                - predicted_aqi
            )

            ledger.at[
                index,
                "category_correct",
            ] = (
                actual_category
                == predicted_category
            )

            ledger.at[
                index,
                "category_distance",
            ] = category_distance

            ledger.at[
                index,
                "adjacent_category_correct",
            ] = (
                bool(
                    category_distance
                    <= 1
                )
                if np.isfinite(
                    category_distance
                )
                else pd.NA
            )

            ledger.at[
                index,
                "evaluated_at",
            ] = now

        self._save_ledger(
            ledger
        )

        return ledger

    def performance_summary(
        self,
        city: str | None = None,
    ) -> pd.DataFrame:
        ledger = self.load_ledger()

        if ledger.empty:
            return pd.DataFrame()

        evaluated = ledger[
            ledger[
                "actual_aqi"
            ]
            .notna()
        ].copy()

        if city is not None:
            evaluated = evaluated[
                evaluated["city"]
                .astype(str)
                .str.casefold()
                == str(city).casefold()
            ]

        if evaluated.empty:
            return pd.DataFrame()

        rows = []

        for horizon, group in (
            evaluated.groupby(
                "horizon_hours"
            )
        ):
            errors = (
                group[
                    "absolute_error"
                ]
                .astype(float)
                .to_numpy()
            )

            y_true = (
                group[
                    "actual_aqi"
                ]
                .astype(float)
                .to_numpy()
            )

            y_pred = (
                group[
                    "predicted_aqi"
                ]
                .astype(float)
                .to_numpy()
            )

            rows.append(
                {
                    "horizon_hours": int(
                        horizon
                    ),
                    "evaluated_forecasts": len(group),
                    "mae": float(
                        np.mean(
                            errors
                        )
                    ),
                    "rmse": float(
                        np.sqrt(
                            np.mean(
                                (
                                    y_true
                                    - y_pred
                                )
                                ** 2
                            )
                        )
                    ),
                    "within_10_aqi_pct": float(
                        np.mean(
                            errors <= 10
                        )
                        * 100.0
                    ),
                    "within_20_aqi_pct": float(
                        np.mean(
                            errors <= 20
                        )
                        * 100.0
                    ),
                    "within_30_aqi_pct": float(
                        np.mean(
                            errors <= 30
                        )
                        * 100.0
                    ),
                    "category_accuracy_pct": float(
                        group[
                            "category_correct"
                        ]
                        .astype(bool)
                        .mean()
                        * 100.0
                    ),
                    "adjacent_category_accuracy_pct": float(
                        group[
                            "adjacent_category_correct"
                        ]
                        .astype(bool)
                        .mean()
                        * 100.0
                    ),
                }
            )

        return (
            pd.DataFrame(
                rows
            )
            .sort_values(
                "horizon_hours"
            )
            .reset_index(
                drop=True
            )
        )