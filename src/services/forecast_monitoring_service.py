from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.features.aqi.calculator import AQICalculator

logger = logging.getLogger(__name__)


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

    DATETIME_COLUMNS = [
        "forecast_created_at",
        "target_timestamp",
        "evaluated_at",
    ]

    NUMERIC_COLUMNS = [
        "horizon_hours",
        "predicted_aqi",
        "actual_aqi",
        "absolute_error",
        "category_distance",
    ]

    BOOLEAN_COLUMNS = [
        "category_correct",
        "adjacent_category_correct",
    ]

    REMOTE_TIMEOUT_SECONDS = 20

    def __init__(
        self,
        ledger_path: Path | None = None,
    ) -> None:
        self.remote_ledger_url = (
            os.getenv(
                "MONITORING_LEDGER_URL"
            )
            if self._is_vercel()
            else None
        )

        env_ledger_path = os.getenv(
            "FORECAST_LEDGER_PATH"
        )

        if ledger_path is not None:
            resolved_path = Path(
                ledger_path
            )
        elif env_ledger_path:
            resolved_path = Path(
                env_ledger_path
            )
        else:
            resolved_path = (
                self._resolve_ledger_path()
            )

        self.ledger_path = resolved_path

        self.ledger_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    @staticmethod
    def _is_vercel() -> bool:
        return bool(
            os.getenv("VERCEL")
        )

    @property
    def remote_read_only(
        self,
    ) -> bool:
        return bool(
            self.remote_ledger_url
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

    @staticmethod
    def _parse_boolean(
        value,
    ):
        if pd.isna(value):
            return pd.NA

        if isinstance(
            value,
            (bool, np.bool_),
        ):
            return bool(value)

        text = (
            str(value)
            .strip()
            .casefold()
        )

        if text in {
            "true",
            "1",
            "yes",
        }:
            return True

        if text in {
            "false",
            "0",
            "no",
        }:
            return False

        return pd.NA

    def _empty_ledger(
        self,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            columns=(
                self.REQUIRED_LEDGER_COLUMNS
            )
        )

    def _normalize_ledger(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        df = df.copy()

        for column in (
            self.REQUIRED_LEDGER_COLUMNS
        ):
            if column not in df.columns:
                df[column] = pd.NA

        df = df[
            self.REQUIRED_LEDGER_COLUMNS
        ].copy()

        for column in (
            self.DATETIME_COLUMNS
        ):
            df[column] = pd.to_datetime(
                df[column],
                utc=True,
                errors="coerce",
            )

        for column in (
            self.NUMERIC_COLUMNS
        ):
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        for column in (
            self.BOOLEAN_COLUMNS
        ):
            df[column] = (
                df[column]
                .map(
                    self._parse_boolean
                )
                .astype("boolean")
            )

        return df

    def _read_local_ledger(
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

        return self._normalize_ledger(
            df
        )

    def _load_remote_ledger(
        self,
    ) -> pd.DataFrame:
        if not self.remote_ledger_url:
            raise RuntimeError(
                "Remote monitoring ledger URL "
                "is not configured."
            )

        response = requests.get(
            self.remote_ledger_url,
            timeout=(
                self.REMOTE_TIMEOUT_SECONDS
            ),
        )

        response.raise_for_status()

        df = pd.read_csv(
            StringIO(
                response.text
            )
        )

        df = self._normalize_ledger(
            df
        )

        df.to_csv(
            self.ledger_path,
            index=False,
        )

        return df

    def load_ledger(
        self,
    ) -> pd.DataFrame:
        if self.remote_read_only:
            try:
                return (
                    self._load_remote_ledger()
                )

            except (
                requests.RequestException,
                ValueError,
                TypeError,
                pd.errors.ParserError,
            ) as exc:
                logger.warning(
                    "Unable to load durable "
                    "remote forecast ledger: %s",
                    exc,
                )

                if self.ledger_path.exists():
                    logger.info(
                        "Using cached forecast "
                        "ledger from %s",
                        self.ledger_path,
                    )

                    return (
                        self._read_local_ledger()
                    )

                logger.warning(
                    "No cached monitoring ledger "
                    "is available."
                )

                return self._empty_ledger()

        return self._read_local_ledger()

    def _save_ledger(
        self,
        df: pd.DataFrame,
    ) -> None:
        if self.remote_read_only:
            logger.debug(
                "Skipping forecast ledger write "
                "because the configured remote "
                "monitoring source is read-only."
            )
            return

        df = self._normalize_ledger(
            df
        )

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
        if self.remote_read_only:
            return self.load_ledger()

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
                        row[
                            "horizon_hours"
                        ]
                    ),
                )
                for _, row
                in ledger.iterrows()
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
        if self.remote_read_only:
            return self.load_ledger()

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

        rows = []

        for horizon in (
            24,
            48,
            72,
        ):
            if ledger.empty:
                horizon_ledger = (
                    self._empty_ledger()
                )
            else:
                horizon_ledger = (
                    ledger[
                        ledger[
                            "horizon_hours"
                        ]
                        == horizon
                    ].copy()
                )

            if city is not None:
                horizon_ledger = (
                    horizon_ledger[
                        horizon_ledger[
                            "city"
                        ]
                        .astype(str)
                        .str.casefold()
                        == str(city).casefold()
                    ]
                )

            evaluated = (
                horizon_ledger[
                    horizon_ledger[
                        "actual_aqi"
                    ]
                    .notna()
                ].copy()
            )

            pending = (
                horizon_ledger[
                    horizon_ledger[
                        "actual_aqi"
                    ]
                    .isna()
                    & horizon_ledger[
                        "target_timestamp"
                    ]
                    .notna()
                ].copy()
            )

            next_maturity_at = None

            if not pending.empty:
                next_timestamp = (
                    pending[
                        "target_timestamp"
                    ]
                    .min()
                )

                if pd.notna(
                    next_timestamp
                ):
                    next_maturity_at = (
                        self
                        ._to_utc_timestamp(
                            next_timestamp
                        )
                        .isoformat()
                    )

            evaluated_forecasts = len(
                evaluated
            )

            status = (
                "live_metrics_available"
                if evaluated_forecasts > 0
                else "awaiting_matured_forecasts"
            )

            base_row = {
                "horizon_hours": horizon,
                "evaluated_forecasts": (
                    evaluated_forecasts
                ),
                "status": status,
                "next_maturity_at": (
                    next_maturity_at
                ),
                "mae": None,
                "rmse": None,
                "within_10_aqi_pct": None,
                "within_20_aqi_pct": None,
                "within_30_aqi_pct": None,
                "category_accuracy_pct": None,
                "adjacent_category_accuracy_pct": None,
            }

            if evaluated.empty:
                rows.append(
                    base_row
                )
                continue

            errors = (
                evaluated[
                    "absolute_error"
                ]
                .astype(float)
                .to_numpy()
            )

            y_true = (
                evaluated[
                    "actual_aqi"
                ]
                .astype(float)
                .to_numpy()
            )

            y_pred = (
                evaluated[
                    "predicted_aqi"
                ]
                .astype(float)
                .to_numpy()
            )

            category_accuracy = (
                evaluated[
                    "category_correct"
                ]
                .astype("boolean")
                .mean()
            )

            adjacent_accuracy = (
                evaluated[
                    "adjacent_category_correct"
                ]
                .astype("boolean")
                .mean()
            )

            base_row.update(
                {
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
                        category_accuracy
                        * 100.0
                    ),
                    "adjacent_category_accuracy_pct": float(
                        adjacent_accuracy
                        * 100.0
                    ),
                }
            )

            rows.append(
                base_row
            )

        return pd.DataFrame(
            rows
        )