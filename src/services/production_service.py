from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.features.aqi.target_builder import AQITargetBuilder
from src.services.explanation_service import (
    AQIExplanationService,
)
from src.services.feature_service import (
    AQIFeatureService,
)
from src.services.forecast_monitoring_service import (
    ForecastMonitoringService,
)
from src.services.live_feature_pipeline import (
    LiveFeaturePipeline,
)
from src.services.live_history_service import (
    LiveHistoryService,
)
from src.services.prediction_service import (
    AQIPrediction,
    AQIPredictionService,
)


@dataclass(frozen=True)
class AQIAlert:
    level: str
    severity: int
    message: str


@dataclass(frozen=True)
class ProductionForecast:
    horizon: str
    predicted_aqi: float
    predicted_category: str
    model_name: str
    model_alias: str
    alert: AQIAlert
    explanation: Any


@dataclass(frozen=True)
class ProductionResult:
    city: str
    timestamp: pd.Timestamp

    current_aqi: float
    current_category: str
    current_alert: AQIAlert

    pm2_5: float
    pm10: float
    temperature: float
    humidity: float
    wind_speed: float

    data_source: str

    forecasts: tuple[
        ProductionForecast,
        ...,
    ]

    live_performance: pd.DataFrame

    history_rows: int
    feature_count: int


class AQIProductionService:
    """
    Unified production orchestration service.

    Production flow
    ---------------
    1. Fetch/update/recover live history.
    2. Calculate the latest AQI.
    3. Engineer the exact 56 model features.
    4. Write the latest feature row to Feast.
    5. Retrieve the feature row from Feast.
    6. Predict 24h, 48h, and 72h AQI using
       the MLflow champion models.
    7. Generate SHAP explanations.
    8. Build AQI-enriched monitoring history.
    9. Evaluate previously matured forecasts.
    10. Record the newly generated forecasts.
    11. Return dashboard-ready output.
    """

    EXPECTED_FEATURE_COUNT = 56

    def __init__(
        self,
    ) -> None:
        self.history_service = (
            LiveHistoryService()
        )

        self.target_builder = (
            AQITargetBuilder()
        )

        self.feature_pipeline = (
            LiveFeaturePipeline()
        )

        self.feature_service = (
            AQIFeatureService()
        )

        self.prediction_service = (
            AQIPredictionService()
        )

        self.explanation_service = (
            AQIExplanationService()
        )

        self.monitoring_service = (
            ForecastMonitoringService()
        )

    @staticmethod
    def _to_utc_timestamp(
        value,
    ) -> pd.Timestamp:
        timestamp = pd.Timestamp(
            value
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

        return timestamp

    @staticmethod
    def _build_alert(
        aqi: float,
    ) -> AQIAlert:
        value = float(
            aqi
        )

        if value <= 50:
            return AQIAlert(
                level="Good",
                severity=0,
                message=(
                    "Air quality is good. "
                    "No health alert is required."
                ),
            )

        if value <= 100:
            return AQIAlert(
                level="Moderate",
                severity=1,
                message=(
                    "Air quality is moderate. "
                    "Unusually sensitive people "
                    "may consider limiting "
                    "prolonged outdoor exposure."
                ),
            )

        if value <= 150:
            return AQIAlert(
                level="Sensitive Group Advisory",
                severity=2,
                message=(
                    "Air quality may be unhealthy "
                    "for sensitive groups. "
                    "Consider reducing prolonged "
                    "or heavy outdoor activity."
                ),
            )

        if value <= 200:
            return AQIAlert(
                level="Health Alert",
                severity=3,
                message=(
                    "Air quality is unhealthy. "
                    "Reduce prolonged outdoor "
                    "exposure, especially for "
                    "sensitive groups."
                ),
            )

        if value <= 300:
            return AQIAlert(
                level="Serious Health Alert",
                severity=4,
                message=(
                    "Air quality is very unhealthy. "
                    "Avoid unnecessary prolonged "
                    "outdoor exposure."
                ),
            )

        return AQIAlert(
            level="Hazardous Emergency Alert",
            severity=5,
            message=(
                "Air quality is hazardous. "
                "Avoid outdoor exposure where "
                "possible and follow applicable "
                "public-health guidance."
            ),
        )

    @staticmethod
    def _require_columns(
        df: pd.DataFrame,
        columns: list[str],
        name: str,
    ) -> None:
        missing = [
            column
            for column in columns
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{name} is missing required "
                f"columns: {missing}"
            )

    @staticmethod
    def _extract_source(
        history: pd.DataFrame,
    ) -> str:
        if history.empty:
            return "unknown"

        if "source" not in history.columns:
            return "unknown"

        value = history.iloc[-1]["source"]

        if pd.isna(value):
            return "unknown"

        return str(value)

    def _prepare_live_data(
        self,
        city: str,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        history = (
            self.history_service
            .ensure_current_history(
                city
            )
        )

        if history.empty:
            raise ValueError(
                f"No live history available "
                f"for {city}."
            )

        self._require_columns(
            history,
            [
                "timestamp",
            ],
            "Live history",
        )

        result = (
            self.feature_pipeline
            .build_latest_features(
                history
            )
        )

        if not isinstance(
            result,
            tuple,
        ):
            raise TypeError(
                "LiveFeaturePipeline must "
                "return a tuple."
            )

        if len(result) != 2:
            raise ValueError(
                "LiveFeaturePipeline must return "
                "(enriched_row, feature_row)."
            )

        (
            enriched_row,
            feature_row,
        ) = result

        if not isinstance(
            enriched_row,
            pd.DataFrame,
        ):
            raise TypeError(
                "enriched_row must be a "
                "pandas DataFrame."
            )

        if not isinstance(
            feature_row,
            pd.DataFrame,
        ):
            raise TypeError(
                "feature_row must be a "
                "pandas DataFrame."
            )

        if enriched_row.shape[0] != 1:
            raise ValueError(
                "Expected exactly one latest "
                "enriched row. "
                f"Received shape: "
                f"{enriched_row.shape}"
            )

        expected_shape = (
            1,
            self.EXPECTED_FEATURE_COUNT,
        )

        if (
            feature_row.shape
            != expected_shape
        ):
            raise ValueError(
                "Expected model feature shape "
                f"{expected_shape}, received "
                f"{feature_row.shape}."
            )

        self._require_columns(
            enriched_row,
            [
                "timestamp",
                "aqi_current",
                "aqi_category_derived",
                "pm2_5",
                "pm10",
                "temperature",
                "humidity",
                "wind_speed",
            ],
            "Enriched live row",
        )

        return (
            history,
            enriched_row,
            feature_row,
        )

    def _write_and_read_feast(
        self,
        city: str,
        timestamp: pd.Timestamp,
        feature_row: pd.DataFrame,
    ) -> pd.DataFrame:
        self.feature_service \
            .write_online_features(
                city=city,
                event_timestamp=timestamp,
                feature_row=feature_row,
            )

        feast_features = (
            self.feature_service
            .get_online_features(
                city
            )
        )

        if not isinstance(
            feast_features,
            pd.DataFrame,
        ):
            raise TypeError(
                "Feast online retrieval must "
                "return a pandas DataFrame."
            )

        expected_shape = (
            1,
            self.EXPECTED_FEATURE_COUNT,
        )

        if (
            feast_features.shape
            != expected_shape
        ):
            raise ValueError(
                "Expected Feast feature shape "
                f"{expected_shape}, received "
                f"{feast_features.shape}."
            )

        return feast_features

    def _build_forecasts(
        self,
        feature_row: pd.DataFrame,
    ) -> tuple[
        ProductionForecast,
        ...,
    ]:
        predictions = (
            self.prediction_service
            .predict_all(
                feature_row
            )
        )

        if len(predictions) != 3:
            raise ValueError(
                "Expected exactly three "
                "production forecasts."
            )

        forecasts = []

        for prediction in predictions:
            if not isinstance(
                prediction,
                AQIPrediction,
            ):
                raise TypeError(
                    "Prediction service returned "
                    "an unexpected result type."
                )

            explanation = (
                self.explanation_service
                .explain_single(
                    feature_row=feature_row,
                    horizon=prediction.horizon,
                    top_n=5,
                )
            )

            alert = self._build_alert(
                prediction.predicted_aqi
            )

            forecast = ProductionForecast(
                horizon=str(
                    prediction.horizon
                ),
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
                alert=alert,
                explanation=explanation,
            )

            forecasts.append(
                forecast
            )

        return tuple(
            forecasts
        )

    @staticmethod
    def _predictions_from_forecasts(
        forecasts: tuple[
            ProductionForecast,
            ...,
        ],
    ) -> list[
        AQIPrediction
    ]:
        predictions = []

        for forecast in forecasts:
            predictions.append(
                AQIPrediction(
                    horizon=(
                        forecast.horizon
                    ),
                    predicted_aqi=(
                        forecast.predicted_aqi
                    ),
                    predicted_category=(
                        forecast.predicted_category
                    ),
                    model_name=(
                        forecast.model_name
                    ),
                    model_alias=(
                        forecast.model_alias
                    ),
                )
            )

        return predictions

    def _prepare_monitoring_history(
        self,
        history: pd.DataFrame,
        enriched_row: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Derive AQI for the complete available monitoring history.

        The same AQITargetBuilder used elsewhere in the project
        is used here so historical actual AQI values follow the
        exact same 24-hour rolling PM2.5/PM10 methodology.

        This allows matured forecasts to be evaluated even when
        a production run was skipped at an earlier target hour.
        """

        self._require_columns(
            history,
            [
                "city",
                "timestamp",
                "pm2_5",
                "pm10",
            ],
            "Monitoring history",
        )

        self._require_columns(
            enriched_row,
            [
                "timestamp",
                "aqi_current",
            ],
            "Enriched monitoring row",
        )

        monitoring_history = (
            history.copy()
        )

        monitoring_history[
            "timestamp"
        ] = pd.to_datetime(
            monitoring_history[
                "timestamp"
            ],
            utc=True,
            errors="raise",
        )

        (
            monitoring_history,
            _,
        ) = self.target_builder.build(
            monitoring_history,
            drop_incomplete_targets=False,
        )

        monitoring_history[
            "timestamp"
        ] = pd.to_datetime(
            monitoring_history[
                "timestamp"
            ],
            utc=True,
            errors="raise",
        )

        latest_timestamp = (
            self._to_utc_timestamp(
                enriched_row.iloc[0][
                    "timestamp"
                ]
            )
        )

        latest_aqi = float(
            enriched_row.iloc[0][
                "aqi_current"
            ]
        )

        latest_mask = (
            monitoring_history[
                "timestamp"
            ]
            == latest_timestamp
        )

        if not latest_mask.any():
            raise ValueError(
                "Latest enriched timestamp is missing "
                "from monitoring history."
            )

        derived_latest = (
            monitoring_history.loc[
                latest_mask,
                "aqi_current",
            ].iloc[-1]
        )

        if pd.isna(
            derived_latest
        ):
            raise ValueError(
                "Latest AQI could not be derived "
                "from monitoring history."
            )

        if (
            abs(
                float(derived_latest)
                - latest_aqi
            )
            > 1e-6
        ):
            raise ValueError(
                "AQI mismatch between full-history "
                "target builder and live feature "
                "pipeline. "
                f"Target builder="
                f"{float(derived_latest)}, "
                f"live pipeline={latest_aqi}."
            )

        return (
            monitoring_history
            .sort_values(
                "timestamp"
            )
            .drop_duplicates(
                subset=[
                    "timestamp",
                ],
                keep="last",
            )
            .reset_index(
                drop=True
            )
        )

    def run(
        self,
        city: str,
    ) -> ProductionResult:
        (
            history,
            enriched_row,
            direct_feature_row,
        ) = self._prepare_live_data(
            city
        )

        latest = (
            enriched_row.iloc[0]
        )

        timestamp = (
            self._to_utc_timestamp(
                latest[
                    "timestamp"
                ]
            )
        )

        feature_row = (
            self._write_and_read_feast(
                city=city,
                timestamp=timestamp,
                feature_row=(
                    direct_feature_row
                ),
            )
        )

        forecasts = (
            self._build_forecasts(
                feature_row
            )
        )

        monitoring_history = (
            self._prepare_monitoring_history(
                history=history,
                enriched_row=enriched_row,
            )
        )

        self.monitoring_service \
            .evaluate_available_forecasts(
                city=city,
                history=(
                    monitoring_history
                ),
                evaluated_at=(
                    timestamp
                ),
            )

        monitoring_predictions = (
            self._predictions_from_forecasts(
                forecasts
            )
        )

        self.monitoring_service \
            .record_forecasts(
                city=city,
                forecast_created_at=(
                    timestamp
                ),
                predictions=(
                    monitoring_predictions
                ),
            )

        live_performance = (
            self.monitoring_service
            .performance_summary(
                city=city
            )
        )

        current_aqi = float(
            latest[
                "aqi_current"
            ]
        )

        current_category = str(
            latest[
                "aqi_category_derived"
            ]
        )

        current_alert = (
            self._build_alert(
                current_aqi
            )
        )

        return ProductionResult(
            city=str(
                city
            ),
            timestamp=timestamp,
            current_aqi=(
                current_aqi
            ),
            current_category=(
                current_category
            ),
            current_alert=(
                current_alert
            ),
            pm2_5=float(
                latest[
                    "pm2_5"
                ]
            ),
            pm10=float(
                latest[
                    "pm10"
                ]
            ),
            temperature=float(
                latest[
                    "temperature"
                ]
            ),
            humidity=float(
                latest[
                    "humidity"
                ]
            ),
            wind_speed=float(
                latest[
                    "wind_speed"
                ]
            ),
            data_source=(
                self._extract_source(
                    history
                )
            ),
            forecasts=(
                forecasts
            ),
            live_performance=(
                live_performance
            ),
            history_rows=len(history),
            feature_count=int(
                feature_row.shape[1]
            ),
        )