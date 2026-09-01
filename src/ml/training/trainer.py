from dataclasses import dataclass
from typing import Any

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd

from src.features.engineer import AQIFeatureEngineer
from src.ml.evaluation import (
    RegressionMetrics,
    calculate_regression_metrics,
)
from src.ml.models import (
    PersistenceBaseline,
    create_random_forest_model,
    create_ridge_model,
    create_xgboost_model,
)


@dataclass(frozen=True)
class ModelEvaluation:
    model_name: str
    horizon: str
    validation_metrics: RegressionMetrics
    test_metrics: RegressionMetrics
    model: Any | None


class AQIModelTrainer:
    TARGETS = {
        "24h": "target_aqi_24h",
        "48h": "target_aqi_48h",
        "72h": "target_aqi_72h",
    }

    MODEL_FACTORIES = {
        "ridge": create_ridge_model,
        "random_forest": create_random_forest_model,
        "xgboost": create_xgboost_model,
    }

    def __init__(
        self,
        experiment_name: str = "pearls-aqi-forecasting",
    ) -> None:
        self.feature_columns = (
            AQIFeatureEngineer.get_model_feature_columns()
        )

        mlflow.set_experiment(
            experiment_name
        )

    def train_single(
        self,
        model_name: str,
        horizon: str,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> ModelEvaluation:
        if horizon not in self.TARGETS:
            raise ValueError(
                f"Unsupported horizon: {horizon}"
            )

        target_column = self.TARGETS[horizon]

        if model_name == "persistence":
            return self._evaluate_baseline(
                validation_df=validation_df,
                test_df=test_df,
                target_column=target_column,
                horizon=horizon,
            )

        if model_name not in self.MODEL_FACTORIES:
            raise ValueError(
                f"Unsupported model: {model_name}"
            )

        model = self.MODEL_FACTORIES[
            model_name
        ]()

        return self._train_model(
            model_name=model_name,
            model=model,
            horizon=horizon,
            target_column=target_column,
            train_df=train_df,
            validation_df=validation_df,
            test_df=test_df,
        )

    def _evaluate_baseline(
        self,
        validation_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_column: str,
        horizon: str,
    ) -> ModelEvaluation:
        validation_predictions = (
            PersistenceBaseline.predict(
                validation_df
            )
        )

        test_predictions = (
            PersistenceBaseline.predict(
                test_df
            )
        )

        validation_metrics = (
            calculate_regression_metrics(
                validation_df[target_column],
                validation_predictions,
            )
        )

        test_metrics = (
            calculate_regression_metrics(
                test_df[target_column],
                test_predictions,
            )
        )

        with mlflow.start_run(
            run_name=f"persistence_{horizon}"
        ):
            mlflow.log_param(
                "model_name",
                "persistence",
            )

            mlflow.log_param(
                "horizon",
                horizon,
            )

            self._log_metrics(
                validation_metrics,
                test_metrics,
            )

        return ModelEvaluation(
            model_name="persistence",
            horizon=horizon,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            model=None,
        )

    def _train_model(
        self,
        model_name: str,
        model: Any,
        horizon: str,
        target_column: str,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> ModelEvaluation:
        x_train = train_df[
            self.feature_columns
        ].astype(float)

        y_train = train_df[
            target_column
        ].astype(float)

        x_validation = validation_df[
            self.feature_columns
        ].astype(float)

        y_validation = validation_df[
            target_column
        ].astype(float)

        x_test = test_df[
            self.feature_columns
        ].astype(float)

        y_test = test_df[
            target_column
        ].astype(float)

        model.fit(
            x_train,
            y_train,
        )

        validation_predictions = (
            model.predict(
                x_validation
            )
        )

        test_predictions = (
            model.predict(
                x_test
            )
        )

        validation_metrics = (
            calculate_regression_metrics(
                y_validation,
                validation_predictions,
            )
        )

        test_metrics = (
            calculate_regression_metrics(
                y_test,
                test_predictions,
            )
        )

        with mlflow.start_run(
            run_name=f"{model_name}_{horizon}"
        ):
            mlflow.log_param(
                "model_name",
                model_name,
            )

            mlflow.log_param(
                "horizon",
                horizon,
            )

            mlflow.log_param(
                "feature_count",
                len(self.feature_columns),
            )

            self._log_metrics(
                validation_metrics,
                test_metrics,
            )

            self._log_model(
                model_name=model_name,
                model=model,
                x_train=x_train,
            )

        return ModelEvaluation(
            model_name=model_name,
            horizon=horizon,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            model=model,
        )

    @staticmethod
    def _log_model(
        model_name: str,
        model: Any,
        x_train: pd.DataFrame,
    ) -> None:
        input_example = (
            x_train
            .head(5)
            .astype(float)
        )

        if model_name == "xgboost":
            mlflow.xgboost.log_model(
                xgb_model=model,
                name="model",
                input_example=input_example,
                model_format="json",
            )
        else:
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                input_example=input_example,
            )

    @staticmethod
    def _log_metrics(
        validation_metrics: RegressionMetrics,
        test_metrics: RegressionMetrics,
    ) -> None:
        mlflow.log_metric(
            "val_mae",
            validation_metrics.mae,
        )

        mlflow.log_metric(
            "val_rmse",
            validation_metrics.rmse,
        )

        mlflow.log_metric(
            "val_r2",
            validation_metrics.r2,
        )

        mlflow.log_metric(
            "test_mae",
            test_metrics.mae,
        )

        mlflow.log_metric(
            "test_rmse",
            test_metrics.rmse,
        )

        mlflow.log_metric(
            "test_r2",
            test_metrics.r2,
        )