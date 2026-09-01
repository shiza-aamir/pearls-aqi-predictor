from __future__ import annotations

import gc
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from tensorflow import keras

from src.ml.models.cnn import build_cnn_model
from src.ml.models.cnn_lstm import build_cnn_lstm_model
from src.ml.models.gru import build_gru_model
from src.ml.training.sequence_builder import SequenceDataset


@dataclass(frozen=True)
class DeepLearningMetrics:
    mae: float
    rmse: float
    r2: float


class DeepLearningTrainer:
    TRACKING_URI = "sqlite:///mlflow.db"
    EXPERIMENT_NAME = "pearls-aqi-deep-learning"

    SUPPORTED_MODELS = {
        "cnn",
        "gru",
        "cnn_lstm",
    }

    def __init__(
        self,
        model_name: str,
        horizon: str,
        sequence_length: int,
        feature_count: int,
    ) -> None:
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model '{model_name}'. "
                f"Expected one of {sorted(self.SUPPORTED_MODELS)}"
            )

        self.model_name = model_name
        self.horizon = horizon
        self.sequence_length = sequence_length
        self.feature_count = feature_count

        self._configure_tensorflow()

        mlflow.set_tracking_uri(self.TRACKING_URI)
        mlflow.set_experiment(self.EXPERIMENT_NAME)

    @staticmethod
    def _configure_tensorflow() -> None:
        try:
            tf.config.threading.set_intra_op_parallelism_threads(2)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        except RuntimeError:
            pass

    def build_model(self) -> keras.Model:
        if self.model_name == "cnn":
            return build_cnn_model(
                sequence_length=self.sequence_length,
                feature_count=self.feature_count,
            )

        if self.model_name == "gru":
            return build_gru_model(
                sequence_length=self.sequence_length,
                feature_count=self.feature_count,
            )

        if self.model_name == "cnn_lstm":
            return build_cnn_lstm_model(
                sequence_length=self.sequence_length,
                feature_count=self.feature_count,
            )

        raise RuntimeError("Model selection failed.")

    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> DeepLearningMetrics:
        mae = mean_absolute_error(
            y_true,
            y_pred,
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )

        r2 = r2_score(
            y_true,
            y_pred,
        )

        return DeepLearningMetrics(
            mae=float(mae),
            rmse=float(rmse),
            r2=float(r2),
        )

    def train(
        self,
        train_data: SequenceDataset,
        validation_data: SequenceDataset,
        test_data: SequenceDataset,
        epochs: int = 30,
        batch_size: int = 32,
    ) -> tuple[
        keras.Model,
        DeepLearningMetrics,
        DeepLearningMetrics,
    ]:
        tf.keras.backend.clear_session()
        gc.collect()

        model = self.build_model()

        output_directory = (
            Path("artifacts")
            / "deep_learning"
            / self.model_name
            / self.horizon
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint_path = (
            output_directory
            / "best_model.keras"
        )

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                min_delta=0.0001,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=0.00001,
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_loss",
                save_best_only=True,
            ),
        ]

        run_name = (
            f"{self.model_name}_"
            f"{self.horizon}"
        )

        with mlflow.start_run(
            run_name=run_name
        ):
            mlflow.log_params(
                {
                    "model_type": self.model_name,
                    "forecast_horizon": self.horizon,
                    "sequence_length": self.sequence_length,
                    "feature_count": self.feature_count,
                    "epochs_max": epochs,
                    "batch_size": batch_size,
                    "optimizer": "adam",
                    "learning_rate": 0.001,
                    "loss": "mse",
                    "tensorflow_version": tf.__version__,
                }
            )

            history = model.fit(
                train_data.X,
                train_data.y,
                validation_data=(
                    validation_data.X,
                    validation_data.y,
                ),
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                verbose=2,
                callbacks=callbacks,
            )

            validation_predictions = (
                model.predict(
                    validation_data.X,
                    batch_size=batch_size,
                    verbose=0,
                )
                .reshape(-1)
            )

            test_predictions = (
                model.predict(
                    test_data.X,
                    batch_size=batch_size,
                    verbose=0,
                )
                .reshape(-1)
            )

            validation_predictions = np.clip(
                validation_predictions,
                0.0,
                500.0,
            )

            test_predictions = np.clip(
                test_predictions,
                0.0,
                500.0,
            )

            validation_metrics = self.calculate_metrics(
                validation_data.y,
                validation_predictions,
            )

            test_metrics = self.calculate_metrics(
                test_data.y,
                test_predictions,
            )

            completed_epochs = len(
                history.history["loss"]
            )

            mlflow.log_metrics(
                {
                    "val_mae": validation_metrics.mae,
                    "val_rmse": validation_metrics.rmse,
                    "val_r2": validation_metrics.r2,
                    "test_mae": test_metrics.mae,
                    "test_rmse": test_metrics.rmse,
                    "test_r2": test_metrics.r2,
                    "epochs_completed": completed_epochs,
                }
            )

            final_model_path = (
                output_directory
                / "final_model.keras"
            )

            model.save(
                final_model_path
            )

            metrics_payload = {
                "model": self.model_name,
                "horizon": self.horizon,
                "validation": asdict(
                    validation_metrics
                ),
                "test": asdict(
                    test_metrics
                ),
                "epochs_completed": completed_epochs,
            }

            metrics_path = (
                output_directory
                / "metrics.json"
            )

            metrics_path.write_text(
                json.dumps(
                    metrics_payload,
                    indent=2,
                ),
                encoding="utf-8",
            )

            history_payload = {
                key: [
                    float(value)
                    for value in values
                ]
                for key, values
                in history.history.items()
            }

            history_path = (
                output_directory
                / "history.json"
            )

            history_path.write_text(
                json.dumps(
                    history_payload,
                    indent=2,
                ),
                encoding="utf-8",
            )

            mlflow.tensorflow.log_model(
                model=model,
                name="model",
            )

            mlflow.log_artifacts(
                str(output_directory),
                artifact_path="deep_learning_artifacts",
            )

        gc.collect()

        return (
            model,
            validation_metrics,
            test_metrics,
        )