from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers


def build_cnn_model(
    sequence_length: int,
    feature_count: int,
) -> keras.Model:
    inputs = keras.Input(
        shape=(
            sequence_length,
            feature_count,
        ),
        name="aqi_sequence",
    )

    x = layers.Conv1D(
        filters=32,
        kernel_size=3,
        padding="same",
        activation="relu",
    )(inputs)

    x = layers.MaxPooling1D(
        pool_size=2,
    )(x)

    x = layers.Conv1D(
        filters=16,
        kernel_size=3,
        padding="same",
        activation="relu",
    )(x)

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(
        units=16,
        activation="relu",
    )(x)

    outputs = layers.Dense(
        units=1,
        name="aqi_prediction",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="aqi_1d_cnn",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse",
    )

    return model