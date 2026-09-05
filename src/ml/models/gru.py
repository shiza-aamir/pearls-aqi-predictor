from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers


def build_gru_model(
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

    x = layers.GRU(
        units=32,
        return_sequences=False,
    )(inputs)

    x = layers.Dropout(
        rate=0.20
    )(x)

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
        name="aqi_gru",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse",
    )

    return model