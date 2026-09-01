import numpy as np
import pandas as pd

from src.ml.training.sequence_builder import (
    SEQUENCE_FEATURES,
    AQISequenceBuilder,
)


def create_test_dataframe(
    row_count: int = 100,
) -> pd.DataFrame:
    timestamps = pd.date_range(
        start="2026-01-01 00:00:00",
        periods=row_count,
        freq="h",
    )

    dataframe = pd.DataFrame(
        {
            "city": ["Islamabad"] * row_count,
            "timestamp": timestamps,
            "target_aqi_24h": np.linspace(
                100,
                200,
                row_count,
            ),
            "target_aqi_48h": np.linspace(
                110,
                210,
                row_count,
            ),
            "target_aqi_72h": np.linspace(
                120,
                220,
                row_count,
            ),
        }
    )

    for index, feature in enumerate(
        SEQUENCE_FEATURES
    ):
        dataframe[feature] = (
            np.arange(
                row_count,
                dtype=float,
            )
            + index
            + 1.0
        )

    return dataframe


def test_sequence_shape() -> None:
    dataframe = create_test_dataframe(
        row_count=100
    )

    builder = AQISequenceBuilder(
        sequence_length=72
    )

    builder.fit_scaler(
        dataframe
    )

    dataset = builder.build(
        dataframe=dataframe,
        horizon="24h",
    )

    assert dataset.X.shape == (
        29,
        72,
        len(SEQUENCE_FEATURES),
    )

    assert dataset.y.shape == (
        29,
    )


def test_scaler_is_fitted_only_when_requested() -> None:
    builder = AQISequenceBuilder(
        sequence_length=72
    )

    assert builder.is_fitted is False

    dataframe = create_test_dataframe()

    builder.fit_scaler(
        dataframe
    )

    assert builder.is_fitted is True


def test_all_values_are_finite() -> None:
    dataframe = create_test_dataframe()

    builder = AQISequenceBuilder(
        sequence_length=72
    )

    builder.fit_scaler(
        dataframe
    )

    dataset = builder.build(
        dataframe=dataframe,
        horizon="48h",
    )

    assert np.isfinite(
        dataset.X
    ).all()

    assert np.isfinite(
        dataset.y
    ).all()