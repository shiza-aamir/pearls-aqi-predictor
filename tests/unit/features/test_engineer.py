import pandas as pd
import pytest

from src.features.engineer import (
    AQIFeatureEngineer,
)


def create_dataset(
    hours: int = 150,
) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2026-01-01",
        periods=hours,
        freq="h",
    )

    rows = []

    city_config = {
        "Islamabad": {
            "offset": 0,
            "latitude": 33.6844,
            "longitude": 73.0479,
        },
        "Lahore": {
            "offset": 50,
            "latitude": 31.5204,
            "longitude": 74.3587,
        },
    }

    for city, config in city_config.items():
        offset = config["offset"]

        for i, timestamp in enumerate(
            timestamps
        ):
            rows.append(
                {
                    "city": city,
                    "timestamp": timestamp,
                    "latitude": config[
                        "latitude"
                    ],
                    "longitude": config[
                        "longitude"
                    ],
                    "temperature": 20 + i % 10,
                    "humidity": 40 + i % 30,
                    "precipitation": 0.0,
                    "wind_speed": 3 + i % 5,
                    "wind_direction": float(
                        i * 15 % 360
                    ),
                    "pressure": 1010 + i % 5,
                    "pm2_5": float(
                        30 + i + offset
                    ),
                    "pm10": float(
                        50 + i + offset
                    ),
                    "carbon_monoxide": 500.0,
                    "nitrogen_dioxide": 20.0,
                    "sulphur_dioxide": 10.0,
                    "ozone": 50.0,
                    "aqi_current": float(
                        100 + i + offset
                    ),
                }
            )

    return pd.DataFrame(rows)


def test_preserves_city_count() -> None:
    result, summary = (
        AQIFeatureEngineer()
        .transform(
            create_dataset()
        )
    )

    assert (
        result["city"].nunique()
        == 2
    )

    assert summary.cities == 2


def test_spatial_features_are_model_features() -> None:
    features = (
        AQIFeatureEngineer
        .get_model_feature_columns()
    )

    assert "latitude" in features
    assert "longitude" in features


def test_coordinates_are_preserved() -> None:
    result, _ = (
        AQIFeatureEngineer()
        .transform(
            create_dataset()
        )
    )

    islamabad = result[
        result["city"] == "Islamabad"
    ]

    lahore = result[
        result["city"] == "Lahore"
    ]

    assert (
        islamabad["latitude"]
        .eq(33.6844)
        .all()
    )

    assert (
        islamabad["longitude"]
        .eq(73.0479)
        .all()
    )

    assert (
        lahore["latitude"]
        .eq(31.5204)
        .all()
    )

    assert (
        lahore["longitude"]
        .eq(74.3587)
        .all()
    )


def test_aqi_lag_1h_is_correct() -> None:
    result, _ = (
        AQIFeatureEngineer()
        .transform(
            create_dataset()
        )
    )

    islamabad = result[
        result["city"] == "Islamabad"
    ].reset_index(drop=True)

    assert (
        islamabad.loc[
            10,
            "aqi_lag_1h",
        ]
        ==
        islamabad.loc[
            9,
            "aqi_current",
        ]
    )


def test_aqi_lag_does_not_cross_city() -> None:
    result, _ = (
        AQIFeatureEngineer()
        .transform(
            create_dataset()
        )
    )

    lahore = result[
        result["city"] == "Lahore"
    ].reset_index(drop=True)

    assert pd.isna(
        lahore.loc[
            0,
            "aqi_lag_1h",
        ]
    )


def test_rolling_mean_uses_past_and_current_values() -> None:
    result, _ = (
        AQIFeatureEngineer()
        .transform(
            create_dataset()
        )
    )

    islamabad = result[
        result["city"] == "Islamabad"
    ].reset_index(drop=True)

    expected = (
        islamabad.loc[
            0:5,
            "aqi_current",
        ]
        .mean()
    )

    actual = islamabad.loc[
        5,
        "aqi_rolling_mean_6h",
    ]

    assert actual == pytest.approx(
        expected
    )


def test_time_features_are_created() -> None:
    result, _ = (
        AQIFeatureEngineer()
        .transform(
            create_dataset()
        )
    )

    expected_columns = [
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "month_sin",
        "month_cos",
        "is_weekend",
    ]

    for column in expected_columns:
        assert column in result.columns


def test_weather_features_are_created() -> None:
    result, _ = (
        AQIFeatureEngineer()
        .transform(
            create_dataset()
        )
    )

    expected_columns = [
        "wind_direction_sin",
        "wind_direction_cos",
        "temp_humidity_interaction",
        "stagnation_index",
    ]

    for column in expected_columns:
        assert column in result.columns


def test_model_features_do_not_include_targets() -> None:
    features = (
        AQIFeatureEngineer
        .get_model_feature_columns()
    )

    forbidden = [
        "target_aqi_24h",
        "target_aqi_48h",
        "target_aqi_72h",
    ]

    for column in forbidden:
        assert column not in features


def test_model_feature_count_is_56() -> None:
    features = (
        AQIFeatureEngineer
        .get_model_feature_columns()
    )

    assert len(features) == 56


def test_missing_latitude_rejected() -> None:
    df = (
        create_dataset()
        .drop(
            columns=["latitude"]
        )
    )

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        (
            AQIFeatureEngineer()
            .transform(df)
        )


def test_missing_longitude_rejected() -> None:
    df = (
        create_dataset()
        .drop(
            columns=["longitude"]
        )
    )

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        (
            AQIFeatureEngineer()
            .transform(df)
        )


def test_missing_required_column_rejected() -> None:
    df = (
        create_dataset()
        .drop(
            columns=["humidity"]
        )
    )

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        (
            AQIFeatureEngineer()
            .transform(df)
        )


def test_hourly_gap_rejected() -> None:
    df = create_dataset()

    index_to_remove = df.index[
        (
            df["city"]
            == "Islamabad"
        )
    ][20]

    df = df.drop(
        index_to_remove
    )

    with pytest.raises(
        ValueError,
        match="continuous hourly",
    ):
        (
            AQIFeatureEngineer()
            .transform(df)
        )