import pandas as pd
import pytest

from src.features.aqi import AQITargetBuilder


def create_hourly_dataset(
    hours: int = 120,
) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2026-01-01 00:00:00",
        periods=hours,
        freq="h",
    )

    rows = []

    for city, offset in [
        ("Islamabad", 0.0),
        ("Lahore", 50.0),
    ]:
        for index, timestamp in enumerate(timestamps):
            rows.append(
                {
                    "city": city,
                    "timestamp": timestamp,
                    "pm2_5": 20.0 + index * 0.1 + offset,
                    "pm10": 40.0 + index * 0.1 + offset,
                }
            )

    return pd.DataFrame(rows)


def test_builder_preserves_two_cities() -> None:
    df = create_hourly_dataset()

    result, summary = AQITargetBuilder().build(df)

    assert result["city"].nunique() == 2
    assert summary.cities == 2


def test_first_23_hours_have_no_current_aqi() -> None:
    df = create_hourly_dataset()

    result, _ = AQITargetBuilder().build(df)

    islamabad = result[
        result["city"] == "Islamabad"
    ].reset_index(drop=True)

    assert islamabad.loc[
        :22,
        "aqi_current",
    ].isna().all()

    assert pd.notna(
        islamabad.loc[23, "aqi_current"]
    )


def test_24_hour_mean_uses_only_past_and_current_rows() -> None:
    df = create_hourly_dataset()

    result, _ = AQITargetBuilder().build(df)

    islamabad = result[
        result["city"] == "Islamabad"
    ].reset_index(drop=True)

    expected = df[
        df["city"] == "Islamabad"
    ].iloc[:24]["pm2_5"].mean()

    actual = islamabad.loc[
        23,
        "pm2_5_24h_mean",
    ]

    assert actual == pytest.approx(expected)


def test_target_24h_matches_future_current_aqi() -> None:
    df = create_hourly_dataset()

    result, _ = AQITargetBuilder().build(df)

    islamabad = result[
        result["city"] == "Islamabad"
    ].reset_index(drop=True)

    current_index = 30

    assert islamabad.loc[
        current_index,
        "target_aqi_24h",
    ] == islamabad.loc[
        current_index + 24,
        "aqi_current",
    ]


def test_target_48h_matches_future_current_aqi() -> None:
    df = create_hourly_dataset()

    result, _ = AQITargetBuilder().build(df)

    islamabad = result[
        result["city"] == "Islamabad"
    ].reset_index(drop=True)

    current_index = 30

    assert islamabad.loc[
        current_index,
        "target_aqi_48h",
    ] == islamabad.loc[
        current_index + 48,
        "aqi_current",
    ]


def test_target_72h_matches_future_current_aqi() -> None:
    df = create_hourly_dataset()

    result, _ = AQITargetBuilder().build(df)

    islamabad = result[
        result["city"] == "Islamabad"
    ].reset_index(drop=True)

    current_index = 23

    assert islamabad.loc[
        current_index,
        "target_aqi_72h",
    ] == islamabad.loc[
        current_index + 72,
        "aqi_current",
    ]


def test_city_targets_do_not_cross_city_boundaries() -> None:
    df = create_hourly_dataset()

    result, _ = AQITargetBuilder().build(df)

    islamabad = result[
        result["city"] == "Islamabad"
    ].reset_index(drop=True)

    lahore = result[
        result["city"] == "Lahore"
    ].reset_index(drop=True)

    assert (
        islamabad.loc[30, "target_aqi_24h"]
        != lahore.loc[30, "target_aqi_24h"]
    )


def test_last_24_rows_have_no_24h_target() -> None:
    df = create_hourly_dataset()

    result, _ = AQITargetBuilder().build(df)

    islamabad = result[
        result["city"] == "Islamabad"
    ].reset_index(drop=True)

    assert islamabad.tail(24)[
        "target_aqi_24h"
    ].isna().all()


def test_drop_incomplete_targets() -> None:
    df = create_hourly_dataset()

    result, _ = AQITargetBuilder().build(
        df,
        drop_incomplete_targets=True,
    )

    assert result[
        [
            "aqi_current",
            "target_aqi_24h",
            "target_aqi_48h",
            "target_aqi_72h",
        ]
    ].notna().all().all()


def test_hourly_gap_is_rejected() -> None:
    df = create_hourly_dataset()

    islamabad_index = df.index[
        df["city"] == "Islamabad"
    ][20]

    df = df.drop(islamabad_index)

    with pytest.raises(
        ValueError,
        match="continuous hourly",
    ):
        AQITargetBuilder().build(df)


def test_negative_pm25_is_rejected() -> None:
    df = create_hourly_dataset()

    df.loc[0, "pm2_5"] = -1

    with pytest.raises(ValueError):
        AQITargetBuilder().build(df)


def test_missing_required_column_is_rejected() -> None:
    df = create_hourly_dataset().drop(
        columns=["pm10"]
    )

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        AQITargetBuilder().build(df)