import pandas as pd

from src.ml.training.splitter import (
    PurgedTimeSeriesSplitter,
)


def create_dataset(
    hours: int = 1000,
) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2026-01-01",
        periods=hours,
        freq="h",
    )

    rows = []

    for city in [
        "Islamabad",
        "Rawalpindi",
        "Lahore",
    ]:
        for timestamp in timestamps:
            rows.append(
                {
                    "timestamp": timestamp,
                    "city": city,
                    "target_aqi_24h": 100.0,
                    "target_aqi_48h": 110.0,
                    "target_aqi_72h": 120.0,
                }
            )

    return pd.DataFrame(rows)


def test_splits_are_chronological() -> None:
    splitter = PurgedTimeSeriesSplitter()

    train, validation, test, _ = splitter.split(
        create_dataset()
    )

    assert train["timestamp"].max() < (
        validation["timestamp"].min()
    )

    assert validation["timestamp"].max() < (
        test["timestamp"].min()
    )


def test_72_hour_purge_exists() -> None:
    splitter = PurgedTimeSeriesSplitter(
        purge_hours=72
    )

    train, validation, test, _ = splitter.split(
        create_dataset()
    )

    train_gap = (
        validation["timestamp"].min()
        - train["timestamp"].max()
    )

    validation_gap = (
        test["timestamp"].min()
        - validation["timestamp"].max()
    )

    assert train_gap >= pd.Timedelta(
        hours=72
    )

    assert validation_gap >= pd.Timedelta(
        hours=72
    )


def test_excluded_city_removed() -> None:
    splitter = PurgedTimeSeriesSplitter(
        excluded_cities=("Rawalpindi",)
    )

    train, validation, test, _ = splitter.split(
        create_dataset()
    )

    assert "Rawalpindi" not in train["city"].unique()
    assert "Rawalpindi" not in validation["city"].unique()
    assert "Rawalpindi" not in test["city"].unique()


def test_all_remaining_cities_exist_in_every_split() -> None:
    splitter = PurgedTimeSeriesSplitter(
        excluded_cities=("Rawalpindi",)
    )

    train, validation, test, _ = splitter.split(
        create_dataset()
    )

    expected = {
        "Islamabad",
        "Lahore",
    }

    assert set(train["city"].unique()) == expected
    assert set(validation["city"].unique()) == expected
    assert set(test["city"].unique()) == expected