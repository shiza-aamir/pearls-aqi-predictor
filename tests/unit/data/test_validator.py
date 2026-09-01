from pathlib import Path

import pandas as pd
import pytest

from src.data.validation import DatasetValidator


@pytest.fixture
def valid_dataset(tmp_path: Path) -> Path:
    timestamps = pd.date_range(
        "2026-01-01 00:00:00",
        periods=48,
        freq="h",
    )

    rows = []

    for city in ["Islamabad", "Lahore"]:
        for index, timestamp in enumerate(timestamps):
            rows.append(
                {
                    "timestamp": timestamp,
                    "city": city,
                    "latitude": 33.6844,
                    "longitude": 73.0479,
                    "pm10": 40.0 + index,
                    "pm2_5": 30.0 + index,
                    "carbon_monoxide": 500.0,
                    "nitrogen_dioxide": 20.0,
                    "sulphur_dioxide": 10.0,
                    "ozone": 60.0,
                    "dust": 5.0,
                    "temperature": 15.0 + index % 10,
                    "humidity": 50.0 + index % 20,
                    "precipitation": 0.0,
                    "wind_speed": 3.0 + index % 4,
                    "wind_direction": float((index * 15) % 360),
                    "pressure": 1010.0 + index % 5,
                    "aqi_category": "Moderate",
                }
            )

    path = tmp_path / "valid.csv"

    pd.DataFrame(rows).to_csv(
        path,
        index=False,
    )

    return path


def test_valid_dataset_structure(valid_dataset: Path) -> None:
    validator = DatasetValidator(valid_dataset)
    report = validator.validate()

    assert report.row_count == 96
    assert report.city_count == 2
    assert report.missing_values == 0
    assert report.duplicate_rows == 0
    assert report.hourly_continuity is True
    assert report.balanced_cities is True
    assert report.expected_rows_per_city == 48
    assert report.required_columns_present is True


def test_numeric_aqi_missing_creates_warning(
    valid_dataset: Path,
) -> None:
    report = DatasetValidator(valid_dataset).validate()

    assert report.numeric_aqi_present is False
    assert any(
        "Numeric AQI target" in warning
        for warning in report.warnings
    )


def test_timezone_missing_creates_warning(
    valid_dataset: Path,
) -> None:
    report = DatasetValidator(valid_dataset).validate()

    assert report.timezone_present is False
    assert any(
        "timezone" in warning.lower()
        for warning in report.warnings
    )


def test_missing_required_column_fails(
    valid_dataset: Path,
) -> None:
    df = pd.read_csv(valid_dataset)
    df = df.drop(columns=["pm2_5"])
    df.to_csv(valid_dataset, index=False)

    report = DatasetValidator(valid_dataset).validate()

    assert report.status == "FAIL"
    assert report.required_columns_present is False
    assert "pm2_5" in report.missing_required_columns


def test_non_hourly_gap_detected(
    valid_dataset: Path,
) -> None:
    df = pd.read_csv(valid_dataset)

    islamabad_indices = df.index[
        df["city"] == "Islamabad"
    ]

    df = df.drop(islamabad_indices[10])
    df.to_csv(valid_dataset, index=False)

    report = DatasetValidator(valid_dataset).validate()

    assert report.hourly_continuity is False


def test_invalid_humidity_detected(
    valid_dataset: Path,
) -> None:
    df = pd.read_csv(valid_dataset)
    df.loc[0, "humidity"] = 120
    df.to_csv(valid_dataset, index=False)

    report = DatasetValidator(valid_dataset).validate()

    assert "humidity" in report.invalid_numeric_ranges
    assert report.invalid_numeric_ranges["humidity"] == 1


def test_missing_dataset_raises_error(
    tmp_path: Path,
) -> None:
    validator = DatasetValidator(
        tmp_path / "missing.csv"
    )

    with pytest.raises(FileNotFoundError):
        validator.validate()