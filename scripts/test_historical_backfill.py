from __future__ import annotations

from pathlib import Path
from time import sleep

import numpy as np
import pandas as pd
import requests


CITY = "Islamabad"
LATITUDE = 33.6844
LONGITUDE = 73.0479

START_DATE = "2022-08-01"
END_DATE = "2026-02-04"

EXISTING_DATA_PATH = Path(
    "data/historical/pakistan_air_quality_final_clean.csv"
)

OUTPUT_DIR = Path(
    "data/historical/backfill_test"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "islamabad_openmeteo_backfill_test.csv"
)

COMPARISON_PATH = (
    OUTPUT_DIR
    / "islamabad_overlap_comparison.csv"
)

WEATHER_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
)

AIR_QUALITY_URL = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
)

WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
]

AIR_VARIABLES = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "dust",
]

WEATHER_RENAME = {
    "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity",
    "wind_speed_10m": "wind_speed",
    "wind_direction_10m": "wind_direction",
    "surface_pressure": "pressure",
}

COMPARE_COLUMNS = [
    "temperature",
    "humidity",
    "precipitation",
    "wind_speed",
    "wind_direction",
    "pressure",
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "dust",
]


def fetch_json(
    url: str,
    params: dict,
) -> dict:
    response = requests.get(
        url,
        params=params,
        timeout=120,
    )

    response.raise_for_status()

    return response.json()


def fetch_weather() -> pd.DataFrame:
    print("\nDownloading weather...")

    payload = fetch_json(
        WEATHER_URL,
        {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "hourly": ",".join(
                WEATHER_VARIABLES
            ),
            "timezone": "GMT",
            "wind_speed_unit": "ms",
        },
    )

    if "hourly" not in payload:
        raise RuntimeError(
            f"No hourly weather data returned: {payload}"
        )

    dataframe = pd.DataFrame(
        payload["hourly"]
    )

    dataframe["timestamp"] = pd.to_datetime(
        dataframe.pop("time"),
        utc=True,
    )

    dataframe = dataframe.rename(
        columns=WEATHER_RENAME
    )

    print(
        f"Weather rows: {len(dataframe):,}"
    )

    return dataframe


def fetch_air_quality() -> pd.DataFrame:
    print("\nDownloading air quality...")

    payload = fetch_json(
        AIR_QUALITY_URL,
        {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "hourly": ",".join(
                AIR_VARIABLES
            ),
            "timezone": "GMT",
            "domains": "cams_global",
        },
    )

    if "hourly" not in payload:
        raise RuntimeError(
            f"No hourly air-quality data returned: {payload}"
        )

    dataframe = pd.DataFrame(
        payload["hourly"]
    )

    dataframe["timestamp"] = pd.to_datetime(
        dataframe.pop("time"),
        utc=True,
    )

    print(
        f"Air-quality rows: {len(dataframe):,}"
    )

    return dataframe


def combine_data(
    weather: pd.DataFrame,
    air: pd.DataFrame,
) -> pd.DataFrame:
    combined = weather.merge(
        air,
        on="timestamp",
        how="inner",
        validate="one_to_one",
    )

    combined.insert(
        0,
        "city",
        CITY,
    )

    combined.insert(
        1,
        "latitude",
        LATITUDE,
    )

    combined.insert(
        2,
        "longitude",
        LONGITUDE,
    )

    return combined


def validate_backfill(
    dataframe: pd.DataFrame,
) -> None:
    print("\n" + "=" * 90)
    print("BACKFILL QUALITY")
    print("=" * 90)

    print(
        f"Rows: {len(dataframe):,}"
    )

    print(
        f"Start: {dataframe['timestamp'].min()}"
    )

    print(
        f"End:   {dataframe['timestamp'].max()}"
    )

    print(
        "Duplicate timestamps:",
        dataframe["timestamp"]
        .duplicated()
        .sum(),
    )

    missing = (
        dataframe[COMPARE_COLUMNS]
        .isna()
        .sum()
    )

    print("\nMissing values:")
    print(
        missing.to_string()
    )

    ordered = dataframe.sort_values(
        "timestamp"
    )

    differences = (
        ordered["timestamp"]
        .diff()
        .dropna()
    )

    gaps = (
        differences
        != pd.Timedelta(hours=1)
    ).sum()

    print(
        f"\nNon-hourly gaps: {gaps}"
    )

    print("\nStatistics:")

    print(
        dataframe[
            COMPARE_COLUMNS
        ]
        .describe()
        .T[
            [
                "mean",
                "std",
                "min",
                "max",
            ]
        ]
        .round(3)
        .to_string()
    )


def load_existing() -> pd.DataFrame:
    dataframe = pd.read_csv(
        EXISTING_DATA_PATH
    )

    dataframe = dataframe[
        dataframe["city"] == CITY
    ].copy()

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"]
    )

    return dataframe


def compare_overlap(
    existing: pd.DataFrame,
    new: pd.DataFrame,
) -> pd.DataFrame:
    new = new.copy()

    new["timestamp"] = (
        new["timestamp"]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
    )

    merged = existing[
        [
            "timestamp",
            *COMPARE_COLUMNS,
        ]
    ].merge(
        new[
            [
                "timestamp",
                *COMPARE_COLUMNS,
            ]
        ],
        on="timestamp",
        how="inner",
        suffixes=(
            "_existing",
            "_new",
        ),
    )

    print("\n" + "=" * 90)
    print("OVERLAP COMPARISON")
    print("=" * 90)

    print(
        f"Existing Islamabad rows: "
        f"{len(existing):,}"
    )

    print(
        f"Matched timestamps: "
        f"{len(merged):,}"
    )

    rows = []

    for variable in COMPARE_COLUMNS:
        existing_column = (
            f"{variable}_existing"
        )

        new_column = (
            f"{variable}_new"
        )

        pair = merged[
            [
                existing_column,
                new_column,
            ]
        ].dropna()

        if pair.empty:
            correlation = np.nan
            mae = np.nan
            old_mean = np.nan
            new_mean = np.nan

        else:
            correlation = (
                pair[
                    existing_column
                ]
                .corr(
                    pair[
                        new_column
                    ]
                )
            )

            mae = (
                pair[
                    existing_column
                ]
                .sub(
                    pair[
                        new_column
                    ]
                )
                .abs()
                .mean()
            )

            old_mean = (
                pair[
                    existing_column
                ].mean()
            )

            new_mean = (
                pair[
                    new_column
                ].mean()
            )

        rows.append(
            {
                "variable": variable,
                "matched_rows": len(pair),
                "correlation": correlation,
                "mae": mae,
                "existing_mean": old_mean,
                "new_mean": new_mean,
            }
        )

    comparison = pd.DataFrame(
        rows
    )

    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    return comparison


def main() -> None:
    print("=" * 90)
    print(
        "PEARLS AQI - ISLAMABAD "
        "HISTORICAL BACKFILL TEST"
    )
    print("=" * 90)

    print(
        f"Range: {START_DATE} -> {END_DATE}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    weather = fetch_weather()

    sleep(1)

    air = fetch_air_quality()

    combined = combine_data(
        weather,
        air,
    )

    validate_backfill(
        combined
    )

    combined.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nSaved:\n{OUTPUT_PATH}"
    )

    existing = load_existing()

    comparison = compare_overlap(
        existing,
        combined,
    )

    comparison.to_csv(
        COMPARISON_PATH,
        index=False,
    )

    print(
        f"\nSaved:\n{COMPARISON_PATH}"
    )

    print("\nTest complete.")
    print(
        "Do NOT replace the main historical "
        "dataset yet."
    )


if __name__ == "__main__":
    main()