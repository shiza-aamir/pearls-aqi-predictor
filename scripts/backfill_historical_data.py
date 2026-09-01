from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests


START_DATE = date(2022, 8, 4)

# Initially use a conservative completed historical cutoff.
# We will later extend this through the normal incremental ingestion pipeline.
END_DATE = date(2026, 8, 28)

WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

BASE_DIR = Path("data/historical/openmeteo")
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"

MASTER_CSV = BASE_DIR / "pakistan_aqi_weather_2022_2026.csv"
MASTER_PARQUET = BASE_DIR / "pakistan_aqi_weather_2022_2026.parquet"
REPORT_PATH = BASE_DIR / "backfill_report.json"

REQUEST_TIMEOUT = 120
MAX_RETRIES = 5
REQUEST_DELAY_SECONDS = 1.0


CITIES = {
    "Faisalabad": (31.4504, 73.1350),
    "Islamabad": (33.6844, 73.0479),
    "Karachi": (24.8607, 67.0011),
    "Lahore": (31.5204, 74.3587),
    "Multan": (30.1575, 71.5249),
    "Peshawar": (34.0151, 71.5249),
    "Quetta": (30.1798, 66.9750),
    "Rahim Yar Khan": (28.4212, 70.2989),
    "Rawalpindi": (33.5651, 73.0169),
    "Sialkot": (32.4945, 74.5229),
}


WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "pressure_msl",
]


AIR_QUALITY_VARIABLES = [
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
    "pressure_msl": "pressure",
}


FINAL_COLUMNS = [
    "timestamp",
    "city",
    "latitude",
    "longitude",
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "dust",
    "temperature",
    "humidity",
    "precipitation",
    "wind_speed",
    "wind_direction",
    "pressure",
]


def slugify(city: str) -> str:
    return (
        city.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def yearly_chunks(
    start: date,
    end: date,
):
    current = start

    while current <= end:
        chunk_end = min(
            date(current.year, 12, 31),
            end,
        )

        yield current, chunk_end

        current = chunk_end + timedelta(days=1)


def request_json(
    url: str,
    params: dict,
) -> dict:
    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

            response.raise_for_status()

            payload = response.json()

            if payload.get("error"):
                raise RuntimeError(
                    payload.get(
                        "reason",
                        str(payload),
                    )
                )

            time.sleep(
                REQUEST_DELAY_SECONDS
            )

            return payload

        except Exception as error:
            last_error = error

            if attempt == MAX_RETRIES:
                break

            wait_seconds = min(
                2 ** attempt,
                30,
            )

            print(
                f"    Request failed "
                f"(attempt {attempt}/"
                f"{MAX_RETRIES}): "
                f"{error}"
            )

            print(
                f"    Retrying in "
                f"{wait_seconds}s..."
            )

            time.sleep(
                wait_seconds
            )

    raise RuntimeError(
        f"API request failed after "
        f"{MAX_RETRIES} attempts: "
        f"{last_error}"
    )


def payload_to_dataframe(
    payload: dict,
) -> pd.DataFrame:
    hourly = payload.get("hourly")

    if not hourly:
        raise RuntimeError(
            "API response did not contain "
            "hourly data."
        )

    dataframe = pd.DataFrame(
        hourly
    )

    if "time" not in dataframe.columns:
        raise RuntimeError(
            "Hourly response has no "
            "'time' column."
        )

    dataframe["timestamp"] = pd.to_datetime(
        dataframe.pop("time"),
        utc=True,
    )

    return dataframe


def download_weather_chunk(
    city: str,
    latitude: float,
    longitude: float,
    start: date,
    end: date,
) -> pd.DataFrame:
    payload = request_json(
        WEATHER_URL,
        {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "hourly": ",".join(
                WEATHER_VARIABLES
            ),
            "timezone": "GMT",
            "wind_speed_unit": "kmh",
        },
    )

    dataframe = payload_to_dataframe(
        payload
    )

    dataframe = dataframe.rename(
        columns=WEATHER_RENAME
    )

    return dataframe


def download_air_quality_chunk(
    city: str,
    latitude: float,
    longitude: float,
    start: date,
    end: date,
) -> pd.DataFrame:
    payload = request_json(
        AIR_QUALITY_URL,
        {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "hourly": ",".join(
                AIR_QUALITY_VARIABLES
            ),
            "timezone": "GMT",
            "domains": "cams_global",
        },
    )

    return payload_to_dataframe(
        payload
    )


def load_or_download_chunk(
    kind: str,
    city: str,
    latitude: float,
    longitude: float,
    start: date,
    end: date,
) -> pd.DataFrame:
    city_slug = slugify(city)

    city_raw_dir = (
        RAW_DIR
        / city_slug
    )

    city_raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        f"{kind}_"
        f"{start.isoformat()}_"
        f"{end.isoformat()}.csv"
    )

    path = (
        city_raw_dir
        / filename
    )

    if path.exists():
        print(
            f"    Cache hit: {path.name}"
        )

        dataframe = pd.read_csv(
            path
        )

        dataframe["timestamp"] = (
            pd.to_datetime(
                dataframe["timestamp"],
                utc=True,
            )
        )

        return dataframe

    print(
        f"    Downloading {kind}: "
        f"{start} -> {end}"
    )

    if kind == "weather":
        dataframe = (
            download_weather_chunk(
                city,
                latitude,
                longitude,
                start,
                end,
            )
        )

    elif kind == "air_quality":
        dataframe = (
            download_air_quality_chunk(
                city,
                latitude,
                longitude,
                start,
                end,
            )
        )

    else:
        raise ValueError(
            f"Unknown data kind: {kind}"
        )

    dataframe.to_csv(
        path,
        index=False,
    )

    return dataframe


def download_city(
    city: str,
    latitude: float,
    longitude: float,
) -> pd.DataFrame:
    print("\n" + "=" * 90)
    print(f"CITY: {city}")
    print("=" * 90)

    weather_frames = []
    air_frames = []

    for chunk_start, chunk_end in yearly_chunks(
        START_DATE,
        END_DATE,
    ):
        weather = load_or_download_chunk(
            kind="weather",
            city=city,
            latitude=latitude,
            longitude=longitude,
            start=chunk_start,
            end=chunk_end,
        )

        air = load_or_download_chunk(
            kind="air_quality",
            city=city,
            latitude=latitude,
            longitude=longitude,
            start=chunk_start,
            end=chunk_end,
        )

        weather_frames.append(
            weather
        )

        air_frames.append(
            air
        )

    weather = pd.concat(
        weather_frames,
        ignore_index=True,
    )

    air = pd.concat(
        air_frames,
        ignore_index=True,
    )

    weather = (
        weather
        .drop_duplicates(
            subset=["timestamp"]
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    air = (
        air
        .drop_duplicates(
            subset=["timestamp"]
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    combined = weather.merge(
        air,
        on="timestamp",
        how="inner",
        validate="one_to_one",
    )

    combined.insert(
        1,
        "city",
        city,
    )

    combined.insert(
        2,
        "latitude",
        latitude,
    )

    combined.insert(
        3,
        "longitude",
        longitude,
    )

    combined = combined[
        FINAL_COLUMNS
    ]

    return combined


def validate_city(
    city: str,
    dataframe: pd.DataFrame,
) -> dict:
    dataframe = (
        dataframe
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    differences = (
        dataframe["timestamp"]
        .diff()
        .dropna()
    )

    non_hourly_gaps = int(
        (
            differences
            != pd.Timedelta(hours=1)
        ).sum()
    )

    missing = {
        column: int(value)
        for column, value
        in dataframe.isna().sum().items()
        if value > 0
    }

    duplicates = int(
        dataframe[
            "timestamp"
        ]
        .duplicated()
        .sum()
    )

    report = {
        "city": city,
        "rows": int(
            len(dataframe)
        ),
        "start": str(
            dataframe[
                "timestamp"
            ].min()
        ),
        "end": str(
            dataframe[
                "timestamp"
            ].max()
        ),
        "duplicate_timestamps": (
            duplicates
        ),
        "non_hourly_gaps": (
            non_hourly_gaps
        ),
        "missing_values": missing,
    }

    print(
        f"\n  Rows: "
        f"{report['rows']:,}"
    )

    print(
        f"  Range: "
        f"{report['start']} "
        f"-> {report['end']}"
    )

    print(
        f"  Duplicate timestamps: "
        f"{duplicates}"
    )

    print(
        f"  Non-hourly gaps: "
        f"{non_hourly_gaps}"
    )

    if missing:
        print(
            f"  Missing values: "
            f"{missing}"
        )
    else:
        print(
            "  Missing values: 0"
        )

    return report


def save_city_processed(
    city: str,
    dataframe: pd.DataFrame,
) -> None:
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        PROCESSED_DIR
        / f"{slugify(city)}.parquet"
    )

    dataframe.to_parquet(
        path,
        index=False,
    )

    print(
        f"  Saved: {path}"
    )


def build_master(
    city_frames: list[pd.DataFrame],
) -> pd.DataFrame:
    master = pd.concat(
        city_frames,
        ignore_index=True,
    )

    master = (
        master
        .sort_values(
            [
                "city",
                "timestamp",
            ]
        )
        .reset_index(drop=True)
    )

    duplicates = int(
        master.duplicated(
            subset=[
                "city",
                "timestamp",
            ]
        ).sum()
    )

    if duplicates:
        raise RuntimeError(
            f"Master dataset contains "
            f"{duplicates} duplicate "
            f"city/timestamp rows."
        )

    return master


def save_master(
    master: pd.DataFrame,
) -> None:
    BASE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\nSaving master CSV..."
    )

    master.to_csv(
        MASTER_CSV,
        index=False,
    )

    print(
        f"Saved: {MASTER_CSV}"
    )

    print(
        "\nSaving master Parquet..."
    )

    master.to_parquet(
        MASTER_PARQUET,
        index=False,
    )

    print(
        f"Saved: {MASTER_PARQUET}"
    )


def print_master_summary(
    master: pd.DataFrame,
) -> None:
    print("\n" + "=" * 90)
    print("MASTER DATASET SUMMARY")
    print("=" * 90)

    print(
        f"Rows: {len(master):,}"
    )

    print(
        f"Columns: {len(master.columns)}"
    )

    print(
        f"Cities: "
        f"{master['city'].nunique()}"
    )

    print(
        f"Start: "
        f"{master['timestamp'].min()}"
    )

    print(
        f"End: "
        f"{master['timestamp'].max()}"
    )

    print(
        "\nRows per city:"
    )

    print(
        master[
            "city"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nMissing values:"
    )

    print(
        master
        .isna()
        .sum()
        .to_string()
    )

    print(
        "\nNumeric statistics:"
    )

    numeric_columns = [
        column
        for column in FINAL_COLUMNS
        if column not in {
            "timestamp",
            "city",
        }
    ]

    print(
        master[
            numeric_columns
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


def main() -> None:
    print("=" * 90)
    print(
        "PEARLS AQI - FULL HISTORICAL "
        "BACKFILL"
    )
    print("=" * 90)

    print(
        f"Range: "
        f"{START_DATE} -> {END_DATE}"
    )

    print(
        f"Cities: {len(CITIES)}"
    )

    print(
        "\nWeather configuration:"
    )

    print(
        "  Pressure: pressure_msl"
    )

    print(
        "  Wind speed: km/h"
    )

    print(
        "  Timezone: UTC/GMT"
    )

    print(
        "\nAir-quality configuration:"
    )

    print(
        "  Domain: CAMS Global"
    )

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    city_frames = []
    reports = []

    for city, coordinates in CITIES.items():
        latitude, longitude = (
            coordinates
        )

        dataframe = download_city(
            city,
            latitude,
            longitude,
        )

        report = validate_city(
            city,
            dataframe,
        )

        save_city_processed(
            city,
            dataframe,
        )

        city_frames.append(
            dataframe
        )

        reports.append(
            report
        )

    master = build_master(
        city_frames
    )

    save_master(
        master
    )

    print_master_summary(
        master
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "start_date": (
                    START_DATE.isoformat()
                ),
                "end_date": (
                    END_DATE.isoformat()
                ),
                "weather_source": (
                    "Open-Meteo Historical "
                    "Weather API"
                ),
                "air_quality_source": (
                    "Open-Meteo Air Quality "
                    "API / CAMS Global"
                ),
                "weather_pressure": (
                    "pressure_msl"
                ),
                "wind_speed_unit": (
                    "km/h"
                ),
                "timezone": "UTC",
                "cities": reports,
                "master_rows": int(
                    len(master)
                ),
            },
            file,
            indent=2,
        )

    print(
        f"\nSaved report: "
        f"{REPORT_PATH}"
    )

    print("\n" + "=" * 90)
    print("BACKFILL COMPLETE")
    print("=" * 90)

    print(
        "\nDo NOT replace the production "
        "training dataset yet."
    )

    print(
        "Review this backfill report and "
        "cross-city consistency first."
    )


if __name__ == "__main__":
    main()