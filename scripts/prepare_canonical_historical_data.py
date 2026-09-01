from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


SOURCE_PATH = Path(
    "data/historical/openmeteo/"
    "pakistan_aqi_weather_2022_2026.parquet"
)

OUTPUT_DIR = Path(
    "data/historical/canonical"
)

CANONICAL_CSV = (
    OUTPUT_DIR
    / "pakistan_aqi_weather_canonical_2022_2026.csv"
)

CANONICAL_PARQUET = (
    OUTPUT_DIR
    / "pakistan_aqi_weather_canonical_2022_2026.parquet"
)

MODEL_CSV = (
    OUTPUT_DIR
    / "pakistan_aqi_weather_model_input_2022_2026.csv"
)

MODEL_PARQUET = (
    OUTPUT_DIR
    / "pakistan_aqi_weather_model_input_2022_2026.parquet"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "canonical_cleaning_report.json"
)


POLLUTANTS = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "dust",
]


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


EXCLUDED_TRAINING_CITIES = [
    "Rawalpindi",
]


def load_source() -> pd.DataFrame:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Source dataset not found:\n"
            f"{SOURCE_PATH}"
        )

    dataframe = pd.read_parquet(
        SOURCE_PATH
    )

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"],
        utc=True,
    )

    return dataframe


def validate_schema(
    dataframe: pd.DataFrame,
) -> None:
    missing_columns = [
        column
        for column in FINAL_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )

    duplicate_count = int(
        dataframe.duplicated(
            subset=[
                "city",
                "timestamp",
            ]
        ).sum()
    )

    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate "
            "city/timestamp rows."
        )

    missing_count = int(
        dataframe[
            FINAL_COLUMNS
        ].isna().sum().sum()
    )

    if missing_count:
        raise ValueError(
            f"Found {missing_count} missing values."
        )


def clean_negative_pollutants(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    cleaned = dataframe.copy()

    report = {}

    print("\nNegative pollutant corrections:")

    for pollutant in POLLUTANTS:
        mask = (
            cleaned[pollutant] < 0
        )

        count = int(
            mask.sum()
        )

        minimum_before = (
            float(
                cleaned.loc[
                    mask,
                    pollutant,
                ].min()
            )
            if count
            else None
        )

        report[pollutant] = {
            "values_clipped": count,
            "minimum_before": (
                minimum_before
            ),
            "replacement": 0.0,
        }

        if count:
            cleaned.loc[
                mask,
                pollutant,
            ] = 0.0

        print(
            f"  {pollutant:<20} "
            f"{count:>6} values clipped"
        )

    return cleaned, report


def validate_cleaned(
    dataframe: pd.DataFrame,
) -> None:
    remaining_negative = {}

    for pollutant in POLLUTANTS:
        count = int(
            (
                dataframe[pollutant] < 0
            ).sum()
        )

        if count:
            remaining_negative[
                pollutant
            ] = count

    if remaining_negative:
        raise ValueError(
            "Negative pollutant values remain: "
            f"{remaining_negative}"
        )

    missing_count = int(
        dataframe.isna().sum().sum()
    )

    if missing_count:
        raise ValueError(
            f"Cleaned dataset contains "
            f"{missing_count} missing cells."
        )

    duplicate_count = int(
        dataframe.duplicated(
            subset=[
                "city",
                "timestamp",
            ]
        ).sum()
    )

    if duplicate_count:
        raise ValueError(
            f"Cleaned dataset contains "
            f"{duplicate_count} duplicate rows."
        )


def create_model_dataset(
    canonical: pd.DataFrame,
) -> pd.DataFrame:
    modelling = canonical[
        ~canonical["city"].isin(
            EXCLUDED_TRAINING_CITIES
        )
    ].copy()

    modelling = (
        modelling
        .sort_values(
            [
                "city",
                "timestamp",
            ]
        )
        .reset_index(drop=True)
    )

    return modelling


def save_outputs(
    canonical: pd.DataFrame,
    modelling: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\nSaving canonical full dataset..."
    )

    canonical.to_csv(
        CANONICAL_CSV,
        index=False,
    )

    canonical.to_parquet(
        CANONICAL_PARQUET,
        index=False,
    )

    print(
        f"  {CANONICAL_CSV}"
    )

    print(
        f"  {CANONICAL_PARQUET}"
    )

    print(
        "\nSaving model-input dataset..."
    )

    modelling.to_csv(
        MODEL_CSV,
        index=False,
    )

    modelling.to_parquet(
        MODEL_PARQUET,
        index=False,
    )

    print(
        f"  {MODEL_CSV}"
    )

    print(
        f"  {MODEL_PARQUET}"
    )


def main() -> None:
    print("=" * 90)
    print(
        "PEARLS AQI - PREPARE CANONICAL "
        "HISTORICAL DATA"
    )
    print("=" * 90)

    dataframe = load_source()

    print(
        f"\nSource rows: "
        f"{len(dataframe):,}"
    )

    print(
        f"Source cities: "
        f"{dataframe['city'].nunique()}"
    )

    print(
        f"Range: "
        f"{dataframe['timestamp'].min()} "
        f"-> "
        f"{dataframe['timestamp'].max()}"
    )

    validate_schema(
        dataframe
    )

    canonical, correction_report = (
        clean_negative_pollutants(
            dataframe
        )
    )

    canonical = (
        canonical[
            FINAL_COLUMNS
        ]
        .sort_values(
            [
                "city",
                "timestamp",
            ]
        )
        .reset_index(drop=True)
    )

    validate_cleaned(
        canonical
    )

    modelling = create_model_dataset(
        canonical
    )

    expected_model_rows = (
        len(canonical)
        - sum(
            canonical["city"].eq(city).sum()
            for city
            in EXCLUDED_TRAINING_CITIES
        )
    )

    if len(modelling) != expected_model_rows:
        raise RuntimeError(
            "Unexpected model dataset row count."
        )

    save_outputs(
        canonical,
        modelling,
    )

    report = {
        "source_path": str(
            SOURCE_PATH
        ),
        "canonical_rows": int(
            len(canonical)
        ),
        "canonical_cities": int(
            canonical["city"].nunique()
        ),
        "model_rows": int(
            len(modelling)
        ),
        "model_cities": int(
            modelling["city"].nunique()
        ),
        "excluded_training_cities": (
            EXCLUDED_TRAINING_CITIES
        ),
        "date_range": {
            "start": str(
                canonical[
                    "timestamp"
                ].min()
            ),
            "end": str(
                canonical[
                    "timestamp"
                ].max()
            ),
        },
        "negative_value_corrections": (
            correction_report
        ),
        "extreme_values_policy": (
            "Retained. High pollutant values "
            "were not clipped or winsorized."
        ),
        "raw_source_modified": False,
    }

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    print("\n" + "=" * 90)
    print("CANONICAL DATASET READY")
    print("=" * 90)

    print(
        f"\nCanonical rows: "
        f"{len(canonical):,}"
    )

    print(
        f"Canonical cities: "
        f"{canonical['city'].nunique()}"
    )

    print(
        f"\nModel-input rows: "
        f"{len(modelling):,}"
    )

    print(
        f"Model-input cities: "
        f"{modelling['city'].nunique()}"
    )

    print(
        "\nExcluded from model training:"
    )

    for city in EXCLUDED_TRAINING_CITIES:
        print(
            f"  - {city}"
        )

    print(
        "\nRemaining negative pollutant "
        f"values: "
        f"{int((canonical[POLLUTANTS] < 0).sum().sum())}"
    )

    print(
        "\nMissing cells: "
        f"{int(canonical.isna().sum().sum())}"
    )

    print(
        "\nDuplicate city/timestamps: "
        f"{int(canonical.duplicated(['city', 'timestamp']).sum())}"
    )

    print(
        f"\nCleaning report:\n"
        f"{REPORT_PATH}"
    )


if __name__ == "__main__":
    main()