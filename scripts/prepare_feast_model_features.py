from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.features.engineer import AQIFeatureEngineer


INPUT_PATH = Path(
    "data/processed/expanded/ml_ready_aqi_dataset.parquet"
)

OUTPUT_PATH = Path(
    "feature_repo/feature_repo/data/model_features.parquet"
)

REPORT_PATH = Path(
    "artifacts/feature_store/expanded_feast_dataset_report.json"
)

EXPECTED_ROWS = 319_473
EXPECTED_CITIES = 9
EXPECTED_FEATURES = 56

EXPECTED_START = pd.Timestamp(
    "2022-08-07 23:00:00+00:00"
)

EXPECTED_END = pd.Timestamp(
    "2026-08-25 23:00:00+00:00"
)


def main() -> None:
    print("=" * 80)
    print(
        "PEARLS AQI - PREPARE EXPANDED FEAST FEATURES"
    )
    print("=" * 80)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Expanded ML-ready dataset not found: "
            f"{INPUT_PATH}"
        )

    df = pd.read_parquet(
        INPUT_PATH
    )

    feature_columns = (
        AQIFeatureEngineer
        .get_model_feature_columns()
    )

    if len(feature_columns) != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_FEATURES} features, "
            f"got {len(feature_columns)}."
        )

    required_columns = [
        "timestamp",
        "city",
        *feature_columns,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Expanded ML dataset is missing "
            f"required columns: {missing_columns}"
        )

    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} rows, "
            f"got {len(df):,}."
        )

    city_count = int(
        df["city"].nunique()
    )

    if city_count != EXPECTED_CITIES:
        raise ValueError(
            f"Expected {EXPECTED_CITIES} cities, "
            f"got {city_count}."
        )

    cities = sorted(
        df["city"]
        .astype(str)
        .unique()
        .tolist()
    )

    if "Rawalpindi" in cities:
        raise ValueError(
            "Rawalpindi must not be present in "
            "the expanded model feature dataset."
        )

    feast_df = df[
        required_columns
    ].copy()

    feast_df = feast_df.rename(
        columns={
            "city": "city_id",
            "timestamp": "event_timestamp",
        }
    )

    feast_df[
        "event_timestamp"
    ] = pd.to_datetime(
        feast_df["event_timestamp"],
        utc=True,
        errors="raise",
    )

    start_timestamp = (
        feast_df[
            "event_timestamp"
        ].min()
    )

    end_timestamp = (
        feast_df[
            "event_timestamp"
        ].max()
    )

    if start_timestamp != EXPECTED_START:
        raise ValueError(
            "Unexpected expanded Feast start. "
            f"Expected {EXPECTED_START}, "
            f"got {start_timestamp}."
        )

    if end_timestamp != EXPECTED_END:
        raise ValueError(
            "Unexpected expanded Feast end. "
            f"Expected {EXPECTED_END}, "
            f"got {end_timestamp}."
        )

    feast_df["created"] = (
        feast_df["event_timestamp"]
    )

    feast_df["city_id"] = (
        feast_df["city_id"]
        .astype(str)
    )

    for column in feature_columns:
        feast_df[column] = (
            pd.to_numeric(
                feast_df[column],
                errors="raise",
            )
            .astype("float64")
        )

    null_counts = (
        feast_df[
            feature_columns
        ]
        .isnull()
        .sum()
    )

    columns_with_nulls = (
        null_counts[
            null_counts > 0
        ]
        .index
        .tolist()
    )

    if columns_with_nulls:
        raise ValueError(
            "Null values found in model features: "
            f"{columns_with_nulls}"
        )

    duplicate_count = int(
        feast_df.duplicated(
            subset=[
                "city_id",
                "event_timestamp",
            ]
        ).sum()
    )

    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate "
            "city/timestamp rows."
        )

    rows_per_city = (
        feast_df
        .groupby("city_id")
        .size()
        .sort_index()
    )

    if rows_per_city.nunique() != 1:
        raise ValueError(
            "Cities do not contain an equal "
            "number of Feast feature rows."
        )

    expected_rows_per_city = (
        EXPECTED_ROWS
        // EXPECTED_CITIES
    )

    if not (
        rows_per_city
        == expected_rows_per_city
    ).all():
        raise ValueError(
            "Unexpected rows per city. "
            f"Expected {expected_rows_per_city:,}."
        )

    feast_df = (
        feast_df
        .sort_values(
            [
                "city_id",
                "event_timestamp",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    feast_df.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    report = {
        "dataset_type": (
            "EXPANDED_FEAST_MODEL_FEATURES"
        ),
        "source": str(INPUT_PATH),
        "destination": str(OUTPUT_PATH),
        "rows": int(len(feast_df)),
        "cities": city_count,
        "city_names": cities,
        "rows_per_city": {
            city: int(count)
            for city, count
            in rows_per_city.items()
        },
        "feature_count": len(
            feature_columns
        ),
        "feature_columns": (
            feature_columns
        ),
        "missing_feature_cells": int(
            feast_df[
                feature_columns
            ]
            .isnull()
            .sum()
            .sum()
        ),
        "duplicate_entity_timestamp_rows": (
            duplicate_count
        ),
        "start": str(
            start_timestamp
        ),
        "end": str(
            end_timestamp
        ),
        "rawalpindi_excluded": (
            "Rawalpindi"
            not in cities
        ),
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

    print("\nValidation:")
    print(
        f"  Source rows:       {len(df):,}"
    )
    print(
        f"  Feast rows:        {len(feast_df):,}"
    )
    print(
        f"  Cities:            {city_count}"
    )
    print(
        f"  Rows/city:         "
        f"{expected_rows_per_city:,}"
    )
    print(
        f"  Features:          "
        f"{len(feature_columns)}"
    )
    print(
        "  Missing features:  0"
    )
    print(
        "  Duplicate keys:    0"
    )
    print(
        f"  Start:             {start_timestamp}"
    )
    print(
        f"  End:               {end_timestamp}"
    )
    print(
        "  Rawalpindi:        excluded"
    )

    print(
        f"\nSaved:\n  {OUTPUT_PATH}"
    )

    print(
        f"  {REPORT_PATH}"
    )

    print("\n" + "=" * 80)
    print(
        "EXPANDED FEAST FEATURE DATASET: PASS"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()