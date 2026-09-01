from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.features.engineer import AQIFeatureEngineer


INPUT_PATH = Path(
    "data/processed/expanded/"
    "historical_with_aqi_targets.parquet"
)

OUTPUT_DIR = Path(
    "data/processed/expanded"
)

OUTPUT_PARQUET = (
    OUTPUT_DIR
    / "ml_ready_aqi_dataset.parquet"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "ml_ready_aqi_dataset.csv"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "ml_dataset_report.json"
)

TARGET_COLUMNS = [
    "target_aqi_24h",
    "target_aqi_48h",
    "target_aqi_72h",
]


def load_data() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input dataset not found:\n{INPUT_PATH}"
        )

    dataframe = pd.read_parquet(
        INPUT_PATH
    )

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"],
        utc=True,
    )

    dataframe = (
        dataframe
        .sort_values(
            [
                "city",
                "timestamp",
            ]
        )
        .reset_index(drop=True)
    )

    return dataframe


def prepare_engineer_input(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    if "aqi" not in dataframe.columns:
        raise ValueError(
            "Expected source column 'aqi' "
            "was not found."
        )

    dataframe["aqi_current"] = (
        dataframe["aqi"]
        .astype("Float64")
    )

    return dataframe


def validate_expected_features(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    missing = [
        column
        for column in feature_columns
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            "Feature engineer did not create "
            f"expected columns: {missing}"
        )

    if len(feature_columns) != 56:
        raise ValueError(
            "Expected exactly 56 model "
            f"features, got {len(feature_columns)}."
        )

    duplicates = [
        column
        for column in feature_columns
        if feature_columns.count(column) > 1
    ]

    if duplicates:
        raise ValueError(
            "Duplicate model feature names: "
            f"{sorted(set(duplicates))}"
        )


def build_complete_rows(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    required = (
        feature_columns
        + TARGET_COLUMNS
    )

    complete_mask = (
        dataframe[
            required
        ]
        .notna()
        .all(axis=1)
    )

    result = (
        dataframe.loc[
            complete_mask
        ]
        .copy()
        .sort_values(
            [
                "city",
                "timestamp",
            ]
        )
        .reset_index(drop=True)
    )

    return result


def validate_city_boundaries(
    source: pd.DataFrame,
    engineered: pd.DataFrame,
) -> None:
    print(
        "\nChecking 72-hour city warm-up..."
    )

    for city in sorted(
        source["city"].unique()
    ):
        source_city = (
            source[
                source["city"] == city
            ]
            .sort_values("timestamp")
        )

        engineered_city = (
            engineered[
                engineered["city"] == city
            ]
            .sort_values("timestamp")
        )

        if engineered_city.empty:
            raise AssertionError(
                f"{city}: no usable ML rows."
            )

        source_start = (
            source_city["timestamp"].min()
        )

        first_usable = (
            engineered_city["timestamp"].min()
        )

        expected_earliest = (
            source_start
            + pd.Timedelta(hours=72)
        )

        if first_usable < expected_earliest:
            raise AssertionError(
                f"{city}: first usable row "
                f"{first_usable} appears before "
                f"72h lag warm-up "
                f"{expected_earliest}."
            )

        print(
            f"  {city:<18} "
            f"first usable: "
            f"{first_usable}"
        )

    print(
        "City feature-boundary check: PASS"
    )


def validate_feature_nulls(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    missing_features = int(
        dataframe[
            feature_columns
        ]
        .isna()
        .sum()
        .sum()
    )

    missing_targets = int(
        dataframe[
            TARGET_COLUMNS
        ]
        .isna()
        .sum()
        .sum()
    )

    if missing_features:
        raise AssertionError(
            "ML-ready dataset still contains "
            f"{missing_features} missing "
            "feature cells."
        )

    if missing_targets:
        raise AssertionError(
            "ML-ready dataset still contains "
            f"{missing_targets} missing "
            "target cells."
        )

    print(
        "\nMissing model feature cells: 0"
    )

    print(
        "Missing target cells: 0"
    )


def summarize_by_city(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        dataframe
        .groupby("city")
        .agg(
            rows=(
                "timestamp",
                "size",
            ),
            start=(
                "timestamp",
                "min",
            ),
            end=(
                "timestamp",
                "max",
            ),
        )
    )

    print(
        "\nROWS BY CITY"
    )

    print(
        summary.to_string()
    )

    return summary


def summarize_by_year(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    temp = dataframe.copy()

    temp["year"] = (
        temp["timestamp"].dt.year
    )

    summary = (
        temp
        .groupby("year")
        .agg(
            rows=(
                "timestamp",
                "size",
            ),
            cities=(
                "city",
                "nunique",
            ),
        )
    )

    print(
        "\nROWS BY YEAR"
    )

    print(
        summary.to_string()
    )

    return summary


def main() -> None:
    print("=" * 90)
    print(
        "PEARLS AQI - EXPANDED "
        "ML DATASET BUILDER"
    )
    print("=" * 90)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    source = load_data()

    print(
        f"\nSource rows: "
        f"{len(source):,}"
    )

    print(
        f"Cities: "
        f"{source['city'].nunique()}"
    )

    print(
        f"Range: "
        f"{source['timestamp'].min()} "
        f"-> "
        f"{source['timestamp'].max()}"
    )

    engineer_input = (
        prepare_engineer_input(
            source
        )
    )

    engineer = (
        AQIFeatureEngineer()
    )

    print(
        "\nRunning existing "
        "AQIFeatureEngineer..."
    )

    engineered, summary = (
        engineer.transform(
            engineer_input
        )
    )

    feature_columns = (
        engineer.get_model_feature_columns()
    )

    validate_expected_features(
        engineered,
        feature_columns,
    )

    print(
        f"\nFeature engineer input rows: "
        f"{summary.input_rows:,}"
    )

    print(
        f"Feature engineer output rows: "
        f"{summary.output_rows:,}"
    )

    print(
        f"Cities: "
        f"{summary.cities}"
    )

    print(
        f"Model feature count: "
        f"{summary.feature_count}"
    )

    complete = build_complete_rows(
        engineered,
        feature_columns,
    )

    validate_city_boundaries(
        source,
        complete,
    )

    validate_feature_nulls(
        complete,
        feature_columns,
    )

    city_summary = (
        summarize_by_city(
            complete
        )
    )

    year_summary = (
        summarize_by_year(
            complete
        )
    )

    print(
        "\nSaving ML-ready dataset..."
    )

    complete.to_parquet(
        OUTPUT_PARQUET,
        index=False,
    )

    complete.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    report = {
        "input_path": str(
            INPUT_PATH
        ),
        "input_rows": int(
            len(source)
        ),
        "engineered_rows": int(
            len(engineered)
        ),
        "ml_ready_rows": int(
            len(complete)
        ),
        "cities": int(
            complete["city"]
            .nunique()
        ),
        "feature_count": int(
            len(feature_columns)
        ),
        "feature_columns": (
            feature_columns
        ),
        "target_columns": (
            TARGET_COLUMNS
        ),
        "start": str(
            complete[
                "timestamp"
            ].min()
        ),
        "end": str(
            complete[
                "timestamp"
            ].max()
        ),
        "rows_by_city": {
            city: int(
                row["rows"]
            )
            for city, row
            in city_summary.iterrows()
        },
        "rows_by_year": {
            str(year): int(
                row["rows"]
            )
            for year, row
            in year_summary.iterrows()
        },
        "rawalpindi_excluded": True,
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
    print("ML DATASET BUILD COMPLETE")
    print("=" * 90)

    print(
        f"\nML-ready rows: "
        f"{len(complete):,}"
    )

    print(
        f"Cities: "
        f"{complete['city'].nunique()}"
    )

    print(
        f"Features: "
        f"{len(feature_columns)}"
    )

    print(
        f"Range: "
        f"{complete['timestamp'].min()} "
        f"-> "
        f"{complete['timestamp'].max()}"
    )

    print(
        f"\nSaved:\n"
        f"{OUTPUT_PARQUET}"
    )

    print(
        f"\nReport:\n"
        f"{REPORT_PATH}"
    )


if __name__ == "__main__":
    main()