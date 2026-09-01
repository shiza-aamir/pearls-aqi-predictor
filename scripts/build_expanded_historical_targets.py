from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.aqi.calculator import AQICalculator


INPUT_PATH = Path(
    "data/historical/canonical/"
    "pakistan_aqi_weather_model_input_2022_2026.parquet"
)

OUTPUT_DIR = Path(
    "data/processed/expanded"
)

OUTPUT_PARQUET = (
    OUTPUT_DIR
    / "historical_with_aqi_targets.parquet"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "historical_with_aqi_targets.csv"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "target_build_report.json"
)

HORIZONS = [24, 48, 72]


def load_data() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input dataset not found:\n{INPUT_PATH}"
        )

    dataframe = pd.read_parquet(INPUT_PATH)

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"],
        utc=True,
    )

    dataframe = (
        dataframe
        .sort_values(
            ["city", "timestamp"]
        )
        .reset_index(drop=True)
    )

    return dataframe


def validate_hourly_continuity(
    dataframe: pd.DataFrame,
) -> None:
    problems = []

    for city, group in dataframe.groupby(
        "city",
        sort=False,
    ):
        differences = (
            group["timestamp"]
            .diff()
            .dropna()
        )

        invalid = int(
            (
                differences
                != pd.Timedelta(hours=1)
            ).sum()
        )

        if invalid:
            problems.append(
                {
                    "city": city,
                    "non_hourly_gaps": invalid,
                }
            )

    if problems:
        raise ValueError(
            f"Hourly continuity failure: {problems}"
        )

    print("Hourly continuity: PASS")


def build_24h_particle_averages(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    print(
        "\nCalculating trailing 24-hour "
        "PM2.5 and PM10 averages..."
    )

    dataframe["pm2_5_24h_avg"] = (
        dataframe
        .groupby(
            "city",
            sort=False,
        )["pm2_5"]
        .transform(
            lambda series: (
                series
                .rolling(
                    window=24,
                    min_periods=24,
                )
                .mean()
            )
        )
    )

    dataframe["pm10_24h_avg"] = (
        dataframe
        .groupby(
            "city",
            sort=False,
        )["pm10"]
        .transform(
            lambda series: (
                series
                .rolling(
                    window=24,
                    min_periods=24,
                )
                .mean()
            )
        )
    )

    return dataframe


def build_current_aqi(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    valid_mask = (
        dataframe[
            [
                "pm2_5_24h_avg",
                "pm10_24h_avg",
            ]
        ]
        .notna()
        .all(axis=1)
    )

    dataframe["pm2_5_aqi"] = np.nan
    dataframe["pm10_aqi"] = np.nan
    dataframe["aqi"] = np.nan
    dataframe["aqi_category"] = None
    dataframe["dominant_pollutant"] = None

    print(
        "Calculating EPA particulate AQI..."
    )

    valid_indices = dataframe.index[
        valid_mask
    ]

    for index in valid_indices:
        pm25 = float(
            dataframe.at[
                index,
                "pm2_5_24h_avg",
            ]
        )

        pm10 = float(
            dataframe.at[
                index,
                "pm10_24h_avg",
            ]
        )

        result = (
            AQICalculator.calculate_aqi(
                pm25=pm25,
                pm10=pm10,
            )
        )

        dataframe.at[
            index,
            "pm2_5_aqi",
        ] = result.pm25_aqi

        dataframe.at[
            index,
            "pm10_aqi",
        ] = result.pm10_aqi

        dataframe.at[
            index,
            "aqi",
        ] = result.aqi

        dataframe.at[
            index,
            "aqi_category",
        ] = result.category

        dataframe.at[
            index,
            "dominant_pollutant",
        ] = result.dominant_pollutant

    dataframe["pm2_5_aqi"] = (
        dataframe["pm2_5_aqi"]
        .astype("Float64")
    )

    dataframe["pm10_aqi"] = (
        dataframe["pm10_aqi"]
        .astype("Float64")
    )

    dataframe["aqi"] = (
        dataframe["aqi"]
        .astype("Float64")
    )

    return dataframe


def build_future_targets(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    print(
        "\nBuilding direct future AQI targets..."
    )

    for horizon in HORIZONS:
        target_column = (
            f"target_aqi_{horizon}h"
        )

        dataframe[target_column] = (
            dataframe
            .groupby(
                "city",
                sort=False,
            )["aqi"]
            .shift(-horizon)
            .astype("Float64")
        )

        print(
            f"  {target_column}"
        )

    return dataframe


def validate_city_boundaries(
    dataframe: pd.DataFrame,
) -> None:
    print(
        "\nChecking city-boundary leakage..."
    )

    for city, group in dataframe.groupby(
        "city",
        sort=False,
    ):
        group = (
            group
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        for horizon in HORIZONS:
            target_column = (
                f"target_aqi_{horizon}h"
            )

            tail = group.tail(horizon)

            if tail[
                target_column
            ].notna().any():
                raise AssertionError(
                    f"{city}: "
                    f"{target_column} leaks "
                    "across city boundary."
                )

    print(
        "City-boundary target check: PASS"
    )


def validate_target_alignment(
    dataframe: pd.DataFrame,
) -> None:
    print(
        "Checking exact target alignment..."
    )

    lookup = (
        dataframe[
            [
                "city",
                "timestamp",
                "aqi",
            ]
        ]
        .set_index(
            [
                "city",
                "timestamp",
            ]
        )["aqi"]
    )

    for horizon in HORIZONS:
        target_column = (
            f"target_aqi_{horizon}h"
        )

        candidates = dataframe[
            dataframe[
                target_column
            ].notna()
        ]

        sample_size = min(
            1000,
            len(candidates),
        )

        sample = candidates.sample(
            n=sample_size,
            random_state=42,
        )

        for row in sample.itertuples():
            future_timestamp = (
                row.timestamp
                + pd.Timedelta(
                    hours=horizon
                )
            )

            expected = lookup.loc[
                (
                    row.city,
                    future_timestamp,
                )
            ]

            actual = getattr(
                row,
                target_column,
            )

            if not np.isclose(
                float(actual),
                float(expected),
                rtol=0.0,
                atol=1e-9,
            ):
                raise AssertionError(
                    "Target alignment failed: "
                    f"{row.city}, "
                    f"{row.timestamp}, "
                    f"{horizon}h, "
                    f"actual={actual}, "
                    f"expected={expected}"
                )

    print(
        "Exact target alignment: PASS"
    )


def validate_initial_aqi_window(
    dataframe: pd.DataFrame,
) -> None:
    print(
        "Checking initial 24-hour AQI window..."
    )

    for city, group in dataframe.groupby(
        "city",
        sort=False,
    ):
        group = (
            group
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        first_23 = group.iloc[:23]

        if first_23["aqi"].notna().any():
            raise AssertionError(
                f"{city}: AQI exists before "
                "24 observations were available."
            )

        if pd.isna(
            group.iloc[23]["aqi"]
        ):
            raise AssertionError(
                f"{city}: AQI missing at the "
                "first valid 24-hour position."
            )

    print(
        "Initial rolling-window check: PASS"
    )


def print_aqi_distribution(
    dataframe: pd.DataFrame,
) -> None:
    print(
        "\nAQI DISTRIBUTION"
    )

    print(
        dataframe["aqi"]
        .dropna()
        .astype(float)
        .describe(
            percentiles=[
                0.01,
                0.05,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
        .round(3)
        .to_string()
    )


def print_category_distribution(
    dataframe: pd.DataFrame,
) -> None:
    print(
        "\nAQI CATEGORY DISTRIBUTION"
    )

    counts = (
        dataframe[
            "aqi_category"
        ]
        .value_counts()
    )

    percentages = (
        dataframe[
            "aqi_category"
        ]
        .value_counts(
            normalize=True
        )
        * 100
    )

    result = pd.DataFrame(
        {
            "rows": counts,
            "percentage": percentages,
        }
    )

    print(
        result.round(3).to_string()
    )


def main() -> None:
    print("=" * 90)
    print(
        "PEARLS AQI - EXPANDED "
        "HISTORICAL TARGET BUILDER"
    )
    print("=" * 90)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = load_data()

    print(
        f"\nInput rows: "
        f"{len(dataframe):,}"
    )

    print(
        f"Cities: "
        f"{dataframe['city'].nunique()}"
    )

    print(
        f"Range: "
        f"{dataframe['timestamp'].min()} "
        f"-> "
        f"{dataframe['timestamp'].max()}"
    )

    validate_hourly_continuity(
        dataframe
    )

    dataframe = (
        build_24h_particle_averages(
            dataframe
        )
    )

    dataframe = build_current_aqi(
        dataframe
    )

    dataframe = build_future_targets(
        dataframe
    )

    validate_initial_aqi_window(
        dataframe
    )

    validate_city_boundaries(
        dataframe
    )

    validate_target_alignment(
        dataframe
    )

    print_aqi_distribution(
        dataframe
    )

    print_category_distribution(
        dataframe
    )

    current_aqi_mask = (
        dataframe["aqi"].notna()
    )

    targets_only_mask = (
        dataframe[
            [
                "target_aqi_24h",
                "target_aqi_48h",
                "target_aqi_72h",
            ]
        ]
        .notna()
        .all(axis=1)
    )

    complete_model_base_mask = (
        current_aqi_mask
        & targets_only_mask
    )

    current_aqi_count = int(
        current_aqi_mask.sum()
    )

    targets_only_count = int(
        targets_only_mask.sum()
    )

    complete_model_base_count = int(
        complete_model_base_mask.sum()
    )

    target_missing = {
        f"{horizon}h": int(
            dataframe[
                f"target_aqi_{horizon}h"
            ]
            .isna()
            .sum()
        )
        for horizon in HORIZONS
    }

    report = {
        "input_rows": int(
            len(dataframe)
        ),
        "cities": int(
            dataframe["city"].nunique()
        ),
        "start": str(
            dataframe["timestamp"].min()
        ),
        "end": str(
            dataframe["timestamp"].max()
        ),
        "current_aqi_rows": (
            current_aqi_count
        ),
        "rows_with_all_future_targets": (
            targets_only_count
        ),
        "rows_with_current_aqi_and_all_targets": (
            complete_model_base_count
        ),
        "rows_without_current_aqi": int(
            dataframe["aqi"]
            .isna()
            .sum()
        ),
        "target_missing": target_missing,
        "aqi_min": float(
            dataframe["aqi"]
            .dropna()
            .min()
        ),
        "aqi_max": float(
            dataframe["aqi"]
            .dropna()
            .max()
        ),
        "rawalpindi_excluded": True,
        "aqi_method": (
            "US EPA particulate AQI using "
            "city-grouped trailing 24-hour "
            "PM2.5 and PM10 averages."
        ),
        "target_method": (
            "Direct city-grouped +24h, "
            "+48h and +72h shifts of "
            "calculated AQI."
        ),
    }

    print(
        "\nSaving target dataset..."
    )

    dataframe.to_parquet(
        OUTPUT_PARQUET,
        index=False,
    )

    dataframe.to_csv(
        OUTPUT_CSV,
        index=False,
    )

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
    print("TARGET BUILD COMPLETE")
    print("=" * 90)

    print(
        f"\nTotal rows: "
        f"{len(dataframe):,}"
    )

    print(
        f"Current AQI available: "
        f"{current_aqi_count:,}"
    )

    print(
        f"Rows with all future targets: "
        f"{targets_only_count:,}"
    )

    print(
        "Rows with current AQI + "
        f"all targets: "
        f"{complete_model_base_count:,}"
    )

    for horizon in HORIZONS:
        print(
            f"Missing {horizon}h target: "
            f"{target_missing[str(horizon) + 'h']:,}"
        )

    print(
        f"\nSaved:\n{OUTPUT_PARQUET}"
    )

    print(
        f"\nReport:\n{REPORT_PATH}"
    )


if __name__ == "__main__":
    main()