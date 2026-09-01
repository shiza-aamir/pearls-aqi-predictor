from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path(
    "data/historical/openmeteo/"
    "pakistan_aqi_weather_2022_2026.parquet"
)

OUTPUT_DIR = Path(
    "artifacts/backfill_audit"
)

SUMMARY_PATH = (
    OUTPUT_DIR / "dataset_summary.csv"
)

NEGATIVE_PATH = (
    OUTPUT_DIR / "negative_values.csv"
)

EXTREME_PATH = (
    OUTPUT_DIR / "extreme_values.csv"
)

PAIRWISE_PATH = (
    OUTPUT_DIR / "city_pollutant_similarity.csv"
)

CITY_STATS_PATH = (
    OUTPUT_DIR / "city_pollutant_statistics.csv"
)

REPORT_PATH = (
    OUTPUT_DIR / "audit_report.json"
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


WEATHER_COLUMNS = [
    "temperature",
    "humidity",
    "precipitation",
    "wind_speed",
    "wind_direction",
    "pressure",
]


ALL_NUMERIC = [
    "latitude",
    "longitude",
    *POLLUTANTS,
    *WEATHER_COLUMNS,
]


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    dataframe = pd.read_parquet(
        DATA_PATH
    )

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"],
        utc=True,
    )

    return dataframe


def basic_audit(
    dataframe: pd.DataFrame,
) -> dict:
    print("\n" + "=" * 90)
    print("BASIC DATASET AUDIT")
    print("=" * 90)

    duplicate_rows = int(
        dataframe.duplicated(
            subset=[
                "city",
                "timestamp",
            ]
        ).sum()
    )

    missing_values = int(
        dataframe.isna().sum().sum()
    )

    city_counts = (
        dataframe["city"]
        .value_counts()
        .sort_index()
    )

    print(
        f"Rows: {len(dataframe):,}"
    )

    print(
        f"Columns: {len(dataframe.columns)}"
    )

    print(
        f"Cities: "
        f"{dataframe['city'].nunique()}"
    )

    print(
        f"Start: "
        f"{dataframe['timestamp'].min()}"
    )

    print(
        f"End: "
        f"{dataframe['timestamp'].max()}"
    )

    print(
        f"Duplicate city/timestamps: "
        f"{duplicate_rows}"
    )

    print(
        f"Missing cells: "
        f"{missing_values}"
    )

    print("\nRows per city:")
    print(
        city_counts.to_string()
    )

    summary = (
        dataframe[ALL_NUMERIC]
        .describe(
            percentiles=[
                0.01,
                0.05,
                0.25,
                0.50,
                0.75,
                0.95,
                0.99,
                0.999,
            ]
        )
        .T
    )

    summary.to_csv(
        SUMMARY_PATH
    )

    return {
        "rows": int(
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
        "duplicates": duplicate_rows,
        "missing_cells": missing_values,
    }


def audit_negative_values(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    print("\n" + "=" * 90)
    print("NEGATIVE POLLUTANT VALUES")
    print("=" * 90)

    records = []

    for pollutant in POLLUTANTS:
        negative = dataframe[
            dataframe[pollutant] < 0
        ]

        count = int(
            len(negative)
        )

        percentage = (
            count
            / len(dataframe)
            * 100
        )

        minimum = (
            float(
                negative[pollutant].min()
            )
            if count
            else np.nan
        )

        cities = (
            ", ".join(
                sorted(
                    negative["city"]
                    .unique()
                    .tolist()
                )
            )
            if count
            else ""
        )

        records.append(
            {
                "variable": pollutant,
                "negative_count": count,
                "percentage": percentage,
                "minimum": minimum,
                "cities": cities,
            }
        )

    result = pd.DataFrame(
        records
    )

    print(
        result.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )

    result.to_csv(
        NEGATIVE_PATH,
        index=False,
    )

    return result


def audit_extreme_values(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    print("\n" + "=" * 90)
    print("POLLUTANT EXTREME VALUES")
    print("=" * 90)

    records = []

    for pollutant in POLLUTANTS:
        series = dataframe[
            pollutant
        ].dropna()

        quantiles = series.quantile(
            [
                0.95,
                0.99,
                0.999,
            ]
        )

        max_index = series.idxmax()

        records.append(
            {
                "variable": pollutant,
                "mean": float(
                    series.mean()
                ),
                "p95": float(
                    quantiles.loc[0.95]
                ),
                "p99": float(
                    quantiles.loc[0.99]
                ),
                "p999": float(
                    quantiles.loc[0.999]
                ),
                "max": float(
                    series.max()
                ),
                "max_city": str(
                    dataframe.loc[
                        max_index,
                        "city",
                    ]
                ),
                "max_timestamp": str(
                    dataframe.loc[
                        max_index,
                        "timestamp",
                    ]
                ),
            }
        )

    result = pd.DataFrame(
        records
    )

    print(
        result.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.3f}"
            ),
        )
    )

    result.to_csv(
        EXTREME_PATH,
        index=False,
    )

    return result


def city_pollutant_statistics(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    statistics = (
        dataframe
        .groupby("city")[
            POLLUTANTS
        ]
        .agg(
            [
                "mean",
                "std",
                "min",
                "max",
            ]
        )
    )

    statistics.columns = [
        f"{variable}_{statistic}"
        for variable, statistic
        in statistics.columns
    ]

    statistics = (
        statistics.reset_index()
    )

    statistics.to_csv(
        CITY_STATS_PATH,
        index=False,
    )

    return statistics


def compare_city_pair(
    dataframe: pd.DataFrame,
    city_a: str,
    city_b: str,
    pollutant: str,
) -> dict:
    left = (
        dataframe[
            dataframe["city"] == city_a
        ][
            [
                "timestamp",
                pollutant,
            ]
        ]
        .rename(
            columns={
                pollutant: "value_a"
            }
        )
    )

    right = (
        dataframe[
            dataframe["city"] == city_b
        ][
            [
                "timestamp",
                pollutant,
            ]
        ]
        .rename(
            columns={
                pollutant: "value_b"
            }
        )
    )

    merged = left.merge(
        right,
        on="timestamp",
        how="inner",
        validate="one_to_one",
    )

    if merged.empty:
        return {
            "city_a": city_a,
            "city_b": city_b,
            "pollutant": pollutant,
            "matched_rows": 0,
            "correlation": np.nan,
            "mae": np.nan,
            "exact_match_pct": np.nan,
            "near_match_pct": np.nan,
        }

    correlation = (
        merged["value_a"]
        .corr(
            merged["value_b"]
        )
    )

    differences = (
        merged["value_a"]
        .sub(
            merged["value_b"]
        )
        .abs()
    )

    exact_match = float(
        np.isclose(
            merged["value_a"],
            merged["value_b"],
            rtol=0.0,
            atol=1e-12,
        ).mean()
        * 100
    )

    near_match = float(
        np.isclose(
            merged["value_a"],
            merged["value_b"],
            rtol=1e-5,
            atol=0.05,
        ).mean()
        * 100
    )

    return {
        "city_a": city_a,
        "city_b": city_b,
        "pollutant": pollutant,
        "matched_rows": int(
            len(merged)
        ),
        "correlation": float(
            correlation
        ),
        "mae": float(
            differences.mean()
        ),
        "exact_match_pct": exact_match,
        "near_match_pct": near_match,
    }


def audit_city_similarity(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    print("\n" + "=" * 90)
    print("CROSS-CITY POLLUTANT SIMILARITY")
    print("=" * 90)

    cities = sorted(
        dataframe["city"]
        .unique()
        .tolist()
    )

    records = []

    for city_a, city_b in combinations(
        cities,
        2,
    ):
        for pollutant in POLLUTANTS:
            records.append(
                compare_city_pair(
                    dataframe,
                    city_a,
                    city_b,
                    pollutant,
                )
            )

    result = pd.DataFrame(
        records
    )

    result.to_csv(
        PAIRWISE_PATH,
        index=False,
    )

    suspicious = (
        result[
            (
                result[
                    "correlation"
                ]
                >= 0.999
            )
            |
            (
                result[
                    "near_match_pct"
                ]
                >= 99.0
            )
        ]
        .sort_values(
            [
                "city_a",
                "city_b",
                "pollutant",
            ]
        )
    )

    if suspicious.empty:
        print(
            "No near-duplicate city/"
            "pollutant pairs detected."
        )
    else:
        print(
            suspicious.to_string(
                index=False,
                float_format=lambda x: (
                    f"{x:.6f}"
                ),
            )
        )

    print(
        f"\nFull similarity report saved to:"
        f"\n{PAIRWISE_PATH}"
    )

    return result


def inspect_islamabad_rawalpindi(
    similarity: pd.DataFrame,
) -> None:
    pair = similarity[
        (
            similarity["city_a"]
            .isin(
                [
                    "Islamabad",
                    "Rawalpindi",
                ]
            )
        )
        &
        (
            similarity["city_b"]
            .isin(
                [
                    "Islamabad",
                    "Rawalpindi",
                ]
            )
        )
    ]

    print("\n" + "=" * 90)
    print("ISLAMABAD VS RAWALPINDI")
    print("=" * 90)

    if pair.empty:
        print(
            "Pair not found."
        )

        return

    print(
        pair.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 90)
    print(
        "PEARLS AQI - HISTORICAL "
        "BACKFILL AUDIT"
    )
    print("=" * 90)

    dataframe = load_dataset()

    basic = basic_audit(
        dataframe
    )

    negative = audit_negative_values(
        dataframe
    )

    extremes = audit_extreme_values(
        dataframe
    )

    city_pollutant_statistics(
        dataframe
    )

    similarity = audit_city_similarity(
        dataframe
    )

    inspect_islamabad_rawalpindi(
        similarity
    )

    suspicious_pairs = (
        similarity[
            (
                similarity["correlation"]
                >= 0.999
            )
            |
            (
                similarity[
                    "near_match_pct"
                ]
                >= 99.0
            )
        ]
    )

    report = {
        "dataset": basic,
        "negative_pollutant_rows": {
            row["variable"]: int(
                row["negative_count"]
            )
            for _, row
            in negative.iterrows()
        },
        "suspicious_similarity_rows": int(
            len(suspicious_pairs)
        ),
        "maximum_values": {
            row["variable"]: float(
                row["max"]
            )
            for _, row
            in extremes.iterrows()
        },
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
    print("AUDIT COMPLETE")
    print("=" * 90)

    print(
        f"\nReports saved under:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()