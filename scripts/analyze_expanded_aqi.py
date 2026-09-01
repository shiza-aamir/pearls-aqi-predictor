from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "data/processed/expanded/"
    "historical_with_aqi_targets.parquet"
)

OUTPUT_DIR = Path(
    "artifacts/eda/expanded"
)


def load_data() -> pd.DataFrame:
    dataframe = pd.read_parquet(INPUT_PATH)

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"],
        utc=True,
    )

    dataframe = dataframe[
        dataframe["aqi"].notna()
    ].copy()

    dataframe["aqi"] = (
        dataframe["aqi"].astype(float)
    )

    dataframe["year"] = (
        dataframe["timestamp"].dt.year
    )

    dataframe["month"] = (
        dataframe["timestamp"].dt.month
    )

    return dataframe


def summarize_by_year(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    result = (
        dataframe
        .groupby("year")["aqi"]
        .agg(
            rows="count",
            mean="mean",
            median="median",
            std="std",
            minimum="min",
            maximum="max",
        )
        .round(2)
    )

    print("\n" + "=" * 90)
    print("AQI BY YEAR")
    print("=" * 90)
    print(result.to_string())

    result.to_csv(
        OUTPUT_DIR / "aqi_by_year.csv"
    )

    return result


def summarize_by_city(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    result = (
        dataframe
        .groupby("city")["aqi"]
        .agg(
            rows="count",
            mean="mean",
            median="median",
            std="std",
            minimum="min",
            maximum="max",
        )
        .sort_values(
            "mean",
            ascending=False,
        )
        .round(2)
    )

    print("\n" + "=" * 90)
    print("AQI BY CITY")
    print("=" * 90)
    print(result.to_string())

    result.to_csv(
        OUTPUT_DIR / "aqi_by_city.csv"
    )

    return result


def summarize_by_month(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    result = (
        dataframe
        .groupby("month")["aqi"]
        .agg(
            rows="count",
            mean="mean",
            median="median",
            std="std",
            minimum="min",
            maximum="max",
        )
        .round(2)
    )

    print("\n" + "=" * 90)
    print("AQI BY MONTH")
    print("=" * 90)
    print(result.to_string())

    result.to_csv(
        OUTPUT_DIR / "aqi_by_month.csv"
    )

    return result


def summarize_categories_by_year(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    counts = pd.crosstab(
        dataframe["year"],
        dataframe["aqi_category"],
    )

    percentages = (
        pd.crosstab(
            dataframe["year"],
            dataframe["aqi_category"],
            normalize="index",
        )
        * 100
    ).round(2)

    print("\n" + "=" * 90)
    print("AQI CATEGORY % BY YEAR")
    print("=" * 90)
    print(percentages.to_string())

    counts.to_csv(
        OUTPUT_DIR
        / "aqi_category_counts_by_year.csv"
    )

    percentages.to_csv(
        OUTPUT_DIR
        / "aqi_category_percent_by_year.csv"
    )

    return percentages


def summarize_categories_by_month(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    percentages = (
        pd.crosstab(
            dataframe["month"],
            dataframe["aqi_category"],
            normalize="index",
        )
        * 100
    ).round(2)

    print("\n" + "=" * 90)
    print("AQI CATEGORY % BY MONTH")
    print("=" * 90)
    print(percentages.to_string())

    percentages.to_csv(
        OUTPUT_DIR
        / "aqi_category_percent_by_month.csv"
    )

    return percentages


def summarize_dominant_pollutant(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    counts = (
        dataframe[
            "dominant_pollutant"
        ]
        .value_counts()
    )

    percentages = (
        dataframe[
            "dominant_pollutant"
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
    ).round(2)

    print("\n" + "=" * 90)
    print("DOMINANT PARTICULATE POLLUTANT")
    print("=" * 90)
    print(result.to_string())

    result.to_csv(
        OUTPUT_DIR
        / "dominant_pollutant.csv"
    )

    return result


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 90)
    print("PEARLS AQI - EXPANDED DATA EDA")
    print("=" * 90)

    dataframe = load_data()

    print(
        f"\nRows with AQI: "
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

    summarize_by_year(dataframe)
    summarize_by_city(dataframe)
    summarize_by_month(dataframe)

    summarize_categories_by_year(
        dataframe
    )

    summarize_categories_by_month(
        dataframe
    )

    summarize_dominant_pollutant(
        dataframe
    )

    print("\n" + "=" * 90)
    print("EDA COMPLETE")
    print("=" * 90)

    print(
        f"\nOutputs saved under:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()