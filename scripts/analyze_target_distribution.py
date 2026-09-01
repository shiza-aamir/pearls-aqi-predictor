from pathlib import Path

import pandas as pd


TRAIN_PATH = Path(
    "data/processed/splits/train.parquet"
)

VALIDATION_PATH = Path(
    "data/processed/splits/validation.parquet"
)

TEST_PATH = Path(
    "data/processed/splits/test.parquet"
)

OUTPUT_DIR = Path(
    "artifacts/model_comparison"
)

TARGETS = {
    "24h": "target_aqi_24h",
    "48h": "target_aqi_48h",
    "72h": "target_aqi_72h",
}


def load_split(
    name: str,
    path: Path,
) -> pd.DataFrame:
    dataframe = pd.read_parquet(path)

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"]
    )

    dataframe["split"] = name

    return dataframe


def get_category(
    value: float,
) -> str:
    if value <= 50:
        return "Good"

    if value <= 100:
        return "Moderate"

    if value <= 150:
        return "USG"

    if value <= 200:
        return "Unhealthy"

    if value <= 300:
        return "Very Unhealthy"

    return "Hazardous"


def summarize_target(
    dataframe: pd.DataFrame,
    split_name: str,
    horizon: str,
    target_column: str,
) -> dict:
    series = dataframe[
        target_column
    ].dropna().astype(float)

    return {
        "split": split_name,
        "horizon": horizon,
        "count": int(series.count()),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std()),
        "min": float(series.min()),
        "q25": float(series.quantile(0.25)),
        "q75": float(series.quantile(0.75)),
        "max": float(series.max()),
    }


def build_category_distribution(
    dataframe: pd.DataFrame,
    split_name: str,
    horizon: str,
    target_column: str,
) -> list[dict]:
    working = dataframe[
        [target_column]
    ].dropna().copy()

    working["category"] = (
        working[target_column]
        .astype(float)
        .apply(get_category)
    )

    counts = (
        working["category"]
        .value_counts()
    )

    total = len(working)

    category_order = [
        "Good",
        "Moderate",
        "USG",
        "Unhealthy",
        "Very Unhealthy",
        "Hazardous",
    ]

    rows = []

    for category in category_order:
        count = int(
            counts.get(
                category,
                0,
            )
        )

        percentage = (
            100.0 * count / total
            if total > 0
            else 0.0
        )

        rows.append(
            {
                "split": split_name,
                "horizon": horizon,
                "category": category,
                "count": count,
                "percentage": percentage,
            }
        )

    return rows


def print_summary(
    summary_df: pd.DataFrame,
) -> None:
    print(
        "\n"
        + "=" * 100
    )

    print(
        "TARGET DISTRIBUTION SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.2f}"
            ),
        )
    )


def print_categories(
    categories_df: pd.DataFrame,
) -> None:
    for horizon in TARGETS:
        print(
            "\n"
            + "=" * 100
        )

        print(
            f"{horizon.upper()} "
            "AQI CATEGORY DISTRIBUTION"
        )

        print(
            "=" * 100
        )

        subset = categories_df[
            categories_df["horizon"]
            == horizon
        ]

        pivot = subset.pivot(
            index="category",
            columns="split",
            values="percentage",
        )

        category_order = [
            "Good",
            "Moderate",
            "USG",
            "Unhealthy",
            "Very Unhealthy",
            "Hazardous",
        ]

        pivot = pivot.reindex(
            category_order
        )

        print(
            pivot.to_string(
                float_format=lambda value: (
                    f"{value:.2f}%"
                ),
            )
        )


def main() -> None:
    print("=" * 100)
    print(
        "PEARLS AQI - TARGET DISTRIBUTION ANALYSIS"
    )
    print("=" * 100)

    splits = {
        "train": load_split(
            "train",
            TRAIN_PATH,
        ),
        "validation": load_split(
            "validation",
            VALIDATION_PATH,
        ),
        "test": load_split(
            "test",
            TEST_PATH,
        ),
    }

    summary_rows = []
    category_rows = []

    for split_name, dataframe in (
        splits.items()
    ):
        print(
            f"\n{split_name.upper()}:"
        )

        print(
            f"  Rows: "
            f"{len(dataframe):,}"
        )

        print(
            f"  Start: "
            f"{dataframe['timestamp'].min()}"
        )

        print(
            f"  End:   "
            f"{dataframe['timestamp'].max()}"
        )

        for horizon, target_column in (
            TARGETS.items()
        ):
            summary_rows.append(
                summarize_target(
                    dataframe=dataframe,
                    split_name=split_name,
                    horizon=horizon,
                    target_column=target_column,
                )
            )

            category_rows.extend(
                build_category_distribution(
                    dataframe=dataframe,
                    split_name=split_name,
                    horizon=horizon,
                    target_column=target_column,
                )
            )

    summary_df = pd.DataFrame(
        summary_rows
    )

    categories_df = pd.DataFrame(
        category_rows
    )

    print_summary(
        summary_df
    )

    print_categories(
        categories_df
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        OUTPUT_DIR
        / "target_distribution_summary.csv"
    )

    category_path = (
        OUTPUT_DIR
        / "target_category_distribution.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    categories_df.to_csv(
        category_path,
        index=False,
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "FILES SAVED"
    )

    print(
        "=" * 100
    )

    print(
        summary_path
    )

    print(
        category_path
    )


if __name__ == "__main__":
    main()