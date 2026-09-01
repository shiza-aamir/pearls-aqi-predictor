from pathlib import Path

import pandas as pd

from src.ml.training.splitter import (
    PurgedTimeSeriesSplitter,
)


INPUT_PATH = Path(
    "data/processed/ml_ready_aqi_dataset.parquet"
)

OUTPUT_DIR = Path(
    "data/processed/splits"
)


def main() -> None:
    df = pd.read_parquet(INPUT_PATH)

    splitter = PurgedTimeSeriesSplitter(
        train_ratio=0.70,
        validation_ratio=0.15,
        purge_hours=72,
        excluded_cities=("Rawalpindi",),
    )

    train, validation, test, summary = (
        splitter.split(df)
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train.to_parquet(
        OUTPUT_DIR / "train.parquet",
        index=False,
    )

    validation.to_parquet(
        OUTPUT_DIR / "validation.parquet",
        index=False,
    )

    test.to_parquet(
        OUTPUT_DIR / "test.parquet",
        index=False,
    )

    print("=" * 60)
    print("PURGED CHRONOLOGICAL SPLIT COMPLETE")
    print("=" * 60)

    print(
        f"Train:      {summary.train_rows:,} rows"
    )
    print(
        f"Validation: {summary.validation_rows:,} rows"
    )
    print(
        f"Test:       {summary.test_rows:,} rows"
    )

    print(
        f"\nTrain period:\n"
        f"{summary.train_start} -> "
        f"{summary.train_end}"
    )

    print(
        f"\nValidation period:\n"
        f"{summary.validation_start} -> "
        f"{summary.validation_end}"
    )

    print(
        f"\nTest period:\n"
        f"{summary.test_start} -> "
        f"{summary.test_end}"
    )

    print(
        f"\nPurged between splits: "
        f"{summary.purge_hours} hours"
    )

    print(
        "\nExcluded from model training:"
    )

    for city in summary.excluded_cities:
        print(f"  - {city}")

    print("\nCities used:")
    print(
        ", ".join(
            sorted(train["city"].unique())
        )
    )


if __name__ == "__main__":
    main()