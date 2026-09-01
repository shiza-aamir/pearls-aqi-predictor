from pathlib import Path

import pandas as pd

from src.features.engineer import AQIFeatureEngineer


INPUT_PATH = Path(
    "data/processed/historical_aqi_with_targets.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/ml_ready_aqi_dataset.parquet"
)


TARGET_COLUMNS = [
    "target_aqi_24h",
    "target_aqi_48h",
    "target_aqi_72h",
]


def main() -> None:
    df = pd.read_parquet(INPUT_PATH)

    engineer = AQIFeatureEngineer()

    engineered_df, summary = engineer.transform(df)

    feature_columns = engineer.get_model_feature_columns()

    required_training_columns = (
        feature_columns + TARGET_COLUMNS
    )

    clean_df = engineered_df.dropna(
        subset=required_training_columns
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean_df.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print("=" * 60)
    print("ML DATASET BUILD COMPLETE")
    print("=" * 60)

    print(f"Input rows: {summary.input_rows}")
    print(f"Engineered rows: {summary.output_rows}")
    print(f"Cities: {summary.cities}")
    print(f"Model features: {summary.feature_count}")
    print(f"Final ML-ready rows: {len(clean_df)}")

    print("\nTargets:")
    for target in TARGET_COLUMNS:
        print(f"  - {target}")

    print("\nMissing values in model data:")

    missing = clean_df[
        required_training_columns
    ].isna().sum().sum()

    print(f"  {missing}")

    print("\nCity distribution:")
    print(
        clean_df["city"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        f"\nDataset written to:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()