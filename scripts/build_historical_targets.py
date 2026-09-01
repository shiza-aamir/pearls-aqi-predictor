from pathlib import Path

import pandas as pd

from src.features.aqi import AQITargetBuilder


INPUT_PATH = Path(
    "data/historical/pakistan_air_quality_final_clean.csv"
)

OUTPUT_PATH = Path(
    "data/processed/historical_aqi_with_targets.parquet"
)


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    builder = AQITargetBuilder()

    result, summary = builder.build(
        df,
        drop_incomplete_targets=False,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print("=" * 60)
    print("AQI TARGET BUILD COMPLETE")
    print("=" * 60)

    print(f"Input rows: {summary.input_rows}")
    print(f"Output rows: {summary.output_rows}")
    print(f"Cities: {summary.cities}")
    print(
        f"Rows with current AQI: "
        f"{summary.rows_with_current_aqi}"
    )
    print(
        f"Rows with all forecast targets: "
        f"{summary.rows_with_all_targets}"
    )

    print("\nTarget columns:")
    print("  - target_aqi_24h")
    print("  - target_aqi_48h")
    print("  - target_aqi_72h")

    print(
        f"\nProcessed dataset written to:\n"
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()