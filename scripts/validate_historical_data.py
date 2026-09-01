import json
from pathlib import Path

from src.data.validation import DatasetValidator


DATASET_PATH = Path(
    "data/historical/pakistan_air_quality_final_clean.csv"
)

REPORT_PATH = Path(
    "artifacts/data_validation_report.json"
)


def main() -> None:
    validator = DatasetValidator(DATASET_PATH)
    report = validator.validate()

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report.to_dict(),
            file,
            indent=4,
        )

    print("=" * 60)
    print("PEARLS AQI HISTORICAL DATASET VALIDATION")
    print("=" * 60)

    print(f"Status: {report.status}")
    print(f"Rows: {report.row_count}")
    print(f"Columns: {report.column_count}")
    print(f"Cities: {report.city_count}")
    print(f"Missing values: {report.missing_values}")
    print(f"Duplicate rows: {report.duplicate_rows}")
    print(f"Hourly continuity: {report.hourly_continuity}")
    print(f"Balanced cities: {report.balanced_cities}")
    print(f"Rows per city: {report.expected_rows_per_city}")
    print(f"Timezone present: {report.timezone_present}")
    print(
        "Weather variability valid: "
        f"{report.weather_variability_valid}"
    )
    print(f"Numeric AQI present: {report.numeric_aqi_present}")
    print(
        "AQI category present: "
        f"{report.aqi_category_present}"
    )

    if report.invalid_numeric_ranges:
        print("\nInvalid numeric ranges:")

        for column, count in report.invalid_numeric_ranges.items():
            print(f"  {column}: {count}")

    if report.warnings:
        print("\nWarnings:")

        for warning in report.warnings:
            print(f"  - {warning}")

    if report.errors:
        print("\nErrors:")

        for error in report.errors:
            print(f"  - {error}")

    print(
        f"\nJSON report written to: {REPORT_PATH}"
    )


if __name__ == "__main__":
    main()