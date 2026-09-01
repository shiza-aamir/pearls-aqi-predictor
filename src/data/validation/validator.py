from pathlib import Path

import pandas as pd

from src.data.validation.report import ValidationReport


class DatasetValidator:
    REQUIRED_COLUMNS = {
        "timestamp",
        "city",
        "latitude",
        "longitude",
        "pm10",
        "pm2_5",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
        "dust",
        "temperature",
        "humidity",
        "precipitation",
        "wind_speed",
        "wind_direction",
        "pressure",
    }

    NUMERIC_RANGE_RULES = {
        "pm10": (0, None),
        "pm2_5": (0, None),
        "carbon_monoxide": (0, None),
        "nitrogen_dioxide": (0, None),
        "sulphur_dioxide": (0, None),
        "ozone": (0, None),
        "dust": (0, None),
        "humidity": (0, 100),
        "precipitation": (0, None),
        "wind_speed": (0, None),
        "wind_direction": (0, 360),
        "pressure": (800, 1200),
    }

    WEATHER_COLUMNS = [
        "temperature",
        "humidity",
        "precipitation",
        "wind_speed",
        "wind_direction",
        "pressure",
    ]

    def __init__(self, dataset_path: str | Path) -> None:
        self.dataset_path = Path(dataset_path)

    def load(self) -> pd.DataFrame:
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.dataset_path}"
            )

        return pd.read_csv(self.dataset_path)

    def validate(self) -> ValidationReport:
        df = self.load()

        errors: list[str] = []
        warnings: list[str] = []

        missing_required_columns = sorted(
            self.REQUIRED_COLUMNS - set(df.columns)
        )

        required_columns_present = not missing_required_columns

        if missing_required_columns:
            errors.append(
                "Missing required columns: "
                + ", ".join(missing_required_columns)
            )

        missing_values = int(df.isna().sum().sum())
        duplicate_rows = int(df.duplicated().sum())

        if missing_values > 0:
            warnings.append(
                f"Dataset contains {missing_values} missing values."
            )

        if duplicate_rows > 0:
            warnings.append(
                f"Dataset contains {duplicate_rows} duplicate rows."
            )

        timestamp_valid = True
        timezone_present = False

        if "timestamp" in df.columns:
            parsed_timestamp = pd.to_datetime(
                df["timestamp"],
                errors="coerce",
            )

            invalid_timestamps = int(parsed_timestamp.isna().sum())

            if invalid_timestamps > 0:
                timestamp_valid = False
                errors.append(
                    f"{invalid_timestamps} timestamps could not be parsed."
                )
            else:
                df["timestamp"] = parsed_timestamp
                timezone_present = df["timestamp"].dt.tz is not None

                if not timezone_present:
                    warnings.append(
                        "Timestamp column does not contain timezone information."
                    )

        city_count = (
            int(df["city"].nunique())
            if "city" in df.columns
            else 0
        )

        hourly_continuity = False
        balanced_cities = False
        expected_rows_per_city: int | None = None

        if timestamp_valid and {"city", "timestamp"}.issubset(df.columns):
            sorted_df = df.sort_values(["city", "timestamp"])

            gaps = (
                sorted_df.groupby("city")["timestamp"]
                .diff()
                .dropna()
            )

            hourly_continuity = bool(
                gaps.eq(pd.Timedelta(hours=1)).all()
            )

            if not hourly_continuity:
                warnings.append(
                    "One or more cities contain non-hourly timestamp gaps."
                )

            city_counts = df.groupby("city").size()

            if not city_counts.empty:
                balanced_cities = city_counts.nunique() == 1

                if balanced_cities:
                    expected_rows_per_city = int(city_counts.iloc[0])
                else:
                    warnings.append(
                        "Cities contain different numbers of observations."
                    )

        invalid_numeric_ranges = self._validate_numeric_ranges(df)

        if invalid_numeric_ranges:
            warnings.append(
                "One or more numeric columns contain values outside "
                "the configured validation ranges."
            )

        weather_variability_valid = self._validate_weather_variability(df)

        if not weather_variability_valid:
            warnings.append(
                "Weather variables show suspiciously low temporal variability."
            )

        numeric_aqi_present = "aqi" in df.columns

        if not numeric_aqi_present:
            warnings.append(
                "Numeric AQI target is not present and must be derived."
            )

        aqi_category_present = "aqi_category" in df.columns

        if not aqi_category_present:
            warnings.append(
                "AQI category column is not present."
            )

        return ValidationReport(
            dataset_path=str(self.dataset_path),
            row_count=len(df),
            column_count=len(df.columns),
            city_count=city_count,
            missing_values=missing_values,
            duplicate_rows=duplicate_rows,
            hourly_continuity=hourly_continuity,
            balanced_cities=balanced_cities,
            expected_rows_per_city=expected_rows_per_city,
            timezone_present=timezone_present,
            weather_variability_valid=weather_variability_valid,
            numeric_aqi_present=numeric_aqi_present,
            aqi_category_present=aqi_category_present,
            required_columns_present=required_columns_present,
            missing_required_columns=missing_required_columns,
            invalid_numeric_ranges=invalid_numeric_ranges,
            warnings=warnings,
            errors=errors,
        )

    def _validate_numeric_ranges(
        self,
        df: pd.DataFrame,
    ) -> dict[str, int]:
        invalid_counts: dict[str, int] = {}

        for column, (minimum, maximum) in self.NUMERIC_RANGE_RULES.items():
            if column not in df.columns:
                continue

            values = pd.to_numeric(df[column], errors="coerce")

            invalid = values.isna()

            if minimum is not None:
                invalid |= values < minimum

            if maximum is not None:
                invalid |= values > maximum

            count = int(invalid.sum())

            if count > 0:
                invalid_counts[column] = count

        return invalid_counts

    def _validate_weather_variability(
        self,
        df: pd.DataFrame,
    ) -> bool:
        required = {"city", "timestamp"}

        if not required.issubset(df.columns):
            return False

        available_weather_columns = [
            column
            for column in self.WEATHER_COLUMNS
            if column in df.columns
        ]

        if not available_weather_columns:
            return False

        working_df = df.copy()

        if not pd.api.types.is_datetime64_any_dtype(
            working_df["timestamp"]
        ):
            working_df["timestamp"] = pd.to_datetime(
                working_df["timestamp"],
                errors="coerce",
            )

        working_df = working_df.dropna(subset=["timestamp"])
        working_df["validation_date"] = working_df["timestamp"].dt.date

        unique_counts = (
            working_df.groupby(["city", "validation_date"])[
                available_weather_columns
            ]
            .nunique()
        )

        variable_columns = [
            column
            for column in available_weather_columns
            if unique_counts[column].max() > 1
        ]

        non_precipitation_columns = [
            column
            for column in available_weather_columns
            if column != "precipitation"
        ]

        if not non_precipitation_columns:
            return True

        return all(
            column in variable_columns
            for column in non_precipitation_columns
        )