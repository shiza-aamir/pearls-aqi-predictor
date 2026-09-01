from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ValidationReport:
    dataset_path: str
    row_count: int
    column_count: int
    city_count: int
    missing_values: int
    duplicate_rows: int
    hourly_continuity: bool
    balanced_cities: bool
    expected_rows_per_city: int | None
    timezone_present: bool
    weather_variability_valid: bool
    numeric_aqi_present: bool
    aqi_category_present: bool
    required_columns_present: bool
    missing_required_columns: list[str] = field(default_factory=list)
    invalid_numeric_ranges: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.errors:
            return "FAIL"

        if self.warnings:
            return "REVIEW"

        return "PASS"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["status"] = self.status
        return result