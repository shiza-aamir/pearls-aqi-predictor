import pytest

from src.features.aqi import AQICalculator


def test_pm25_good_upper_boundary() -> None:
    assert AQICalculator.calculate_pm25_aqi(9.0) == 50


def test_pm25_moderate_upper_boundary() -> None:
    assert AQICalculator.calculate_pm25_aqi(35.4) == 100


def test_pm25_sensitive_group_upper_boundary() -> None:
    assert AQICalculator.calculate_pm25_aqi(55.4) == 150


def test_pm25_unhealthy_upper_boundary() -> None:
    assert AQICalculator.calculate_pm25_aqi(125.4) == 200


def test_pm25_very_unhealthy_upper_boundary() -> None:
    assert AQICalculator.calculate_pm25_aqi(225.4) == 300


def test_pm25_hazardous_upper_boundary() -> None:
    assert AQICalculator.calculate_pm25_aqi(325.4) == 500


def test_pm25_above_scale_is_capped_at_500() -> None:
    assert AQICalculator.calculate_pm25_aqi(500.0) == 500


def test_pm10_good_upper_boundary() -> None:
    assert AQICalculator.calculate_pm10_aqi(54.0) == 50


def test_pm10_moderate_upper_boundary() -> None:
    assert AQICalculator.calculate_pm10_aqi(154.0) == 100


def test_pm10_sensitive_group_upper_boundary() -> None:
    assert AQICalculator.calculate_pm10_aqi(254.0) == 150


def test_pm10_unhealthy_upper_boundary() -> None:
    assert AQICalculator.calculate_pm10_aqi(354.0) == 200


def test_pm10_very_unhealthy_upper_boundary() -> None:
    assert AQICalculator.calculate_pm10_aqi(424.0) == 300


def test_pm10_hazardous_upper_boundary() -> None:
    assert AQICalculator.calculate_pm10_aqi(604.0) == 500


def test_pm25_is_truncated_to_one_decimal() -> None:
    assert AQICalculator.calculate_pm25_aqi(
        35.49
    ) == AQICalculator.calculate_pm25_aqi(35.4)


def test_pm10_is_truncated_to_integer() -> None:
    assert AQICalculator.calculate_pm10_aqi(
        154.9
    ) == AQICalculator.calculate_pm10_aqi(154.0)


def test_final_aqi_uses_dominant_pollutant() -> None:
    result = AQICalculator.calculate_aqi(
        pm25=55.4,
        pm10=54.0,
    )

    assert result.aqi == 150
    assert result.pm25_aqi == 150
    assert result.pm10_aqi == 50
    assert result.dominant_pollutant == "pm2_5"
    assert result.category == "Unhealthy for Sensitive Groups"


@pytest.mark.parametrize(
    ("aqi", "expected"),
    [
        (0, "Good"),
        (50, "Good"),
        (51, "Moderate"),
        (100, "Moderate"),
        (101, "Unhealthy for Sensitive Groups"),
        (150, "Unhealthy for Sensitive Groups"),
        (151, "Unhealthy"),
        (200, "Unhealthy"),
        (201, "Very Unhealthy"),
        (300, "Very Unhealthy"),
        (301, "Hazardous"),
        (500, "Hazardous"),
    ],
)
def test_aqi_categories(
    aqi: int,
    expected: str,
) -> None:
    assert AQICalculator.category_from_aqi(aqi) == expected


def test_negative_pm25_rejected() -> None:
    with pytest.raises(ValueError):
        AQICalculator.calculate_pm25_aqi(-0.1)


def test_negative_pm10_rejected() -> None:
    with pytest.raises(ValueError):
        AQICalculator.calculate_pm10_aqi(-1)


def test_negative_aqi_category_rejected() -> None:
    with pytest.raises(ValueError):
        AQICalculator.category_from_aqi(-1)