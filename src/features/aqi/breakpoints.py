from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class AQIBreakpoint:
    concentration_low: float
    concentration_high: float
    aqi_low: int
    aqi_high: int


PM25_BREAKPOINTS: Final[tuple[AQIBreakpoint, ...]] = (
    AQIBreakpoint(0.0, 9.0, 0, 50),
    AQIBreakpoint(9.1, 35.4, 51, 100),
    AQIBreakpoint(35.5, 55.4, 101, 150),
    AQIBreakpoint(55.5, 125.4, 151, 200),
    AQIBreakpoint(125.5, 225.4, 201, 300),
    AQIBreakpoint(225.5, 325.4, 301, 500),
)


PM10_BREAKPOINTS: Final[tuple[AQIBreakpoint, ...]] = (
    AQIBreakpoint(0.0, 54.0, 0, 50),
    AQIBreakpoint(55.0, 154.0, 51, 100),
    AQIBreakpoint(155.0, 254.0, 101, 150),
    AQIBreakpoint(255.0, 354.0, 151, 200),
    AQIBreakpoint(355.0, 424.0, 201, 300),
    AQIBreakpoint(425.0, 604.0, 301, 500),
)