from __future__ import annotations

import numpy as np

from src.services.feature_service import (
    AQIFeatureService,
)


CITIES = [
    "Faisalabad",
    "Islamabad",
    "Karachi",
    "Lahore",
    "Multan",
    "Peshawar",
    "Quetta",
    "Rahim Yar Khan",
    "Sialkot",
]


def main() -> None:
    print("=" * 80)
    print(
        "PEARLS AQI - EXPANDED FEAST ONLINE TEST"
    )
    print("=" * 80)

    service = (
        AQIFeatureService()
    )

    if len(service.feature_columns) != 56:
        raise RuntimeError(
            "Expected exactly 56 Feast features."
        )

    for city in CITIES:
        features = (
            service.get_online_features(
                city
            )
        )

        if features.shape != (
            1,
            56,
        ):
            raise RuntimeError(
                f"{city}: expected shape "
                f"(1, 56), got "
                f"{features.shape}."
            )

        values = (
            features.to_numpy()
        )

        if not np.isfinite(
            values
        ).all():
            raise RuntimeError(
                f"{city}: non-finite "
                "online features detected."
            )

        print(
            f"{city:<18} "
            f"PASS | "
            f"PM2.5={features.iloc[0]['pm2_5']:.2f} | "
            f"AQI lag24="
            f"{features.iloc[0]['aqi_lag_24h']:.2f}"
        )

    print("\n" + "=" * 80)
    print(
        "EXPANDED FEAST ONLINE TEST: PASS"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()