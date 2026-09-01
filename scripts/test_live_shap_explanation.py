from __future__ import annotations

from src.services.explanation_service import (
    AQIExplanationService,
)
from src.services.feature_service import (
    AQIFeatureService,
)


CITY = "Islamabad"

HORIZONS = [
    "24h",
    "48h",
    "72h",
]


def main() -> None:
    print("=" * 80)
    print(
        "PEARLS AQI - FEAST + SHAP "
        "EXPLANATION TEST"
    )
    print("=" * 80)

    feature_service = (
        AQIFeatureService()
    )

    explanation_service = (
        AQIExplanationService()
    )

    features = (
        feature_service
        .get_online_features(
            CITY
        )
    )

    print(
        f"\nCity: {CITY}"
    )

    print(
        f"Features retrieved: "
        f"{features.shape[1]}"
    )

    for horizon in HORIZONS:
        explanation = (
            explanation_service
            .explain_single(
                feature_row=features,
                horizon=horizon,
                top_n=8,
            )
        )

        print("\n" + "-" * 80)

        print(
            f"{horizon} FORECAST"
        )

        print(
            f"Prediction: "
            f"{explanation.predicted_aqi:.2f}"
        )

        print(
            f"SHAP base:  "
            f"{explanation.base_value:.2f}"
        )

        print(
            "\nTop contributors:"
        )

        for (
            rank,
            contribution,
        ) in enumerate(
            explanation.contributions,
            start=1,
        ):
            symbol = (
                "+"
                if contribution.shap_value > 0
                else ""
            )

            print(
                f"  {rank}. "
                f"{contribution.feature:<30} "
                f"value="
                f"{contribution.feature_value:>10.3f} | "
                f"SHAP="
                f"{symbol}"
                f"{contribution.shap_value:.3f} | "
                f"{contribution.direction}"
            )

    print("\n" + "=" * 80)
    print(
        "FEAST + SHAP EXPLANATION: PASS"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()