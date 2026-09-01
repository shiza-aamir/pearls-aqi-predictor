from __future__ import annotations

import numpy as np

from src.services.explanation_service import (
    AQIExplanationService,
)
from src.services.live_feature_pipeline import (
    LiveFeaturePipeline,
)
from src.services.live_history_service import (
    LiveHistoryService,
)
from src.services.prediction_service import (
    AQIPredictionService,
)


CITY = "Islamabad"


def main() -> None:
    print("=" * 80)
    print(
        "PEARLS AQI - LIVE PRODUCTION "
        "SHAP EXPLANATION TEST"
    )
    print("=" * 80)

    history_service = (
        LiveHistoryService()
    )

    feature_pipeline = (
        LiveFeaturePipeline()
    )

    prediction_service = (
        AQIPredictionService()
    )

    explanation_service = (
        AQIExplanationService()
    )

    print(
        f"\nPreparing live features for {CITY}..."
    )

    history = (
        history_service
        .ensure_current_history(
            CITY
        )
    )

    latest, features = (
        feature_pipeline
        .build_latest_features(
            history
        )
    )

    row = latest.iloc[0]

    print(
        f"Timestamp:   {row['timestamp']}"
    )
    print(
        f"Current AQI: {row['aqi_current']:.0f}"
    )
    print(
        f"PM2.5:      {row['pm2_5']:.2f}"
    )
    print(
        f"PM10:       {row['pm10']:.2f}"
    )
    print(
        f"Features:   {features.shape[1]}"
    )

    if features.shape != (1, 56):
        raise RuntimeError(
            "Expected exactly one row "
            "with 56 model features."
        )

    predictions = (
        prediction_service.predict_all(
            features
        )
    )

    if len(predictions) != 3:
        raise RuntimeError(
            "Expected exactly 3 predictions."
        )

    print(
        "\nPredictions + SHAP explanations:"
    )

    for prediction in predictions:
        explanation = (
            explanation_service
            .explain_single(
                feature_row=features,
                horizon=prediction.horizon,
                top_n=5,
            )
        )

        print(
            "\n"
            + "-" * 80
        )

        print(
            f"Horizon:       "
            f"{prediction.horizon}"
        )

        print(
            f"Prediction:    "
            f"{prediction.predicted_aqi:.2f}"
        )

        print(
            f"Category:      "
            f"{prediction.predicted_category}"
        )

        print(
            f"Model alias:   "
            f"{prediction.model_alias}"
        )

        print(
            f"SHAP base:     "
            f"{explanation.base_value:.2f}"
        )

        if not np.isclose(
            prediction.predicted_aqi,
            explanation.predicted_aqi,
            atol=1e-5,
        ):
            raise RuntimeError(
                f"{prediction.horizon}: prediction "
                "service and SHAP model disagree. "
                f"Prediction service="
                f"{prediction.predicted_aqi}, "
                f"SHAP service="
                f"{explanation.predicted_aqi}."
            )

        contribution_sum = sum(
            contribution.shap_value
            for contribution
            in explanation.contributions
        )

        print(
            "\nTop model contributions:"
        )

        for rank, contribution in enumerate(
            explanation.contributions,
            start=1,
        ):
            sign = (
                "+"
                if contribution.shap_value > 0
                else ""
            )

            print(
                f"  {rank}. "
                f"{contribution.feature:<30} "
                f"value="
                f"{contribution.feature_value:>10.3f} "
                f"SHAP="
                f"{sign}"
                f"{contribution.shap_value:>8.3f} "
                f"({contribution.direction})"
            )

        print(
            f"\nTop-5 SHAP contribution sum: "
            f"{contribution_sum:+.3f}"
        )

    print(
        "\n" + "=" * 80
    )

    print(
        "LIVE PRODUCTION SHAP TEST: PASS"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()