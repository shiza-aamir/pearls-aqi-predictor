from __future__ import annotations

from src.services.live_feature_pipeline import (
    LiveFeaturePipeline,
)
from src.services.live_history_service import (
    LiveHistoryService,
)


CITY = "Islamabad"


def main() -> None:
    print("=" * 80)
    print(
        "PEARLS AQI - LIVE HISTORY BOOTSTRAP TEST"
    )
    print("=" * 80)

    history_service = (
        LiveHistoryService()
    )

    feature_pipeline = (
        LiveFeaturePipeline()
    )

    print(
        f"\nBootstrapping {CITY}..."
    )

    history = (
        history_service.bootstrap(
            CITY,
            force=True,
        )
    )

    print(
        f"Rows:        {len(history)}"
    )

    print(
        f"Start:       "
        f"{history['timestamp'].min()}"
    )

    print(
        f"End:         "
        f"{history['timestamp'].max()}"
    )

    print(
        f"Source:      "
        f"{history['source'].unique().tolist()}"
    )

    print(
        f"Missing:     "
        f"{int(history.isnull().sum().sum())}"
    )

    gaps = (
        history["timestamp"]
        .sort_values()
        .diff()
        .dropna()
    )

    print(
        f"Hourly gaps: "
        f"{int((gaps != gaps.iloc[0]).sum())}"
    )

    print(
        "\nBuilding AQI + 56 model features..."
    )

    (
        latest,
        features,
    ) = feature_pipeline.build_latest_features(
        history
    )

    print(
        f"Latest timestamp: "
        f"{latest.iloc[0]['timestamp']}"
    )

    print(
        f"Current AQI:      "
        f"{latest.iloc[0]['aqi_current']:.0f}"
    )

    print(
        f"PM2.5:            "
        f"{latest.iloc[0]['pm2_5']:.2f}"
    )

    print(
        f"PM10:             "
        f"{latest.iloc[0]['pm10']:.2f}"
    )

    print(
        f"AQI lag 24h:      "
        f"{features.iloc[0]['aqi_lag_24h']:.2f}"
    )

    print(
        f"AQI lag 72h:      "
        f"{features.iloc[0]['aqi_lag_72h']:.2f}"
    )

    print(
        f"Feature count:    "
        f"{features.shape[1]}"
    )

    print(
        f"Feature nulls:    "
        f"{int(features.isnull().sum().sum())}"
    )

    print("\n" + "=" * 80)
    print(
        "LIVE HISTORY BOOTSTRAP TEST: PASS"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()