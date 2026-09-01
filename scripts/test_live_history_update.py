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
        "PEARLS AQI - OPENWEATHER "
        "LIVE HISTORY UPDATE TEST"
    )
    print("=" * 80)

    history_service = (
        LiveHistoryService()
    )

    feature_pipeline = (
        LiveFeaturePipeline()
    )

    before = history_service.load(
        CITY
    )

    print(
        f"\nBefore rows:   {len(before)}"
    )
    print(
        f"Before latest: "
        f"{before['timestamp'].max()}"
    )

    print(
        "\nFetching and upserting "
        "OpenWeather observation..."
    )

    after = (
        history_service
        .update_from_openweather(
            CITY
        )
    )

    print(
        f"\nAfter rows:    {len(after)}"
    )
    print(
        f"After latest:  "
        f"{after['timestamp'].max()}"
    )

    latest_source = (
        after
        .sort_values("timestamp")
        .iloc[-1]["source"]
    )

    print(
        f"Latest source: "
        f"{latest_source}"
    )

    (
        latest,
        features,
    ) = (
        feature_pipeline
        .build_latest_features(
            after
        )
    )

    print(
        "\nLatest engineered observation:"
    )

    print(
        f"Timestamp:     "
        f"{latest.iloc[0]['timestamp']}"
    )
    print(
        f"Current AQI:   "
        f"{latest.iloc[0]['aqi_current']:.0f}"
    )
    print(
        f"PM2.5:         "
        f"{latest.iloc[0]['pm2_5']:.2f}"
    )
    print(
        f"PM10:          "
        f"{latest.iloc[0]['pm10']:.2f}"
    )
    print(
        f"AQI lag 24h:   "
        f"{features.iloc[0]['aqi_lag_24h']:.2f}"
    )
    print(
        f"AQI lag 72h:   "
        f"{features.iloc[0]['aqi_lag_72h']:.2f}"
    )
    print(
        f"Features:      "
        f"{features.shape[1]}"
    )
    print(
        f"Nulls:         "
        f"{int(features.isnull().sum().sum())}"
    )

    if latest_source != "openweather_live":
        raise RuntimeError(
            "Latest observation was not "
            "stored as OpenWeather live data."
        )

    if features.shape != (1, 56):
        raise RuntimeError(
            "Expected exactly 56 "
            "production features."
        )

    print("\n" + "=" * 80)
    print(
        "LIVE HISTORY UPDATE TEST: PASS"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()