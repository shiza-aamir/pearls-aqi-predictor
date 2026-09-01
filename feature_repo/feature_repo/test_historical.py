from datetime import datetime, timezone

import pandas as pd
from feast import FeatureStore


def main() -> None:
    store = FeatureStore(repo_path=".")

    entity_df = pd.DataFrame(
        {
            "city_id": [
                "Islamabad",
                "Islamabad",
            ],
            "event_timestamp": [
                datetime(
                    2026,
                    1,
                    15,
                    12,
                    0,
                    tzinfo=timezone.utc,
                ),
                datetime(
                    2026,
                    2,
                    1,
                    20,
                    0,
                    tzinfo=timezone.utc,
                ),
            ],
        }
    )

    feature_refs = [
        "aqi_features:latitude",
        "aqi_features:longitude",
        "aqi_features:temperature",
        "aqi_features:humidity",
        "aqi_features:pm2_5",
        "aqi_features:pm10",
        "aqi_features:aqi_lag_24h",
        "aqi_features:aqi_rolling_mean_24h",
    ]

    historical_df = (
        store.get_historical_features(
            entity_df=entity_df,
            features=feature_refs,
        )
        .to_df()
    )

    print("=" * 70)
    print("FEAST HISTORICAL FEATURE TEST")
    print("=" * 70)

    print(historical_df.to_string(index=False))

    print("\nRows:", len(historical_df))
    print("Columns:", len(historical_df.columns))

    feature_columns = [
        column
        for column in historical_df.columns
        if column
        not in {
            "city_id",
            "event_timestamp",
        }
    ]

    missing_count = int(
        historical_df[
            feature_columns
        ]
        .isnull()
        .sum()
        .sum()
    )

    print("Missing feature values:", missing_count)

    print("=" * 70)


if __name__ == "__main__":
    main()