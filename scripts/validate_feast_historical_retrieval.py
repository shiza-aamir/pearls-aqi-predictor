from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from feast import FeatureStore

from src.features.engineer import AQIFeatureEngineer


SOURCE_PATH = Path(
    "data/processed/expanded/ml_ready_aqi_dataset.parquet"
)

FEATURE_REPO_PATH = Path(
    "feature_repo/feature_repo"
)

REPORT_PATH = Path(
    "artifacts/feature_store/"
    "historical_retrieval_validation.json"
)

SAMPLE_OUTPUT_PATH = Path(
    "artifacts/feature_store/"
    "historical_retrieval_sample.parquet"
)

FEATURE_VIEW_NAME = "aqi_features"

TARGET_COLUMNS = [
    "target_aqi_24h",
    "target_aqi_48h",
    "target_aqi_72h",
]

SAMPLE_TIMESTAMPS_PER_CITY = 10

RANDOM_SEED = 42

ABSOLUTE_TOLERANCE = 1e-9


def main() -> None:
    print("=" * 80)
    print(
        "PEARLS AQI - FEAST HISTORICAL "
        "RETRIEVAL VALIDATION"
    )
    print("=" * 80)

    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {SOURCE_PATH}"
        )

    source = pd.read_parquet(
        SOURCE_PATH
    )

    source["timestamp"] = pd.to_datetime(
        source["timestamp"],
        utc=True,
        errors="raise",
    )

    feature_columns = (
        AQIFeatureEngineer
        .get_model_feature_columns()
    )

    if len(feature_columns) != 56:
        raise RuntimeError(
            f"Expected 56 model features, "
            f"got {len(feature_columns)}."
        )

    required = [
        "timestamp",
        "city",
        *feature_columns,
        *TARGET_COLUMNS,
    ]

    missing = [
        column
        for column in required
        if column not in source.columns
    ]

    if missing:
        raise ValueError(
            "Expanded ML dataset is missing: "
            f"{missing}"
        )

    print(
        f"\nSource rows:     {len(source):,}"
    )
    print(
        f"Source cities:   {source['city'].nunique()}"
    )
    print(
        f"Model features:  {len(feature_columns)}"
    )

    samples = []

    for city, city_df in source.groupby(
        "city",
        sort=True,
    ):
        city_df = (
            city_df
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        if (
            len(city_df)
            < SAMPLE_TIMESTAMPS_PER_CITY
        ):
            raise RuntimeError(
                f"Insufficient rows for {city}."
            )

        city_sample = city_df.sample(
            n=SAMPLE_TIMESTAMPS_PER_CITY,
            random_state=RANDOM_SEED,
        )

        samples.append(
            city_sample
        )

    expected = (
        pd.concat(
            samples,
            ignore_index=True,
        )
        .sort_values(
            [
                "city",
                "timestamp",
            ]
        )
        .reset_index(drop=True)
    )

    print(
        f"Validation rows: {len(expected):,}"
    )

    entity_df = expected[
        [
            "city",
            "timestamp",
        ]
    ].copy()

    entity_df = entity_df.rename(
        columns={
            "city": "city_id",
            "timestamp": "event_timestamp",
        }
    )

    feature_refs = [
        f"{FEATURE_VIEW_NAME}:{column}"
        for column in feature_columns
    ]

    store = FeatureStore(
        repo_path=str(
            FEATURE_REPO_PATH
        )
    )

    print(
        "\nRequesting point-in-time "
        "historical features from Feast..."
    )

    retrieval_job = (
        store.get_historical_features(
            entity_df=entity_df,
            features=feature_refs,
        )
    )

    retrieved = (
        retrieval_job
        .to_df()
    )

    retrieved[
        "event_timestamp"
    ] = pd.to_datetime(
        retrieved[
            "event_timestamp"
        ],
        utc=True,
        errors="raise",
    )

    if len(retrieved) != len(expected):
        raise RuntimeError(
            "Historical retrieval row-count "
            "mismatch. "
            f"Expected {len(expected)}, "
            f"got {len(retrieved)}."
        )

    feast_missing = [
        column
        for column in feature_columns
        if column not in retrieved.columns
    ]

    if feast_missing:
        raise RuntimeError(
            "Feast did not return features: "
            f"{feast_missing}"
        )

    retrieved = retrieved.rename(
        columns={
            "city_id": "city",
            "event_timestamp": "timestamp",
        }
    )

    expected_compare = expected[
        [
            "city",
            "timestamp",
            *feature_columns,
            *TARGET_COLUMNS,
        ]
    ].copy()

    retrieved_compare = retrieved[
        [
            "city",
            "timestamp",
            *feature_columns,
        ]
    ].copy()

    expected_compare = (
        expected_compare
        .sort_values(
            [
                "city",
                "timestamp",
            ]
        )
        .reset_index(drop=True)
    )

    retrieved_compare = (
        retrieved_compare
        .sort_values(
            [
                "city",
                "timestamp",
            ]
        )
        .reset_index(drop=True)
    )

    expected_keys = expected_compare[
        [
            "city",
            "timestamp",
        ]
    ]

    retrieved_keys = retrieved_compare[
        [
            "city",
            "timestamp",
        ]
    ]

    if not expected_keys.equals(
        retrieved_keys
    ):
        raise RuntimeError(
            "Entity/timestamp keys changed "
            "during Feast retrieval."
        )

    comparison_results = {}

    failed_features = []

    maximum_difference = 0.0

    for feature in feature_columns:
        expected_values = (
            expected_compare[
                feature
            ]
            .astype(float)
            .to_numpy()
        )

        retrieved_values = (
            retrieved_compare[
                feature
            ]
            .astype(float)
            .to_numpy()
        )

        if not np.isfinite(
            retrieved_values
        ).all():
            raise RuntimeError(
                f"Feast returned non-finite "
                f"values for {feature}."
            )

        differences = np.abs(
            expected_values
            - retrieved_values
        )

        feature_max_difference = float(
            differences.max()
        )

        maximum_difference = max(
            maximum_difference,
            feature_max_difference,
        )

        matches = bool(
            np.allclose(
                expected_values,
                retrieved_values,
                rtol=0.0,
                atol=ABSOLUTE_TOLERANCE,
            )
        )

        comparison_results[
            feature
        ] = {
            "matches": matches,
            "max_absolute_difference": (
                feature_max_difference
            ),
        }

        if not matches:
            failed_features.append(
                feature
            )

    if failed_features:
        raise RuntimeError(
            "Feast historical values differ "
            "from the versioned ML dataset for: "
            f"{failed_features}"
        )

    training_sample = (
        retrieved_compare
        .merge(
            expected_compare[
                [
                    "city",
                    "timestamp",
                    *TARGET_COLUMNS,
                ]
            ],
            on=[
                "city",
                "timestamp",
            ],
            how="left",
            validate="one_to_one",
        )
    )

    if (
        training_sample[
            TARGET_COLUMNS
        ]
        .isnull()
        .any()
        .any()
    ):
        raise RuntimeError(
            "Targets were not attached "
            "successfully."
        )

    expected_training_columns = [
        "city",
        "timestamp",
        *feature_columns,
        *TARGET_COLUMNS,
    ]

    training_sample = training_sample[
        expected_training_columns
    ]

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_sample.to_parquet(
        SAMPLE_OUTPUT_PATH,
        index=False,
    )

    report = {
        "validation_type": (
            "FEAST_POINT_IN_TIME_HISTORICAL_RETRIEVAL"
        ),
        "source_dataset": str(
            SOURCE_PATH
        ),
        "feature_view": (
            FEATURE_VIEW_NAME
        ),
        "source_rows": int(
            len(source)
        ),
        "source_cities": int(
            source["city"].nunique()
        ),
        "validation_rows": int(
            len(expected)
        ),
        "samples_per_city": (
            SAMPLE_TIMESTAMPS_PER_CITY
        ),
        "feature_count": int(
            len(feature_columns)
        ),
        "target_columns": (
            TARGET_COLUMNS
        ),
        "absolute_tolerance": (
            ABSOLUTE_TOLERANCE
        ),
        "maximum_absolute_difference": (
            maximum_difference
        ),
        "failed_features": (
            failed_features
        ),
        "all_features_match": (
            len(failed_features) == 0
        ),
        "point_in_time_retrieval_verified": True,
        "targets_attached_after_retrieval": True,
        "model_retrained": False,
        "final_holdout_reused_for_selection": False,
        "comparison": (
            comparison_results
        ),
    }

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    print("\nValidation results:")
    print(
        f"  Retrieved rows:       "
        f"{len(retrieved_compare)}"
    )
    print(
        f"  Features checked:     "
        f"{len(feature_columns)}"
    )
    print(
        f"  Features matched:     "
        f"{len(feature_columns) - len(failed_features)}"
    )
    print(
        f"  Failed features:      "
        f"{len(failed_features)}"
    )
    print(
        f"  Maximum difference:   "
        f"{maximum_difference:.12g}"
    )
    print(
        f"  Targets attached:     "
        f"{len(TARGET_COLUMNS)}"
    )
    print(
        f"  Training sample rows: "
        f"{len(training_sample)}"
    )

    print(
        f"\nValidation report:\n  "
        f"{REPORT_PATH}"
    )

    print(
        f"\nTraining sample:\n  "
        f"{SAMPLE_OUTPUT_PATH}"
    )

    print("\n" + "=" * 80)
    print(
        "FEAST HISTORICAL RETRIEVAL "
        "VALIDATION: PASS"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()