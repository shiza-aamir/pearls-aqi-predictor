from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "data/processed/expanded/"
    "ml_ready_aqi_dataset.parquet"
)

OUTPUT_DIR = Path(
    "data/splits/expanded"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "walk_forward_split_report.json"
)

PURGE_HOURS = 72


FOLDS = [
    {
        "name": "fold_1",
        "train_start": "2022-08-07 23:00:00+00:00",
        "train_end": "2023-06-30 23:00:00+00:00",
        "validation_start": "2023-07-04 00:00:00+00:00",
        "validation_end": "2023-12-31 23:00:00+00:00",
    },
    {
        "name": "fold_2",
        "train_start": "2022-08-07 23:00:00+00:00",
        "train_end": "2024-06-30 23:00:00+00:00",
        "validation_start": "2024-07-04 00:00:00+00:00",
        "validation_end": "2024-12-31 23:00:00+00:00",
    },
    {
        "name": "fold_3",
        "train_start": "2022-08-07 23:00:00+00:00",
        "train_end": "2025-06-30 23:00:00+00:00",
        "validation_start": "2025-07-04 00:00:00+00:00",
        "validation_end": "2025-12-31 23:00:00+00:00",
    },
]


FINAL_TEST = {
    "train_start": "2022-08-07 23:00:00+00:00",
    "train_end": "2025-12-31 23:00:00+00:00",
    "test_start": "2026-01-04 00:00:00+00:00",
    "test_end": "2026-08-25 23:00:00+00:00",
}


def load_data() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input dataset not found:\n{INPUT_PATH}"
        )

    dataframe = pd.read_parquet(
        INPUT_PATH
    )

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"],
        utc=True,
    )

    dataframe = (
        dataframe
        .sort_values(
            [
                "timestamp",
                "city",
            ]
        )
        .reset_index(drop=True)
    )

    return dataframe


def slice_period(
    dataframe: pd.DataFrame,
    start: str,
    end: str,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    return (
        dataframe[
            dataframe["timestamp"]
            .between(
                start_ts,
                end_ts,
                inclusive="both",
            )
        ]
        .copy()
        .reset_index(drop=True)
    )


def validate_purge_gap(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    label: str,
) -> None:
    if train.empty or validation.empty:
        raise ValueError(
            f"{label}: empty train or validation split."
        )

    train_end = (
        train["timestamp"].max()
    )

    validation_start = (
        validation["timestamp"].min()
    )

    gap = (
        validation_start
        - train_end
    )

    minimum_gap = pd.Timedelta(
        hours=PURGE_HOURS
    )

    if gap < minimum_gap:
        raise AssertionError(
            f"{label}: purge gap too small. "
            f"Found {gap}, expected at least "
            f"{minimum_gap}."
        )

    print(
        f"  Purge gap: {gap}"
    )


def validate_cities(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    label: str,
) -> None:
    train_cities = set(
        train["city"].unique()
    )

    validation_cities = set(
        validation["city"].unique()
    )

    if train_cities != validation_cities:
        raise AssertionError(
            f"{label}: city sets differ."
        )

    if len(train_cities) != 9:
        raise AssertionError(
            f"{label}: expected 9 cities, "
            f"got {len(train_cities)}."
        )


def validate_overlap(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    label: str,
) -> None:
    overlap = (
        train[
            [
                "city",
                "timestamp",
            ]
        ]
        .merge(
            validation[
                [
                    "city",
                    "timestamp",
                ]
            ],
            on=[
                "city",
                "timestamp",
            ],
            how="inner",
        )
    )

    if not overlap.empty:
        raise AssertionError(
            f"{label}: train/validation overlap detected."
        )


def save_split(
    dataframe: pd.DataFrame,
    path: Path,
) -> None:
    dataframe.to_parquet(
        path,
        index=False,
    )


def describe_split(
    dataframe: pd.DataFrame,
) -> dict:
    return {
        "rows": int(
            len(dataframe)
        ),
        "cities": int(
            dataframe["city"].nunique()
        ),
        "start": str(
            dataframe["timestamp"].min()
        ),
        "end": str(
            dataframe["timestamp"].max()
        ),
        "mean_target_24h": float(
            dataframe[
                "target_aqi_24h"
            ].mean()
        ),
        "mean_target_48h": float(
            dataframe[
                "target_aqi_48h"
            ].mean()
        ),
        "mean_target_72h": float(
            dataframe[
                "target_aqi_72h"
            ].mean()
        ),
    }


def build_walk_forward_folds(
    dataframe: pd.DataFrame,
) -> dict:
    report = {}

    for definition in FOLDS:
        name = definition["name"]

        print(
            "\n" + "=" * 90
        )

        print(
            name.upper()
        )

        print(
            "=" * 90
        )

        train = slice_period(
            dataframe,
            definition[
                "train_start"
            ],
            definition[
                "train_end"
            ],
        )

        validation = slice_period(
            dataframe,
            definition[
                "validation_start"
            ],
            definition[
                "validation_end"
            ],
        )

        validate_purge_gap(
            train,
            validation,
            name,
        )

        validate_cities(
            train,
            validation,
            name,
        )

        validate_overlap(
            train,
            validation,
            name,
        )

        fold_dir = (
            OUTPUT_DIR / name
        )

        fold_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        train_path = (
            fold_dir
            / "train.parquet"
        )

        validation_path = (
            fold_dir
            / "validation.parquet"
        )

        save_split(
            train,
            train_path,
        )

        save_split(
            validation,
            validation_path,
        )

        train_report = (
            describe_split(
                train
            )
        )

        validation_report = (
            describe_split(
                validation
            )
        )

        report[name] = {
            "train": train_report,
            "validation": (
                validation_report
            ),
        }

        print(
            f"  Train rows: "
            f"{len(train):,}"
        )

        print(
            f"  Train range: "
            f"{train_report['start']} "
            f"-> "
            f"{train_report['end']}"
        )

        print(
            f"  Validation rows: "
            f"{len(validation):,}"
        )

        print(
            f"  Validation range: "
            f"{validation_report['start']} "
            f"-> "
            f"{validation_report['end']}"
        )

        print(
            "  Validation target means:"
        )

        print(
            f"    24h: "
            f"{validation_report['mean_target_24h']:.2f}"
        )

        print(
            f"    48h: "
            f"{validation_report['mean_target_48h']:.2f}"
        )

        print(
            f"    72h: "
            f"{validation_report['mean_target_72h']:.2f}"
        )

    return report


def build_final_test(
    dataframe: pd.DataFrame,
) -> dict:
    print(
        "\n" + "=" * 90
    )

    print(
        "FINAL TEST SPLIT"
    )

    print(
        "=" * 90
    )

    train = slice_period(
        dataframe,
        FINAL_TEST[
            "train_start"
        ],
        FINAL_TEST[
            "train_end"
        ],
    )

    test = slice_period(
        dataframe,
        FINAL_TEST[
            "test_start"
        ],
        FINAL_TEST[
            "test_end"
        ],
    )

    validate_purge_gap(
        train,
        test,
        "final_test",
    )

    validate_cities(
        train,
        test,
        "final_test",
    )

    validate_overlap(
        train,
        test,
        "final_test",
    )

    final_dir = (
        OUTPUT_DIR / "final"
    )

    final_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_split(
        train,
        final_dir
        / "train.parquet",
    )

    save_split(
        test,
        final_dir
        / "test.parquet",
    )

    train_report = (
        describe_split(
            train
        )
    )

    test_report = (
        describe_split(
            test
        )
    )

    print(
        f"  Train rows: "
        f"{len(train):,}"
    )

    print(
        f"  Train range: "
        f"{train_report['start']} "
        f"-> "
        f"{train_report['end']}"
    )

    print(
        f"  Test rows: "
        f"{len(test):,}"
    )

    print(
        f"  Test range: "
        f"{test_report['start']} "
        f"-> "
        f"{test_report['end']}"
    )

    print(
        "  Test target means:"
    )

    print(
        f"    24h: "
        f"{test_report['mean_target_24h']:.2f}"
    )

    print(
        f"    48h: "
        f"{test_report['mean_target_48h']:.2f}"
    )

    print(
        f"    72h: "
        f"{test_report['mean_target_72h']:.2f}"
    )

    return {
        "train": train_report,
        "test": test_report,
    }


def main() -> None:
    print(
        "=" * 90
    )

    print(
        "PEARLS AQI - EXPANDED "
        "WALK-FORWARD SPLITS"
    )

    print(
        "=" * 90
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = load_data()

    print(
        f"\nInput rows: "
        f"{len(dataframe):,}"
    )

    print(
        f"Cities: "
        f"{dataframe['city'].nunique()}"
    )

    print(
        f"Range: "
        f"{dataframe['timestamp'].min()} "
        f"-> "
        f"{dataframe['timestamp'].max()}"
    )

    fold_report = (
        build_walk_forward_folds(
            dataframe
        )
    )

    final_report = (
        build_final_test(
            dataframe
        )
    )

    report = {
        "purge_hours": (
            PURGE_HOURS
        ),
        "folds": fold_report,
        "final": final_report,
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

    print(
        "\n" + "=" * 90
    )

    print(
        "SPLIT CREATION COMPLETE"
    )

    print(
        "=" * 90
    )

    print(
        f"\nReport:\n"
        f"{REPORT_PATH}"
    )


if __name__ == "__main__":
    main()