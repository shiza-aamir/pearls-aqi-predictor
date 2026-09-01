from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.services.explanation_service import (
    AQIExplanationService,
)


DATA_PATH = Path(
    "data/processed/expanded/"
    "ml_ready_aqi_dataset.parquet"
)

OUTPUT_ROOT = Path(
    "artifacts/explainability"
)

SAMPLE_SIZE = 2_000
RANDOM_SEED = 42

HORIZONS = [
    "24h",
    "48h",
    "72h",
]


def main() -> None:
    print("=" * 80)
    print(
        "PEARLS AQI - SHAP GLOBAL EXPLAINABILITY"
    )
    print("=" * 80)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_parquet(
        DATA_PATH
    )

    service = (
        AQIExplanationService()
    )

    feature_columns = (
        service.feature_columns
    )

    missing = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Dataset missing features: {missing}"
        )

    if len(df) < SAMPLE_SIZE:
        raise RuntimeError(
            "Dataset is smaller than SHAP "
            "sample size."
        )

    sample = (
        df.sample(
            n=SAMPLE_SIZE,
            random_state=RANDOM_SEED,
        )
        .sort_index()
        .reset_index(drop=True)
    )

    sample_features = sample[
        feature_columns
    ].copy()

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"\nDataset rows: {len(df):,}"
    )
    print(
        f"SHAP sample:  {len(sample):,}"
    )
    print(
        f"Features:     {len(feature_columns)}"
    )

    overall_report = {
        "analysis": (
            "GLOBAL_XGBOOST_SHAP_EXPLAINABILITY"
        ),
        "dataset": str(DATA_PATH),
        "dataset_rows": int(
            len(df)
        ),
        "sample_rows": int(
            len(sample)
        ),
        "random_seed": RANDOM_SEED,
        "feature_count": int(
            len(feature_columns)
        ),
        "horizons": {},
    }

    for horizon in HORIZONS:
        print("\n" + "-" * 80)
        print(
            f"ANALYZING {horizon}"
        )
        print("-" * 80)

        horizon_dir = (
            OUTPUT_ROOT / horizon
        )

        horizon_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            prepared,
            shap_values,
            base_value,
        ) = service.shap_values(
            sample_features,
            horizon,
        )

        mean_abs_shap = np.mean(
            np.abs(shap_values),
            axis=0,
        )

        importance = pd.DataFrame(
            {
                "feature": (
                    feature_columns
                ),
                "mean_abs_shap": (
                    mean_abs_shap
                ),
            }
        )

        importance = (
            importance
            .sort_values(
                "mean_abs_shap",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        importance[
            "rank"
        ] = (
            np.arange(
                1,
                len(importance) + 1,
            )
        )

        importance = importance[
            [
                "rank",
                "feature",
                "mean_abs_shap",
            ]
        ]

        importance_path = (
            horizon_dir
            / "global_feature_importance.csv"
        )

        importance.to_csv(
            importance_path,
            index=False,
        )

        plt.figure()

        shap.summary_plot(
            shap_values,
            prepared,
            feature_names=feature_columns,
            max_display=20,
            show=False,
        )

        plt.tight_layout()

        summary_path = (
            horizon_dir
            / "shap_summary.png"
        )

        plt.savefig(
            summary_path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close()

        plt.figure()

        shap.summary_plot(
            shap_values,
            prepared,
            feature_names=feature_columns,
            plot_type="bar",
            max_display=20,
            show=False,
        )

        plt.tight_layout()

        bar_path = (
            horizon_dir
            / "shap_importance_bar.png"
        )

        plt.savefig(
            bar_path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close()

        top_10 = (
            importance
            .head(10)
        )

        print(
            f"Base value: {base_value:.4f}"
        )

        print("\nTop 10 features:")

        for row in top_10.itertuples(
            index=False
        ):
            print(
                f"  {row.rank:02}. "
                f"{row.feature:<30} "
                f"{row.mean_abs_shap:.4f}"
            )

        horizon_report = {
            "horizon": horizon,
            "base_value": base_value,
            "sample_rows": int(
                len(prepared)
            ),
            "feature_count": int(
                len(feature_columns)
            ),
            "top_10_features": (
                top_10
                .to_dict(
                    orient="records"
                )
            ),
            "importance_csv": str(
                importance_path
            ),
            "summary_plot": str(
                summary_path
            ),
            "bar_plot": str(
                bar_path
            ),
        }

        report_path = (
            horizon_dir
            / "shap_report.json"
        )

        with report_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                horizon_report,
                file,
                indent=2,
            )

        overall_report[
            "horizons"
        ][horizon] = (
            horizon_report
        )

    overall_path = (
        OUTPUT_ROOT
        / "shap_global_report.json"
    )

    with overall_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            overall_report,
            file,
            indent=2,
        )

    print("\n" + "=" * 80)
    print(
        "GLOBAL SHAP EXPLAINABILITY: PASS"
    )
    print("=" * 80)

    print(
        f"\nArtifacts:\n  {OUTPUT_ROOT}"
    )


if __name__ == "__main__":
    main()