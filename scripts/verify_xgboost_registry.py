from __future__ import annotations

import mlflow
from mlflow import MlflowClient


TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "pearls-aqi-xgboost"

EXPECTED = {
    "champion-24h": "24h",
    "champion-48h": "48h",
    "champion-72h": "72h",
}


def main() -> None:
    mlflow.set_tracking_uri(
        TRACKING_URI
    )

    client = MlflowClient()

    versions = client.search_model_versions(
        filter_string=(
            f"name = '{MODEL_NAME}'"
        )
    )

    print("=" * 70)
    print("PEARLS AQI - XGBOOST REGISTRY")
    print("=" * 70)

    print(
        f"\nModel: {MODEL_NAME}"
    )

    print(
        f"Versions found: {len(versions)}"
    )

    if not versions:
        raise RuntimeError(
            "No XGBoost model versions found."
        )

    for version in sorted(
        versions,
        key=lambda item: int(
            item.version
        ),
    ):
        details = (
            client.get_model_version(
                name=MODEL_NAME,
                version=version.version,
            )
        )

        print("\n" + "-" * 70)

        print(
            f"Version: {details.version}"
        )

        print(
            "Horizon:",
            details.tags.get(
                "forecast_horizon"
            ),
        )

        print(
            "Type:",
            details.tags.get(
                "model_type"
            ),
        )

        print(
            "Status:",
            details.tags.get(
                "deployment_status"
            ),
        )

        print(
            "Aliases:",
            details.aliases,
        )

    print("\n" + "=" * 70)
    print("ALIAS VERIFICATION")
    print("=" * 70)

    for alias, expected_horizon in (
        EXPECTED.items()
    ):
        version = (
            client.get_model_version_by_alias(
                name=MODEL_NAME,
                alias=alias,
            )
        )

        actual_horizon = (
            version.tags.get(
                "forecast_horizon"
            )
        )

        if actual_horizon != expected_horizon:
            raise RuntimeError(
                f"@{alias}: expected "
                f"{expected_horizon}, got "
                f"{actual_horizon}."
            )

        print(
            f"@{alias} -> "
            f"v{version.version} "
            f"({actual_horizon})"
        )

    print(
        "\nRegistry verification: PASS"
    )


if __name__ == "__main__":
    main()