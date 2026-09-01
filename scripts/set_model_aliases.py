from __future__ import annotations

import mlflow
from mlflow import MlflowClient


TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "pearls-aqi-random-forest"

HORIZON_ALIASES = {
    "24h": "champion-24h",
    "48h": "champion-48h",
    "72h": "champion-72h",
}


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)

    client = MlflowClient()

    versions = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )

    if not versions:
        raise RuntimeError(
            f"No versions found for '{MODEL_NAME}'."
        )

    assigned = set()

    print("=" * 70)
    print("SETTING MODEL ALIASES")
    print("=" * 70)

    for version in versions:
        details = client.get_model_version(
            name=MODEL_NAME,
            version=version.version,
        )

        horizon = details.tags.get(
            "forecast_horizon"
        )

        if horizon not in HORIZON_ALIASES:
            continue

        alias = HORIZON_ALIASES[horizon]

        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias=alias,
            version=version.version,
        )

        assigned.add(horizon)

        print(
            f"{horizon}: version "
            f"{version.version} -> @{alias}"
        )

    missing = (
        set(HORIZON_ALIASES) - assigned
    )

    if missing:
        raise RuntimeError(
            "Missing registered model versions "
            f"for: {sorted(missing)}"
        )

    print("=" * 70)
    print("ALIASES SET SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()