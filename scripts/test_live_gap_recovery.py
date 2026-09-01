from __future__ import annotations

import shutil

import pandas as pd

from src.services.live_history_service import (
    LiveHistoryService,
)


CITY = "Islamabad"


def main() -> None:
    print("=" * 80)
    print(
        "PEARLS AQI - LIVE GAP RECOVERY TEST"
    )
    print("=" * 80)

    service = LiveHistoryService()

    path = service.get_history_path(
        CITY
    )

    backup_path = path.with_suffix(
        ".backup.parquet"
    )

    shutil.copy2(
        path,
        backup_path,
    )

    print(
        f"\nBackup created: {backup_path}"
    )

    try:
        original = service.load(
            CITY
        )

        print(
            f"Original rows:   {len(original)}"
        )
        print(
            f"Original latest: "
            f"{original['timestamp'].max()}"
        )

        # Simulate the local process having stopped
        # several hours ago.
        stale = (
            original.iloc[:-4]
            .copy()
            .reset_index(drop=True)
        )

        if len(stale) < 97:
            raise RuntimeError(
                "Not enough rows to construct "
                "the stale-history test."
            )

        service.save(
            CITY,
            stale,
        )

        print(
            f"\nStale rows:      {len(stale)}"
        )
        print(
            f"Stale latest:    "
            f"{stale['timestamp'].max()}"
        )

        print(
            "\nRunning automatic recovery..."
        )

        recovered = (
            service.ensure_current_history(
                CITY
            )
        )

        gaps = (
            recovered["timestamp"]
            .sort_values()
            .diff()
            .dropna()
        )

        invalid_gaps = int(
            (
                gaps
                != pd.Timedelta(hours=1)
            ).sum()
        )

        print(
            f"\nRecovered rows:  {len(recovered)}"
        )
        print(
            f"Recovered start: "
            f"{recovered['timestamp'].min()}"
        )
        print(
            f"Recovered latest:"
            f" {recovered['timestamp'].max()}"
        )
        print(
            f"Invalid gaps:    {invalid_gaps}"
        )
        print(
            "Sources:         "
            f"{recovered['source'].value_counts().to_dict()}"
        )

        if invalid_gaps != 0:
            raise RuntimeError(
                "Gap recovery left discontinuities."
            )

        if (
            recovered.iloc[-1]["source"]
            != "openweather_live"
        ):
            raise RuntimeError(
                "Latest observation is not "
                "OpenWeather live data."
            )

        print("\n" + "=" * 80)
        print(
            "LIVE GAP RECOVERY TEST: PASS"
        )
        print("=" * 80)

    finally:
        if backup_path.exists():
            shutil.copy2(
                backup_path,
                path,
            )

            backup_path.unlink()

            print(
                "\nOriginal history restored."
            )


if __name__ == "__main__":
    main()