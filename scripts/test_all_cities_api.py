from __future__ import annotations

import time

import httpx


BASE_URL = "http://127.0.0.1:8000/api/v1"


CITIES = [
    "Faisalabad",
    "Islamabad",
    "Karachi",
    "Lahore",
    "Multan",
    "Peshawar",
    "Quetta",
    "Rahim Yar Khan",
    "Sialkot",
]


def main() -> None:
    print("=" * 90)
    print("PEARLS AQI - ALL CITIES REST API TEST")
    print("=" * 90)

    passed = 0
    failed = 0

    with httpx.Client(timeout=120.0) as client:
        for index, city in enumerate(CITIES, start=1):
            print()
            print(f"[{index}/{len(CITIES)}] {city}")

            started = time.perf_counter()

            try:
                response = client.get(
                    f"{BASE_URL}/forecast/{city}"
                )

                response.raise_for_status()

                payload = response.json()

                assert payload["city"] == city
                assert 0 <= payload["current"]["aqi"] <= 500
                assert payload["metadata"]["feature_count"] == 56
                assert len(payload["forecasts"]) == 3

                horizons = [
                    item["horizon_hours"]
                    for item in payload["forecasts"]
                ]

                assert horizons == [24, 48, 72]

                for item in payload["forecasts"]:
                    assert 0 <= item["aqi"] <= 500
                    assert item["category"]
                    assert item["model"]["alias"]
                    assert item["explanation"]["top_features"]

                elapsed = time.perf_counter() - started

                print(
                    f"PASS | Current AQI: "
                    f"{payload['current']['aqi']:.2f} | "
                    f"{elapsed:.2f}s"
                )

                passed += 1

            except Exception as exc:
                elapsed = time.perf_counter() - started

                print(
                    f"FAIL | {elapsed:.2f}s | "
                    f"{type(exc).__name__}: {exc}"
                )

                failed += 1

    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"Passed: {passed}/{len(CITIES)}")
    print(f"Failed: {failed}/{len(CITIES)}")

    if failed:
        raise SystemExit(1)

    print()
    print("ALL CITIES REST API TEST: PASS")


if __name__ == "__main__":
    main()