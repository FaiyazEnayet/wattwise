"""Regenerate the anonymous synthetic fixture for public CI.

Writes tests/fixtures/synthetic/ from the declared synthetic process
in wattwise.synthetic (invented parameters, seeded generator; no
measured household data is read). Output is deterministic: rerunning
this script reproduces the committed files byte for byte, which the
WW12 tests assert.

Usage: python scripts/make_synthetic_fixture.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wattwise import household as hh  # noqa: E402
from wattwise import synthetic  # noqa: E402

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "synthetic"


def _intervals_csv(frame) -> bytes:
    lines = ["timestamp,kwh"]
    for ts, value in frame["kwh"].items():
        lines.append(f"{ts:%Y-%m-%dT%H:%M:%S},{value:.4f}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _bills_csv(bills) -> bytes:
    lines = ["period_start,period_end,kwh,bill_usd"]
    for row in bills.itertuples():
        lines.append(
            f"{row.period_start:%Y-%m-%d},{row.period_end:%Y-%m-%d},"
            f"{row.kwh:.4f},{row.bill_usd:.2f}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _manifest() -> dict:
    homes = []
    for hid in sorted(synthetic.SYNTHETIC_HOMES):
        spec = synthetic.SYNTHETIC_HOMES[hid]
        homes.append({
            "household_id": hid,
            "seed": spec["seed"],
            "tariff": spec["tariff"],
            "intervals_file": f"{hid}_intervals.csv",
            "bills_file": f"{hid}_bills.csv",
        })
    return {
        "schema": "wattwise-synthetic-fixture/1",
        "generation": {
            "method": synthetic.GENERATION_METHOD,
            "window": "62 days of hourly intervals from 2026-01-05",
            "rounding": "energy rounded to 4 decimal places; bills to cents",
        },
        "households": homes,
    }


def render_fixture_files() -> dict:
    """Map of fixture-relative filename to exact bytes."""
    files = {}
    datasets = []
    for hid in sorted(synthetic.SYNTHETIC_HOMES):
        intervals = synthetic.make_synthetic_intervals(hid)
        bills = synthetic.make_synthetic_bills(hid, intervals)
        files[f"{hid}_intervals.csv"] = _intervals_csv(intervals)
        files[f"{hid}_bills.csv"] = _bills_csv(bills)
        datasets.append(hh.HouseholdDataset(
            household_id=hid, intervals=intervals, bills=bills,
            tariff=hh.tariff_from_spec(synthetic.SYNTHETIC_HOMES[hid]["tariff"])))
    files["households.json"] = (
        json.dumps(_manifest(), indent=2, sort_keys=True) + "\n").encode("utf-8")
    files["example_report.json"] = (
        hh.report_bytes(hh.build_household_report(datasets)) + b"\n")
    return files


def main() -> int:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for rel, data in render_fixture_files().items():
        (FIXTURE_DIR / rel).write_bytes(data)
        print(f"wrote tests/fixtures/synthetic/{rel} ({len(data)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
