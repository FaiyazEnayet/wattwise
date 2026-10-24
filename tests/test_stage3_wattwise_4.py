"""Stage 3 WW4: effective-dated tariff versions and per-year holidays.

Pins:
- `federal_holidays(year)` reproduces the archived 2026 set exactly
  and generates the correct observed-date set for 2027.
- The Rate194 legacy path resolves holidays per timestamp year by
  default instead of defaulting to the 2026-only set.
- The tariff schedule archive selects the schedule in force on a
  date, replay validates an explicit schedule version's date range,
  monthly supply history from the actual bills prices intervals, and
  every counterfactual carries its tariff version.
"""
import datetime as dt

import pandas as pd
import pytest

from wattwise.decision_analysis import (
    TariffSchedule,
    build_report,
    monthly_supply_history,
    schedule_in_force,
    tariff_replay,
)
from wattwise.rates import FEDERAL_HOLIDAYS_2026, Rate194, federal_holidays

ROOT = __file__.rsplit("/", 2)[0]


def test_stage3_wattwise_4_holidays_2026_matches_archive():
    assert federal_holidays(2026) == FEDERAL_HOLIDAYS_2026


def test_stage3_wattwise_4_holidays_2027_observed_dates():
    # OPM 2027: Juneteenth (Sat) observed Fri Jun 18, Independence Day
    # (Sun) observed Mon Jul 5, Christmas (Sat) observed Fri Dec 24,
    # and New Year's Day 2028 (Sat) observed Fri Dec 31, 2027.
    expected = frozenset(
        dt.date(2027, m, d)
        for m, d in [
            (1, 1), (1, 18), (2, 15), (5, 31), (6, 18), (7, 5),
            (9, 6), (10, 11), (11, 11), (11, 25), (12, 24), (12, 31),
        ]
    )
    assert federal_holidays(2027) == expected


def test_stage3_wattwise_4_rate194_defaults_to_per_year_holidays():
    rate = Rate194()
    # Mon Jul 5 2027 is the observed Independence Day holiday: the
    # 2026-only default would call 4 PM peak; per-year it is off-peak.
    assert rate.energy_charge(1.0, dt.datetime(2027, 7, 5, 16)) == pytest.approx(
        rate.energy_charge(1.0, dt.datetime(2027, 7, 5, 10))
    )
    # Tue Jul 6 2027, 4 PM: genuinely peak, priced above the morning.
    assert rate.energy_charge(1.0, dt.datetime(2027, 7, 6, 16)) > rate.energy_charge(
        1.0, dt.datetime(2027, 7, 6, 10)
    )
    # 2026 regression: Columbus Day still off-peak by default.
    assert rate.energy_charge(1.0, dt.datetime(2026, 10, 12, 16)) == pytest.approx(
        rate.energy_charge(1.0, dt.datetime(2026, 10, 12, 10))
    )


def test_stage3_wattwise_4_schedule_in_force_boundaries():
    current = schedule_in_force(dt.date(2026, 6, 1))
    assert current.version == "2026.1"
    with pytest.raises(LookupError):
        schedule_in_force(dt.date(2025, 12, 31))
    # Fixture archive with a closed 2026 entry (the live 2026.1 is
    # open-ended; a successor closes it when published).
    fixture_2026 = TariffSchedule(
        version="2026.1",
        effective_from=dt.date(2026, 1, 1),
        effective_to=dt.date(2026, 12, 31),
        plans=("194",),
        source="fixture",
    )
    future = TariffSchedule(
        version="2027.1",
        effective_from=dt.date(2027, 1, 1),
        effective_to=None,
        plans=("194",),
        source="fixture",
    )
    archive = (fixture_2026, future)
    assert schedule_in_force(dt.date(2026, 12, 31), archive).version == "2026.1"
    assert schedule_in_force(dt.date(2027, 1, 1), archive).version == "2027.1"


def _tiny_intervals(start, periods=8):
    idx = pd.date_range(start, periods=periods, freq="15min")
    return pd.DataFrame({"kwh": 0.25, "kw": 1.0}, index=idx)


def test_stage3_wattwise_4_replay_validates_schedule_range():
    iv26 = _tiny_intervals("2026-06-01 16:00")
    costs_ok, _ = tariff_replay(iv26, "194", schedule_version="2026.1")
    costs_default, _ = tariff_replay(iv26, "194")
    assert costs_ok.equals(costs_default)
    iv25 = _tiny_intervals("2025-10-06 16:00")
    with pytest.raises(ValueError, match="2026.1"):
        tariff_replay(iv25, "194", schedule_version="2026.1")


def test_stage3_wattwise_4_supply_history_from_actual_bills():
    details = pd.DataFrame(
        {
            "period_start": ["2026-01-05", "2026-02-03"],
            "kwh": [200.0, 250.0],
            "supply_usd": [24.0, 35.0],
        }
    )
    hist = monthly_supply_history(details)
    assert hist == {"2026-01": pytest.approx(0.12), "2026-02": pytest.approx(0.14)}
    iv = _tiny_intervals("2026-01-12 16:00")
    costs_hist, _ = tariff_replay(iv, "194", supply_history=hist)
    costs_scalar, _ = tariff_replay(iv, "194", supply=0.12)
    assert costs_hist.equals(costs_scalar)
    with pytest.raises(ValueError, match="supply history"):
        tariff_replay(_tiny_intervals("2026-03-02 16:00"), "194",
                      supply_history=hist)


def test_stage3_wattwise_4_counterfactuals_carry_tariff_version():
    report = build_report(ROOT)
    assert report["tariffs"], "report has no tariff counterfactuals"
    for entry in report["tariffs"]:
        assert entry["tariff_version"] == "2026.1"
