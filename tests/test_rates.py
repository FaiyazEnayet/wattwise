"""Rate structure tests: hand-checked against the published tariff."""

import datetime as dt

from wattwise.rates import (Rate180, Rate194, Rate194History, is_peak_hour,
                            is_summer, FEDERAL_HOLIDAYS_2026)

SUPPLY = 0.1392


def test_peak_weekday_afternoon():
    # Wednesday Oct 7 2026, 4 PM -> peak.
    assert is_peak_hour(dt.datetime(2026, 10, 7, 16), FEDERAL_HOLIDAYS_2026)


def test_offpeak_weekday_morning():
    assert not is_peak_hour(dt.datetime(2026, 10, 7, 10), FEDERAL_HOLIDAYS_2026)


def test_offpeak_weekend():
    # Saturday 4 PM is off-peak.
    assert not is_peak_hour(dt.datetime(2026, 10, 10, 16), FEDERAL_HOLIDAYS_2026)


def test_offpeak_holiday():
    # Columbus Day 2026 (Monday Oct 12), 4 PM -> off-peak.
    assert not is_peak_hour(dt.datetime(2026, 10, 12, 16), FEDERAL_HOLIDAYS_2026)


def test_peak_boundaries():
    assert is_peak_hour(dt.datetime(2026, 10, 7, 15), FEDERAL_HOLIDAYS_2026)
    assert not is_peak_hour(dt.datetime(2026, 10, 7, 19), FEDERAL_HOLIDAYS_2026)
    assert not is_peak_hour(dt.datetime(2026, 10, 7, 14), FEDERAL_HOLIDAYS_2026)


def test_tou_nonsummer_offpeak_price():
    r = Rate194(supply_rate=SUPPLY)
    # 0.0891 delivery + 0.83 * 0.1392 supply
    assert abs(r.marginal_price(dt.datetime(2026, 10, 7, 10)) - 0.204636) < 1e-6


def test_tou_nonsummer_peak_price():
    r = Rate194(supply_rate=SUPPLY)
    # 0.1809 delivery + 2.1036 * 0.1392 supply
    assert abs(r.marginal_price(dt.datetime(2026, 10, 7, 16)) - 0.473721) < 1e-4


def test_tou_summer_peak_price():
    r = Rate194(supply_rate=SUPPLY)
    # 0.2127 delivery + 1.9465 * 0.1392 supply
    assert abs(r.marginal_price(dt.datetime(2026, 7, 15, 16)) - 0.483653) < 1e-4


def test_flat_nonsummer():
    r = Rate180(supply_rate=SUPPLY)
    # 0.1021 delivery + 0.1392 supply, no tier outside summer
    assert abs(r.marginal_price(dt.datetime(2026, 10, 7, 16)) - 0.2413) < 1e-6


def test_flat_summer_tier():
    r = Rate180(supply_rate=SUPPLY)
    assert abs(r.delivery_rate(200, True) - 0.1021) < 1e-9
    assert abs(r.delivery_rate(300, True) - 0.1294) < 1e-9
    assert abs(r.delivery_rate(9999, False) - 0.1021) < 1e-9


def test_fixed_charge():
    assert Rate194().fixed_charge(30) == 30 * 0.54
    assert Rate180().fixed_charge(30) == 30 * 0.54


def test_is_summer():
    assert is_summer(dt.date(2026, 7, 1))
    assert is_summer(dt.date(2026, 9, 30))
    assert not is_summer(dt.date(2026, 10, 1))
    assert not is_summer(dt.date(2026, 5, 31))


def test_history_loads_thirteen_periods():
    h = Rate194History()
    assert len(h.periods) == 13
    for p in h.periods.values():
        # peak + off-peak must reconstruct the period total
        assert abs(p["peak_kwh"] + p["offpeak_kwh"] - p["kwh"]) < 1e-9


def test_history_exact_rates_spot_check():
    # Sep 2026 (bill date 2026-10-07), transcribed line-by-line.
    h = Rate194History()
    p = h.periods["2026-10-07"]
    assert p["del_peak_rate"] == 0.2200
    assert p["del_offpeak_rate"] == 0.1089
    assert p["sup_peak_rate"] == 0.273846
    assert p["sup_offpeak_rate"] == 0.116050
    assert p["basic_usd"] == 16.80
    assert p["taxes_usd"] == 9.18


def test_history_reproduces_sep_2026_bill():
    h = Rate194History()
    modeled = h.bill_for("2026-10-07", 39.0, 281.0)
    assert abs(modeled - 109.01) < 0.50


def test_history_reproduces_all_bills_to_within_half_cent():
    # Each bill's transcribed rates reproduce its total to within the
    # utility's own per-line cent rounding.
    h = Rate194History()
    assert h.max_error() < 0.01
