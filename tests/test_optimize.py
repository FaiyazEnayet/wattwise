"""Optimizer tests: toy cases with known optima."""

import numpy as np
import pandas as pd

from wattwise.optimize import optimize_load_shift, optimize_month
from wattwise.rates import Rate194


def _day(flex_kwh=2.0, flex_hour=0):
    """One day: flexible kWh sitting in flex_hour, plus 1 kWh baseload."""
    idx = pd.date_range("2026-10-07", periods=24, freq="h")
    end_use = pd.DataFrame(0.0, index=idx,
                           columns=["baseload", "water_heater"])
    end_use["baseload"] = 1.0 / 24
    end_use.loc[idx[flex_hour], "water_heater"] = flex_kwh
    hourly = end_use.sum(axis=1)
    return hourly, end_use


def test_shifts_to_cheapest_hour():
    hourly, end_use = _day(flex_kwh=2.0, flex_hour=0)
    prices = np.full(24, 0.10)
    prices[0] = 0.50  # expensive midnight hour
    r = optimize_load_shift(hourly, end_use, prices,
                            flexible={"water_heater": 2.0})
    # Known optimum: all 2 kWh move to a $0.10 hour. Baseload (1 kWh
    # spread over the day) stays: (0.50 + 23*0.10)/24 = $0.1167.
    assert abs(r.cost_after - (0.20 + 0.1167)) < 1e-4
    assert abs(r.cost_before - (1.00 + 0.1167)) < 1e-4
    assert abs(r.savings - 0.80) < 1e-6
    # Nothing scheduled in the expensive hour.
    assert r.schedule["water_heater"].iloc[0] < 1e-6


def test_window_constraint_respected():
    hourly, end_use = _day(flex_kwh=2.0, flex_hour=0)
    prices = np.full(24, 0.10)
    prices[0] = 0.50
    # Window only allows the expensive hour -> no shift possible.
    r = optimize_load_shift(hourly, end_use, prices,
                            flexible={"water_heater": 2.0}, window=(0, 1))
    assert abs(r.savings) < 1e-6
    assert abs(r.schedule["water_heater"].iloc[0] - 2.0) < 1e-6


def test_energy_conserved():
    hourly, end_use = _day()
    prices = np.linspace(0.1, 0.5, 24)
    r = optimize_load_shift(hourly, end_use, prices,
                            flexible={"water_heater": 2.0})
    assert abs(r.schedule["water_heater"].sum() - 2.0) < 1e-6
    assert abs(r.hourly_optimized.sum() - hourly.sum()) < 1e-6


def test_power_cap_respected():
    hourly, end_use = _day(flex_kwh=6.0, flex_hour=0)
    prices = np.full(24, 0.10)
    prices[0] = 0.50
    r = optimize_load_shift(hourly, end_use, prices,
                            flexible={"water_heater": 2.0})
    # 6 kWh at 2 kW cap needs 3 cheap hours.
    assert (r.schedule["water_heater"] <= 2.0 + 1e-6).all()
    assert abs(r.schedule["water_heater"].sum() - 6.0) < 1e-6


def test_flat_rate_never_saves():
    hourly, end_use = _day()
    prices = np.full(24, 0.2413)  # flat
    r = optimize_load_shift(hourly, end_use, prices,
                            flexible={"water_heater": 2.0})
    assert abs(r.savings) < 1e-6


def test_month_aggregation_matches_daily():
    from wattwise import make_demo_profile
    hourly, end_use, _ = make_demo_profile(seed=7)
    month = hourly.iloc[-3 * 24:]
    eu = end_use.iloc[-3 * 24:]
    rate = Rate194()
    agg = optimize_month(month, eu, rate)
    # Recompute by hand over the same 3 days.
    tot_b = tot_a = 0.0
    for d in pd.unique(month.index.date):
        m = month.index.date == d
        h, e = month.loc[m], eu.loc[m]
        p = np.array([rate.marginal_price(ts) for ts in h.index[:24]])
        r = optimize_load_shift(h, e, p)
        tot_b += r.cost_before
        tot_a += r.cost_after
    assert abs(agg.cost_before - tot_b) < 1e-6
    assert abs(agg.cost_after - tot_a) < 1e-6
