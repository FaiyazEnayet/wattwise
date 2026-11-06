"""WW3: consecutive-cycle scheduling in the Python optimizer.

The LP optimizer (optimize_load_shift) is an idealized lower bound: it can
split an appliance cycle across non-adjacent hours. The MILP path
(optimize_consecutive_cycles) must schedule each noninterruptible appliance
as one contiguous block at nameplate power, respect an optional household
power cap, match brute-force enumeration on tiny instances, and never beat
the relaxed LP lower bound beyond tolerance.

Fixtures are synthetic toy days; no measured usage data is involved.
"""

import itertools

import numpy as np
import pandas as pd
import pytest

from wattwise.optimize import optimize_consecutive_cycles, optimize_load_shift

COLS = ["baseload", "water_heater", "dishwasher", "laundry"]


def _day(entries):
    """One day: 1 kWh baseload spread over the day plus flexible kWh.

    entries: {appliance: (kwh, hour)} placed in a single hour.
    """
    idx = pd.date_range("2026-10-07", periods=24, freq="h")
    end_use = pd.DataFrame(0.0, index=idx, columns=COLS)
    end_use["baseload"] = 1.0 / 24
    for name, (kwh, hour) in entries.items():
        end_use.loc[idx[hour], name] = kwh
    hourly = end_use.sum(axis=1)
    return hourly, end_use


def _blocks(schedule, name):
    """Hours where the appliance runs, as a sorted list."""
    return [h for h in range(24) if schedule[name].iloc[h] > 1e-9]


def _is_contiguous(hours):
    return hours == list(range(hours[0], hours[0] + len(hours)))


def test_lp_splits_cycle_milp_does_not():
    hourly, end_use = _day({"dishwasher": (2.2, 0)})
    prices = np.full(24, 0.50)
    prices[2] = 0.05
    prices[5] = 0.05
    flex = {"dishwasher": 1.1}
    lp = optimize_load_shift(hourly, end_use, prices, flexible=flex)
    # The LP splits the 2-hour cycle across the two cheap, non-adjacent hours.
    assert abs(lp.schedule["dishwasher"].iloc[2] - 1.1) < 1e-6
    assert abs(lp.schedule["dishwasher"].iloc[5] - 1.1) < 1e-6
    r = optimize_consecutive_cycles(hourly, end_use, prices, flexible=flex)
    hours = _blocks(r.schedule, "dishwasher")
    assert len(hours) == 2
    assert _is_contiguous(hours)
    assert all(abs(r.schedule["dishwasher"].iloc[h] - 1.1) < 1e-6 for h in hours)
    assert abs(r.schedule["dishwasher"].sum() - 2.2) < 1e-6


def test_power_cap_respected():
    hourly, end_use = _day({"dishwasher": (2.2, 0), "laundry": (5.6, 0)})
    prices = np.full(24, 0.50)
    prices[3] = 0.05
    flex = {"dishwasher": 1.1, "laundry": 2.8}
    lp = optimize_load_shift(hourly, end_use, prices, flexible=flex)
    assert lp.hourly_optimized.max() > 3.0 + 1e-9
    r = optimize_consecutive_cycles(hourly, end_use, prices, flexible=flex,
                                    power_cap_kw=3.0)
    assert (r.hourly_optimized <= 3.0 + 1e-9).all()
    for name in ("dishwasher", "laundry"):
        hours = _blocks(r.schedule, name)
        assert _is_contiguous(hours)
        assert all(abs(r.schedule[name].iloc[h] - flex[name]) < 1e-6
                   for h in hours)


def test_matches_enumeration():
    hourly, end_use = _day({"dishwasher": (2.2, 0), "laundry": (2.8, 0)})
    prices = np.array([0.40, 0.10, 0.30, 0.20, 0.50, 0.15] + [0.90] * 18)
    flex = {"dishwasher": 1.1, "laundry": 2.8}
    r = optimize_consecutive_cycles(hourly, end_use, prices, flexible=flex,
                                    window=(0, 6))
    inflex = np.full(24, 1.0 / 24)
    best = None
    for d_start in range(0, 5):
        for l_start in range(0, 6):
            load = inflex.copy()
            for h in range(d_start, d_start + 2):
                load[h] += 1.1
            load[l_start] += 2.8
            cost = float((load * prices).sum())
            if best is None or cost < best:
                best = cost
    assert best is not None
    assert abs(r.cost_after - best) < 1e-6
    assert r.start_hours["dishwasher"] in range(0, 5)
    assert r.start_hours["laundry"] in range(0, 6)


def test_milp_cost_not_below_lp_bound():
    hourly, end_use = _day({"dishwasher": (2.2, 0)})
    prices = np.full(24, 0.50)
    prices[2] = 0.05
    prices[5] = 0.05
    flex = {"dishwasher": 1.1}
    lp = optimize_load_shift(hourly, end_use, prices, flexible=flex)
    r = optimize_consecutive_cycles(hourly, end_use, prices, flexible=flex)
    assert abs(r.lower_bound_cost - lp.cost_after) < 1e-6
    assert r.cost_after >= r.lower_bound_cost - 1e-6
    # The split LP bound is strictly cheaper here: 2.2 kWh at $0.05.
    assert abs(lp.cost_after - (0.11 + float((np.full(24, 1.0 / 24) * prices).sum()))) < 1e-6
    assert r.cost_after > lp.cost_after


def test_interruptible_appliance_stays_continuous():
    hourly, end_use = _day({"water_heater": (9.0, 0), "dishwasher": (2.2, 0)})
    prices = np.full(24, 0.50)
    prices[2] = 0.05
    prices[5] = 0.05
    flex = {"water_heater": 4.5, "dishwasher": 1.1}
    r = optimize_consecutive_cycles(hourly, end_use, prices, flexible=flex)
    # Thermal storage may split across the two cheap non-adjacent hours.
    assert abs(r.schedule["water_heater"].iloc[2] - 4.5) < 1e-6
    assert abs(r.schedule["water_heater"].iloc[5] - 4.5) < 1e-6
    # The dishwasher cycle stays in one contiguous block.
    hours = _blocks(r.schedule, "dishwasher")
    assert len(hours) == 2
    assert _is_contiguous(hours)


def test_window_respected():
    hourly, end_use = _day({"dishwasher": (2.2, 0)})
    prices = np.full(24, 0.10)
    prices[11] = 0.05
    prices[12] = 0.20  # makes the 10:00 start uniquely cheapest
    r = optimize_consecutive_cycles(hourly, end_use, prices,
                                    flexible={"dishwasher": 1.1},
                                    window=(10, 14))
    hours = _blocks(r.schedule, "dishwasher")
    assert hours == [10, 11]
    assert r.start_hours["dishwasher"] == 10


def test_non_integer_cycle_length_rejected():
    hourly, end_use = _day({"dishwasher": (2.0, 0)})
    prices = np.full(24, 0.10)
    with pytest.raises(ValueError, match="whole-hour cycle"):
        optimize_consecutive_cycles(hourly, end_use, prices,
                                    flexible={"dishwasher": 1.1})


def test_zero_demand_appliance_skipped():
    hourly, end_use = _day({"water_heater": (2.0, 0)})
    prices = np.full(24, 0.10)
    prices[3] = 0.05
    r = optimize_consecutive_cycles(hourly, end_use, prices)
    assert (r.schedule["dishwasher"].abs() < 1e-9).all()
    assert (r.schedule["laundry"].abs() < 1e-9).all()
    assert r.start_hours["dishwasher"] is None
    assert abs(r.schedule["water_heater"].sum() - 2.0) < 1e-6


def test_infeasible_power_cap_raises():
    hourly, end_use = _day({"dishwasher": (2.2, 0)})
    prices = np.full(24, 0.10)
    with pytest.raises(RuntimeError, match="MILP failed"):
        optimize_consecutive_cycles(hourly, end_use, prices,
                                    flexible={"dishwasher": 1.1},
                                    power_cap_kw=0.01)
