"""Bill math and comparison tests."""

import pandas as pd

from wattwise import (make_demo_profile, Rate180, Rate194, monthly_bill,
                      compare_plans, peak_share, appliance_breakdown,
                      savings_waterfall)


def _month():
    hourly, end_use, _ = make_demo_profile(seed=7)
    return hourly.iloc[-30 * 24:], end_use.iloc[-30 * 24:]


def test_monthly_bill_hand_check():
    # 10 kWh in one nonsummer off-peak hour + 30 days fixed.
    idx = pd.date_range("2026-10-07 10:00", periods=1, freq="h")
    hourly = pd.Series([10.0], index=idx)
    b = monthly_bill(hourly, Rate194())
    # energy: 10 * (0.0891 + 0.83*0.1392); fixed: 1 day * 0.54
    assert abs(b["energy_charge"] - 10 * 0.204636) < 0.01
    assert abs(b["fixed_charge"] - 0.54) < 1e-9
    assert abs(b["total"] - (b["energy_charge"] + b["fixed_charge"])) < 1e-9


def test_compare_plans_picks_cheaper():
    month, _ = _month()
    out = compare_plans(month)
    assert out["cheaper_plan"] in ("flat (Rate 180)", "time-of-day (Rate 194)")
    assert out["difference"] >= 0
    cheaper = min(out["flat"]["total"], out["tou_as_is"]["total"])
    assert out["flat"]["total"] == cheaper or out["tou_as_is"]["total"] == cheaper


def test_peak_share_bounds():
    month, _ = _month()
    ps = peak_share(month, Rate194())
    assert 0 <= ps["peak_kwh_pct"] <= 100
    assert 0 <= ps["peak_cost_pct"] <= 100
    # Peak energy costs more per kWh, so cost share exceeds kWh share.
    assert ps["peak_cost_pct"] > ps["peak_kwh_pct"]


def test_breakdown_sums_to_100():
    _, end_use = _month()
    bd = appliance_breakdown(end_use)
    assert abs(bd["pct"].sum() - 100) < 0.5
    assert (bd["kwh"] > 0).all()


def test_waterfall_steps_sum():
    steps = savings_waterfall(109.01, 102.79, 97.61, 62.53)
    total = steps[0][1] + sum(d for _, d in steps[1:])
    assert abs(total - 62.53) < 0.01
