"""What-if scenario tests: direction and sanity of each scenario."""

from wattwise import (make_demo_profile, Rate194, solar_scenario,
                      ev_scenario, heat_pump_scenario)


def _month():
    hourly, _, temp = make_demo_profile(seed=7)
    return hourly.iloc[-30 * 24:], temp.iloc[-30 * 24:]


def test_solar_reduces_bill():
    month, _ = _month()
    out = solar_scenario(month, Rate194(), system_kw=6.0)
    assert out["bill_after"] < out["bill_before"]
    assert out["monthly_saving"] > 0
    assert out["monthly_gen_kwh"] > 0
    assert out["simple_payback_years"] > 0


def test_solar_bigger_system_saves_more():
    month, _ = _month()
    small = solar_scenario(month, Rate194(), system_kw=4.0)
    big = solar_scenario(month, Rate194(), system_kw=8.0)
    assert big["monthly_saving"] > small["monthly_saving"]


def test_ev_adds_cost_but_cheap_per_mile():
    month, _ = _month()
    out = ev_scenario(month, Rate194())
    assert out["added_monthly_cost"] > 0
    assert out["extra_kwh_per_month"] > 0
    # Overnight off-peak EV fuel should beat gasoline by a mile.
    assert out["fuel_cost_per_mile_cents"] < 10.0


def test_heat_pump_adds_winterish_load():
    month, temp = _month()
    out = heat_pump_scenario(month, temp, Rate194(), cop=3.0)
    assert out["added_monthly_cost"] >= 0
    # Better COP -> less added cost.
    bad = heat_pump_scenario(month, temp, Rate194(), cop=1.5)
    assert out["added_monthly_cost"] <= bad["added_monthly_cost"]


def test_assumptions_documented():
    month, temp = _month()
    for out in (solar_scenario(month, Rate194()),
                ev_scenario(month, Rate194()),
                heat_pump_scenario(month, temp, Rate194())):
        assert isinstance(out["assumptions"], str) and len(out["assumptions"]) > 10
