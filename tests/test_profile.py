"""Synthetic profile tests: shape, calibration, determinism."""

import pandas as pd

from wattwise import make_demo_profile, Rate180
from wattwise.analyze import monthly_bill


def test_shape_and_index():
    hourly, end_use, temp = make_demo_profile(seed=7)
    assert len(hourly) == 90 * 24
    assert isinstance(hourly.index, pd.DatetimeIndex)
    assert (end_use.sum(axis=1) - hourly).abs().max() < 1e-9
    assert (hourly >= 0).all()


def test_bill_calibration_near_109():
    hourly, _, _ = make_demo_profile(seed=7)
    bill = monthly_bill(hourly.iloc[-30 * 24:], Rate180())["total"]
    assert abs(bill - 109.01) < 3.0, f"bill={bill}"


def test_deterministic():
    h1, _, _ = make_demo_profile(seed=7)
    h2, _, _ = make_demo_profile(seed=7)
    assert (h1 - h2).abs().max() == 0.0
    h3, _, _ = make_demo_profile(seed=8)
    assert (h1 - h3).abs().max() > 0.0


def test_end_uses_present():
    _, end_use, _ = make_demo_profile(seed=7)
    for col in ["baseload", "hvac", "water_heater", "dishwasher",
                "laundry", "cooking", "lighting_plug"]:
        assert col in end_use.columns
        assert end_use[col].sum() > 0


def test_weather_reasonable():
    _, _, temp = make_demo_profile(seed=7)
    assert 20 < temp.mean() < 80
    assert temp.max() - temp.min() > 10  # real diurnal + system swing
