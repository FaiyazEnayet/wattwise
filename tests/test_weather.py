"""Weather correlation tests: known slopes are recovered."""

import numpy as np
import pandas as pd

from wattwise.weather import temperature_sensitivity, degree_days


def _synthetic_weather_load(seed=5, days=90, cool_slope=1.5, heat_slope=0.8,
                            temp_mean=75.0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-07-14", periods=days * 24, freq="h")
    temp = temp_mean + 8 * np.sin(2 * np.pi * (idx.hour - 9) / 24) \
        + 6 * np.sin(2 * np.pi * np.arange(len(idx)) / (24 * 6)) \
        + rng.normal(0, 1.0, len(idx))
    temp_f = pd.Series(temp, index=idx)
    dd = degree_days(temp_f)
    daily_base = 10 + rng.normal(0, 0.5, days)
    daily_kwh = daily_base + cool_slope * dd["cdd"] + heat_slope * dd["hdd"]
    hourly = pd.Series(np.repeat((daily_kwh / 24).to_numpy(), 24), index=idx)
    return hourly, temp_f


def test_recovers_cooling_slope():
    hourly, temp = _synthetic_weather_load(cool_slope=1.5, heat_slope=0.0)
    out = temperature_sensitivity(hourly, temp)
    assert abs(out["cooling_kwh_per_cdd"] - 1.5) < 0.25
    assert out["r_squared"] > 0.8


def test_recovers_heating_slope():
    hourly, temp = _synthetic_weather_load(cool_slope=0.0, heat_slope=0.8,
                                           temp_mean=50.0)
    out = temperature_sensitivity(hourly, temp)
    assert abs(out["heating_kwh_per_hdd"] - 0.8) < 0.25


def test_degree_days_definition():
    idx = pd.date_range("2026-10-01", periods=48, freq="h")
    temp = pd.Series([70.0] * 24 + [60.0] * 24, index=idx)
    dd = degree_days(temp)
    assert abs(dd["cdd"].iloc[0] - 5.0) < 1e-9
    assert abs(dd["hdd"].iloc[1] - 5.0) < 1e-9
    assert dd["cdd"].iloc[1] == 0.0
