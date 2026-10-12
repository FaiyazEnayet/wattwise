"""Forecast tests: model recovers known patterns, backtest is sane."""

import numpy as np
import pandas as pd
import pytest

from wattwise.forecast import BillForecaster, forecast_month_bill
from wattwise.rates import Rate194


def _synthetic(days=60, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-08-01", periods=days * 24, freq="h")
    dow = idx.weekday
    # 10 kWh/day base + weekend bump + trend + noise
    daily = 10 + 2 * (dow >= 5).astype(float) + 0.02 * np.arange(len(idx)) / 24
    hourly = pd.Series(
        np.clip(daily / 24 + rng.normal(0, 0.05, len(idx)), 0.1, None),
        index=idx)
    return hourly


def test_recovers_weekly_pattern():
    hourly = _synthetic()
    model = BillForecaster().fit(hourly)
    # 2026-10-01 is a Thursday: index 2 = Saturday, 3 = Sunday.
    fc = model.predict_days(pd.date_range("2026-10-01", periods=7, freq="D"))
    assert fc["forecast_kwh"].iloc[2] > fc["forecast_kwh"].iloc[0]
    assert fc["forecast_kwh"].iloc[3] > fc["forecast_kwh"].iloc[4]


def test_backtest_error_bounded():
    hourly = _synthetic(days=90)
    cut = hourly.index[0] + pd.Timedelta(days=60)
    model = BillForecaster().fit(hourly.loc[:cut])
    test_days = pd.date_range(cut + pd.Timedelta(days=1), periods=7, freq="D")
    fc = model.predict_days(test_days)
    actual = hourly.loc[test_days[0]:test_days[-1] + pd.Timedelta(hours=23)].sum()
    err = abs(fc["forecast_kwh"].sum() - actual) / actual
    assert err < 0.15, f"err={err}"


def test_intervals_ordered():
    hourly = _synthetic()
    model = BillForecaster().fit(hourly)
    fc = model.predict_days(pd.date_range("2026-10-01", periods=5, freq="D"))
    assert ((fc["lo_kwh"] <= fc["forecast_kwh"])
            & (fc["forecast_kwh"] <= fc["hi_kwh"])).all()


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        BillForecaster().predict_days(
            pd.date_range("2026-10-01", periods=2, freq="D"))


def test_month_bill_structure():
    hourly = _synthetic(days=40)
    out = forecast_month_bill(hourly, Rate194())
    assert out["bill_lo"] <= out["bill_forecast"] <= out["bill_hi"]
    assert out["days_so_far"] + len(out["daily_forecast"]) == out["days_in_month"]
    assert out["forecast_kwh"] > 0
