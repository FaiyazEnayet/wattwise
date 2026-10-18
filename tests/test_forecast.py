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


def test_forecast_continues_time_coordinates():
    """WW1: fit on 56 days of exact 10+2t load; the next-week forecast
    must be 122,124,...,134, not 10,12,...,22. Time coordinates persist
    across fit and predict."""
    # 56 days of exact linear load: day t uses 10 + 2*t kWh.
    days = pd.date_range("2026-08-01", periods=56, freq="D")
    daily_kwh = 10 + 2 * np.arange(56, dtype=float)
    # Expand to hourly (constant within each day).
    idx = pd.date_range("2026-08-01", periods=56 * 24, freq="h")
    hourly = pd.Series(
        np.repeat(daily_kwh / 24, 24), index=idx)

    model = BillForecaster(n_fourier=0).fit(hourly)
    future = pd.date_range("2026-09-26", periods=7, freq="D")
    pred = model.predict_days(future)
    expected = 10 + 2 * np.arange(56, 63, dtype=float)
    assert pred["forecast_kwh"].to_numpy() == pytest.approx(expected, rel=1e-6)


def test_fourier_phase_continuous_across_fit_predict():
    """WW1: weekly Fourier terms use the same phase in predict as in fit."""
    from wattwise.forecast import _design_matrix

    fit_dates = pd.date_range("2026-08-01", periods=28, freq="D")
    pred_dates = pd.date_range("2026-08-29", periods=7, freq="D")
    origin = fit_dates[0]
    X_fit = _design_matrix(fit_dates, np.zeros(28), np.zeros(28),
                           n_fourier=2, origin=origin)
    X_pred = _design_matrix(pred_dates, np.zeros(7), np.zeros(7),
                            n_fourier=2, origin=origin)
    # The sin(2*pi*t/7) column for t=28..34 must continue t=0..27's phase.
    # Column layout: [1, t, dow dummies x6, sin1, cos1, sin2, cos2, cdd, hdd]
    sin_col = 1 + 1 + 6  # index of sin(2*pi*1*t/7)
    t_pred = np.arange(28, 35, dtype=float)
    assert X_pred[:, sin_col] == pytest.approx(
        np.sin(2 * np.pi * t_pred / 7.0))
