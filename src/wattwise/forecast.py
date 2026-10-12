"""Time-series bill forecasting.

Model: ordinary least squares on daily energy with
  - linear trend,
  - day-of-week effects,
  - Fourier terms (weekly seasonality),
  - heating/cooling degree-days from temperature.
This is the workhorse core of Prophet-style forecasting, without the
Bayesian machinery: interpretable, fast, and honest prediction intervals
from residual variance.

Given a partial month of hourly data, it forecasts each remaining day,
sums to a month-end kWh projection, and prices it through the rate.
"""

import calendar
import datetime as dt

import numpy as np
import pandas as pd


def _design_matrix(dates: pd.DatetimeIndex, cdd: np.ndarray,
                   hdd: np.ndarray, n_fourier: int = 2) -> np.ndarray:
    n = len(dates)
    dow = dates.weekday.to_numpy()
    t = np.arange(n, dtype=float)
    cols = [np.ones(n), t]
    # Day-of-week dummies (drop Monday as reference).
    for d in range(1, 7):
        cols.append((dow == d).astype(float))
    # Weekly Fourier terms.
    for k in range(1, n_fourier + 1):
        cols.append(np.sin(2 * np.pi * k * t / 7.0))
        cols.append(np.cos(2 * np.pi * k * t / 7.0))
    cols.append(cdd)
    cols.append(hdd)
    return np.column_stack(cols)


def _degree_days(temp_f: pd.Series) -> pd.DataFrame:
    daily = temp_f.resample("D").mean()
    cdd = np.clip(daily - 65.0, 0, None)
    hdd = np.clip(65.0 - daily, 0, None)
    return pd.DataFrame({"cdd": cdd, "hdd": hdd}, index=daily.index)


class BillForecaster:
    """Fit on history, forecast the rest of the current month."""

    def __init__(self, n_fourier: int = 2):
        self.n_fourier = n_fourier
        self.coef_ = None
        self.resid_std_ = None

    def fit(self, hourly_kwh: pd.Series, temp_f: pd.Series = None):
        daily = hourly_kwh.resample("D").sum()
        daily = daily[daily > 0]
        if temp_f is not None:
            dd = _degree_days(temp_f)
            dd = dd.reindex(daily.index, fill_value=0.0)
        else:
            dd = pd.DataFrame({"cdd": 0.0, "hdd": 0.0}, index=daily.index)
        X = _design_matrix(daily.index, dd["cdd"].to_numpy(),
                           dd["hdd"].to_numpy(), self.n_fourier)
        y = daily.to_numpy()
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        self.coef_ = coef
        resid = y - X @ coef
        self.resid_std_ = float(np.std(resid, ddof=X.shape[1]))
        self._last_date = daily.index[-1]
        return self

    def predict_days(self, dates: pd.DatetimeIndex,
                     temp_f: pd.Series = None) -> pd.DataFrame:
        """Point forecast + 80% interval per day."""
        if self.coef_ is None:
            raise RuntimeError("Call fit() first.")
        if temp_f is not None:
            dd = _degree_days(temp_f).reindex(dates, fill_value=0.0)
        else:
            dd = pd.DataFrame({"cdd": 0.0, "hdd": 0.0}, index=dates)
        X = _design_matrix(dates, dd["cdd"].to_numpy(), dd["hdd"].to_numpy(),
                           self.n_fourier)
        mu = X @ self.coef_
        mu = np.clip(mu, 0.5, None)  # houses don't use negative energy
        half = 1.2816 * self.resid_std_
        return pd.DataFrame({"forecast_kwh": mu,
                             "lo_kwh": np.clip(mu - half, 0.5, None),
                             "hi_kwh": mu + half}, index=dates)


def forecast_month_bill(hourly_kwh: pd.Series, rate,
                        temp_f: pd.Series = None) -> dict:
    """Forecast the current month's bill from partial-month data.

    Returns actual kWh so far, forecast remaining kWh with interval,
    and the priced bill (actual + forecast) with bands.
    """
    last = hourly_kwh.index[-1]
    month_start = last.replace(day=1, hour=0, minute=0, second=0)
    days_in_month = calendar.monthrange(last.year, last.month)[1]
    month_end = last.replace(day=days_in_month)

    model = BillForecaster().fit(hourly_kwh, temp_f)
    remaining = pd.date_range(last.date() + dt.timedelta(days=1),
                              month_end.date(), freq="D")
    actual_kwh = float(hourly_kwh.loc[month_start:].sum())
    days_so_far = (last.date() - month_start.date()).days + 1

    if len(remaining):
        fc = model.predict_days(remaining, temp_f)
        proj = float(fc["forecast_kwh"].sum())
        lo = float(fc["lo_kwh"].sum())
        hi = float(fc["hi_kwh"].sum())
    else:
        fc, proj, lo, hi = None, 0.0, 0.0, 0.0

    total_kwh = actual_kwh + proj
    summer = 6 <= last.month <= 9

    def price(kwh):
        from .rates import Rate180, Rate194
        if isinstance(rate, Rate180):
            return rate.energy_charge(kwh, summer) + rate.fixed_charge(days_in_month)
        # TOU: price the projection at the observed average marginal mix.
        hist = hourly_kwh.loc[month_start:]
        avg_price = sum(rate.marginal_price(ts) * v
                        for ts, v in hist.items()) / max(hist.sum(), 1e-9)
        return kwh * avg_price + rate.fixed_charge(days_in_month)

    return {
        "days_so_far": days_so_far,
        "days_in_month": days_in_month,
        "actual_kwh": actual_kwh,
        "forecast_kwh": proj,
        "forecast_lo_kwh": lo,
        "forecast_hi_kwh": hi,
        "bill_forecast": price(total_kwh),
        "bill_lo": price(actual_kwh + lo),
        "bill_hi": price(actual_kwh + hi),
        "daily_forecast": fc,
        "resid_std_kwh": model.resid_std_,
    }
