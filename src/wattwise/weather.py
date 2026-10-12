"""Weather correlation: how much of the bill is the weather's fault.

Regresses daily energy on heating/cooling degree-days to split the load
into a weather-independent base plus temperature-driven heating and
cooling slopes (kWh per degree-day). The correlation it reports is the
share of day-to-day variance temperature explains.
"""

import numpy as np
import pandas as pd


def degree_days(temp_f: pd.Series, base: float = 65.0) -> pd.DataFrame:
    daily = temp_f.resample("D").mean()
    return pd.DataFrame({
        "cdd": np.clip(daily - base, 0, None),
        "hdd": np.clip(base - daily, 0, None),
        "temp_mean": daily,
    })


def temperature_sensitivity(hourly_kwh: pd.Series,
                            temp_f: pd.Series) -> dict:
    """Fit kwh/day = base + cool_slope * CDD + heat_slope * HDD.

    Returns slopes, R^2, and the weather-driven share of total energy.
    """
    daily = hourly_kwh.resample("D").sum()
    daily = daily[daily > 0]
    dd = degree_days(temp_f).reindex(daily.index, fill_value=0.0)
    X = np.column_stack([np.ones(len(daily)), dd["cdd"], dd["hdd"]])
    y = daily.to_numpy()
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    base, cool, heat = (float(c) for c in coef)
    weather_kwh = float(np.sum(np.clip(yhat - base, 0, None)))
    return {
        "base_kwh_per_day": base,
        "cooling_kwh_per_cdd": max(cool, 0.0),
        "heating_kwh_per_hdd": max(heat, 0.0),
        "r_squared": float(r2),
        "weather_share": float(weather_kwh / max(y.sum(), 1e-9)),
        "daily": pd.DataFrame({"kwh": y, "cdd": dd["cdd"].to_numpy(),
                               "hdd": dd["hdd"].to_numpy(),
                               "fitted": yhat}, index=daily.index),
    }
