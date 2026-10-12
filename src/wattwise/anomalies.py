"""Anomaly detection with plain-English explanations.

Flags days whose usage is far from the household's own same-weekday
baseline (z-score), then explains each flag: which hours drove the
excess and which appliances they look like, via the disaggregator.
Output reads like: "Tuesday Oct 6 ran 40% hot — an extra 5.1 kWh,
mostly 6-9 PM, looks like water heater + dryer."
"""

import datetime as dt

import numpy as np
import pandas as pd

from .disagg import disaggregate_day


def daily_anomalies(hourly: pd.Series, z_thresh: float = 2.5,
                    min_history: int = 14) -> pd.DataFrame:
    """Flag anomalous days. Returns date, kwh, baseline, z, pct_over."""
    daily = hourly.resample("D").sum()
    daily = daily[daily > 0]
    rows = []
    for date, kwh in daily.items():
        same_wd = daily[daily.index.weekday == date.weekday()]
        same_wd = same_wd[same_wd.index != date]
        if len(same_wd) < min_history:
            continue
        mu, sd = same_wd.mean(), same_wd.std()
        if sd < 1e-9:
            continue
        z = (kwh - mu) / sd
        if abs(z) >= z_thresh:
            rows.append({"date": date.date(), "kwh": float(kwh),
                         "baseline_kwh": float(mu), "z": float(z),
                         "pct_over": float(100 * (kwh - mu) / mu)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("z", ascending=False).reset_index(drop=True)
    return out


def explain_anomaly(hourly: pd.Series, date: dt.date, shapes: dict,
                    baseline_profile: np.ndarray) -> dict:
    """Explain one flagged day: excess hours + likely appliances.

    Compares the day's disaggregation against the baseline's: the
    appliances whose attributed energy grew most are the likely drivers.
    """
    day = hourly.loc[hourly.index.date == date].to_numpy()[:24]
    excess = day - baseline_profile
    excess_kwh = float(np.clip(excess, 0, None).sum())

    attr_day, _ = disaggregate_day(day, shapes)
    attr_base, _ = disaggregate_day(np.clip(baseline_profile, 0, None), shapes)
    growth = {n: attr_day[n] - attr_base[n] for n in attr_day
              if n != "baseload"}  # baseload is the residual catcher, never a driver
    drivers = sorted(growth.items(), key=lambda kv: kv[1], reverse=True)[:3]
    drivers = [(n, round(v, 2)) for n, v in drivers if v > 0.05]

    # Describe the peak excess as a clock range around the top hour.
    ex = np.clip(excess, 0, None)
    top = int(np.argmax(ex))

    def fmt(h):
        h = h % 24
        suffix = "AM" if h < 12 else "PM"
        h12 = h % 12 or 12
        return f"{h12} {suffix}"

    span = f"{fmt(top)}-{fmt(top + 1)}"
    names = ", ".join(n.replace("_", " ") for n, _ in drivers) or "no single appliance"
    pct = 100 * excess_kwh / max(baseline_profile.sum(), 1e-9)
    text = (f"{date:%A %b %d} ran {pct:.0f}% hot: an extra {excess_kwh:.1f} kWh, "
            f"peaking {span} — looks like {names}.")
    # Top contributing hours for charts.
    top_hours = np.argsort(ex)[-4:][::-1]
    top_hours = [int(h) for h in top_hours if ex[h] > 0.05]
    return {"date": date, "excess_kwh": round(excess_kwh, 2),
            "pct_over": round(pct, 1), "peak_hours": top_hours,
            "likely_appliances": drivers, "text": text}


def typical_weekday_profile(hourly: pd.Series, weekday: int) -> np.ndarray:
    """Mean 24-h profile for a weekday, excluding anomalous days."""
    daily = hourly.resample("D").sum()
    mu, sd = daily.mean(), daily.std()
    ok_days = {d.date() for d, v in daily.items() if abs(v - mu) < 2.5 * sd}
    sel = hourly[[ts.date() in ok_days and ts.weekday() == weekday
                  for ts in hourly.index]]
    if len(sel) == 0:
        return np.zeros(24)
    mat = sel.to_numpy().reshape(-1, 24)
    return mat.mean(axis=0)
