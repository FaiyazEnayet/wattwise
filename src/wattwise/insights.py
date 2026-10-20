"""Decision-support insights on measured smart-meter data.

Six report sections, all computed from the shipped demo data (no
synthetic inputs): bill forecast, peak-hour drift alert, what-if
scenarios, seasonal outlook, cost-driver ranking, and anomaly
detection. Every dollar figure uses the exact per-period Rate 194
rates from the utility's own bill details (Rate194History), so the
numbers are measurements plus model arithmetic, not guesses.

All functions take optional pre-loaded data arguments so tests can
inject fixtures; the report() entry point uses the shipped files.
"""

import datetime as dt

import numpy as np
import pandas as pd

from .measured import (load_billing_history, load_appliance_estimates,
                       load_intervals, load_monthly)
from .rates import (Rate194History, is_peak_hour, FEDERAL_HOLIDAYS_2026)

# Peak drift alert threshold: flag when the current period's peak
# share is more than 15% above the 3-period baseline.
PEAK_DRIFT_THRESHOLD = 0.15

# Forecast range: +/-15% of the point estimate, roughly the observed
# month-to-month swing outside the seasonal trend.
FORECAST_BAND = 0.15


def _latest_bill_date(rates: Rate194History) -> str:
    """Most recent bill date in the measured rate history."""
    return sorted(rates.periods.keys())[-1]


def _period_peak_share(intervals: pd.DataFrame,
                       start: dt.date, end: dt.date) -> tuple:
    """Total and peak kWh over [start, end] from 15-min intervals.

    Peak follows the Rate 194 window: weekdays 3-7 PM, federal
    holidays excluded. Returns (total_kwh, peak_kwh).
    """
    sel = intervals[(intervals.index.date >= start)
                    & (intervals.index.date <= end)]
    if sel.empty:
        return 0.0, 0.0
    total = float(sel["kwh"].sum())
    peak_mask = [is_peak_hour(ts, FEDERAL_HOLIDAYS_2026)
                 for ts in sel.index]
    peak = float(sel.loc[peak_mask, "kwh"].sum()) if any(peak_mask) else 0.0
    return total, peak


def forecast_next_bill(intervals: pd.DataFrame = None,
                       rates: Rate194History = None) -> dict:
    """Project the next 30-day bill from the trailing trend plus the
    seasonal factor.

    Method: trailing-30-day kWh from the interval export, scaled by
    last year's seasonal factor (next-30-days / trailing-30-days on
    the same dates a year earlier), then priced with the most recent
    period's exact rates. The band is +/-15% of the estimate.
    """
    iv = load_intervals() if intervals is None else intervals
    rh = Rate194History() if rates is None else rates
    bill_date = _latest_bill_date(rh)
    p = rh.periods[bill_date]

    last_ts = iv.index.max()
    trailing = iv[iv.index >= last_ts - pd.Timedelta(days=30)]
    trail_kwh = float(trailing["kwh"].sum())

    # Seasonal factor from the same window a year earlier.
    prior_last = last_ts - pd.Timedelta(days=365)
    prior_trail = iv[(iv.index >= prior_last - pd.Timedelta(days=30))
                     & (iv.index <= prior_last)]
    prior_next = iv[(iv.index > prior_last)
                    & (iv.index <= prior_last + pd.Timedelta(days=30))]
    seasonal_factor = (float(prior_next["kwh"].sum())
                       / max(float(prior_trail["kwh"].sum()), 1e-9))
    if not np.isfinite(seasonal_factor) or seasonal_factor <= 0:
        seasonal_factor = 1.0

    proj_kwh = trail_kwh * seasonal_factor
    # Peak share: use the trailing window's measured peak fraction.
    t_start, t_end = trailing.index.min().date(), trailing.index.max().date()
    _, trail_peak = _period_peak_share(iv, t_start, t_end)
    peak_share = trail_peak / max(trail_kwh, 1e-9)

    peak_kwh = proj_kwh * peak_share
    estimate = rh.bill_for(bill_date, peak_kwh, proj_kwh - peak_kwh)
    return {
        "estimate_usd": round(estimate, 2),
        "low_usd": round(estimate * (1 - FORECAST_BAND), 2),
        "high_usd": round(estimate * (1 + FORECAST_BAND), 2),
        "projected_kwh": round(proj_kwh, 1),
        "trailing_30d_kwh": round(trail_kwh, 1),
        "seasonal_factor": round(seasonal_factor, 3),
        "rates_from_bill_date": bill_date,
        "method": ("trailing 30-day interval total x prior-year "
                   "seasonal factor, priced at the latest measured "
                   "period's exact rates; band +/-15%"),
    }


def peak_drift(intervals: pd.DataFrame = None,
               monthly: pd.DataFrame = None,
               rates: Rate194History = None) -> dict:
    """Compare the latest complete period's peak usage against the
    3-period baseline. Flags when peak share is >15% above baseline.
    """
    mn = load_monthly() if monthly is None else monthly
    rh = Rate194History() if rates is None else rates
    bill_date = _latest_bill_date(rh)
    p = rh.periods[bill_date]

    # Baseline: average of the three periods before the anchor period
    # that carry measured or estimated peak splits.
    with_peak = mn[mn["kwh_peak"].notna()].copy()
    with_peak = with_peak.sort_values("period_end").tail(4)
    baseline = with_peak.iloc[:-1]
    current = with_peak.iloc[-1]

    base_peak = float(baseline["kwh_peak"].sum())
    base_total = float(baseline["kwh_total"].sum())
    base_share = base_peak / max(base_total, 1e-9)
    cur_share = float(current["kwh_peak"]) / max(float(current["kwh_total"]),
                                                1e-9)
    drift = (cur_share - base_share) / max(base_share, 1e-9)
    flagged = drift > PEAK_DRIFT_THRESHOLD

    # Dollar impact: what the current period's peak kWh cost vs the
    # same kWh at the baseline share.
    cur_kwh = float(current["kwh_total"])
    cur_peak_kwh = float(current["kwh_peak"])
    base_peak_kwh = cur_kwh * base_share
    peak_premium = (p["del_peak_rate"] + p["sup_peak_rate"]
                    - p["del_offpeak_rate"] - p["sup_offpeak_rate"])
    impact = (cur_peak_kwh - base_peak_kwh) * peak_premium
    return {
        "baseline_peak_share": round(base_share, 4),
        "current_peak_share": round(cur_share, 4),
        "drift": round(drift, 4),
        "flagged": bool(flagged),
        "impact_usd": round(impact, 2),
        "period": (str(current["period_start"].date()),
                   str(current["period_end"].date())),
        "rates_from_bill_date": bill_date,
    }


def what_if(peak_shift_pct: float = 0.0, cooling_cut_pct: float = 0.0,
            total_cut_pct: float = 0.0,
            rates: Rate194History = None,
            anchor_peak_kwh: float = 39.0,
            anchor_offpeak_kwh: float = 281.0) -> dict:
    """Dollar impact of usage scenarios, priced at the latest measured
    period's exact rates (the Sep 2026 anchor: 39 peak / 281 off-peak
    kWh, bill $109.01).

    - peak_shift_pct: move X% of peak kWh into off-peak hours
      (same total usage, different timing).
    - cooling_cut_pct: cut the cooling end use by X%. Cooling is the
      anchor period's largest cost driver at $18 of the $82 energy
      charges (22.0%); its kWh is taken as 22.0% of the period total,
      reduced proportionally across peak and off-peak.
    - total_cut_pct: cut ALL usage by X%.

    Each scenario is independent; combining is left to the caller.
    """
    rh = Rate194History() if rates is None else rates
    bill_date = _latest_bill_date(rh)
    base_bill = rh.bill_for(bill_date, anchor_peak_kwh,
                            anchor_offpeak_kwh)
    total = anchor_peak_kwh + anchor_offpeak_kwh
    out = {"baseline_bill_usd": round(base_bill, 2),
           "rates_from_bill_date": bill_date}
    if peak_shift_pct:
        shift = anchor_peak_kwh * peak_shift_pct / 100.0
        new_bill = rh.bill_for(bill_date, anchor_peak_kwh - shift,
                               anchor_offpeak_kwh + shift)
        out["peak_shift"] = {"pct": peak_shift_pct,
                             "new_bill_usd": round(new_bill, 2),
                             "saving_usd": round(base_bill - new_bill, 2)}
    if cooling_cut_pct:
        cool_kwh = 0.22 * total
        cut = cool_kwh * cooling_cut_pct / 100.0
        peak_frac = anchor_peak_kwh / max(total, 1e-9)
        new_peak = anchor_peak_kwh - cut * peak_frac
        new_off = anchor_offpeak_kwh - cut * (1 - peak_frac)
        new_bill = rh.bill_for(bill_date, new_peak, new_off)
        out["cooling_cut"] = {"pct": cooling_cut_pct,
                              "new_bill_usd": round(new_bill, 2),
                              "saving_usd": round(base_bill - new_bill, 2)}
    if total_cut_pct:
        factor = 1 - total_cut_pct / 100.0
        new_bill = rh.bill_for(bill_date, anchor_peak_kwh * factor,
                               anchor_offpeak_kwh * factor)
        out["total_cut"] = {"pct": total_cut_pct,
                            "new_bill_usd": round(new_bill, 2),
                            "saving_usd": round(base_bill - new_bill, 2)}
    return out


def seasonal_outlook(monthly: pd.DataFrame = None,
                     rates: Rate194History = None,
                     n_months: int = 4) -> list:
    """Expected kWh and bill for the next n months, from last year's
    same-month actuals priced at the latest measured rates (the rate
    change between periods is the documented limitation).
    """
    mn = load_monthly() if monthly is None else monthly
    rh = Rate194History() if rates is None else rates
    bill_date = _latest_bill_date(rh)
    p = rh.periods[bill_date]

    last_end = mn["period_end"].max()
    rows = []
    for i in range(1, n_months + 1):
        target = last_end + pd.offsets.MonthBegin(i)
        same_last_year = mn[
            (mn["period_start"].dt.month == target.month)
            & (mn["period_start"] < last_end - pd.Timedelta(days=300))]
        if same_last_year.empty:
            continue
        row = same_last_year.sort_values("period_start").iloc[-1]
        kwh = float(row["kwh_total"]) if pd.notna(row["kwh_total"]) else None
        if kwh is None:
            continue
        peak_kwh = (float(row["kwh_peak"]) if pd.notna(row["kwh_peak"])
                    else kwh * 0.122)
        bill = rh.bill_for(bill_date, peak_kwh, kwh - peak_kwh)
        rows.append({
            "month": target.strftime("%Y-%m"),
            "expected_kwh": round(kwh, 1),
            "expected_bill_usd": round(bill, 2),
            "source": ("measured" if row["kwh_source"] == "measured"
                       else "portal chart estimate"),
        })
    return rows


def cost_drivers(appliance: pd.DataFrame = None) -> list:
    """Rank end uses by dollar cost in the anchor period (Sep 2026),
    with the saving from a 10% and 20% cut of each.

    The utility's estimates are dollar figures that exclude taxes and
    fees; a 10% end-use cut saves about 10% of the category dollars.
    """
    est = load_appliance_estimates() if appliance is None else appliance
    sel = est[est["period_start"] == "2026-09-02"]
    sel = sel[sel["category"] != "taxes_fees"]
    ranked = (sel.groupby("category")["dollars_usd"].sum()
              .sort_values(ascending=False))
    return [{"category": cat,
             "dollars_usd": round(float(val), 2),
             "save_10pct_usd": round(float(val) * 0.10, 2),
             "save_20pct_usd": round(float(val) * 0.20, 2)}
            for cat, val in ranked.items()]


def detect_anomalies(intervals: pd.DataFrame = None,
                     std_threshold: float = 2.0,
                     window_days: int = 14) -> list:
    """Flag days where usage is >2 std above the rolling 14-day mean
    for that weekday, with excess kWh and dollars.

    The dollar figure prices the excess at the latest measured
    period's average energy $/kWh.
    """
    iv = load_intervals() if intervals is None else intervals
    rh = Rate194History()
    bill_date = _latest_bill_date(rh)
    p = rh.periods[bill_date]
    avg_price = (p["delivery_usd"] + p["supply_usd"]) / max(p["kwh"], 1e-9)

    daily = iv["kwh"].resample("D").sum()
    items = list(daily.items())
    flagged = []
    for i, (day, kwh) in enumerate(items):
        weekday = day.weekday()
        # Rolling window of same-weekday history before this day.
        same_wd = [v for d, v in items[:i]
                   if d.weekday() == weekday]
        hist = same_wd[-window_days:]
        if len(hist) < 4:
            continue
        mean = float(np.mean(hist))
        std = float(np.std(hist))
        if std == 0:
            continue
        z = (float(kwh) - mean) / std
        if z > std_threshold:
            excess = float(kwh) - mean
            flagged.append({
                "date": day.strftime("%Y-%m-%d"),
                "kwh": round(float(kwh), 2),
                "baseline_kwh": round(mean, 2),
                "excess_kwh": round(excess, 2),
                "excess_usd": round(excess * avg_price, 2),
                "z_score": round(z, 2),
            })
    return flagged


def report() -> str:
    """Full six-section insights report for the latest period."""
    lines = []
    lines.append("=== WattWise insights (measured data) ===")

    fc = forecast_next_bill()
    lines.append("\n-- Bill forecast (next 30 days) --")
    lines.append(
        f"Estimate ${fc['estimate_usd']} (range "
        f"${fc['low_usd']}-${fc['high_usd']}), {fc['projected_kwh']} kWh. "
        f"Trailing 30d: {fc['trailing_30d_kwh']} kWh, seasonal factor "
        f"{fc['seasonal_factor']}. {fc['method']}.")

    pd_ = peak_drift()
    lines.append("\n-- Peak-hour drift --")
    status = "FLAGGED" if pd_["flagged"] else "ok"
    lines.append(
        f"Peak share {pd_['current_peak_share']:.1%} vs baseline "
        f"{pd_['baseline_peak_share']:.1%} ({pd_['drift']:+.1%}) [{status}]. "
        f"Cost impact ${pd_['impact_usd']:+.2f}.")

    wi = what_if(peak_shift_pct=20.0, cooling_cut_pct=20.0,
                 total_cut_pct=10.0)
    lines.append("\n-- What-if (anchor period rates) --")
    lines.append(f"Baseline bill ${wi['baseline_bill_usd']}.")
    for key in ("peak_shift", "cooling_cut", "total_cut"):
        s = wi[key]
        lines.append(f"  {key}: {s['pct']}% -> bill ${s['new_bill_usd']}, "
                     f"saves ${s['saving_usd']}.")

    lines.append("\n-- Seasonal outlook --")
    for row in seasonal_outlook():
        lines.append(f"  {row['month']}: {row['expected_kwh']} kWh, "
                     f"~${row['expected_bill_usd']} ({row['source']}).")

    lines.append("\n-- Cost drivers (anchor period) --")
    for d in cost_drivers()[:5]:
        lines.append(f"  {d['category']}: ${d['dollars_usd']} "
                     f"(10% cut saves ${d['save_10pct_usd']}, "
                     f"20% saves ${d['save_20pct_usd']}).")

    anoms = detect_anomalies()
    lines.append(f"\n-- Anomalies ({len(anoms)} flagged days) --")
    for a in anoms[:10]:
        lines.append(f"  {a['date']}: {a['kwh']} kWh vs baseline "
                     f"{a['baseline_kwh']} (+{a['excess_kwh']} kWh, "
                     f"+${a['excess_usd']}, z={a['z_score']}).")
    return "\n".join(lines)
