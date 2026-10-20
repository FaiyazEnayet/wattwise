"""Measured PSEG Long Island data for this household, and a profile
anchored to it.

Two measured datasets ship with the repo (demo/pseg_bills.csv and
demo/pseg_appliance_estimates.csv), extracted from PSEG Long Island
emails with all identifying fields stripped:

- A full year of 15-minute smart-meter intervals
  (demo/pseg_interval_15min.csv): 37,976 rows, 2025-09-19 to
  2026-10-19, 4,086.55 kWh. Aggregated to hourly, the intervals
  supply the intraday SHAPE of the measured profile (own-day shape
  where the export covers the date, season/day-type average shape
  otherwise). With end-exclusive periods and detailed line-item
  totals, interval sums are within 0.49% of billed kWh for all 12
  fully covered periods; the
  holiday-adjusted weekday 3-7 PM peak reproduces the billed peak
  kWh within a few kWh, validating the rate window.
- Weekly Usage Summary emails give cumulative kWh per billing period
  (complete for the Jul, Aug and Sep 2026 periods). Differencing
  consecutive snapshots yields 12 measured sub-period increments
  (demo/pseg_subperiod.csv) covering 2026-07-02 to 2026-10-17,
  used as the daily-total anchors.
- Bill emails plus payment confirmations give the amount due per
  period for 13 billed periods, Aug 2025 to Oct 2026 ($1,394.07
  total). Where no bill email exists, the period dates are inferred
  from the payment date with a ~9-day lag calibrated on periods
  where both exist, and are labeled as such.
- The utility's appliance breakdowns are dollar ESTIMATES produced by
  its disaggregation partner from meter data. They exclude the daily
  service charge and taxes, so they do not sum to the bill.
- Tariff facts (demo/pseg_rates.csv): Time-of-Day Rate 194; peak
  3-7 PM weekdays at ~2x; Power Supply Charge rose significantly in
  Feb 2026 (expected to stay elevated for a few months).

Plus portal-sourced data from the utility's customer portal
(screenshots, all identifying fields stripped):

- Fourteen exact bills (2025-08-01 through 2026-10-02, $1,493.31
  total): exact bill dates and days-in-bill; period dates chained
  backward from the known 2026-07-02 anchor, matching every portal
  day count and the verified 2026-02-03 and 2025-10-02 boundaries.
- Monthly kWh for twelve periods from the portal's yearly energy
  chart: measured totals for Jul/Aug/Sep 2026 (619.39 / 427.24 /
  320.25 kWh), chart-estimated totals for the nine preceding
  periods, peak/off-peak splits (39 / 281 kWh measured for Sep 2026
  per the portal's monthly summary; chart estimates elsewhere) and
  the chart's average temperature curve. Chart estimates are labeled
  as such in demo/pseg_monthly.csv.
- The Sep 2026 rows now also carry the $27 taxes/fees line from the
  portal's bill breakdown; taxes are excluded when computing
  appliance kWh shares (they are not an end use).
- The Aug 2026 appliance rows were updated to the portal's
  bill-comparison values (cooling 37, laundry 12, cooking 8), which
  reconcile exactly to the portal's $99 Aug energy charges ($19 peak
  + $80 off-peak); the earlier energy-insights email snapshot values
  (35 / 14 / 7) summed to $98 and have been superseded. A $28
  taxes/fees row was added for reconciliation.
- One real hourly day from the portal: `demo/pseg_hourly_oct17.csv`
  holds the 24 hourly costs for Saturday 2026-10-17 (all off-peak,
  Saturday) plus the chart's average temperature. The day is bimodal:
  an early-afternoon peak (~1 PM), a smaller late-evening peak
  (~8 PM), a 3-4 AM trough, and elevated late-night hours. The
  seeded generator's intraday windows were sanity-checked against
  this day and adjusted (see profile.py); it remains a single day,
  not a measured average profile.

make_measured_profile() anchors the measured 15-minute intervals to
the September 2026 period: each day's hourly totals come from the
real smart-meter shape (the date's own hourly profile where the
interval export covers it, the season/day-type average profile
otherwise), scaled so each day matches the measured sub-period
daily totals and the month totals exactly 320.25 kWh. End-use
columns are split from the hourly totals using the seeded demo
generator's per-hour appliance proportions, then rescaled so the
30-day totals match the measured appliance mix. Period totals,
the appliance mix and the bills are measured; the appliance-level
intraday SPLIT is still modeled (no submetered end-use data
exists). That split is deliberate and is stated wherever the
numbers are quoted.
"""

import datetime as dt
import os

import pandas as pd

from .profile import make_demo_profile

_DEMO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "..", "demo")

# The most recent complete measured period (PSEG billing period
# 2026-09-02 to 2026-10-02; appliance insights cover 09-02 to 10-01).
ANCHOR_PERIOD_START = dt.date(2026, 9, 2)
ANCHOR_PERIOD_END = dt.date(2026, 10, 1)  # inclusive, 30 days
ANCHOR_KWH = 320.25          # weekly usage summary, 9/2 - 10/1
ANCHOR_BILL_USD = 109.01     # actual bill, Time-of-Day Rate 194
ANCHOR_PEAK_COST_USD = 19.0  # peak-hour cost for that period, per the
# subject line of the utility's Oct 13 energy-insights email ("You
# spent $19 in peak hours"). An earlier extraction read $17 from the
# email body; that figure is the above/below-average comparison
# value (aboveBelowCostValue), not peak cost.

# Fully specified Sep 2026 period from the portal's monthly summary
# (Sep 02 - Oct 01, 2026): 281 kWh off-peak ($63), 39 kWh peak ($19),
# of which the top contributors were cooling (30% of peak) and
# always-on (13% of peak). Total energy charges $82; $27 taxes/fees.
ANCHOR_PEAK_KWH = 39.0
ANCHOR_OFFPEAK_KWH = 281.0
ANCHOR_ENERGY_CHARGES_USD = 82.0
ANCHOR_TAXES_FEES_USD = 27.0

# Utility appliance categories -> wattwise end uses. ``other`` has no
# direct counterpart; it is split between water heating and the
# dishwasher in proportion to the seeded model's ratio. ``taxes_fees``
# is carried in the appliance-estimates CSV for reconciliation but is
# not an end use and is excluded from share mapping (KeyError would
# otherwise fire, and taxes carry no energy information).
CATEGORY_TO_END_USES = {
    "cooling": ["hvac"],
    "laundry": ["laundry"],
    "always_on": ["baseload"],
    "refrigeration": ["baseload"],
    "entertainment": ["lighting_plug"],
    "lighting": ["lighting_plug"],
    "cooking": ["cooking"],
    "other": ["water_heater", "dishwasher"],
}


def load_billing_history(path: str = None) -> pd.DataFrame:
    """Real PSEG billing periods: dates, days, kWh (when the weekly
    summaries cover the full period) and the amount billed."""
    path = path or os.path.join(_DEMO_DIR, "pseg_bills.csv")
    df = pd.read_csv(path, parse_dates=["period_start", "period_end"])
    return df


def load_appliance_estimates(path: str = None) -> pd.DataFrame:
    """Utility appliance dollar estimates per billing period."""
    path = path or os.path.join(_DEMO_DIR, "pseg_appliance_estimates.csv")
    df = pd.read_csv(path, parse_dates=["period_start", "period_end"])
    return df


def load_subperiods(path: str = None) -> pd.DataFrame:
    """Measured sub-period kWh increments from differenced weekly
    usage snapshots (Jul-Oct 2026)."""
    path = path or os.path.join(_DEMO_DIR, "pseg_subperiod.csv")
    df = pd.read_csv(path,
                     parse_dates=["subperiod_start", "subperiod_end"])
    return df


def load_rates(path: str = None) -> pd.DataFrame:
    """Tariff facts extracted from utility emails (Time-of-Day Rate
    194 details, power-supply events)."""
    path = path or os.path.join(_DEMO_DIR, "pseg_rates.csv")
    return pd.read_csv(path)


def load_hourly_oct17(path: str = None) -> pd.DataFrame:
    """Real hourly costs for 2026-10-17 (Saturday, all off-peak) from
    the portal's hourly cost chart, with the chart's average
    temperature. Costs are chart reads of the portal's bars, labeled
    as such; the day total ($2.15) is the cost of one measured day,
    not a monthly figure."""
    path = path or os.path.join(_DEMO_DIR, "pseg_hourly_oct17.csv")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_monthly(path: str = None) -> pd.DataFrame:
    """Monthly billing periods with kWh totals, peak/off-peak splits
    and average temperature: measured for Jul/Aug/Sep 2026, portal
    chart estimates for the nine preceding periods, labeled in
    ``kwh_source``. Pre-Oct-2025 periods have bills but no kWh."""
    path = path or os.path.join(_DEMO_DIR, "pseg_monthly.csv")
    return pd.read_csv(path, parse_dates=["period_start", "period_end"])


def anchor_appliance_shares(end_use_seed: pd.DataFrame) -> dict:
    """Wattwise end-use kWh shares for the anchor period, derived from
    the utility's dollar estimates for 2026-09-02 to 2026-10-01.

    Dollar shares are used as kWh shares; within one billing period
    under one tariff that is an approximation (peak hours price some
    appliances higher), and it is labeled as such.
    """
    est = load_appliance_estimates()
    sel = est[est["period_start"] == pd.Timestamp(ANCHOR_PERIOD_START)]
    # taxes_fees is not an end use: exclude it so it neither distorts
    # the shares nor trips the category mapping.
    sel = sel[sel["category"] != "taxes_fees"]
    dollars = sel.groupby("category")["dollars_usd"].sum()
    total = dollars.sum()
    shares = {}
    for category, amount in dollars.items():
        targets = CATEGORY_TO_END_USES[category]
        part = amount / total
        if len(targets) == 1:
            shares[targets[0]] = shares.get(targets[0], 0.0) + part
        else:
            seed_totals = {t: float(end_use_seed[t].sum()) for t in targets}
            seed_sum = sum(seed_totals.values())
            for t in targets:
                shares[t] = shares.get(t, 0.0) + part * seed_totals[t] / seed_sum
    return shares


def load_intervals(path: str = None) -> pd.DataFrame:
    """Full year of 15-minute smart-meter intervals (2025-09-19 to
    2026-10-19, 37,976 rows), the My Account CSV export transcribed
    verbatim (timestamps, kWh and kW only). Index: interval start
    timestamps; columns: kwh, kw."""
    path = path or os.path.join(_DEMO_DIR, "pseg_interval_15min.csv")
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["Start"], format="%m/%d/%Y %I:%M:%S %p")
    df = df.drop(columns=["Start"]).rename(columns={"kWh": "kwh",
                                                   "kW": "kw"})
    return df.set_index("ts").sort_index()


_SEASON_MONTHS = {12: "winter", 1: "winter", 2: "winter",
                  3: "spring", 4: "spring", 5: "spring",
                  6: "summer", 7: "summer", 8: "summer",
                  9: "fall", 10: "fall", 11: "fall"}


def interval_hourly_shape(season: str, weekend: bool,
                          intervals: pd.DataFrame = None):
    """Average hourly share vector (24 values, sums to 1) from the
    15-minute interval export for a season and day type. Each day is
    normalized to shares before averaging, so high-use days do not
    dominate the shape."""
    iv = load_intervals() if intervals is None else intervals
    by_day = iv["kwh"].groupby([iv.index.date, iv.index.hour]).sum()
    days = by_day.rename("kwh").reset_index()
    days.columns = ["date", "hour", "kwh"]
    days["date"] = pd.to_datetime(days["date"])
    sel = days[(days["date"].dt.month.map(_SEASON_MONTHS) == season)
               & ((days["date"].dt.weekday >= 5) == weekend)]
    pivot = sel.pivot_table(index="date", columns="hour", values="kwh")
    pivot = pivot.reindex(columns=range(24), fill_value=0.0)
    shares = pivot.div(pivot.sum(axis=1), axis=0)
    return shares.mean(axis=0).to_numpy()


def interval_reconciliation() -> list:
    """Per-billing-period reconciliation of the interval export
    against the billed figures: interval kWh summed over the period,
    and the interval-derived peak kWh (weekdays 3-7 PM excluding
    federal holidays, the Rate 194 window). One row per billing
    period in pseg_bills.csv."""
    from .decision_analysis import peak_mask
    iv = load_intervals()
    is_peak = peak_mask(iv.index)
    peak_kwh = iv.loc[is_peak, "kwh"]
    rows = []
    bills = load_billing_history()
    # The later line-item extraction supersedes earlier chart estimates.
    detail_path = os.path.join(_DEMO_DIR, "pseg_rates_detailed.csv")
    if os.path.exists(detail_path):
        detail = pd.read_csv(detail_path, parse_dates=["period_start"])
        exact = detail.set_index("period_start")["kwh"]
        bills["kwh"] = bills["period_start"].map(exact).fillna(bills["kwh"])
    for _, b in bills.iterrows():
        end = b["period_end"]  # exclusive, as established by portal day counts
        seg = iv[(iv.index >= b["period_start"]) & (iv.index < end)]
        rows.append({
            "period_start": str(b["period_start"].date()),
            "period_end": str(b["period_end"].date()),
            "billed_kwh": (None if pd.isna(b["kwh"])
                           else float(b["kwh"])),
            "interval_kwh": round(float(seg["kwh"].sum()), 2),
            "interval_peak_kwh": round(
                float(peak_kwh[(peak_kwh.index >= b["period_start"])
                               & (peak_kwh.index < end)].sum()), 2),
            "n_intervals": int(len(seg)),
        })
    return rows


def make_measured_profile(seed: int = 7):
    """30-day profile for the measured Sep 2026 period.

    Returns (hourly_kwh Series, end_use DataFrame, weather Series) for
    2026-09-02..2026-10-01. Measured anchors:
      1. Daily totals: each sub-period's daily mean matches the
         measured sub-period average from the differenced weekly
         usage snapshots (pseg_subperiod.csv); the month totals
         exactly ANCHOR_KWH. September declines over the month.
      2. Intraday shape: the real 15-minute smart-meter export
         (demo/pseg_interval_15min.csv). Sep 19 - Oct 1 use each
         date's own measured hourly profile; Sep 2 - 18 use the
         measured fall season/day-type average hourly shape.
      3. Appliance totals: end-use 30-day totals match the utility's
         dollar-share appliance mix. The appliance SPLIT within each
         hour uses the seeded generator's per-hour proportions --
         the one modeled element left (no submetered data exists).
    """
    _, end_use_seed, temp = make_demo_profile(
        seed=seed, days=30, end=ANCHOR_PERIOD_END)
    shares = anchor_appliance_shares(end_use_seed)

    # --- Daily totals from the sub-period trend anchor (as before).
    scaled = end_use_seed.copy()
    for col in scaled.columns:
        current = float(scaled[col].sum())
        scaled[col] = scaled[col] * (shares[col] * ANCHOR_KWH / current)
    subs = load_subperiods()
    subs = subs[(subs["subperiod_start"] >= pd.Timestamp(ANCHOR_PERIOD_START))
                & (subs["subperiod_start"] <= pd.Timestamp(ANCHOR_PERIOD_END))]
    for _, row in subs.iterrows():
        day_mask = (scaled.index.date >= row["subperiod_start"].date()) & (
            scaled.index.date <= row["subperiod_end"].date())
        num_days = len(set(d for d in scaled.index.date[day_mask]))
        current_mean = float(scaled.loc[day_mask].sum().sum() / num_days)
        scaled.loc[day_mask] = scaled.loc[day_mask] * (
            row["avg_daily_kwh"] / current_mean)
    scaled = scaled * (ANCHOR_KWH / float(scaled.sum().sum()))
    daily_target = scaled.sum(axis=1).groupby(scaled.index.date).sum()

    # --- Hourly totals from the measured interval shape.
    iv = load_intervals()
    by_day = iv["kwh"].groupby([iv.index.date, iv.index.hour]).sum()
    own_days = set(by_day.index.get_level_values(0))
    fall_weekday = interval_hourly_shape("fall", False, iv)
    fall_weekend = interval_hourly_shape("fall", True, iv)
    target_hourly = scaled.sum(axis=1).copy()
    for day, dtotal in daily_target.items():
        if day in own_days:
            day_hours = by_day.loc[day].reindex(range(24), fill_value=0.0)
            shape = day_hours.to_numpy() / float(day_hours.sum())
        else:
            shape = (fall_weekend if pd.Timestamp(day).weekday() >= 5
                     else fall_weekday)
        day_mask = target_hourly.index.date == day
        target_hourly.loc[day_mask] = dtotal * shape

    # --- End-use split: seeded per-hour proportions, then balance
    # column totals (measured appliance mix) against daily totals.
    prop = end_use_seed.div(end_use_seed.sum(axis=1), axis=0).fillna(0.0)
    eu = prop.mul(target_hourly, axis=0)
    for _ in range(3):
        for col in eu.columns:
            eu[col] = eu[col] * (shares[col] * ANCHOR_KWH
                                 / float(eu[col].sum()))
        now = eu.sum(axis=1).groupby(eu.index.date).sum()
        for day, dtotal in daily_target.items():
            day_mask = eu.index.date == day
            eu.loc[day_mask] = eu.loc[day_mask] * (dtotal / float(now[day]))
    eu = eu * (ANCHOR_KWH / float(eu.sum().sum()))
    total = eu.sum(axis=1)
    total.name = "kwh"
    return total, eu, temp


def history_summary() -> dict:
    """Aggregate facts about the measured billing history."""
    bills = load_billing_history()
    billed = bills[bills["bill_usd"].notna()]
    measured = bills[bills["kwh_source"] == "measured"]
    charted = bills[bills["kwh_source"] == "portal chart estimate"]
    summary = {
        "billed_periods": int(len(billed)),
        "bills_total_usd": round(float(billed["bill_usd"].sum()), 2),
        "measured_kwh_periods": int(len(measured)),
        "measured_kwh_total": round(float(measured["kwh"].sum()), 2),
        "chart_estimated_kwh_periods": int(len(charted)),
        "chart_estimated_kwh_total": round(float(charted["kwh"].sum()), 2),
        "first_period_start": str(bills["period_start"].min().date()),
        "last_period_end": str(bills["period_end"].max().date()),
        # Anchor-period time-of-day split, measured by the portal.
        "anchor_peak_kwh": ANCHOR_PEAK_KWH,
        "anchor_offpeak_kwh": ANCHOR_OFFPEAK_KWH,
        "anchor_energy_charges_usd": ANCHOR_ENERGY_CHARGES_USD,
        "anchor_taxes_fees_usd": ANCHOR_TAXES_FEES_USD,
    }
    # Back-compat: full_kwh_* counts only metered periods, as before.
    summary["full_kwh_periods"] = int(len(measured))
    summary["full_kwh_total"] = round(float(measured["kwh"].sum()), 2)
    # Seasonal envelope across all full periods with kWh (measured +
    # chart estimates), excluding the unbilled partial period.
    seasonal = bills[bills["kwh"].notna()
                     & (bills["period_end"] <= pd.Timestamp("2026-10-02"))
                     & (bills["kwh_source"]
                        != "partial (measured through 2026-10-17)")]
    hi = seasonal.loc[seasonal["kwh"].idxmax()]
    lo = seasonal.loc[seasonal["kwh"].idxmin()]
    summary["seasonal_max_kwh"] = float(hi["kwh"])
    summary["seasonal_max_period"] = str(hi["period_start"].date())
    summary["seasonal_min_kwh"] = float(lo["kwh"])
    summary["seasonal_min_period"] = str(lo["period_start"].date())
    summary["seasonal_ratio"] = round(float(hi["kwh"]) / float(lo["kwh"]), 2)
    return summary
