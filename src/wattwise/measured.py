"""Measured PSEG Long Island data for this household, and a profile
anchored to it.

Two measured datasets ship with the repo (demo/pseg_bills.csv and
demo/pseg_appliance_estimates.csv), extracted from PSEG Long Island
email alerts with all identifying fields stripped:

- Weekly Usage Summary emails give cumulative kWh per billing period
  (complete for the Jul, Aug and Sep 2026 periods).
- Bill emails give the amount due per period (Feb-Oct 2026, six
  billed periods; the Sep 2026 bill is $109.01 including $17 of
  peak-hour cost, per the utility's energy-insights email).
- The utility's appliance breakdowns are dollar ESTIMATES produced by
  its disaggregation partner from meter data. They exclude the daily
  service charge and taxes, so they do not sum to the bill.

make_measured_profile() anchors the seeded demo generator to the
September 2026 period: each end-use column is rescaled so the 30-day
total per appliance matches the measured appliance mix and the month
totals exactly 320.25 kWh. The intraday SHAPE of each appliance is
still the modeled seeded shape -- PSEG emails do not contain interval
data -- so hourly detail is modeled, while period totals, the
appliance mix and the bills are measured. That split is deliberate and
is stated wherever the numbers are quoted.
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
ANCHOR_PEAK_COST_USD = 17.0  # peak-hour cost inside that bill

# Utility appliance categories -> wattwise end uses. ``other`` has no
# direct counterpart; it is split between water heating and the
# dishwasher in proportion to the seeded model's ratio. This mapping
# is an estimate, documented here rather than hidden.
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


def anchor_appliance_shares(end_use_seed: pd.DataFrame) -> dict:
    """Wattwise end-use kWh shares for the anchor period, derived from
    the utility's dollar estimates for 2026-09-02 to 2026-10-01.

    Dollar shares are used as kWh shares; within one billing period
    under one tariff that is an approximation (peak hours price some
    appliances higher), and it is labeled as such.
    """
    est = load_appliance_estimates()
    sel = est[est["period_start"] == pd.Timestamp(ANCHOR_PERIOD_START)]
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


def make_measured_profile(seed: int = 7):
    """Seeded 30-day profile rescaled to the measured Sep 2026 period.

    Returns (hourly_kwh Series, end_use DataFrame, weather Series) for
    2026-09-02..2026-10-01. End-use 30-day totals match the measured
    appliance mix; the household total is exactly ANCHOR_KWH.
    """
    _, end_use, temp = make_demo_profile(
        seed=seed, days=30, end=ANCHOR_PERIOD_END)
    shares = anchor_appliance_shares(end_use)
    scaled = end_use.copy()
    for col in scaled.columns:
        current = float(scaled[col].sum())
        target = shares[col] * ANCHOR_KWH
        scaled[col] = scaled[col] * (target / current)
    total = scaled.sum(axis=1)
    total.name = "kwh"
    return total, scaled, temp


def history_summary() -> dict:
    """Aggregate facts about the measured billing history."""
    bills = load_billing_history()
    billed = bills[bills["bill_usd"].notna()]
    with_kwh = bills[bills["kwh"].notna() & (bills["period_end"]
                                             <= pd.Timestamp("2026-10-02"))]
    return {
        "billed_periods": int(len(billed)),
        "bills_total_usd": round(float(billed["bill_usd"].sum()), 2),
        "full_kwh_periods": int(len(with_kwh)),
        "full_kwh_total": round(float(with_kwh["kwh"].sum()), 2),
        "first_period_start": str(bills["period_start"].min().date()),
        "last_period_end": str(bills["period_end"].max().date()),
    }
