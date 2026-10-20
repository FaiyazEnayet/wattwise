"""Tests for the measured PSEG data module.

Regression cover for the real-data layer: billing history integrity,
profile calibration against measured totals, sub-period data, and a
PII scan of the data files that ship in the repo.
"""

import os
import re

import pandas as pd
import pytest

from wattwise.measured import (
    ANCHOR_KWH, load_billing_history, load_monthly, load_intervals,
    interval_reconciliation,
    load_appliance_estimates, load_subperiods, load_rates,
    make_measured_profile, history_summary,
    ANCHOR_PERIOD_START, ANCHOR_PEAK_KWH, ANCHOR_OFFPEAK_KWH,
    ANCHOR_ENERGY_CHARGES_USD, ANCHOR_TAXES_FEES_USD,
)

DEMO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "demo")


def test_billing_history_integrity():
    bills = load_billing_history()
    billed = bills[bills["bill_usd"].notna()]
    # Fourteen exact portal bills, Aug 2025 through Oct 2026.
    assert len(billed) == 14
    assert billed["bill_usd"].sum() == pytest.approx(1493.31, abs=0.01)
    sep = bills[bills["period_start"] == pd.Timestamp("2026-09-02")].iloc[0]
    assert sep["kwh"] == pytest.approx(320.25, abs=0.01)
    assert sep["bill_usd"] == pytest.approx(109.01, abs=0.001)
    assert sep["days"] == 30
    # Exact portal chain: bill dates + days-in-bill, periods chained.
    aug25 = bills[bills["period_start"] == pd.Timestamp("2025-08-01")].iloc[0]
    assert aug25["bill_usd"] == pytest.approx(138.98, abs=0.001)
    assert aug25["days"] == 33
    oct25 = bills[bills["period_start"] == pd.Timestamp("2025-09-03")].iloc[0]
    assert oct25["bill_usd"] == pytest.approx(99.24, abs=0.001)
    assert oct25["days"] == 29
    mar26 = bills[bills["period_start"] == pd.Timestamp("2026-03-03")].iloc[0]
    assert mar26["bill_usd"] == pytest.approx(93.71, abs=0.001)
    assert mar26["days"] == 30
    jun26 = bills[bills["period_start"] == pd.Timestamp("2026-06-02")].iloc[0]
    assert jun26["bill_usd"] == pytest.approx(124.78, abs=0.001)
    # Periods chain: each period_end equals the next period_start, and
    # days equal the (end - start) span, matching the portal counts.
    chained = bills.sort_values("period_start").reset_index(drop=True)
    for i in range(len(chained) - 1):
        assert chained.loc[i, "period_end"] == chained.loc[i + 1, "period_start"]
    spans = (chained["period_end"] - chained["period_start"]).dt.days
    assert (spans == chained["days"]).all()
    # No PII columns may ever appear in the shipped CSV.
    forbidden = {"name", "address", "account", "zip", "card", "email"}
    assert not (set(bills.columns) & forbidden)


def test_monthly_loader_and_seasonal_pattern():
    monthly = load_monthly()
    # 14 billed periods + the unbilled partial Oct-Nov 2026 period.
    assert len(monthly) == 15
    # The Sep 2026 anchor: measured split from the portal monthly summary.
    sep = monthly[monthly["period_start"] == pd.Timestamp("2026-09-02")].iloc[0]
    assert sep["kwh_peak"] == 39
    assert sep["kwh_offpeak"] == 281
    assert sep["kwh_total"] == pytest.approx(320.25, abs=0.01)
    assert sep["bill_usd"] == pytest.approx(109.01, abs=0.001)
    # Measured totals agree with the weekly-summary figures.
    measured = monthly[monthly["kwh_source"].str.startswith("measured",
                                                             na=False)]
    assert set(measured["kwh_total"]) >= {619.39, 427.24}
    # Chart-estimated periods are labeled as estimates, not measured.
    estimated = monthly[monthly["kwh_source"] == "portal chart estimate"]
    assert len(estimated) == 9
    # Seasonal pattern: August peak vs January trough, ~3x swing.
    full = monthly[monthly["kwh_total"].notna()
                   & (monthly["bill_usd"].notna())]
    hi = full.loc[full["kwh_total"].idxmax()]
    lo = full.loc[full["kwh_total"].idxmin()]
    assert hi["period_start"] == pd.Timestamp("2026-07-02")
    assert hi["kwh_total"] == pytest.approx(619.39, abs=0.01)
    assert lo["period_start"] == pd.Timestamp("2025-12-02")
    assert hi["kwh_total"] / lo["kwh_total"] >= 3.0
    # Temperature curve tracks the season: warm peak, cold trough.
    assert hi["avg_temp_f"] >= 70
    assert lo["avg_temp_f"] <= 35
    # Pre-Oct-2025 periods: the Aug 2025 period now has its 472 kWh
    # from the portal bill comparison; the Sep 2025 period still has
    # no kWh (export starts 2025-09-19).
    early = monthly[monthly["period_start"] < pd.Timestamp("2025-10-02")]
    aug25 = early[early["period_start"] == pd.Timestamp("2025-08-01")].iloc[0]
    assert aug25["kwh_total"] == pytest.approx(472.0, abs=0.01)
    assert aug25["kwh_source"] == "portal comparison"
    sep25 = early[early["period_start"] == pd.Timestamp("2025-09-03")].iloc[0]
    assert pd.isna(sep25["kwh_total"])


def test_appliance_estimates_match_sep_anchors():
    est = load_appliance_estimates()
    sel = est[est["period_start"] == pd.Timestamp(ANCHOR_PERIOD_START)]
    # Appliance rows sum to the portal's energy charges ($82); the
    # taxes/fees row ($27) reconciles to the $109 bill.
    appliance = sel[sel["category"] != "taxes_fees"]
    assert appliance["dollars_usd"].sum() == 82
    assert sel[sel["category"] == "taxes_fees"]["dollars_usd"].sum() == 27
    assert sel["dollars_usd"].sum() == 109
    cooling = sel[sel["category"] == "cooling"]["dollars_usd"].sum()
    assert cooling == 18
    # Module constants match the portal's monthly-summary split.
    assert ANCHOR_PEAK_KWH == 39.0
    assert ANCHOR_OFFPEAK_KWH == 281.0
    assert ANCHOR_ENERGY_CHARGES_USD == 82.0
    assert ANCHOR_TAXES_FEES_USD == 27.0


def test_subperiods_reconcile_with_month_totals():
    subs = load_subperiods()
    # Sub-period increments inside Jul/Oct 2026 must sum to the
    # measured billing-period totals.
    jul = subs[subs["subperiod_start"] >= pd.Timestamp("2026-07-02")]
    jul = jul[jul["subperiod_start"] < pd.Timestamp("2026-08-04")]
    assert jul["kwh"].sum() == pytest.approx(619.39, abs=0.05)
    sep = subs[subs["subperiod_start"] >= pd.Timestamp("2026-09-02")]
    sep = sep[sep["subperiod_start"] < pd.Timestamp("2026-10-02")]
    assert sep["kwh"].sum() == pytest.approx(320.25, abs=0.05)
    # All increments are positive; 12 measured sub-periods.
    assert len(subs) == 12
    assert (subs["kwh"] > 0).all()


def test_rates_file_has_tod_facts():
    rates = load_rates()
    topics = set(rates["topic"])
    assert "peak window" in topics
    detail = rates[rates["topic"] == "peak window"]["detail"].iloc[0]
    assert "3-7 PM" in detail


def test_measured_profile_hits_anchor_totals():
    hourly, end_use, _temp = make_measured_profile()
    # 30 days of hourly data, total exactly the measured month.
    assert len(hourly) == 30 * 24
    assert hourly.sum() == pytest.approx(ANCHOR_KWH, abs=0.1)
    # Appliance mix matches the utility's dollar-share estimates.
    assert end_use["hvac"].sum() == pytest.approx(0.2195 * ANCHOR_KWH, abs=1.0)
    assert end_use["laundry"].sum() == pytest.approx(0.1951 * ANCHOR_KWH, abs=1.0)
    # The real mix disagrees with the synthetic demo: water heating is
    # a small share here, dominant there. Guard both directions so a
    # silent reshape cannot sneak through.
    assert end_use["water_heater"].sum() < 0.10 * ANCHOR_KWH


def test_measured_profile_appliance_totals_reconcile():
    _hourly, end_use, _temp = make_measured_profile()
    assert end_use.sum().sum() == pytest.approx(ANCHOR_KWH, abs=0.1)


def test_measured_profile_follows_subperiod_trend():
    hourly, _end_use, _temp = make_measured_profile()
    daily = hourly.groupby(hourly.index.date).sum()
    dates = [pd.Timestamp(d) for d in daily.index]
    daily.index = dates
    subs = load_subperiods()
    subs = subs[subs["subperiod_start"] >= pd.Timestamp("2026-09-02")]
    subs = subs[subs["subperiod_start"] < pd.Timestamp("2026-10-02")]
    # Each sub-period's daily mean must match the measured one; the
    # September profile declines over the month, and the model should
    # reproduce that trend rather than a flat seeded shape.
    for _, row in subs.iterrows():
        sel = daily[(daily.index >= row["subperiod_start"])
                    & (daily.index <= row["subperiod_end"])]
        assert sel.mean() == pytest.approx(row["avg_daily_kwh"], abs=0.15)
    # The measured trend declines: first sub-period above the last.
    first = subs.iloc[0]["avg_daily_kwh"]
    last = subs.iloc[-1]["avg_daily_kwh"]
    assert first > last


def test_history_summary_aggregates():
    s = history_summary()
    assert s["billed_periods"] == 14
    assert s["bills_total_usd"] == pytest.approx(1493.31, abs=0.01)
    assert s["measured_kwh_periods"] == 3
    assert s["measured_kwh_total"] == pytest.approx(1366.88, abs=0.01)
    assert s["chart_estimated_kwh_periods"] == 9
    assert s["chart_estimated_kwh_total"] == pytest.approx(2400.0, abs=0.01)
    assert s["full_kwh_periods"] == 3
    assert s["full_kwh_total"] == pytest.approx(1366.88, abs=0.01)
    assert s["seasonal_max_kwh"] == pytest.approx(619.39, abs=0.01)
    assert s["seasonal_max_period"] == "2026-07-02"
    assert s["seasonal_min_kwh"] == 195
    assert s["seasonal_ratio"] >= 3.0
    assert s["anchor_peak_kwh"] == 39.0
    assert s["anchor_offpeak_kwh"] == 281.0
    assert s["anchor_taxes_fees_usd"] == 27.0


def test_data_files_contain_no_pii():
    """Pattern-based PII scan: no real tokens needed, so this test is
    safe to ship in the public repo. Catches emails, card-like digit
    runs, street-address patterns and account-style labels."""
    leaked = []
    files = ("pseg_bills.csv", "pseg_appliance_estimates.csv",
             "pseg_monthly.csv", "pseg_subperiod.csv", "pseg_rates.csv",
             "pseg_rates_detailed.csv", "pseg_daily_oct2026.csv",
             "pseg_hourly_oct17.csv", "pseg_interval_15min.csv",
             "pseg_interval_validation.csv", "real_metrics.json")
    patterns = {
        "email": re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I),
        "card_run": re.compile(r"\d{15,19}"),
        "street": re.compile(r"\d{2,6}\s+[A-Z][A-Za-z]*\s+"
                             r"(ST|RD|AVE|LANE|DR|BLVD|WAY|CT|PL)\b"),
        "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "phone": re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        "account_label": re.compile(r"ACCOUNT\s*(NUMBER|#|NO\.?)",
                                    re.I),
    }
    for fname in files:
        with open(os.path.join(DEMO, fname)) as f:
            text = f.read()
        for label, rx in patterns.items():
            if rx.search(text):
                leaked.append((fname, label))
    assert not leaked, f"PII patterns in repo data: {leaked}"


def test_intervals_load_and_total():
    iv = load_intervals()
    # Full-year export: 2025-09-19 to 2026-10-19.
    assert len(iv) == 37976
    assert iv["kwh"].sum() == pytest.approx(4086.55, abs=1.0)
    assert set(iv.columns) == {"kwh", "kw"}


def test_interval_reconciliation_matches_billed_kwh():
    """End-exclusive periods and line-item totals remove the extra-day bias."""
    rows = interval_reconciliation()
    full = [r for r in rows if r["billed_kwh"] and r["n_intervals"] >= 28 * 96]
    assert len(full) == 12
    worst = 0.0
    for r in full:
        err = 100 * (r["interval_kwh"] - r["billed_kwh"]) / r["billed_kwh"]
        worst = max(worst, abs(err))
        assert abs(err) < 0.5
    assert worst < 0.5


def test_interval_peak_reproduces_billed_peak():
    """The Rate 194 peak window (weekdays 3-7 PM, federal holidays
    excluded) applied to the intervals reproduces the utility's own
    billed peak kWh per period."""
    rows = interval_reconciliation()
    det = pd.read_csv(os.path.join(DEMO, "pseg_rates_detailed.csv"))
    billed_peak = {r["period_start"]: r["peak_kwh"]
                   for _, r in det.iterrows()}
    for r in rows:
        if r["billed_kwh"] is None or r["n_intervals"] < 28 * 96:
            continue
        assert r["interval_peak_kwh"] == pytest.approx(
            billed_peak[r["period_start"]], abs=6.0)
    # Sep 2026 anchor after correcting the extra-day boundary.
    sep = next(r for r in rows if r["period_start"] == "2026-09-02")
    assert sep["interval_peak_kwh"] == pytest.approx(39.08, abs=0.1)


def test_measured_profile_uses_real_interval_shape():
    """The measured Sep profile is built on the real interval shape,
    not the seeded one: its weekend profile correlates with the
    measured fall-weekend shape, and midday is materially higher
    than the old seeded 20%."""
    from wattwise.measured import interval_hourly_shape
    hourly, _end_use, _temp = make_measured_profile()
    sats = hourly[hourly.index.weekday == 5]
    prof = sats.groupby(sats.index.hour).mean()
    share = (prof / prof.sum()).to_numpy()
    real = interval_hourly_shape("fall", True)
    corr = float(pd.Series(share).corr(pd.Series(real)))
    # The profile mixes own-day shapes (Sep 19/26) with the season
    # average (Sep 5/12), so correlation is high but not exact.
    assert corr > 0.80
    # Real fall weekends: midday 27.5%, 17-21h 26.1%, overnight 19.3%
    # -- the seeded shape was 20.0% / 47.0% / 8.4%.
    assert share[10:15].sum() > 0.25
    assert share[17:22].sum() < 0.30
    assert share[0:6].sum() > 0.12


def test_hourly_oct17_loader():
    from wattwise.measured import load_hourly_oct17
    h = load_hourly_oct17()
    assert len(h) == 24
    assert set(h["tou_period"]) == {"off-peak"}
    assert h["cost_usd"].sum() == pytest.approx(2.15, abs=0.02)
    assert h.loc[h["cost_usd"].idxmax(), "hour"] == 13
