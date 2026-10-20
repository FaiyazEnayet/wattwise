"""Tests for the insights engine (decision-support on measured data).

All tests run against the shipped demo data: the expectations are
about internal consistency (arithmetic, ranges, detection logic),
never about hand-typed dollar figures.
"""

import datetime as dt

import numpy as np
import pandas as pd

from wattwise.insights import (forecast_next_bill, peak_drift, what_if,
                               seasonal_outlook, cost_drivers,
                               detect_anomalies, report)
from wattwise.measured import load_intervals
from wattwise.rates import Rate194History


def test_forecast_in_sane_range():
    fc = forecast_next_bill()
    assert 50.0 < fc["estimate_usd"] < 250.0
    assert fc["low_usd"] < fc["estimate_usd"] < fc["high_usd"]
    assert 150.0 < fc["projected_kwh"] < 700.0
    assert 0.5 < fc["seasonal_factor"] < 2.0
    # Band is +/-15% by construction.
    assert fc["low_usd"] == round(fc["estimate_usd"] * 0.85, 2)


def test_peak_drift_math():
    pd_ = peak_drift()
    assert pd_["baseline_peak_share"] > 0
    assert 0 < pd_["current_peak_share"] < 1
    drift = ((pd_["current_peak_share"] - pd_["baseline_peak_share"])
             / pd_["baseline_peak_share"])
    # The function rounds its outputs to 4 decimals; allow rounding slack.
    assert abs(drift - pd_["drift"]) < 1e-3
    assert pd_["flagged"] == (drift > 0.15)


def test_what_if_matches_manual_calc():
    rates = Rate194History()
    bill_date = sorted(rates.periods.keys())[-1]
    # Shifting 100% of peak off-peak: the saving must equal the peak
    # premium times the peak kWh, computed by hand.
    p = rates.periods[bill_date]
    premium = (p["del_peak_rate"] + p["sup_peak_rate"]
               - p["del_offpeak_rate"] - p["sup_offpeak_rate"])
    out = what_if(peak_shift_pct=100.0, rates=rates)
    manual = rates.bill_for(bill_date, 0.0, 320.0)
    assert abs(out["peak_shift"]["new_bill_usd"] - manual) < 0.01
    assert abs(out["peak_shift"]["saving_usd"] - 39.0 * premium) < 0.01
    # Baseline reproduces the anchor bill.
    assert abs(out["baseline_bill_usd"] - p["bill_usd"]) < 0.01


def test_what_if_cuts_reduce_bill():
    out = what_if(peak_shift_pct=20.0, cooling_cut_pct=20.0,
                  total_cut_pct=10.0)
    for key in ("peak_shift", "cooling_cut", "total_cut"):
        assert out[key]["saving_usd"] > 0
        assert out[key]["new_bill_usd"] < out["baseline_bill_usd"]
    # A total 10% cut saves about 10% of the energy+taxes portion
    # (fixed charges stay), so it must exceed the peak-shift saving.
    assert out["total_cut"]["saving_usd"] > out["peak_shift"]["saving_usd"]


def test_seasonal_outlook_lookup():
    rows = seasonal_outlook()
    assert len(rows) >= 2
    for row in rows:
        assert row["expected_kwh"] > 0
        assert row["expected_bill_usd"] > 0
        assert row["month"] >= "2026-12"
    # December should be cheaper than the summer peak (Aug ~619 kWh).
    assert rows[0]["expected_kwh"] < 300.0


def test_cost_drivers_ranking_and_savings():
    drivers = cost_drivers()
    dollars = [d["dollars_usd"] for d in drivers]
    assert dollars == sorted(dollars, reverse=True)
    top = drivers[0]
    assert top["category"] == "cooling"
    assert abs(top["save_10pct_usd"] - top["dollars_usd"] * 0.10) < 0.01
    assert abs(top["save_20pct_usd"] - top["dollars_usd"] * 0.20) < 0.01
    # Taxes and fees are not an end use.
    assert "taxes_fees" not in [d["category"] for d in drivers]


def test_anomaly_injection_detected():
    iv = load_intervals().copy()
    # Inject a 3x-usage Monday far above its weekday history.
    spike_day = pd.Timestamp("2026-02-02")  # a Monday
    mask = iv.index.normalize() == spike_day
    assert mask.sum() > 0
    iv.loc[mask, "kwh"] = iv.loc[mask, "kwh"] * 3.0
    anoms = detect_anomalies(intervals=iv)
    dates = [a["date"] for a in anoms]
    assert "2026-02-02" in dates
    hit = [a for a in anoms if a["date"] == "2026-02-02"][0]
    assert hit["z_score"] > 2.0
    assert hit["excess_usd"] > 0


def test_report_has_all_sections():
    text = report()
    for section in ("Bill forecast", "Peak-hour drift", "What-if",
                    "Seasonal outlook", "Cost drivers", "Anomalies"):
        assert section in text
