"""Build the demo dataset and freeze measured metrics.

Generates the synthetic 90-day profile, runs the full pipeline
(disaggregation, forecast, anomalies, weather, optimizer, what-ifs),
and writes demo/profile.csv + demo/metrics.json. Every number the
README quotes comes from metrics.json — nothing is hand-typed.
"""

import datetime as dt
import json
import os

import numpy as np
import pandas as pd

from wattwise import (
    Rate180, Rate194, make_demo_profile, disaggregate, learn_shapes,
    forecast_month_bill, daily_anomalies, explain_anomaly,
    typical_weekday_profile, temperature_sensitivity, optimize_month,
    solar_scenario, ev_scenario, heat_pump_scenario,
    monthly_bill, compare_plans, peak_share, appliance_breakdown,
    savings_waterfall, make_measured_profile, history_summary,
)
from wattwise.measured import (ANCHOR_KWH, ANCHOR_BILL_USD,
                               ANCHOR_PEAK_COST_USD)

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(HERE, "..", "demo")


def main():
    os.makedirs(DEMO, exist_ok=True)
    hourly, end_use, temp = make_demo_profile(seed=7)
    hourly.to_csv(os.path.join(DEMO, "profile.csv"), header=True)
    end_use.to_csv(os.path.join(DEMO, "end_use.csv"))
    temp.to_csv(os.path.join(DEMO, "weather.csv"), header=True)

    rate194 = Rate194()
    m = {}

    # --- Bill math on the trailing 30 days ---
    month = hourly.iloc[-30 * 24:]
    m["bill_flat_30d"] = monthly_bill(month, Rate180())["total"]
    m["bill_tou_30d"] = monthly_bill(month, Rate194())["total"]
    m["kwh_30d"] = round(float(month.sum()), 1)
    plans = compare_plans(month)
    m["cheaper_plan"] = plans["cheaper_plan"]
    m["plan_difference"] = plans["difference"]
    ps = peak_share(month, rate194)
    m["peak_kwh_pct"] = ps["peak_kwh_pct"]
    m["peak_cost_pct"] = ps["peak_cost_pct"]

    # --- Optimizer ---
    opt = optimize_month(month, end_use.iloc[-30 * 24:], rate194)
    m["opt_cost_before"] = round(opt.cost_before, 2)
    m["opt_cost_after"] = round(opt.cost_after, 2)
    m["opt_savings"] = round(opt.savings, 2)
    m["opt_savings_pct"] = round(opt.savings_pct, 2)
    m["opt_shifted_kwh"] = round(opt.shifted_kwh, 1)

    # --- Forecast: backtest — fit on 60 days, predict next 7, score ---
    from wattwise.forecast import BillForecaster
    cut = hourly.index[0] + pd.Timedelta(days=60)
    fmodel = BillForecaster().fit(hourly.loc[:cut], temp.loc[:cut])
    test_days = pd.date_range(cut + pd.Timedelta(days=1), periods=7, freq="D")
    fpred = fmodel.predict_days(test_days, temp)
    actual_7 = hourly.loc[test_days[0]:test_days[-1] + pd.Timedelta(days=1)
                          - pd.Timedelta(hours=1)].resample("D").sum()
    m["forecast_7d_kwh"] = round(float(fpred["forecast_kwh"].sum()), 1)
    m["forecast_7d_actual_kwh"] = round(float(actual_7.sum()), 1)
    m["forecast_7d_err_pct"] = round(
        100 * abs(fpred["forecast_kwh"].sum() - actual_7.sum())
        / max(actual_7.sum(), 1e-9), 2)
    # Month-end bill projection from today's vantage point.
    fc = forecast_month_bill(hourly, Rate194(), temp)
    m["forecast_bill"] = round(fc["bill_forecast"], 2)
    m["forecast_bill_lo"] = round(fc["bill_lo"], 2)
    m["forecast_bill_hi"] = round(fc["bill_hi"], 2)

    # --- Disaggregation accuracy (learned shapes, last 30 days) ---
    shapes = learn_shapes(end_use.iloc[:-30 * 24])
    dis = disaggregate(month, shapes=shapes)
    truth = end_use.iloc[-30 * 24:].resample("D").sum()
    truth.index = truth.index.date
    errs = []
    for app in dis.columns:
        if app in truth.columns:
            errs.append(np.abs(dis[app] - truth[app]).mean())
    m["disagg_mae_kwh_per_day"] = round(float(np.mean(errs)), 3)
    m["disagg_recon_rmse"] = round(dis.attrs["mean_rmse_kwh"], 4)

    # --- Anomalies: inject a spike to prove detection works ---
    spiked = hourly.copy()
    spike_day = (hourly.index[-1] - pd.Timedelta(days=9)).date()
    mask = spiked.index.date == spike_day
    evening = (spiked.index.hour >= 18) & (spiked.index.hour < 21)
    spiked.loc[mask & evening] += 1.8  # extra 5.4 kWh that evening
    anoms = daily_anomalies(spiked, min_history=8)
    m["anomaly_detected"] = bool((anoms["date"] == spike_day).any())
    if m["anomaly_detected"]:
        prof = typical_weekday_profile(spiked, spike_day.weekday())
        exp = explain_anomaly(spiked, spike_day, shapes, prof)
        m["anomaly_text"] = exp["text"]
        m["anomaly_excess_kwh"] = exp["excess_kwh"]

    # --- Weather ---
    sens = temperature_sensitivity(hourly, temp)
    m["weather_r2"] = round(sens["r_squared"], 3)
    m["cooling_kwh_per_cdd"] = round(sens["cooling_kwh_per_cdd"], 3)
    m["heating_kwh_per_hdd"] = round(sens["heating_kwh_per_hdd"], 3)
    m["weather_share"] = round(sens["weather_share"], 3)

    # --- What-ifs (on trailing 30 days) ---
    sol = solar_scenario(month, rate194)
    ev = ev_scenario(month, rate194)
    hp = heat_pump_scenario(month, temp.iloc[-30 * 24:], rate194)
    m["solar_monthly_saving"] = sol["monthly_saving"]
    m["solar_payback_years"] = sol["simple_payback_years"]
    m["solar_bill_after"] = sol["bill_after"]
    m["ev_added_monthly_cost"] = ev["added_monthly_cost"]
    m["ev_cost_per_mile_cents"] = ev["fuel_cost_per_mile_cents"]
    m["hp_added_monthly_cost"] = hp["added_monthly_cost"]

    # --- Waterfall ---
    wf = savings_waterfall(m["bill_flat_30d"], m["bill_tou_30d"],
                           m["bill_tou_30d"] - m["opt_savings"],
                           sol["bill_after"])
    m["waterfall"] = [{"label": l, "delta": round(d, 2)} for l, d in wf]

    # --- Appliance breakdown ---
    bd = appliance_breakdown(end_use.iloc[-30 * 24:])
    m["appliance_breakdown"] = [
        {"appliance": i, "kwh": float(r["kwh"]), "pct": float(r["pct"])}
        for i, r in bd.iterrows()
    ]

    m["generated_at"] = dt.datetime.now().isoformat()
    m["demo_note"] = ("SYNTHETIC demo data (seed 7). Real analysis uses "
                      "Green Button XML import.")
    with open(os.path.join(DEMO, "metrics.json"), "w") as f:
        json.dump(m, f, indent=2)
    print(json.dumps({k: v for k, v in m.items()
                      if k not in ("waterfall", "appliance_breakdown")},
                     indent=2))

    build_measured()


def build_measured():
    """Run the pipeline on the measured-anchored Sep 2026 profile and
    write demo/real_metrics.json + demo/real_profile.csv.

    Period totals (320.25 kWh), the appliance mix and the actual bill
    ($109.01, Rate 194) are measured PSEG data; the intraday shape is
    the modeled seeded shape, so modeled bill/optimizer figures below
    are model outputs on a measured-calibrated profile, not meter
    readings.
    """
    hourly, end_use, _temp = make_measured_profile()
    hourly.to_frame().to_csv(os.path.join(DEMO, "real_profile.csv"),
                             header=True)
    rate194 = Rate194()
    rm = {}
    rm["period"] = "2026-09-02 to 2026-10-01 (30 days)"
    rm["measured_kwh"] = ANCHOR_KWH
    rm["measured_bill_actual_usd"] = ANCHOR_BILL_USD
    rm["measured_peak_cost_actual_usd"] = ANCHOR_PEAK_COST_USD
    rm["measured_avg_daily_kwh"] = round(ANCHOR_KWH / 30, 2)

    # What wattwise's tariff model computes on the same usage.
    rm["modeled_bill_tou_usd"] = monthly_bill(hourly, rate194)["total"]
    rm["modeled_bill_flat_usd"] = monthly_bill(hourly, Rate180())["total"]
    plans = compare_plans(hourly)
    rm["cheaper_plan"] = plans["cheaper_plan"]
    rm["plan_difference"] = plans["difference"]
    ps = peak_share(hourly, rate194)
    rm["modeled_peak_kwh_pct"] = ps["peak_kwh_pct"]
    rm["modeled_peak_cost_pct"] = ps["peak_cost_pct"]

    # Load shifting on the measured appliance mix.
    opt = optimize_month(hourly, end_use, rate194)
    rm["opt_cost_before"] = round(opt.cost_before, 2)
    rm["opt_cost_after"] = round(opt.cost_after, 2)
    rm["opt_savings"] = round(opt.savings, 2)
    rm["opt_savings_pct"] = round(opt.savings_pct, 2)
    rm["opt_shifted_kwh"] = round(opt.shifted_kwh, 1)

    bd = appliance_breakdown(end_use)
    rm["appliance_breakdown"] = [
        {"appliance": i, "kwh": float(r["kwh"]), "pct": float(r["pct"])}
        for i, r in bd.iterrows()
    ]
    rm["history"] = history_summary()
    rm["generated_at"] = dt.datetime.now().isoformat()
    rm["data_note"] = (
        "MEASURED PSEG Long Island data. Period kWh totals, appliance "
        "mix and bills come from utility emails (weekly usage "
        "summaries, bill notices, appliance dollar estimates); see "
        "demo/pseg_bills.csv and demo/pseg_appliance_estimates.csv. "
        "Intraday hourly shape is modeled (seeded), not metered "
        "interval data, so modeled bills and optimizer savings are "
        "model outputs on a measured-calibrated profile.")
    with open(os.path.join(DEMO, "real_metrics.json"), "w") as f:
        json.dump(rm, f, indent=2)
    print(json.dumps(rm, indent=2))


if __name__ == "__main__":
    main()
