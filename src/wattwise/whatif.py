"""What-if scenarios: solar, EV, heat pump.

Each scenario starts from the household's measured (or demo) load and
recomputes the bill under a physical change, with every assumption
stated up front. These are planning estimates, not quotes.
"""

import numpy as np
import pandas as pd


def _tou_bill_from_hourly(net_hourly: pd.Series, rate) -> float:
    """Bill for an hourly net series under a TOU rate.

    Net metering: within each hour, solar offsets load 1:1; exported kWh
    earn no separate credit here (excess is simply zeroed), which is the
    conservative reading. Fixed charge always applies.
    """
    days = (net_hourly.index[-1] - net_hourly.index[0]).days + 1
    energy = sum(rate.energy_charge(max(v, 0.0), ts)
                 for ts, v in net_hourly.items())
    return energy + rate.fixed_charge(days)


def solar_scenario(hourly: pd.Series, rate, system_kw: float = 6.0,
                   seed: int = 11) -> dict:
    """What if the roof had solar? 6 kW system, simple clear-sky model.

    Generation follows a sine hump peaking at 1 PM, scaled by nameplate
    and a random daily cloud factor (0.35-1.0). Assumption, documented.
    """
    rng = np.random.default_rng(seed)
    hours = hourly.index
    h = hours.hour.to_numpy()
    # Daylight hump: zero at 6 AM / 8 PM, peak 1 PM.
    sun = np.clip(np.sin(np.pi * (h - 6) / 14.0), 0, None)
    day_factor = pd.Series(hours.date).map(
        dict(zip(pd.unique(hours.date),
                 rng.uniform(0.35, 1.0, len(pd.unique(hours.date))))))
    gen_kw = system_kw * 0.82 * sun * day_factor.to_numpy()  # 0.82 derate
    gen = pd.Series(np.clip(gen_kw, 0, None), index=hours)

    days = (hours[-1] - hours[0]).days + 1
    base_bill = _tou_bill_from_hourly(hourly, rate)
    net = hourly - gen
    solar_bill = _tou_bill_from_hourly(net, rate)
    monthly_saving = base_bill - solar_bill
    annual_saving = monthly_saving * 365.0 / days
    # Installed cost assumption, documented: $2.80/W before incentives.
    install_cost = system_kw * 1000 * 2.80
    payback_years = install_cost / max(annual_saving, 1e-9)
    return {
        "scenario": f"{system_kw:.0f} kW rooftop solar",
        "monthly_gen_kwh": float(gen.sum() * 30.0 / days),
        "self_consumed_pct": float(100 * np.clip(hourly - net.clip(lower=0),
                                                 0, None).sum()
                                   / max(gen.sum(), 1e-9)),
        "bill_before": round(base_bill, 2),
        "bill_after": round(solar_bill, 2),
        "monthly_saving": round(monthly_saving, 2),
        "est_install_cost": round(install_cost, 2),
        "simple_payback_years": round(payback_years, 1),
        "assumptions": ("6 kW, 0.82 derate, random cloud 0.35-1.0, "
                        "$2.80/W installed, no export credit, no incentives"),
    }


def ev_scenario(hourly: pd.Series, rate, miles_per_day: float = 30.0,
                kwh_per_mile: float = 0.30) -> dict:
    """What if you drove electric? Adds overnight charging (11 PM-6 AM).

    30 miles/day at 0.30 kWh/mile ~= 9 kWh/night, all off-peak.
    Assumption, documented.
    """
    hours = hourly.index
    h = hours.hour.to_numpy()
    night = (h >= 23) | (h < 6)
    night_hours = night.sum()
    ev_kwh_total = miles_per_day * kwh_per_mile * \
        ((hours[-1] - hours[0]).days + 1)
    per_hour = ev_kwh_total / max(night_hours, 1)
    added = pd.Series(np.where(night, per_hour, 0.0), index=hours)
    new_load = hourly + added
    days = (hours[-1] - hours[0]).days + 1
    base_bill = _tou_bill_from_hourly(hourly, rate)
    ev_bill = _tou_bill_from_hourly(new_load, rate)
    extra_kwh_month = float(added.sum() * 30.0 / days)
    return {
        "scenario": f"EV, {miles_per_day:.0f} mi/day",
        "extra_kwh_per_month": round(extra_kwh_month, 1),
        "bill_before": round(base_bill, 2),
        "bill_after": round(ev_bill, 2),
        "added_monthly_cost": round(ev_bill - base_bill, 2),
        "fuel_cost_per_mile_cents": round(
            100 * (ev_bill - base_bill) / max(extra_kwh_month / 0.30, 1e-9), 1),
        "assumptions": ("0.30 kWh/mile, charged 11 PM-6 AM (off-peak), "
                        "no demand charge modeled"),
    }


def heat_pump_scenario(hourly: pd.Series, temp_f: pd.Series, rate,
                       cop: float = 3.0, heat_kwh_per_hdd: float = 2.2) -> dict:
    """What if space heating ran on a heat pump?

    Adds HDD-driven electric heat at the given COP. heat_kwh_per_hdd is
    the thermal load per heating degree-day (assumption, documented);
    electricity = thermal / COP.
    """
    hours = hourly.index
    daily_temp = temp_f.resample("D").mean()
    hdd = np.clip(65.0 - daily_temp, 0, None)
    # Spread each day's heating across 6 AM - 10 PM.
    added = pd.Series(0.0, index=hours)
    h = hours.hour.to_numpy()
    heat_hours = (h >= 6) & (h < 22)
    day_map = pd.Series(hours.date, index=hours)
    for d, v in hdd.items():
        if v <= 0:
            continue
        mask = (day_map == d.date()) & heat_hours
        added.loc[mask] = v * heat_kwh_per_hdd / cop / max(mask.sum(), 1)
    new_load = hourly + added
    base_bill = _tou_bill_from_hourly(hourly, rate)
    hp_bill = _tou_bill_from_hourly(new_load, rate)
    days = (hours[-1] - hours[0]).days + 1
    return {
        "scenario": f"Heat pump (COP {cop})",
        "extra_kwh_per_month": round(float(added.sum() * 30.0 / days), 1),
        "bill_before": round(base_bill, 2),
        "bill_after": round(hp_bill, 2),
        "added_monthly_cost": round(hp_bill - base_bill, 2),
        "assumptions": (f"{heat_kwh_per_hdd} thermal kWh/HDD, COP {cop}, "
                        "heat 6 AM-10 PM, no equipment cost modeled"),
    }
