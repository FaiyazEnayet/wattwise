"""Bill math, plan comparison, and the savings waterfall."""

import pandas as pd

from .rates import Rate180, Rate194, is_summer


def monthly_bill(hourly: pd.Series, rate) -> dict:
    """Full bill for the hours given: energy + fixed charge."""
    first, last = hourly.index[0], hourly.index[-1]
    days = (last - first).days + 1
    summer = is_summer(first.date())
    if isinstance(rate, Rate180):
        kwh = float(hourly.sum())
        energy = rate.energy_charge(kwh, summer)
    else:
        energy = sum(rate.energy_charge(v, ts) for ts, v in hourly.items())
    fixed = rate.fixed_charge(days)
    return {
        "kwh": float(hourly.sum()),
        "days": days,
        "energy_charge": round(energy, 2),
        "fixed_charge": round(fixed, 2),
        "total": round(energy + fixed, 2),
        "avg_cents_per_kwh": round(100 * (energy + fixed) / max(hourly.sum(), 1e-9), 2),
    }


def compare_plans(hourly: pd.Series, supply_rate: float = None) -> dict:
    """Bill the same load under flat vs time-of-day. Returns both + winner."""
    kw = {} if supply_rate is None else {"supply_rate": supply_rate}
    flat = monthly_bill(hourly, Rate180(**kw))
    tou = monthly_bill(hourly, Rate194(**kw))
    winner = "flat (Rate 180)" if flat["total"] <= tou["total"] else "time-of-day (Rate 194)"
    return {
        "flat": flat,
        "tou_as_is": tou,
        "cheaper_plan": winner,
        "difference": round(abs(flat["total"] - tou["total"]), 2),
    }


def peak_share(hourly: pd.Series, rate: Rate194) -> dict:
    """What fraction of energy (and cost) falls in peak hours?"""
    from .rates import is_peak_hour
    peak_kwh = sum(v for ts, v in hourly.items()
                   if is_peak_hour(ts, rate.holidays))
    total = float(hourly.sum())
    peak_cost = sum(rate.energy_charge(v, ts) for ts, v in hourly.items()
                    if is_peak_hour(ts, rate.holidays))
    total_energy_cost = sum(rate.energy_charge(v, ts) for ts, v in hourly.items())
    return {
        "peak_kwh": round(peak_kwh, 1),
        "peak_kwh_pct": round(100 * peak_kwh / max(total, 1e-9), 1),
        "peak_cost_pct": round(100 * peak_cost / max(total_energy_cost, 1e-9), 1),
    }


def appliance_breakdown(end_use: pd.DataFrame) -> pd.DataFrame:
    """Monthly kWh and share per appliance, sorted high to low."""
    kwh = end_use.sum().sort_values(ascending=False)
    return pd.DataFrame({
        "kwh": kwh.round(1),
        "pct": (100 * kwh / kwh.sum()).round(1),
    })


def savings_waterfall(bill_flat: float, bill_tou: float,
                      bill_optimized: float, bill_solar: float = None) -> list:
    """Ordered (label, $) steps for a waterfall chart."""
    steps = [
        ("Flat rate bill (Rate 180)", bill_flat),
        ("Switch to time-of-day (Rate 194)", bill_tou - bill_flat),
        ("Shift flexible loads off-peak", bill_optimized - bill_tou),
    ]
    if bill_solar is not None:
        steps.append(("Add 6 kW rooftop solar", bill_solar - bill_optimized))
    return steps
