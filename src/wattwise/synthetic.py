"""Anonymous synthetic households for public CI and examples.

Generation method (declared synthetic process)
----------------------------------------------
Every value in this module comes from a seeded generator over invented,
round-number parameters. No measured household data is read, sampled,
perturbed, subsampled, noised or otherwise transformed: the process
below is the whole provenance of the fixture.

For each synthetic home, hour by hour over a fixed 62-day window:

- a synthetic temperature curve: a slow sinusoid over the window plus
  a daily sinusoid, with seeded Gaussian noise;
- a constant base load;
- a Gaussian morning bump and a Gaussian evening bump (the evening
  bump is larger for one home, smaller for the other), scaled on
  weekend days by a declared weekend factor;
- a cold term proportional to how far the synthetic temperature falls
  below a declared threshold;
- seeded Gaussian load noise, floored at a small positive minimum.

Energy is rounded to 4 decimal places. Bills are then priced from the
generated intervals with the home's declared tariff (see
`households.json` in the fixture directory) plus its daily service
charge, rounded to cents, so the fixture bills reconcile with the
fixture intervals exactly up to that cent rounding.

The two homes are interface fixtures. They demonstrate that the
pipeline runs end to end on files that are not the measured household;
they are not evidence about any real household or population.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_START = "2026-01-05T00:00:00"  # a Monday; a round, arbitrary window
_DAYS = 62

# Invented parameters. Nothing here is fitted to, or read from, any
# measured series.
SYNTHETIC_HOMES = {
    "synthetic-a": {
        "seed": 1201,
        "base_kw": 0.12,
        "morning_amp_kw": 0.55, "morning_center_h": 7.5, "morning_width_h": 1.2,
        "evening_amp_kw": 0.95, "evening_center_h": 19.0, "evening_width_h": 2.2,
        "weekend_scale": 1.15,
        "cold_threshold_f": 36.0, "cold_kw_per_f": 0.012,
        "noise_sd_kw": 0.03,
        "tariff": {"kind": "flat", "name": "Synthetic flat",
                   "delivery_per_kwh": 0.11, "supply_per_kwh": 0.13,
                   "daily_service_charge": 0.50},
    },
    "synthetic-b": {
        "seed": 1202,
        "base_kw": 0.09,
        "morning_amp_kw": 0.35, "morning_center_h": 6.8, "morning_width_h": 1.0,
        "evening_amp_kw": 1.25, "evening_center_h": 19.5, "evening_width_h": 1.8,
        "weekend_scale": 0.90,
        "cold_threshold_f": 36.0, "cold_kw_per_f": 0.018,
        "noise_sd_kw": 0.045,
        "tariff": {"kind": "tou", "name": "Synthetic time-of-use",
                   "peak_delivery_per_kwh": 0.21,
                   "offpeak_delivery_per_kwh": 0.10,
                   "peak_supply_per_kwh": 0.28,
                   "offpeak_supply_per_kwh": 0.11,
                   "daily_service_charge": 0.55,
                   "peak_start_hour": 15, "peak_end_hour": 19},
    },
}

GENERATION_METHOD = (
    "Seeded synthetic generator over invented round-number parameters "
    "(constant base load, Gaussian morning and evening bumps, a weekend "
    "scale, a cold term from a synthetic temperature curve, Gaussian "
    "noise; see wattwise.synthetic). No measured household data is "
    "read or transformed."
)


def synthetic_spec(household_id: str) -> dict:
    """The declared generation parameters for one synthetic home."""
    try:
        return SYNTHETIC_HOMES[household_id]
    except KeyError:
        raise ValueError(f"unknown synthetic household {household_id!r}") from None


def make_synthetic_intervals(household_id: str) -> pd.DataFrame:
    """Hourly intervals for one synthetic home (index: timestamp, kwh)."""
    spec = synthetic_spec(household_id)
    rng = np.random.default_rng(spec["seed"])
    index = pd.date_range(_START, periods=_DAYS * 24, freq="h")
    day = np.repeat(np.arange(_DAYS), 24)
    hour = np.tile(np.arange(24), _DAYS)
    weekend = np.asarray(index.day_of_week) >= 5

    temperature = (
        44.0
        + 8.0 * np.sin(2 * np.pi * (day - 10) / _DAYS)
        + 4.5 * np.sin(2 * np.pi * (hour - 14) / 24)
        + rng.normal(0.0, 1.2, len(index))
    )

    def bump(center, width):
        return np.exp(-((hour - center) ** 2) / (2 * width ** 2))

    kw = np.full(len(index), spec["base_kw"])
    bumps = (spec["morning_amp_kw"] * bump(spec["morning_center_h"],
                                           spec["morning_width_h"])
             + spec["evening_amp_kw"] * bump(spec["evening_center_h"],
                                             spec["evening_width_h"]))
    kw = kw + np.where(weekend, bumps * spec["weekend_scale"], bumps)
    kw = kw + spec["cold_kw_per_f"] * np.maximum(
        spec["cold_threshold_f"] - temperature, 0.0)
    kw = kw + rng.normal(0.0, spec["noise_sd_kw"], len(index))
    kwh = np.round(np.maximum(kw, 0.02), 4)
    frame = pd.DataFrame({"kwh": kwh}, index=index)
    frame.index.name = "timestamp"
    return frame


def make_synthetic_bills(household_id: str,
                         intervals: pd.DataFrame = None) -> pd.DataFrame:
    """Two billed periods priced from the synthetic intervals themselves.

    Periods are [start, end): the first 31 days and the remaining 31.
    Each bill is the tariff price of the period's intervals plus the
    daily service charge, rounded to cents.
    """
    from .household import tariff_from_spec

    if intervals is None:
        intervals = make_synthetic_intervals(household_id)
    tariff = tariff_from_spec(synthetic_spec(household_id)["tariff"])
    start = intervals.index.min().normalize()
    bounds = [start, start + pd.Timedelta(days=31),
              start + pd.Timedelta(days=_DAYS)]
    rows = []
    for a, b in zip(bounds, bounds[1:]):
        seg = intervals[(intervals.index >= a) & (intervals.index < b)]
        energy = sum(float(v) * tariff.marginal_price(ts)
                     for ts, v in seg["kwh"].items())
        days = (b - a).days
        rows.append({
            "period_start": a, "period_end": b,
            "kwh": round(float(seg["kwh"].sum()), 4),
            "bill_usd": round(energy + tariff.fixed_charge(days), 2),
        })
    return pd.DataFrame(rows)
