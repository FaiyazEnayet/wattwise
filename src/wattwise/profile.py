"""Synthetic load + weather profiles for the demo house.

Generates 90 days of hourly load for a Long Island home, ending on a
configurable date, plus matching hourly temperatures. The load is built
from end-use components (baseload, HVAC, water heating, cooking,
laundry, dishwasher, lighting) with weekday/weekend behavior and
seeded randomness, then scaled so the trailing 30-day bill under
Rate 180 (flat, nonsummer) lands near $109.

The end-use intraday windows were sanity-checked against one real
hourly day from the portal (Saturday 2026-10-17,
demo/pseg_hourly_oct17.csv) and adjusted toward its bimodal shape
(midday peak ~1 PM, smaller late-evening peak, late-night tail).

DEMO DATA IS SYNTHETIC. Real analysis uses read_greenbutton_xml on
an actual PSEG export; the importer path is what makes this real-world
usable.
"""

import datetime as dt

import numpy as np
import pandas as pd

from .rates import Rate180, is_summer

# Target: $109.01 bill -> energy part after $0.54*30 fixed charge.
TARGET_BILL = 109.01
FIXED_30D = 30 * 0.54


def make_demo_weather(seed: int = 7, days: int = 90,
                      end: dt.date = dt.date(2026, 10, 11)) -> pd.Series:
    """Hourly temperature (F) for coastal Long Island.

    October shoulder season: mild days around 60F with a diurnal swing,
    occasional warm afternoons and cool snaps, plus noise. Seeded.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(end) - pd.Timedelta(days=days - 1)
    hours = pd.date_range(start, start + pd.Timedelta(days=days) - pd.Timedelta(hours=1), freq="h")
    n = len(hours)
    day_of_year = hours.dayofyear.to_numpy()
    # Seasonal baseline drifts down through October (~66F -> ~56F).
    seasonal = 66.0 - 0.12 * (day_of_year - day_of_year[0])
    # Diurnal swing peaks ~3 PM.
    hour = hours.hour.to_numpy()
    diurnal = 6.5 * np.sin(2 * np.pi * (hour - 9) / 24.0)
    # Slow weather systems: 4-7 day warm/cool spells, ±9F.
    t = np.arange(n)
    systems = 6.0 * np.sin(2 * np.pi * t / (24 * 5.5) + 1.2)
    systems += 3.0 * np.sin(2 * np.pi * t / (24 * 3.1) + 4.0)
    noise = rng.normal(0, 1.2, n)
    temp = seasonal + diurnal + systems + noise
    return pd.Series(temp, index=hours, name="temp_f")


def _appliance_shapes(hours: pd.DatetimeIndex, temp_f: pd.Series,
                      rng: np.random.Generator) -> dict:
    """Hourly kW per end use, before calibration scaling."""
    n = len(hours)
    hour = hours.hour.to_numpy()
    weekday = hours.weekday.to_numpy() < 5

    shapes = {}
    # Baseload: fridge cycles + standby, always on with mild ripple.
    shapes["baseload"] = 0.20 + 0.05 * np.sin(2 * np.pi * np.arange(n) / 3.7) \
        + rng.normal(0, 0.015, n)

    # HVAC: shoulder season. Cooling on warm afternoons, heat on cold
    # nights. Proportional to degree-hours around comfort band.
    cdh = np.clip(temp_f.to_numpy() - 70.0, 0, None)  # cooling degree-hours
    hdh = np.clip(58.0 - temp_f.to_numpy(), 0, None)  # heating degree-hours
    shapes["hvac"] = 0.45 * cdh + 0.18 * hdh + rng.normal(0, 0.02, n)

    # Water heater: 4.5 kW element, morning + evening recovery windows.
    # Morning recovery runs at full duty after showers; the evening
    # recovery is spread over 18-21h at partial duty. Shape check
    # against the one real hourly day available (portal, Sat
    # 2026-10-17): the original twin full-duty blocks stacked with
    # cooking, laundry and the dishwasher to put ~47% of a Saturday
    # into 17-21h, versus ~23% measured.
    wh = np.zeros(n)
    wh[(hour >= 6) & (hour < 8)] = 4.5
    wh_duty = np.where((hour >= 18) & (hour < 21),
                       rng.uniform(0.45, 0.65, n),
                       rng.uniform(0.85, 1.0, n))
    wh[(hour >= 18) & (hour < 21)] = 4.5
    shapes["water_heater"] = wh * wh_duty

    # Cooking: evening peak, bigger on weekends. Weekend lunch runs
    # 12-14h (was 12-13h): the measured Saturday peaks at ~1 PM.
    cook = np.zeros(n)
    cook[(hour >= 17) & (hour < 19)] = 1.6
    cook[(hour >= 12) & (hour < 14) & ~weekday] = 1.2
    shapes["cooking"] = cook * rng.uniform(0.8, 1.1, n)

    # Dishwasher: 1.1 kW after dinner, daily.
    dw = np.zeros(n)
    dw[(hour >= 20) & (hour < 22)] = 1.1
    shapes["dishwasher"] = dw * rng.uniform(0.9, 1.05, n)

    # Laundry: washer+dryer 2.8 kW, ~4 days a week. Weekday loads run
    # in the evening; weekend loads run late morning. The measured
    # Saturday (2026-10-17) puts ~31% of the day in 10 AM-2 PM with
    # its peak at ~1 PM, where the evening-only window left midday at
    # ~20% and stacked laundry on top of every other evening load.
    laundry_days = rng.random(len(pd.unique(hours.date))) < 4 / 7
    day_idx = (hours.normalize() - hours.normalize()[0]).days.to_numpy()
    is_ld = laundry_days[day_idx]
    ld = np.zeros(n)
    ld[(hour >= 19) & (hour < 21) & weekday & is_ld] = 2.8
    ld[(hour >= 10) & (hour < 12) & ~weekday & is_ld] = 2.8
    shapes["laundry"] = ld

    # Lighting + plug loads: evening hump, morning bump. The hump
    # runs to midnight (was 11 PM) and weekend nights spill into the
    # midnight hour: the measured Saturday stays at ~5% of the day
    # through 11 PM and its midnight hour alone is ~5.7%.
    light = np.zeros(n)
    light[(hour >= 6) & (hour < 8)] = 0.25
    light[(hour >= 17) & (hour < 24)] = 0.55
    light[~weekday & (hour >= 9) & (hour < 17)] = 0.30
    light[(hour == 0) & ~weekday] = 0.35
    shapes["lighting_plug"] = light * rng.uniform(0.9, 1.1, n)

    for k in shapes:
        shapes[k] = np.clip(shapes[k], 0, None)
    return shapes


def make_demo_profile(seed: int = 7, days: int = 90,
                      end: dt.date = dt.date(2026, 10, 11)):
    """Return (hourly_kwh Series, end_use DataFrame, weather Series).

    The end-use DataFrame holds per-appliance hourly kWh; the series is
    the household total. Calibrated so the trailing 30-day Rate 180
    (nonsummer) bill is within a few dollars of $109.01.
    """
    temp = make_demo_weather(seed=seed, days=days, end=end)
    hours = temp.index
    rng = np.random.default_rng(seed + 1)
    shapes = _appliance_shapes(hours, temp, rng)
    end_use = pd.DataFrame(shapes, index=hours)
    total = end_use.sum(axis=1)

    # Calibrate: scale total so trailing-30-day flat bill ~= TARGET_BILL.
    rate = Rate180()
    month = total.iloc[-30 * 24:]
    summer = is_summer(month.index[0].date())
    kwh_30 = month.sum()

    def bill_for(scale):
        e = (kwh_30 * scale) * (rate.delivery_rate(kwh_30 * scale, summer)
                                + rate.supply_rate)
        return e + rate.fixed_charge(30)

    # Solve scale by bisection (bill is monotonic in scale).
    lo, hi = 0.2, 3.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if bill_for(mid) < TARGET_BILL:
            lo = mid
        else:
            hi = mid
    scale = (lo + hi) / 2
    end_use = end_use * scale
    total = total * scale
    total.name = "kwh"
    return total, end_use, temp
