"""Appliance disaggregation (NILM-lite).

Given only the whole-house hourly meter data, estimate how much each
appliance used. Method: per day, each appliance contributes a fixed
characteristic 24-hour shape (learned from the household's own history
when end-use data exists, else a default library shape); a non-negative
least-squares solve finds the daily energy per appliance that best
reconstructs the meter. Honest about uncertainty: it reports the
reconstruction error alongside the split.
"""

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from .appliances import APPLIANCES


def _default_shapes() -> dict:
    """Characteristic 24-h shapes, normalized to sum to 1."""
    h = np.arange(24)
    shapes = {}
    shapes["baseload"] = np.full(24, 1 / 24)
    hv = np.clip(np.sin(2 * np.pi * (h - 9) / 24), 0, None)
    shapes["hvac"] = hv / hv.sum()
    wh = np.zeros(24); wh[6:8] = 0.5; wh[18:20] = 0.5
    shapes["water_heater"] = wh / wh.sum()
    dw = np.zeros(24); dw[20:22] = 1.0
    shapes["dishwasher"] = dw / dw.sum()
    ld = np.zeros(24); ld[19:21] = 1.0
    shapes["laundry"] = ld / ld.sum()
    ck = np.zeros(24); ck[17:19] = 1.0
    shapes["cooking"] = ck / ck.sum()
    li = np.zeros(24); li[6:8] = 0.2; li[17:23] = 0.8
    shapes["lighting_plug"] = li / li.sum()
    return shapes


def learn_shapes(end_use: pd.DataFrame) -> dict:
    """Learn each appliance's average 24-h shape from end-use history."""
    shapes = {}
    hours = end_use.index.hour.to_numpy()
    for app in APPLIANCES:
        col = end_use[app.name].to_numpy()
        shape = np.array([col[hours == hh].mean() for hh in range(24)])
        s = shape.sum()
        shapes[app.name] = shape / s if s > 0 else np.full(24, 1 / 24)
    return shapes


def disaggregate_day(hourly_kwh: np.ndarray, shapes: dict):
    """Split one day (24 hourly kWh) into per-appliance kWh.

    Returns (attribution dict, rmse of reconstruction).
    """
    names = [a.name for a in APPLIANCES]
    A = np.column_stack([shapes[n] for n in names])  # hourly share per kWh
    x, _ = nnls(A, hourly_kwh)
    recon = A @ x
    rmse = float(np.sqrt(np.mean((recon - hourly_kwh) ** 2)))
    return {n: float(v) for n, v in zip(names, x)}, rmse


def disaggregate(hourly: pd.Series, shapes: dict = None,
                 learned_from: pd.DataFrame = None) -> pd.DataFrame:
    """Disaggregate a full hourly series into daily appliance kWh."""
    if shapes is None:
        shapes = learn_shapes(learned_from) if learned_from is not None \
            else _default_shapes()
    days = pd.unique(hourly.index.date)
    rows, rmses = [], []
    for d in days:
        day = hourly.loc[hourly.index.date == d]
        if len(day) < 24:
            continue
        attr, rmse = disaggregate_day(day.to_numpy()[:24], shapes)
        attr["date"] = d
        rows.append(attr)
        rmses.append(rmse)
    out = pd.DataFrame(rows).set_index("date")
    out.attrs["mean_rmse_kwh"] = float(np.mean(rmses)) if rmses else 0.0
    return out
