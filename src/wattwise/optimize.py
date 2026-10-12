"""Linear-programming load-shifting optimizer.

Flexible appliances (dishwasher, laundry, water heater) don't care *when*
they run inside their allowed windows — but under time-of-day rates, *when*
is money. This solves the scheduling LP:

    minimize  sum over appliances a, hours h of price[h] * x[a,h]
    subject to  sum_h x[a,h] = demand[a]      (do the same work)
                0 <= x[a,h] <= cap[a]        (power limits)
                x[a,h] = 0 outside a's window

Inflexible loads stay exactly where they are. The answer is the cheapest
feasible schedule and the dollar savings versus doing nothing.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from .appliances import BY_NAME


@dataclass
class OptimizationResult:
    hourly_optimized: pd.Series
    cost_before: float
    cost_after: float
    savings: float
    savings_pct: float
    shifted_kwh: float
    schedule: pd.DataFrame  # appliance x hour kWh


def optimize_load_shift(hourly: pd.Series, end_use: pd.DataFrame,
                        price_per_hour: np.ndarray,
                        flexible: dict = None,
                        window: tuple = (0, 24)) -> OptimizationResult:
    """Optimize one day (24 hourly values).

    hourly: 24 hourly kWh (Series, DatetimeIndex not required).
    end_use: DataFrame with per-appliance hourly kWh for the day.
    price_per_hour: 24 marginal $/kWh values.
    flexible: {appliance_name: max_kw}; defaults to all flexible
        appliances at their nameplate power.
    window: global allowed hours for flexible loads, [start, end).
    """
    from .appliances import APPLIANCES
    if flexible is None:
        flexible = {a.name: a.power_kw for a in APPLIANCES if a.flexible}
    names = list(flexible.keys())
    n_app, n_h = len(names), 24
    demand = np.array([end_use[n].sum() for n in names])
    caps = np.array([flexible[n] for n in names])

    allowed = np.zeros((n_app, n_h))
    for i, n in enumerate(names):
        w0, w1 = BY_NAME[n].window
        lo, hi = max(w0, window[0]), min(w1, window[1])
        allowed[i, lo:hi] = 1.0

    # Variables x[a,h] flattened; minimize c^T x.
    c = np.tile(price_per_hour, n_app)
    # Equality: each appliance's energy sums to its demand.
    A_eq = np.zeros((n_app, n_app * n_h))
    for i in range(n_app):
        A_eq[i, i * n_h:(i + 1) * n_h] = 1.0
    # Bounds: 0..cap inside window, 0 outside.
    bounds = []
    for i in range(n_app):
        for h in range(n_h):
            bounds.append((0.0, caps[i] * allowed[i, h]))

    res = linprog(c, A_eq=A_eq, b_eq=demand, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")
    X = res.x.reshape(n_app, n_h)

    inflex = end_use.drop(columns=names, errors="ignore").to_numpy()[:24].sum(axis=1)
    opt_total = inflex + X.sum(axis=0)
    cost_before = float((hourly.to_numpy()[:24] * price_per_hour).sum())
    cost_after = float((opt_total * price_per_hour).sum())
    base_flex = end_use[names].to_numpy()[:24].T
    shifted = float(np.abs(X - base_flex).sum() / 2.0)

    idx = hourly.index[:24] if len(hourly.index) >= 24 else pd.RangeIndex(24)
    schedule = pd.DataFrame(X.T, index=idx, columns=names)
    opt_series = pd.Series(opt_total, index=idx, name="kwh")
    savings = cost_before - cost_after
    return OptimizationResult(
        hourly_optimized=opt_series,
        cost_before=cost_before,
        cost_after=cost_after,
        savings=savings,
        savings_pct=100 * savings / max(cost_before, 1e-9),
        shifted_kwh=shifted,
        schedule=schedule,
    )


def optimize_month(hourly: pd.Series, end_use: pd.DataFrame,
                   rate) -> OptimizationResult:
    """Run the daily optimizer across a month and aggregate."""
    days = pd.unique(hourly.index.date)
    total_before = total_after = shifted = 0.0
    sched_rows = []
    for d in days:
        mask = hourly.index.date == d
        h = hourly.loc[mask]
        if len(h) < 24:
            continue
        eu = end_use.loc[mask]
        prices = np.array([rate.marginal_price(ts) for ts in h.index[:24]])
        r = optimize_load_shift(h, eu, prices)
        total_before += r.cost_before
        total_after += r.cost_after
        shifted += r.shifted_kwh
        sched_rows.append(r.schedule)
    sched = pd.concat(sched_rows) if sched_rows else pd.DataFrame()
    savings = total_before - total_after
    return OptimizationResult(
        hourly_optimized=None,
        cost_before=total_before,
        cost_after=total_after,
        savings=savings,
        savings_pct=100 * savings / max(total_before, 1e-9),
        shifted_kwh=shifted,
        schedule=sched,
    )
