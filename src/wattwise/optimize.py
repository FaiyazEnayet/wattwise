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

The LP is an idealized lower bound: it can split an appliance cycle
across non-adjacent hours and stack flexible load without a household
power cap. For schedules that keep noninterruptible appliances
(dishwasher, laundry) in one contiguous block at nameplate power, use
optimize_consecutive_cycles, which adds binary start-time variables and
keeps this LP as the documented lower bound.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, Bounds, linprog, milp

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
    lower_bound_cost: float | None = None  # relaxed LP cost, MILP results only
    start_hours: dict | None = None  # noninterruptible appliance -> start hour


def optimize_load_shift(hourly: pd.Series, end_use: pd.DataFrame,
                        price_per_hour: np.ndarray,
                        flexible: dict = None,
                        window: tuple = (0, 24)) -> OptimizationResult:
    """Optimize one day (24 hourly values).

    This is the relaxed LP and the documented lower bound for the day: it
    can split an appliance cycle across non-adjacent hours and stack
    flexible load without a household power cap. For one contiguous block
    per noninterruptible appliance, use optimize_consecutive_cycles.

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


def optimize_consecutive_cycles(hourly: pd.Series, end_use: pd.DataFrame,
                                price_per_hour: np.ndarray,
                                flexible: dict = None,
                                window: tuple = (0, 24),
                                noninterruptible: set = None,
                                power_cap_kw: float = None
                                ) -> OptimizationResult:
    """Schedule one day with consecutive cycles for noninterruptible loads.

    This ports the dashboard task scheduler construction to the Python
    path: each noninterruptible appliance (dishwasher, laundry by
    default) runs exactly one contiguous block at nameplate power, picked
    by binary start-time variables. Interruptible flexible loads (water
    heater) stay continuous, as in the LP. The LP from
    optimize_load_shift is kept as the documented lower bound and is
    returned on the result as lower_bound_cost.

    Cycles are whole hours: an appliance demand must be an exact multiple
    of its nameplate power (within 1e-9), otherwise ValueError. Zero
    demand appliances are skipped. power_cap_kw, when given, caps total
    household load (inflexible plus flexible) in every hour.
    """
    from .appliances import APPLIANCES
    if flexible is None:
        flexible = {a.name: a.power_kw for a in APPLIANCES if a.flexible}
    if power_cap_kw is not None and power_cap_kw <= 0:
        raise ValueError("power_cap_kw must be positive")
    names = list(flexible.keys())
    n_h = 24
    demand = np.array([end_use[n].sum() for n in names])
    caps = np.array([flexible[n] for n in names])
    if noninterruptible is None:
        ni_names = {a.name for a in APPLIANCES
                    if a.flexible and a.noninterruptible} & set(names)
    else:
        ni_names = set(noninterruptible)
        unknown = ni_names - set(names)
        if unknown:
            raise ValueError(
                f"noninterruptible appliances not in flexible: {sorted(unknown)}")

    allowed = {}
    for n in names:
        w0, w1 = BY_NAME[n].window
        allowed[n] = (max(w0, window[0]), min(w1, window[1]))

    # Cycle lengths in whole hours, and feasible start hours per appliance.
    cycles = {}
    starts = {}
    for n in ni_names:
        i = names.index(n)
        if demand[i] <= 1e-12:
            cycles[n] = 0
            starts[n] = []
            continue
        length = demand[i] / caps[i]
        if abs(length - round(length)) > 1e-9 or round(length) < 1:
            raise ValueError(
                f"{n} demand {demand[i]:.4f} kWh at {caps[i]} kW is not a "
                f"whole-hour cycle (length {length:.4f} h)")
        cycles[n] = int(round(length))
        lo, hi = allowed[n]
        feasible = [h for h in range(lo, hi - cycles[n] + 1)]
        if not feasible:
            raise ValueError(
                f"{n} has no feasible start for a {cycles[n]} h cycle "
                f"inside window [{lo}, {hi})")
        starts[n] = feasible

    # Variable layout: [start binaries s] [run binaries y] [continuous x].
    ni_active = [n for n in ni_names if cycles[n] > 0]
    int_names = [n for n in names if n not in ni_names]
    s_index, y_index, x_index = {}, {}, {}
    pos = 0
    for n in ni_active:
        s_index[n] = (pos, len(starts[n]))
        pos += len(starts[n])
    for n in ni_active:
        y_index[n] = (pos, n_h)
        pos += n_h
    for n in int_names:
        x_index[n] = (pos, n_h)
        pos += n_h
    n_var = pos

    c = np.zeros(n_var)
    for n in ni_active:
        y0, _ = y_index[n]
        i = names.index(n)
        c[y0:y0 + n_h] = price_per_hour * caps[i]
    for n in int_names:
        x0, _ = x_index[n]
        c[x0:x0 + n_h] = price_per_hour

    integrality = np.zeros(n_var)
    for n in ni_active:
        s0, m = s_index[n]
        y0, _ = y_index[n]
        integrality[s0:s0 + m] = 1
        integrality[y0:y0 + n_h] = 1

    lb = np.zeros(n_var)
    ub = np.ones(n_var)
    for n in int_names:
        x0, _ = x_index[n]
        i = names.index(n)
        lo, hi = allowed[n]
        ub[x0:x0 + n_h] = 0.0
        ub[x0 + lo:x0 + hi] = caps[i]

    rows, lbs, ubs = [], [], []
    # Exactly one start per active noninterruptible appliance.
    for n in ni_active:
        s0, m = s_index[n]
        row = np.zeros(n_var)
        row[s0:s0 + m] = 1.0
        rows.append(row)
        lbs.append(1.0)
        ubs.append(1.0)
    # Run hours follow from the chosen start: y covers the L hours after it.
    for n in ni_active:
        s0, _ = s_index[n]
        y0, _ = y_index[n]
        for h in range(n_h):
            row = np.zeros(n_var)
            row[y0 + h] = 1.0
            for k, start in enumerate(starts[n]):
                if start <= h < start + cycles[n]:
                    row[s0 + k] -= 1.0
            rows.append(row)
            lbs.append(0.0)
            ubs.append(0.0)
    # Interruptible appliances deliver their full demand.
    for n in int_names:
        x0, _ = x_index[n]
        i = names.index(n)
        row = np.zeros(n_var)
        row[x0:x0 + n_h] = 1.0
        rows.append(row)
        lbs.append(demand[i])
        ubs.append(demand[i])
    # Optional household power cap on total load per hour.
    inflex = end_use.drop(columns=names, errors="ignore").to_numpy()[:24].sum(axis=1)
    if power_cap_kw is not None:
        for h in range(n_h):
            row = np.zeros(n_var)
            for n in ni_active:
                y0, _ = y_index[n]
                i = names.index(n)
                row[y0 + h] = caps[i]
            for n in int_names:
                x0, _ = x_index[n]
                row[x0 + h] = 1.0
            rows.append(row)
            lbs.append(-np.inf)
            ubs.append(power_cap_kw - inflex[h])

    res = milp(c, constraints=LinearConstraint(np.array(rows), lbs, ubs),
               integrality=integrality, bounds=Bounds(lb, ub))
    if not res.success:
        raise RuntimeError(f"MILP failed: {res.message}")
    x = res.x

    X = np.zeros((len(names), n_h))
    start_hours = {}
    for n in ni_active:
        y0, _ = y_index[n]
        i = names.index(n)
        X[i] = caps[i] * np.round(x[y0:y0 + n_h])
        s0, _ = s_index[n]
        chosen = int(np.argmax(np.round(x[s0:s0 + len(starts[n])])))
        start_hours[n] = starts[n][chosen]
    for n in ni_names:
        if n not in start_hours:
            start_hours[n] = None
    for n in int_names:
        x0, _ = x_index[n]
        i = names.index(n)
        X[i] = x[x0:x0 + n_h]

    opt_total = inflex + X.sum(axis=0)
    cost_before = float((hourly.to_numpy()[:24] * price_per_hour).sum())
    cost_after = float((opt_total * price_per_hour).sum())
    base_flex = end_use[names].to_numpy()[:24].T
    shifted = float(np.abs(X - base_flex).sum() / 2.0)

    idx = hourly.index[:24] if len(hourly.index) >= 24 else pd.RangeIndex(24)
    schedule = pd.DataFrame(X.T, index=idx, columns=names)
    opt_series = pd.Series(opt_total, index=idx, name="kwh")
    savings = cost_before - cost_after
    lp = optimize_load_shift(hourly, end_use, price_per_hour,
                             flexible=flexible, window=window)
    return OptimizationResult(
        hourly_optimized=opt_series,
        cost_before=cost_before,
        cost_after=cost_after,
        savings=savings,
        savings_pct=100 * savings / max(cost_before, 1e-9),
        shifted_kwh=shifted,
        schedule=schedule,
        lower_bound_cost=lp.cost_after,
        start_hours=start_hours,
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
