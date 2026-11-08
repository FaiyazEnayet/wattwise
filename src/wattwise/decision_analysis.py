"""Auditable household decision analysis on original meter intervals.

Money amounts are USD, energy is kWh per interval, and power is kW.
Observed bills, retrospective accounting, and counterfactuals remain separate.
"""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from pandas.tseries.holiday import USFederalHolidayCalendar
from scipy.optimize import lsq_linear

RATE_SOURCE = 'https://www.psegliny.com/aboutpseglongisland/ratesandtariffs/-/media/A0FDA80A6FE44A45973922422E86BD9E.ashx'
DEFAULT_SUPPLY = 0.139822  # September 2026 published base PSC, scenario input


@dataclass(frozen=True)
class TariffSchedule:
    """One archived, effective-dated version of the delivery tariff.

    `effective_to=None` means the version is still in force. The price
    formulas in `tariff_replay` currently implement the single verified
    schedule below; a successor version is added here (never by editing
    this entry) when a new schedule is published and verified.
    """

    version: str
    effective_from: dt.date
    effective_to: dt.date | None
    plans: tuple
    source: str


TARIFF_SCHEDULES = (
    TariffSchedule(
        version="2026.1",
        effective_from=dt.date(2026, 1, 1),
        effective_to=None,
        plans=("180", "194", "195"),
        source=RATE_SOURCE,
    ),
)


def schedule_in_force(on: dt.date, schedules=TARIFF_SCHEDULES) -> TariffSchedule:
    """The archived schedule in force on `on`; LookupError if none."""
    for s in schedules:
        if s.effective_from <= on and (s.effective_to is None or on <= s.effective_to):
            return s
    raise LookupError(f"no tariff schedule in force on {on}")


def monthly_supply_history(details: pd.DataFrame) -> dict:
    """Blended supply price ($/kWh) per bill month, from actual bills.

    Maps each bill's period-start month 'YYYY-MM' to supply_usd / kWh
    for that bill: the household's realized supply price history, for
    replay scenarios that should not use one declared scalar.
    """
    out = {}
    for _, row in details.iterrows():
        start = pd.Timestamp(row.period_start)
        out[f"{start:%Y-%m}"] = float(row.supply_usd) / float(row.kwh)
    return out


def peak_mask(index):
    """Apply the published weekday window using every represented year."""
    index = pd.DatetimeIndex(index)
    holidays = USFederalHolidayCalendar().holidays(index.min().normalize(), index.max().normalize())
    return ((index.weekday < 5) & (index.hour >= 15) & (index.hour < 19)
            & ~index.normalize().isin(holidays))


def read_intervals(path):
    frame = pd.read_csv(path)
    required = {'Start', 'kWh', 'kW'}
    if not required.issubset(frame):
        raise ValueError('Interval CSV requires Start, kWh and kW columns')
    frame['timestamp'] = pd.to_datetime(frame['Start'], format='%m/%d/%Y %I:%M:%S %p', errors='raise')
    frame = frame.rename(columns={'kWh': 'kwh', 'kW': 'kw'}).set_index('timestamp').sort_index()
    frame = frame[['kwh', 'kw']].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(frame.to_numpy()).all() or (frame < 0).any().any():
        raise ValueError('Nonfinite or negative readings require explicit export/import semantics')
    if frame.index.has_duplicates:
        raise ValueError('Duplicate wall timestamps require meter timezone or fold metadata; never sum blindly')
    return frame


def interval_quality(iv):
    """Retain raw wall times and flag ambiguity rather than invent UTC offsets."""
    daily = iv.kwh.resample('D').agg(['sum', 'count'])
    expected = []
    reasons = []
    for day, row in daily.iterrows():
        a = day.tz_localize('America/New_York')
        b = (day + pd.Timedelta(days=1)).tz_localize('America/New_York')
        n = int((b.tz_convert('UTC') - a.tz_convert('UTC')).total_seconds() / 900)
        expected.append(n)
        if day == iv.index[-1].normalize() and iv.index[-1].hour < 23:
            reasons.append('partial export day')
        elif n != 96:
            reasons.append('DST wall-clock convention unresolved')
        elif row['count'] != n:
            reasons.append('interval coverage gap')
        else:
            reasons.append('complete')
    daily['expected'] = expected
    daily['quality'] = reasons
    daily['usable'] = daily.quality.eq('complete')
    grid = pd.date_range(iv.index.min(), iv.index.max(), freq='15min')
    missing = grid.difference(iv.index)
    return daily, {
        'rows': len(iv), 'total_kwh': float(iv.kwh.sum()),
        'first': str(iv.index.min()), 'last': str(iv.index.max()),
        'usable_days': int(daily.usable.sum()), 'days': len(daily),
        'missing_wall_slots': [str(t) for t in missing],
        'flagged_days': [{'date': str(d.date()), 'count': int(r['count']),
                          'expected': int(r.expected), 'reason': r.quality}
                         for d, r in daily[~daily.usable].iterrows()],
        'timezone': 'America/New_York assumed from utility service area; CSV has no UTC offset',
        'max_kw': float(iv.kw.max()),
        'kw_rounding_max': float((iv.kw - 4 * iv.kwh).abs().max()),
    }


def reconcile_bills(iv, details):
    """Bill boundaries are [start, end), matching portal day counts."""
    records = []
    for _, row in details.iterrows():
        a, b = pd.Timestamp(row.period_start), pd.Timestamp(row.period_end)
        seg = iv[(iv.index >= a) & (iv.index < b)]
        full = iv.index.min() <= a and iv.index.max() + pd.Timedelta(minutes=15) >= b
        peak = float(seg.loc[peak_mask(seg.index), 'kwh'].sum()) if len(seg) else 0.0
        kwh = float(seg.kwh.sum())
        pp = float(row.del_peak_rate + row.sup_peak_rate)
        op = float(row.del_offpeak_rate + row.sup_offpeak_rate)
        energy = row.peak_kwh * pp + row.offpeak_kwh * op
        reconstructed = energy + row.basic_usd + row.mfc_usd + row.der_usd + row.other_taxes_usd
        r = {key: (float(value) if isinstance(value, (int, float, np.number)) else str(value))
             for key, value in row.items()}
        r.update(raw_kwh=kwh, raw_peak_kwh=peak, complete=bool(full),
                 difference_pct=100 * (kwh / row.kwh - 1),
                 peak_price=pp, offpeak_price=op,
                 energy_charge=float(energy), reconstructed=float(reconstructed),
                 reconstruction_error=float(reconstructed - row.bill_usd),
                 shift_ceiling=float(peak * max(pp - op, 0)),
                 observed_peak_share=float(row.peak_kwh / row.kwh))
        records.append(r)
    return records


def tariff_replay(iv, plan='194', supply=DEFAULT_SUPPLY, schedule_version=None,
                  supply_history=None):
    """2026 delivery schedule plus one declared PSC for the same historical load.

    This standardized scenario excludes taxes and adjustments and is not a
    reconstruction of historical alternative bills. Summer 180 blocks reset
    monthly in this scenario; actual billing-day proration needs the tariff.

    WW4 versioning (additive; defaults keep the declared scenario above):
    `schedule_version` pins an archived TariffSchedule and rejects any
    interval outside its effective range, instead of silently pricing
    other years with this schedule. `supply_history` (from
    `monthly_supply_history`) prices each interval with its own month's
    realized supply price and rejects months the history does not cover.
    """
    if not np.isfinite(supply) or supply < 0:
        raise ValueError('Supply rate must be finite and nonnegative')
    if schedule_version is not None:
        match = [s for s in TARIFF_SCHEDULES if s.version == schedule_version]
        if not match:
            raise ValueError(f'Unknown tariff schedule version {schedule_version}')
        sched = match[0]
        days = iv.index.date
        first, last = days.min(), days.max()
        if first < sched.effective_from or (
            sched.effective_to is not None and last > sched.effective_to
        ):
            raise ValueError(
                f'intervals {first}..{last} fall outside tariff schedule '
                f'{sched.version} ({sched.effective_from}..{sched.effective_to})'
            )
    if supply_history is not None:
        keys = [f'{ts:%Y-%m}' for ts in iv.index]
        missing = sorted({k for k in keys if k not in supply_history})
        if missing:
            raise ValueError(f'no supply history for month(s) {missing}')
        supply = np.array([supply_history[k] for k in keys])
    summer = iv.index.month.isin([6, 7, 8, 9])
    peak = peak_mask(iv.index)
    if plan == '194':
        price = np.where(peak, np.where(summer, .2217 + 1.9505 * supply,
                                                      .1885 + 2.134 * supply),
                                      np.where(summer, .1093, .0929) + .83 * supply)
    elif plan == '195':
        night = (iv.index.hour >= 22) | (iv.index.hour < 6)
        price = np.where(night, np.where(summer, .0452, .045) + .60 * supply,
                        np.where(peak, np.where(summer, .2979 + 1.7419 * supply,
                                                          .2440 + 1.9688 * supply),
                                       np.where(summer, .1388, .0929) + supply))
    elif plan == '180':
        # Split the threshold-crossing interval, not a marginal rate on all kWh.
        groups = iv.kwh.groupby(iv.index.to_period('M'))
        before = groups.cumsum() - iv.kwh
        low = np.minimum(iv.kwh, np.maximum(250 - before, 0))
        high = iv.kwh - low
        delivery = np.where(summer, low * .1064 + high * .1348, iv.kwh * .1064)
        price = delivery / np.maximum(iv.kwh.to_numpy(), 1e-12) + supply
    else:
        raise ValueError('Unknown tariff plan')
    energy = iv.kwh.to_numpy() * price
    costs = pd.Series(energy, index=iv.index)
    return costs, np.asarray(price)


def shapley_bill_change(old, new):
    """Exact symmetric attribution over usage, timing, rates and fixed items."""
    def value(state):
        n, timing, rates, fixed = state
        q = (new if n else old)['kwh']
        source = new if timing else old
        p = source['peak_kwh'] / source['kwh']
        source = new if rates else old
        pp, op = source['peak_price'], source['offpeak_price']
        source = new if fixed else old
        return q * (p * pp + (1 - p) * op) + source['basic_usd'] + source['mfc_usd'] + source['der_usd'] + source['other_taxes_usd']
    contributions = np.zeros(4)
    for order in permutations(range(4)):
        state = [0] * 4
        previous = value(state)
        for feature in order:
            state[feature] = 1
            current = value(state)
            contributions[feature] += (current - previous) / 24
            previous = current
    return dict(zip(['Usage volume', 'Peak timing', 'Energy rates', 'Service and other items'], map(float, contributions)))


def _weather_matrix(index, temp):
    return np.column_stack([np.ones(len(index)), *[(index.weekday == d).astype(float) for d in range(1, 7)],
                            np.maximum(np.asarray(temp) - 65, 0)])


def weather_analysis(daily, weather_path):
    if not Path(weather_path).exists():
        return {'available': False, 'reason': 'No matched weather file; upload or fetch weather to enable this analysis'}
    w = pd.read_csv(weather_path, parse_dates=['timestamp']).set_index('timestamp')
    # Require at least 20 observed/reanalysis hours; missing weather is never 0 F.
    wd = w.temperature_f.resample('D').agg(['mean', 'count'])
    joined = daily[daily.usable][['sum']].join(wd.rename(columns={'mean': 'temp_f', 'count': 'weather_hours'})).dropna()
    joined = joined[joined.weather_hours >= 20]
    if len(joined) < 90:
        return {'available': False, 'reason': 'Fewer than 90 complete days with weather'}
    X = _weather_matrix(joined.index, joined.temp_f)
    y = joined['sum'].to_numpy()
    lower = np.full(X.shape[1], -np.inf); lower[-1] = 0
    fit = lsq_linear(X, y, bounds=(lower, np.full(X.shape[1], np.inf)))
    prediction = X @ fit.x
    # Conditional holdout uses realized weather, not a forecast available in advance.
    split = len(y) - 56
    train = lsq_linear(X[:split], y[:split], bounds=(lower, np.full(X.shape[1], np.inf)))
    mae = float(np.mean(np.abs(y[split:] - X[split:] @ train.x)))
    baseline = np.array([np.median(y[:split][joined.index[:split].weekday == d]) for d in joined.index[split:].weekday])
    baseline_mae = float(np.mean(np.abs(y[split:] - baseline)))
    rng = np.random.default_rng(11777)
    slopes = []
    # Moving blocks preserve within-week dependence; this is one household.
    for _ in range(200):
        starts = rng.integers(0, len(y) - 6, int(np.ceil(len(y) / 7)))
        take = np.concatenate([np.arange(a, a + 7) for a in starts])[:len(y)]
        slopes.append(lsq_linear(X[take], y[take], bounds=(lower, np.full(X.shape[1], np.inf))).x[-1])
    base = prediction - fit.x[-1] * np.maximum(joined.temp_f.to_numpy() - 65, 0)
    residual = y - prediction
    scale = max(float(np.median(np.abs(residual - np.median(residual))) * 1.4826), .5)
    alerts = joined.assign(expected=prediction, residual=residual, score=residual / scale)
    alerts = alerts[alerts.score > 3].sort_values('residual', ascending=False).head(10)
    return {'available': True, 'days': len(y), 'cooling_slope': float(fit.x[-1]),
            'slope_interval_80': list(map(float, np.quantile(slopes, [.1, .9]))),
            'conditional_holdout_mae': mae, 'baseline_holdout_mae': baseline_mae,
            'r2_descriptive': float(1 - np.sum(residual**2) / np.sum((y-y.mean())**2)),
            'association_kwh': float(np.sum(prediction - base)),
            'dates': [str(d.date()) for d in joined.index], 'kwh': list(map(float, y)),
            'temperature_f': list(map(float, joined.temp_f)), 'expected': list(map(float, prediction)),
            'residual': list(map(float, residual)),
            'alerts': [{'date': str(d.date()), 'actual': float(r['sum']),
                        'expected': float(r.expected), 'excess_kwh': float(r.residual)} for d,r in alerts.iterrows()],
            'note': 'Cooling association above 65 F with weekday controls; not appliance metering or a causal AC saving. Weather is ERA5 reanalysis for the Long Island area; modeled weather, not a home sensor.'}


def predict_daily(history, dates, method):
    """Three small-data baselines; recursively predict without future readings."""
    history = history.copy().astype(float)
    if method not in {'weekly_naive', 'weekday_median', 'ridge_ar'}:
        raise ValueError('Unknown forecasting model')
    coefficients = None
    def features(s, day):
        lag = lambda n: float(s.get(day-pd.Timedelta(days=n), s.tail(28).median()))
        past = s[s.index < day].tail(7)
        return [1., lag(7), lag(14), lag(28), float(past.mean()),
                *[(day.weekday() == d) * 1. for d in range(1,7)]]
    if method == 'ridge_ar':
        rows, target = [], []
        for day in history.index[28:]:
            rows.append(features(history.loc[:day-pd.Timedelta(days=1)], day))
            target.append(history.loc[day])
        X = np.asarray(rows); target=np.asarray(target)
        penalty = np.eye(X.shape[1]) * 10; penalty[0,0] = .001
        coefficients = np.linalg.solve(X.T@X + penalty, X.T@target)
    result=[]
    for day in pd.DatetimeIndex(dates):
        past = history[history.index < day]
        if method == 'weekly_naive':
            value = past.get(day-pd.Timedelta(days=7), past.tail(28).median())
        elif method == 'weekday_median':
            recent = past.tail(56)
            value = recent[recent.index.weekday == day.weekday()].median()
            if pd.isna(value): value=past.tail(28).median()
        else:
            value = np.asarray(features(past, day)) @ coefficients
        value=max(float(value),0)
        history.loc[day]=value; result.append(value)
    return np.asarray(result)


def forecast_analysis(daily):
    # Excluded days stay absent; lags only use available past observations.
    history = daily.loc[daily.usable, 'sum']
    if len(history) < 180:
        return {'available':False, 'reason':'At least 180 complete days are required'}
    methods=['weekly_naive','weekday_median','ridge_ar']
    origins = history.index[-84:]
    predictions={m:[] for m in methods}
    prediction_paths={m:[] for m in methods}
    for day in origins:
        train=history[history.index < day].tail(240)
        for m in methods:
            path=predict_daily(train,pd.date_range(day,periods=28,freq='D'),m)
            prediction_paths[m].append(path)
            predictions[m].append(float(path[0]))
    actual=history.reindex(origins).to_numpy()
    # First 56 origins choose the model; final 28 remain untouched for reporting.
    chosen=min(methods,key=lambda m:np.mean(np.abs(np.array(predictions[m][:56])-actual[:56])))
    results=[]
    calibration=actual[:56]-np.array(predictions[chosen][:56])
    low,high=np.quantile(calibration,[.1,.9])
    for m in methods:
        errors=actual[-28:]-np.array(predictions[m][-28:])
        results.append({'model':m,'selection_mae':float(np.mean(np.abs(np.array(predictions[m][:56])-actual[:56]))),
                        'holdout_mae':float(np.mean(np.abs(errors))), 'holdout_bias':float(np.mean(-errors))})
    held=np.array(predictions[chosen][-28:])
    coverage=float(np.mean((actual[-28:]>=held+low)&(actual[-28:]<=held+high)))
    horizon_rows=[]
    cumulative_intervals={}
    for horizon in [7,14,28]:
        for method in methods:
            calibration_errors=[]; held_errors=[]
            for i,day in enumerate(origins):
                target_dates=pd.date_range(day,periods=horizon,freq='D')
                observed=history.reindex(target_dates)
                if observed.isna().any(): continue
                error=float(observed.sum()-prediction_paths[method][i][:horizon].sum())
                # No calibration target may extend into the final holdout.
                if i<56 and target_dates[-1]<=origins[55]:calibration_errors.append(error)
                elif i>=56:held_errors.append(error)
            # Same convention as wattwise.tournament._horizon_rows: a
            # horizon with no complete calibration window reports its
            # coverage as null, and one with no complete holdout window
            # (e.g. every 28-day window crossing an excluded DST day)
            # reports its holdout metrics as null, instead of failing
            # on an empty mean or quantile.
            bounds=(np.quantile(calibration_errors,[.1,.9])
                    if calibration_errors else None)
            if bounds is not None and held_errors:
                empirical=float(np.mean((np.array(held_errors)>=bounds[0]) & (np.array(held_errors)<=bounds[1])))
            else:
                empirical=None
            if held_errors:
                mae=float(np.mean(np.abs(held_errors)))
                bias=float(-np.mean(held_errors))
            else:
                mae=None; bias=None
            horizon_rows.append({'model':method,'horizon':horizon,'holdout_origins':len(held_errors),
                                 'calibration_origins':len(calibration_errors),
                                 'mae_kwh':mae,
                                 'bias_kwh':bias,
                                 'coverage_80':empirical})
            if method==chosen and bounds is not None:cumulative_intervals[str(horizon)]=list(map(float,bounds))
    future=pd.date_range(daily.index[-1]+pd.Timedelta(days=1),periods=35,freq='D')
    # If the final export day is partial, forecast it separately for the open bill.
    prediction=predict_daily(history,future,chosen)
    rng=np.random.default_rng(7)
    paths=[]
    for _ in range(600):
        starts=rng.integers(0,len(calibration)-6,5)
        errors=np.concatenate([calibration[a:a+7] for a in starts])
        paths.append(np.maximum(prediction+errors,0))
    paths=np.asarray(paths)
    return {'available':True,'chosen':chosen,'models':results,'holdout_days':28,
            'selection_days':56,'interval_coverage_80':coverage,
            'holdout_dates':[str(d.date()) for d in origins[-28:]],
            'holdout_actual':list(map(float,actual[-28:])),
            'holdout_predicted':list(map(float,held)),
            'dates':[str(d.date()) for d in future], 'prediction':list(map(float,prediction)),
            'lo':list(map(float,np.quantile(paths,.1,axis=0))),
            'hi':list(map(float,np.quantile(paths,.9,axis=0))),
            'seven_day_total':float(prediction[:7].sum()),
            'seven_day_interval_80':[max(0.,float(prediction[:7].sum()+v)) for v in cumulative_intervals['7']],
            'horizon_results':horizon_rows,
            'cumulative_error_bounds':cumulative_intervals,
            'note':'Chronological selection and untouched final 28-day evaluation. Future multiday predictions are recursive. Daily bands resample seven-day residual blocks; seven-day total bounds use chronological cumulative errors whose calibration targets end before the holdout. Origins overlap and are not independent. The 28-day holdout admits at most one complete 28-day forecast per model, and a horizon with no complete holdout window reports null metrics. Partial final day is excluded.'}


def load_shapes(iv, daily):
    usable=set(daily.index[daily.usable].date)
    clean=iv[np.isin(iv.index.date,list(usable))]
    hours=clean.kwh.resample('h').sum()
    matrix=hours.groupby([hours.index.date,hours.index.hour]).sum().unstack().dropna()
    # Reindex excludes gap days which resample would otherwise insert as zero.
    matrix=matrix[np.isin(matrix.index,list(usable))]
    normalized=matrix.div(matrix.sum(axis=1),axis=0).to_numpy()
    rng=np.random.default_rng(3)
    centers=normalized[rng.choice(len(normalized),3,replace=False)].copy()
    for _ in range(40):
        labels=((normalized[:,None,:]-centers[None,:,:])**2).sum(axis=2).argmin(axis=1)
        new=np.array([normalized[labels==k].mean(axis=0) if np.any(labels==k) else centers[k] for k in range(3)])
        if np.allclose(new,centers): break
        centers=new
    return {'hours':list(range(24)), 'centers':(centers*100).tolist(),
            'counts':[int((labels==k).sum()) for k in range(3)],
            'dates':[str(d) for d in matrix.index],'labels':labels.tolist(),
            'note':'Three descriptive k-means groups on normalized complete-day shapes. Cluster labels do not identify occupancy or appliances.'}


def open_bill_forecast(iv, daily, forecast, billing_history, supply=DEFAULT_SUPPLY):
    """Freeze the existing open-bill planning calculation for both UI views.

    Period boundaries come from the declared unbilled history row. The band
    sums daily bounds and is a conservative envelope, not a calibrated
    monthly interval. Service accrues by fractional local calendar day.
    """
    as_of = iv.index[-1] + pd.Timedelta(minutes=15)
    candidates = billing_history[
        billing_history.bill_usd.isna()
        & (billing_history.period_start <= iv.index[-1])
        & (billing_history.period_end > iv.index[-1])]
    if len(candidates) != 1 or not forecast.get('available'):
        return {'available': False, 'reason': 'No unique declared open billing period or daily forecast is available.'}
    row = candidates.iloc[0]
    start, end = pd.Timestamp(row.period_start), pd.Timestamp(row.period_end)
    observed = iv[(iv.index >= start) & (iv.index < end)]
    last_day = iv.index[-1].normalize()
    future_dates = pd.date_range(last_day + pd.Timedelta(days=1), end - pd.Timedelta(days=1))
    positions = {pd.Timestamp(d): i for i, d in enumerate(forecast['dates'])}
    if any(d not in positions for d in future_dates):
        return {'available': False, 'reason': 'The stored daily forecast does not cover the declared open billing period.'}
    peers = daily[(daily.index < last_day) & daily.usable & (daily.index.weekday == last_day.weekday())].tail(8)
    if daily.loc[last_day, 'quality'] != 'complete' and peers.empty:
        return {'available': False, 'reason': 'No complete comparable weekdays are available to finish the partial export day.'}
    peer_values = sorted(peers['sum'].tolist())
    partial = (max(0.0, peer_values[len(peer_values) // 2] - float(daily.loc[last_day, 'sum']))
               if daily.loc[last_day, 'quality'] != 'complete' else 0.0)
    indices = [positions[d] for d in future_dates]
    sofar = float(observed.kwh.sum())
    totals = [sofar + partial + sum(forecast[key][i] for i in indices)
              for key in ['prediction', 'lo', 'hi']]
    energy_cost, _ = tariff_replay(observed, '194', supply)
    if sofar <= 0:
        return {'available': False, 'reason': 'Positive observed energy is required for the declared price-mix assumption.'}
    average_price = float(energy_cost.sum()) / sofar
    days = (end - start).total_seconds() / 86400
    elapsed = (as_of - start).total_seconds() / 86400
    remaining = days - elapsed
    service = days * .56
    projected, lower, upper = [value * average_price + service for value in totals]
    amount_sofar = float(energy_cost.sum()) + elapsed * .56
    models = {m['model']: m for m in forecast['models']}
    money = lambda value: f'${value:,.2f}'
    return {
        'available': True, 'period_start': str(start.date()), 'period_end_exclusive': str(end.date()),
        'period_source': 'demo/pseg_bills.csv unbilled period', 'as_of': str(as_of),
        'observed_kwh': sofar, 'modeled_subtotal_so_far_usd': amount_sofar,
        'days_in_period': days, 'days_elapsed': elapsed, 'days_remaining': remaining,
        'partial_day_completion_kwh': partial, 'projected_kwh': totals[0],
        'projected_subtotal_usd': projected, 'envelope_kwh': totals[1:],
        'envelope_usd': [lower, upper], 'band_type': 'conservative_sum_of_daily_bands',
        'tariff_plan': '194', 'tariff_version': TARIFF_SCHEDULES[0].version, 'base_supply_usd_per_kwh': supply,
        'average_energy_price_usd_per_kwh': average_price, 'service_usd_per_day': .56,
        'weekly_holdout_mae': models['weekly_naive']['holdout_mae'],
        'autoregression_holdout_mae': models['ridge_ar']['holdout_mae'],
        'display': {
            'projected_subtotal_usd': money(projected), 'projected_kwh': f'{totals[0]:,.1f}',
            'observed_kwh': f'{sofar:,.2f}', 'modeled_subtotal_so_far_usd': money(amount_sofar),
            'days_in_period': f'{days:g}', 'days_elapsed': f'{elapsed:.1f}', 'days_remaining': f'{remaining:.1f}',
            'envelope_usd': f'{money(lower)}–{money(upper)}',
            'envelope_kwh': f'{totals[1]:,.1f}–{totals[2]:,.1f}',
            'weekly_holdout_mae': f'{models["weekly_naive"]["holdout_mae"]:.3f}',
            'autoregression_holdout_mae': f'{models["ridge_ar"]["holdout_mae"]:.3f}',
        },
    }


# ---------------------------------------------------------------------------
# Reproducible evidence manifest (P1)
#
# The frozen report carries a manifest fingerprinting every input it was
# built from, the pricing assumptions and the analysis code itself. The
# manifest digest is the reuse gate: scripts/make_dashboard.py refuses
# --reuse-analysis when the stored digest no longer matches the current
# inputs and code, and scripts/verify_frozen_outputs.py rebuilds the
# report from a clean copy of the worktree and compares canonical JSON.
# Only repository-relative paths are recorded; no absolute paths, no
# coordinates, nothing private.

MANIFEST_SCHEMA = 'wattwise-report-manifest/1'

# (role, path under demo/, parser schema, timestamp convention)
REPORT_INPUTS = (
    ('intervals', 'pseg_interval_15min.csv',
     'pseg-interval-csv: Start,kWh,kW with %m/%d/%Y %I:%M:%S %p timestamps',
     'naive wall time, America/New_York assumed; CSV carries no UTC offset'),
    ('bill_details', 'pseg_rates_detailed.csv',
     'pseg-bill-details-csv: one row per billed period with line-item rates',
     'period_start/period_end are local calendar dates; boundaries [start, end)'),
    ('billing_history', 'pseg_bills.csv',
     'pseg-bills-csv: period_start,period_end,bill_usd; an unbilled row has bill_usd empty',
     'period_start/period_end are local calendar dates; boundaries [start, end)'),
    ('appliance_estimates', 'pseg_appliance_estimates.csv',
     'pseg-appliance-estimates-csv: category-level utility estimates',
     'monthly aggregates; no interval timestamps'),
    ('weather', 'household_weather.csv',
     'open-meteo-era5-hourly-csv: timestamp,temperature_f',
     'hourly local time, America/New_York (provider metadata: demo/household_weather.metadata.json)'),
)

# Changing any of these files changes the analysis rules, so the report's
# provenance fingerprint must change with them.
CODE_PATHS = ('src/wattwise/decision_analysis.py',)


class ManifestMismatch(RuntimeError):
    """The stored report does not match the current inputs or code."""


def _sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def code_fingerprint(paths=None):
    """SHA-256 over the analysis source files, in a fixed order."""
    if paths is None:
        paths = [Path(__file__).resolve().parents[2] / rel for rel in CODE_PATHS]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def _csv_data_rows(path):
    """Data-row count of a CSV with a header row (explicit row counts)."""
    with open(path, 'rb') as fh:
        return max(sum(1 for _ in fh) - 1, 0)


def _git_head(root):
    try:
        out = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def build_manifest(root):
    """Fingerprint every report input, the pricing assumptions and the code.

    The returned dict includes its own `manifest_digest`. The git HEAD
    and the library environment are recorded as build annotations only
    and are excluded from the digest: the digest identifies what the
    report was built FROM (inputs, pricing, code), not the machine
    state it was built ON. The commit that records the frozen outputs
    always differs from the commit they were generated at, and library
    patch versions drift between machines; either one would make the
    digest useless as a staleness check.
    """
    root = Path(root)
    entries = []
    for role, name, schema, timestamps in REPORT_INPUTS:
        path = root / 'demo' / name
        if not path.exists():
            entries.append({'role': role, 'path': f'demo/{name}', 'present': False})
            continue
        raw = path.read_bytes()
        entries.append({
            'role': role, 'path': f'demo/{name}', 'present': True,
            'sha256': _sha256_bytes(raw), 'bytes': len(raw),
            'rows': _csv_data_rows(path), 'parser_schema': schema,
            'timestamp_convention': timestamps,
        })
    manifest = {
        'schema': MANIFEST_SCHEMA,
        'inputs': entries,
        'pricing': {
            'supply_usd_per_kwh': DEFAULT_SUPPLY,
            'tariff_schedule_version': TARIFF_SCHEDULES[0].version,
            'tariff_source': RATE_SOURCE,
            'plans': list(TARIFF_SCHEDULES[0].plans),
        },
        'code': {'sha256': code_fingerprint(), 'files': list(CODE_PATHS)},
        'environment': {
            'python': platform.python_version(),
            'pandas': pd.__version__, 'numpy': np.__version__,
            'scipy': scipy.__version__,
        },
    }
    manifest['manifest_digest'] = manifest_digest(manifest)
    manifest['git_head'] = _git_head(root)
    return manifest


def manifest_digest(manifest):
    """Digest over the canonical ordered manifest entries.

    Covers the inputs, the pricing assumptions and the code fingerprint.
    Excludes the digest itself and the build annotations (git HEAD,
    library environment), which describe the build, not its inputs.
    """
    payload = {k: v for k, v in manifest.items()
               if k not in ('manifest_digest', 'git_head', 'environment')}
    return _sha256_bytes(canonical_json_bytes(payload))


def _canonicalize(value):
    if isinstance(value, dict):
        return {str(k): _canonicalize(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(v) for v in value]
    if isinstance(value, (bool, str)) or value is None:
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError('Non-finite value cannot enter a frozen report')
        # Canonical float rendering: 9 significant decimal digits.
        # Cross-library floating point noise is relative (~1e-14), so a
        # relative rendering absorbs it at any magnitude while staying
        # far below claim precision (money is reported to the cent).
        number = float(f'{number:.9g}')
        return 0.0 if number == 0 else number
    raise TypeError(f'Unserializable value in report: {type(value)!r}')


def canonical_json_bytes(obj):
    """Deterministic JSON bytes: sorted keys, canonical floats, no NaN/Inf."""
    return json.dumps(_canonicalize(obj), sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def frozen_report_bytes(report):
    """Canonical bytes for frozen-report comparison.

    Identical to canonical_json_bytes except the manifest's build
    annotations (git_head, environment) are normalized out: the commit
    that records the frozen report always differs from the commit it
    was generated at, and library versions drift between machines, so
    those fields can never be part of a byte-equivalence check (they
    are likewise excluded from the manifest digest).
    """
    if not isinstance(report, dict):
        return canonical_json_bytes(report)
    normalized = copy.deepcopy(report)
    manifest = normalized.get('manifest')
    if isinstance(manifest, dict):
        for key in ('git_head', 'environment'):
            if key in manifest:
                manifest[key] = '<build-annotation>'
    return canonical_json_bytes(normalized)


def validate_report_contract(report):
    """Data contract for a frozen report; raises ValueError on violation."""
    required = {'schema_version', 'dataset_sha256', 'quality', 'daily', 'bills',
                'tariffs', 'weather', 'forecast', 'summary', 'manifest'}
    missing = required - set(report)
    if missing:
        raise ValueError(f'Report is missing required keys: {sorted(missing)}')
    canonical_json_bytes(report)  # rejects non-finite or unserializable values
    manifest = report['manifest']
    if manifest.get('schema') != MANIFEST_SCHEMA:
        raise ValueError('Report manifest has an unknown schema')
    if manifest.get('manifest_digest') != manifest_digest(manifest):
        raise ValueError('Report manifest digest does not match its entries')
    entries = {e['role']: e for e in manifest['inputs']}
    if entries['intervals'].get('rows') != report['quality']['rows']:
        raise ValueError('Interval row count disagrees with the quality block')
    if entries['bill_details'].get('rows') != len(report['bills']):
        raise ValueError('Bill row count disagrees with the bills block')
    if len(report['daily']) != report['quality']['days']:
        raise ValueError('Daily row count disagrees with the quality block')
    return report


def verify_reuse(root, stored_report):
    """No-recompute assertion for --reuse-analysis.

    The stored report may only be reused when its manifest digest equals
    the manifest recomputed from the current inputs, pricing and code.
    Raises ManifestMismatch otherwise (including a report with no
    manifest at all).
    """
    stored = stored_report.get('manifest') if isinstance(stored_report, dict) else None
    if not stored or 'manifest_digest' not in stored:
        raise ManifestMismatch(
            'Stored report has no evidence manifest; rebuild instead of reusing it')
    current = build_manifest(root)
    if stored['manifest_digest'] != current['manifest_digest']:
        raise ManifestMismatch(
            'Stored report manifest digest does not match the current inputs '
            'and code; rebuild instead of reusing it')
    return stored_report

def build_report(root):
    root=Path(root)
    path=root/'demo/pseg_interval_15min.csv'
    iv=read_intervals(path); daily,quality=interval_quality(iv)
    details=pd.read_csv(root/'demo/pseg_rates_detailed.csv')
    bills=reconcile_bills(iv,details)
    replay=iv[(iv.index>=pd.Timestamp('2025-10-01')) & (iv.index<pd.Timestamp('2026-10-01'))]
    tariffs=[]
    for name in ['180','194','195']:
        cost,_=tariff_replay(replay,name)
        monthly=cost.resample('MS').sum()
        tariffs.append({'plan':name,'energy_usd':float(cost.sum()),'service_usd':365*.56,
                        'subtotal_usd':float(cost.sum()+365*.56),
                        'monthly':[float(v) for v in monthly], 'tariff_version':TARIFF_SCHEDULES[0].version, 'supply':DEFAULT_SUPPLY})
    full=[r for r in bills if r['complete']]
    attribution=[{'from':a['period_start'],'to':b['period_start'],
                   'change':b['bill_usd']-a['bill_usd'],
                   'components':shapley_bill_change(a,b)} for a,b in zip(bills,bills[1:])]
    drows=[]
    for day,r in daily.iterrows():
        seg=iv[iv.index.normalize()==day]
        pk=float(seg.loc[peak_mask(seg.index),'kwh'].sum())
        overnight=float(seg.loc[seg.index.hour<6,'kwh'].sum())
        drows.append({'date':str(day.date()),'kwh':float(r['sum']),'peak_kwh':pk,
                      'overnight_kwh':overnight,'kw_max':float(seg.kw.max()),
                      'intervals':int(r['count']),'quality':r.quality})
    hourly=iv.kwh.resample('h').sum()
    weekday_heat=hourly.groupby([hourly.index.weekday,hourly.index.hour]).mean().unstack().reindex(index=range(7),columns=range(24))
    utility_est=pd.read_csv(root/'demo/pseg_appliance_estimates.csv')
    utility_est=utility_est[utility_est.category!='taxes_fees']
    est=[{k:(None if pd.isna(v) else v) for k,v in r.items()} for r in utility_est.to_dict('records')]
    # Compact interval payload preserves all data for browser date exploration.
    summer_days=daily[(daily.index.month.isin([6,7,8])) & daily.usable]
    winter_days=daily[(daily.index.month.isin([12,1,2])) & daily.usable]
    forecast = forecast_analysis(daily)
    history = pd.read_csv(root/'demo/pseg_bills.csv', parse_dates=['period_start', 'period_end'])
    report = {'schema_version':1,'dataset_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'holidays':[d.strftime('%Y-%m-%d') for d in USFederalHolidayCalendar().holidays('2025-01-01','2030-12-31')],
            'household':{'region':'Long Island, NY','ac':'individual room units',
                         'heating':'building paid; excluded','ev':False,'solar':'not confirmed'},
            'quality':quality,'daily':drows,'bills':bills,'attribution':attribution,
            'tariffs':tariffs,'tariff_months':[str(d.date()) for d in pd.date_range('2025-10-01',periods=12,freq='MS')],
            'tariff_note':'Standardized 2026 delivery schedule replay on Oct 2025 to Sep 2026 readings with fixed September 2026 base PSC $0.139822/kWh. Energy and service subtotal only. Taxes, adjustments, eligibility and actual billing-cycle tier proration excluded. Not historical alternative bills.',
            'weather':weather_analysis(daily,root/'demo/household_weather.csv'),
            'forecast':forecast,'open_bill_forecast':open_bill_forecast(iv,daily,forecast,history),
            'shapes':load_shapes(iv,daily),
            'weekday_heatmap':weekday_heat.fillna(0).to_numpy().tolist(),
            'utility_appliance_estimates':est,
            'intervals':{'timestamps':[t.strftime('%Y-%m-%dT%H:%M') for t in iv.index],
                         'kwh':iv.kwh.tolist(),'kw':iv.kw.tolist()},
            'summary':{'full_billing_periods':len(full),
                       'summer_mean_daily_kwh':float(summer_days['sum'].mean()),
                       'winter_mean_daily_kwh':float(winter_days['sum'].mean()),
                       'max_reconciliation_pct':max(abs(r['difference_pct']) for r in full),
                       'max_reconstruction_error':max(abs(r['reconstruction_error']) for r in bills),
                       'total_billed_usd':sum(r['bill_usd'] for r in bills),
                       'shift_ceiling_full_periods_usd':sum(r['shift_ceiling'] for r in full)},
            'sources':{'rates':RATE_SOURCE,'weather':'https://open-meteo.com/en/docs/historical-weather-api',
                       'method':'https://www.energy.gov/cmei/femp/measurement-and-verification-options-federal-energy-and-water-saving-projects'}}
    report['manifest'] = build_manifest(root)
    return validate_report_contract(report)
