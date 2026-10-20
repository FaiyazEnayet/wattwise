"""Auditable household decision analysis on original meter intervals.

Money amounts are USD, energy is kWh per interval, and power is kW.
Observed bills, retrospective accounting, and counterfactuals remain separate.
"""
from __future__ import annotations

import hashlib
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from scipy.optimize import lsq_linear

RATE_SOURCE = 'https://www.psegliny.com/aboutpseglongisland/ratesandtariffs/-/media/A0FDA80A6FE44A45973922422E86BD9E.ashx'
DEFAULT_SUPPLY = 0.139822  # September 2026 published base PSC, scenario input


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


def tariff_replay(iv, plan='194', supply=DEFAULT_SUPPLY):
    """2026 delivery schedule plus one declared PSC for the same historical load.

    This standardized scenario excludes taxes and adjustments and is not a
    reconstruction of historical alternative bills. Summer 180 blocks reset
    monthly in this scenario; actual billing-day proration needs the tariff.
    """
    if not np.isfinite(supply) or supply < 0:
        raise ValueError('Supply rate must be finite and nonnegative')
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
            'note': 'Cooling association above 65 F with weekday controls; not appliance metering or a causal AC saving. Weather is ERA5 reanalysis at approximate town center.'}


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
            bounds=np.quantile(calibration_errors,[.1,.9])
            empirical=float(np.mean((np.array(held_errors)>=bounds[0]) & (np.array(held_errors)<=bounds[1])))
            horizon_rows.append({'model':method,'horizon':horizon,'holdout_origins':len(held_errors),
                                 'calibration_origins':len(calibration_errors),
                                 'mae_kwh':float(np.mean(np.abs(held_errors))),
                                 'bias_kwh':float(-np.mean(held_errors)),
                                 'coverage_80':empirical})
            if method==chosen:cumulative_intervals[str(horizon)]=list(map(float,bounds))
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
            'note':'Chronological selection and untouched final 28-day evaluation. Future multiday predictions are recursive. Daily bands resample seven-day residual blocks; seven-day total bounds use chronological cumulative errors whose calibration targets end before the holdout. Origins overlap and are not independent. The 28-day holdout admits only one complete 28-day forecast per model. Partial final day is excluded.'}


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
                        'monthly':[float(v) for v in monthly], 'supply':DEFAULT_SUPPLY})
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
    return {'schema_version':1,'dataset_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'holidays':[d.strftime('%Y-%m-%d') for d in USFederalHolidayCalendar().holidays('2025-01-01','2030-12-31')],
            'household':{'town':'Port Jefferson NY 11777','ac':'individual room units',
                         'heating':'building paid; excluded','ev':False,'solar':'not confirmed'},
            'quality':quality,'daily':drows,'bills':bills,'attribution':attribution,
            'tariffs':tariffs,'tariff_months':[str(d.date()) for d in pd.date_range('2025-10-01',periods=12,freq='MS')],
            'tariff_note':'Standardized 2026 delivery schedule replay on Oct 2025 to Sep 2026 readings with fixed September 2026 base PSC $0.139822/kWh. Energy and service subtotal only. Taxes, adjustments, eligibility and actual billing-cycle tier proration excluded. Not historical alternative bills.',
            'weather':weather_analysis(daily,root/'demo/household_weather.csv'),
            'forecast':forecast_analysis(daily),'shapes':load_shapes(iv,daily),
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
