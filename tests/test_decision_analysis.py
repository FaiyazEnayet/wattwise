"""Regression tests for accounting, temporal validation and decision analytics."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from wattwise.decision_analysis import (read_intervals, interval_quality, reconcile_bills,
    peak_mask, tariff_replay, shapley_bill_change, predict_daily, build_report,
    open_bill_forecast)
from wattwise.rates import Rate180

ROOT=Path(__file__).resolve().parents[1]


def test_original_data_integrity_and_dst_flags():
    iv=read_intervals(ROOT/'demo/pseg_interval_15min.csv')
    daily,quality=interval_quality(iv)
    assert len(iv)==37976
    assert iv.kwh.sum()==pytest.approx(4086.55)
    assert not daily.loc['2025-11-02','usable']
    assert not daily.loc['2026-03-08','usable']
    assert not daily.iloc[-1].usable
    assert len(quality['missing_wall_slots'])==4


def test_half_open_periods_match_line_item_totals():
    iv=read_intervals(ROOT/'demo/pseg_interval_15min.csv')
    details=pd.read_csv(ROOT/'demo/pseg_rates_detailed.csv')
    rows=reconcile_bills(iv,details)
    full=[r for r in rows if r['complete']]
    assert len(full)==12
    assert max(abs(r['difference_pct']) for r in full)<.5
    sep=full[-1]
    assert sep['raw_kwh']==pytest.approx(320.75)
    assert sep['raw_peak_kwh']==pytest.approx(39.08)


def test_multi_year_holidays_and_peak_boundaries():
    dates=pd.to_datetime(['2025-11-27 16:00','2026-11-26 16:00',
                          '2026-10-20 14:45','2026-10-20 15:00',
                          '2026-10-20 19:00'])
    assert peak_mask(dates).tolist()==[False,False,False,True,False]


def test_summer_flat_block_is_incremental_and_continuous():
    r=Rate180(supply_rate=0)
    assert r.energy_charge(300,True)==pytest.approx(250*.1021+50*.1294)
    assert r.energy_charge(250.001,True)-r.energy_charge(250,True)<.001
    with pytest.raises(ValueError): r.energy_charge(-1,True)


def test_2026_tariff_replay_block_crossing_interval():
    iv=pd.DataFrame({'kwh':[249.,2.],'kw':[0,0]},index=pd.to_datetime(['2026-07-01','2026-07-02']))
    costs,_=tariff_replay(iv,'180',supply=0)
    assert costs.sum()==pytest.approx(250*.1064+1*.1348)
    with pytest.raises(ValueError): tariff_replay(iv,'196')
    with pytest.raises(ValueError): tariff_replay(iv,'194',supply=-1)


def test_shapley_exact_reconciliation():
    iv=read_intervals(ROOT/'demo/pseg_interval_15min.csv')
    rows=reconcile_bills(iv,pd.read_csv(ROOT/'demo/pseg_rates_detailed.csv'))
    a,b=rows[-2:]
    explanation=shapley_bill_change(a,b)
    assert sum(explanation.values())==pytest.approx(b['reconstructed']-a['reconstructed'])
    assert shapley_bill_change(a,a)==pytest.approx({k:0 for k in explanation})


def test_forecast_ignores_future_values_and_preserves_weekly_lags():
    dates=pd.date_range('2026-01-01',periods=180,freq='D')
    history=pd.Series(np.arange(180)%7+5.,index=dates)
    future=pd.date_range(dates[-1]+pd.Timedelta(days=1),periods=14)
    pred=predict_daily(history,future,'weekly_naive')
    assert np.allclose(pred[:7],history.tail(7))
    assert np.allclose(pred[7:],history.tail(7))
    for method in ['weekday_median','ridge_ar']:
        pred=predict_daily(history,future,method)
        assert np.isfinite(pred).all() and (pred>=0).all()


def test_duplicate_timestamps_are_not_silently_double_counted(tmp_path):
    file=tmp_path/'duplicate.csv'
    file.write_text('Start,kWh,kW\n01/01/2026 01:00:00 AM,1,4\n01/01/2026 01:00:00 AM,1,4\n')
    with pytest.raises(ValueError,match='Duplicate'):
        read_intervals(file)


def test_open_bill_preserves_previous_projection_and_excludes_next_period():
    iv = read_intervals(ROOT/'demo/pseg_interval_15min.csv')
    daily, _ = interval_quality(iv)
    forecast = json.loads((ROOT/'demo/analysis.json').read_text())['forecast']
    history = pd.read_csv(ROOT/'demo/pseg_bills.csv', parse_dates=['period_start', 'period_end'])
    result = open_bill_forecast(iv, daily, forecast, history)
    assert result['observed_kwh'] == pytest.approx(183.68)
    assert result['projected_kwh'] == pytest.approx(324.7196155600658)
    assert result['display']['projected_subtotal_usd'] == '$94.67'
    assert result['envelope_usd'] == pytest.approx([79.0453516559592, 104.25857787383438])
    assert result['days_elapsed'] + result['days_remaining'] == 32
    assert result['days_elapsed'] == 17.625
    for i, day in enumerate(forecast['dates']):
        if day >= result['period_end_exclusive']:
            for field in ['prediction', 'lo', 'hi']:
                forecast[field][i] = 1000000
    assert open_bill_forecast(iv, daily, forecast, history) == result


def test_open_bill_refuses_missing_forecast_coverage():
    iv = read_intervals(ROOT/'demo/pseg_interval_15min.csv')
    daily, _ = interval_quality(iv)
    forecast = json.loads((ROOT/'demo/analysis.json').read_text())['forecast']
    forecast['dates'] = forecast['dates'][:5]
    history = pd.read_csv(ROOT/'demo/pseg_bills.csv', parse_dates=['period_start', 'period_end'])
    result = open_bill_forecast(iv, daily, forecast, history)
    assert not result['available']
    assert 'cover' in result['reason']
