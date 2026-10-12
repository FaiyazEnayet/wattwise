"""Anomaly detection tests: injected spikes are found and explained."""

import pandas as pd

from wattwise import make_demo_profile, learn_shapes
from wattwise.anomalies import (daily_anomalies, explain_anomaly,
                                typical_weekday_profile)


def _with_spike(seed=7, spike_kwh_per_h=2.0):
    hourly, end_use, _ = make_demo_profile(seed=seed)
    shapes = learn_shapes(end_use.iloc[:-30 * 24])
    spiked = hourly.copy()
    spike_day = (hourly.index[-1] - pd.Timedelta(days=9)).date()
    mask = (spiked.index.date == spike_day) & (spiked.index.hour >= 18) \
        & (spiked.index.hour < 21)
    spiked.loc[mask] += spike_kwh_per_h
    return spiked, shapes, spike_day


def test_injected_spike_detected():
    spiked, _, spike_day = _with_spike()
    anoms = daily_anomalies(spiked, min_history=8)
    assert (anoms["date"] == spike_day).any()


def test_explanation_names_evening():
    spiked, shapes, spike_day = _with_spike()
    prof = typical_weekday_profile(spiked, spike_day.weekday())
    exp = explain_anomaly(spiked, spike_day, shapes, prof)
    assert "PM" in exp["text"]
    assert exp["excess_kwh"] > 3.0  # injected 6 kWh over 3 hours
    assert len(exp["likely_appliances"]) > 0


def test_clean_profile_has_few_flags():
    hourly, _, _ = make_demo_profile(seed=7)
    anoms = daily_anomalies(hourly, min_history=8, z_thresh=3.5)
    # Seeded demo is well-behaved; allow at most a couple of flags.
    assert len(anoms) <= 3


def test_typical_profile_shape():
    hourly, _, _ = make_demo_profile(seed=7)
    prof = typical_weekday_profile(hourly, 2)  # Wednesday
    assert len(prof) == 24
    assert prof.sum() > 0
    # Evening peak higher than deep night.
    assert prof[18:21].mean() > prof[2:5].mean()
