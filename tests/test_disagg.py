"""Disaggregation tests: learned shapes split the meter plausibly."""

import numpy as np
import pandas as pd

from wattwise import make_demo_profile, learn_shapes
from wattwise.disagg import disaggregate, disaggregate_day, _default_shapes


def test_learned_shapes_normalized():
    _, end_use, _ = make_demo_profile(seed=7)
    shapes = learn_shapes(end_use)
    for name, s in shapes.items():
        assert abs(s.sum() - 1.0) < 1e-9
        assert (s >= 0).all()


def test_default_shapes_normalized():
    for name, s in _default_shapes().items():
        assert abs(s.sum() - 1.0) < 1e-9


def test_day_attribution_sums_close_to_meter():
    hourly, end_use, _ = make_demo_profile(seed=7)
    shapes = learn_shapes(end_use.iloc[:-30 * 24])
    day = hourly.iloc[-24:].to_numpy()
    attr, rmse = disaggregate_day(day, shapes)
    assert abs(sum(attr.values()) - day.sum()) < 0.5 * day.sum() + 1.0
    assert rmse >= 0


def test_month_mae_bounded():
    hourly, end_use, _ = make_demo_profile(seed=7)
    shapes = learn_shapes(end_use.iloc[:-30 * 24])
    dis = disaggregate(hourly.iloc[-30 * 24:], shapes=shapes)
    truth = end_use.iloc[-30 * 24:].resample("D").sum()
    truth.index = truth.index.date
    errs = [np.abs(dis[a] - truth[a]).mean() for a in dis.columns
            if a in truth.columns]
    assert float(np.mean(errs)) < 2.0  # kWh/day per appliance
