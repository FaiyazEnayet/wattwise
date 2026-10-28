"""Stage 3 wattwise FIXED-neural: neural benchmark candidate tests."""

import numpy as np
import pandas as pd
import pytest

from wattwise.neural_forecast import (
    CANDIDATE_CONFIGS,
    _day_features,
    _select_config,
    evaluate_neural,
    neural_path,
)


def _synthetic_daily(n=300, seed=7):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2025-01-01", periods=n, freq="D")
    values = (10.0 + 2.0 * np.sin(2 * np.pi * np.arange(n) / 7.0)
              + 0.01 * np.arange(n) + rng.normal(0, 0.5, n))
    return pd.DataFrame({"sum": values, "usable": True}, index=index)


def _weather_csv(tmp_path, index):
    rows = []
    for day in index:
        for hour in range(24):
            rows.append({"timestamp": day + pd.Timedelta(hours=hour),
                         "temperature_f": 70.0})
    path = tmp_path / "weather.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_stage3_wattwise_neural_features():
    index = pd.date_range("2025-01-01", periods=40, freq="D")
    history = pd.Series(np.arange(1.0, 41.0), index=index)
    cdd = pd.Series(5.0, index=index)
    day = index[30]
    features = _day_features(history[history.index < day], cdd, day)
    assert len(features) == 15
    assert features[0] == 30.0  # lag1
    assert features[1] == 29.0  # lag2
    assert features[2] == 28.0  # lag3
    assert features[3] == 24.0  # lag7
    assert features[4] == 17.0  # lag14
    assert features[5] == 27.0  # rolling 7 mean
    assert features[6] == 16.5  # rolling 28 mean
    assert features[7 + 4] == 1.0  # 2025-01-31 is a Friday
    assert features[7 + 4] == features[7:14].max()
    assert features[14] == 5.0  # CDD
    # A missing lag day surfaces as NaN instead of an invented value.
    gapped = history.drop(index[29])
    missing = _day_features(gapped[gapped.index < day], cdd, day)
    assert np.isnan(missing[0])


def test_stage3_wattwise_neural_determinism():
    daily = _synthetic_daily(200)
    history = daily.loc[daily.usable, "sum"]
    cdd = pd.Series(0.0, index=history.index)
    origin = history.index[150]
    config = CANDIDATE_CONFIGS[0]
    first = neural_path(history, cdd, origin, config, 0, horizon=7)
    second = neural_path(history, cdd, origin, config, 0, horizon=7)
    assert np.array_equal(first, second)


def test_stage3_wattwise_neural_leakage():
    daily = _synthetic_daily(200)
    history = daily.loc[daily.usable, "sum"]
    cdd = pd.Series(0.0, index=history.index)
    origin = history.index[150]
    config = CANDIDATE_CONFIGS[1]
    clean = neural_path(history, cdd, origin, config, 1, horizon=28)
    corrupted = history.copy()
    corrupted.loc[corrupted.index >= origin] *= 100.0
    tampered = neural_path(corrupted, cdd, origin, config, 1, horizon=28)
    assert np.array_equal(clean, tampered)


def test_stage3_wattwise_neural_selection_ignores_holdout():
    daily = _synthetic_daily(300)
    history = daily.loc[daily.usable, "sum"]
    cdd = pd.Series(0.0, index=history.index)
    origins = history.index[-20:]
    actual = history.reindex(origins).to_numpy()
    _, chosen = _select_config(history, cdd, origins, 12, actual)
    corrupted = history.copy()
    corrupted.loc[corrupted.index >= origins[12]] *= 100.0
    _, chosen_tampered = _select_config(corrupted, cdd, origins, 12, actual)
    assert chosen == chosen_tampered


def test_stage3_wattwise_neural_sanity(tmp_path):
    daily = _synthetic_daily(300)
    weather = _weather_csv(tmp_path, daily.index)
    result = evaluate_neural(daily, weather, n_origins=20, n_selection=12)
    assert result["available"] is True
    assert result["protocol"]["origins"] == 20
    assert result["protocol"]["selection_origins"] == 12
    assert result["protocol"]["holdout_origins"] == 8
    assert len(result["one_step"]) == 4
    for row in result["one_step"]:
        assert np.isfinite(row["holdout_mae"]) and row["holdout_mae"] >= 0
        assert np.isfinite(row["holdout_wape"]) and row["holdout_wape"] >= 0
    assert len(result["horizons"]) == 12
    for row in result["horizons"]:
        assert row["coverage_80"] is None or 0.0 <= row["coverage_80"] <= 1.0
        assert row["mae_kwh"] is None or (
            np.isfinite(row["mae_kwh"]) and row["mae_kwh"] >= 0)
        assert row["holdout_origins"] <= 8
    seven_day = [r for r in result["horizons"] if r["horizon"] == 7]
    assert all(r["holdout_origins"] == 2 for r in seven_day)
    assert all(r["calibration_origins"] == 6 for r in seven_day)
    assert "never promoted" in " ".join(result["notes"])
    assert "realized" in " ".join(result["notes"])


def test_stage3_wattwise_neural_short_history(tmp_path):
    daily = _synthetic_daily(100)
    weather = _weather_csv(tmp_path, daily.index)
    result = evaluate_neural(daily, weather)
    assert result["available"] is False
