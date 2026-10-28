"""Stage 3 wattwise E1: forecast model tournament tests."""

import numpy as np
import pandas as pd
import pytest

from wattwise.tournament import (
    HORIZONS,
    MAX_INTERVAL_REGRESSION,
    MAX_RUNTIME_MULTIPLE,
    SEEDS,
    WIN_HORIZONS_REQUIRED,
    WIN_MARGIN,
    ConstrainedTreeRegressor,
    decide,
    run_tournament,
    tournament_day_features,
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


def test_stage3_wattwise_1_features_past_only():
    index = pd.date_range("2025-01-01", periods=40, freq="D")
    history = pd.Series(np.arange(1.0, 41.0), index=index)
    cdd = pd.Series(5.0, index=index)
    day = index[30]
    features = tournament_day_features(history[history.index < day], cdd, day)
    assert len(features) == 16
    assert features[0] == 30.0  # lag1
    assert features[1] == 29.0  # lag2
    assert features[2] == 28.0  # lag3
    assert features[3] == 24.0  # lag7
    assert features[4] == 17.0  # lag14
    assert features[5] == 27.0  # rolling 7 mean
    assert features[6] == 16.5  # rolling 28 mean
    assert features[7 + 4] == 1.0  # 2025-01-31 is a Friday
    assert features[7:14].max() == 1.0
    # Changing weather on or after the target day must not move features.
    future_changed = cdd.copy()
    future_changed.loc[day:] = 99.0
    changed = tournament_day_features(history[history.index < day],
                                      future_changed, day)
    np.testing.assert_array_equal(features, changed)
    # A missing lag day surfaces as NaN instead of an invented value.
    gapped = history.drop(index[29])
    missing = tournament_day_features(gapped[gapped.index < day], cdd, day)
    assert np.isnan(missing[0])


def test_stage3_wattwise_1_tree_respects_constraints():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, (200, 4))
    y = (x[:, 0] > 0).astype(float) * 5.0 + rng.normal(0, 0.1, 200)
    tree = ConstrainedTreeRegressor(max_depth=3, min_samples_leaf=20)
    tree.fit(x, y)
    first = tree.predict(x)
    second = tree.predict(x)
    np.testing.assert_array_equal(first, second)  # deterministic
    assert np.all(np.isfinite(first))
    assert first.min() >= y.min() - 1e-9
    assert first.max() <= y.max() + 1e-9
    assert tree.depth_ <= 3
    assert min(tree.leaf_sizes_) >= 20
    # A constant target collapses to a single leaf.
    flat = ConstrainedTreeRegressor(max_depth=3, min_samples_leaf=20)
    flat.fit(x, np.full(200, 4.0))
    np.testing.assert_allclose(flat.predict(x), 4.0)
    assert flat.depth_ == 0


def test_stage3_wattwise_1_decision_rule_thresholds():
    assert WIN_MARGIN == 0.05
    assert MAX_INTERVAL_REGRESSION == 0.05
    assert MAX_RUNTIME_MULTIPLE == 10.0
    assert WIN_HORIZONS_REQUIRED == 2


def _metric_table(control_mae, challenger_mae, control_score,
                  challenger_score):
    metrics = {
        "weekly_naive": {h: {"mae": control_mae, "interval_score": control_score}
                         for h in HORIZONS},
        "ridge": {h: {"mae": challenger_mae, "interval_score": challenger_score}
                  for h in HORIZONS},
    }
    runtimes = {"weekly_naive": 1.0, "ridge": 2.0}
    return metrics, runtimes


def test_stage3_wattwise_1_decision_rule_win_and_retain():
    metrics, runtimes = _metric_table(2.0, 1.8, 10.0, 10.0)
    outcome = decide(metrics, runtimes)
    assert outcome["decision"] == "REPLACE"
    assert outcome["winner"] == "ridge"
    # Margin miss on one horizon still wins if two horizons clear it.
    metrics["ridge"][28] = {"mae": 1.99, "interval_score": 10.0}
    outcome = decide(metrics, runtimes)
    assert outcome["decision"] == "REPLACE"
    # Interval regression beyond the allowed margin blocks the win.
    metrics, runtimes = _metric_table(2.0, 1.8, 10.0, 10.6)
    outcome = decide(metrics, runtimes)
    assert outcome["decision"] == "RETAIN"
    # A runtime blowup beyond the allowed multiple blocks the win.
    metrics, runtimes = _metric_table(2.0, 1.8, 10.0, 10.0)
    runtimes["ridge"] = 11.0
    outcome = decide(metrics, runtimes)
    assert outcome["decision"] == "RETAIN"
    # A marginal win under the margin retains the control.
    metrics, runtimes = _metric_table(2.0, 1.95, 10.0, 10.0)
    outcome = decide(metrics, runtimes)
    assert outcome["decision"] == "RETAIN"


def test_stage3_wattwise_1_reduced_tournament_synthetic(tmp_path):
    daily = _synthetic_daily(300)
    weather = _weather_csv(tmp_path, daily.index)
    result = run_tournament(daily, weather, n_origins=12, n_selection=8)
    assert result["available"] is True
    assert result["protocol"]["origins"] == 12
    assert result["protocol"]["selection_origins"] == 8
    assert result["protocol"]["holdout_origins"] == 4
    assert result["protocol"]["seeds"] == list(SEEDS)
    models = {row["model"] for row in result["one_step"]}
    assert {"weekly_naive", "weekday_median", "ridge",
            "constrained_tree", "neural", "ensemble"} <= models
    for row in result["horizons"]:
        assert row["horizon"] in HORIZONS
        # Origins are the last days of the series, so a holdout origin
        # admits a complete h-day window only when enough days follow it.
        expected = max(0, 12 - row["horizon"] + 1 - 8)
        assert row["holdout_origins"] == expected
        assert row["mae_kwh"] is None or np.isfinite(row["mae_kwh"])
        assert row["interval_score"] is None or np.isfinite(
            row["interval_score"])
    assert result["decision"]["decision"] in ("RETAIN", "REPLACE")
    # The ensemble path is the mean of its members on the same origins.
    assert result["protocol"]["ensemble"] == "mean of ridge, tree, neural paths"


def test_stage3_wattwise_1_horizon_complete_window_rule():
    from wattwise.tournament import _horizon_rows
    index = pd.date_range("2025-01-01", periods=40, freq="D")
    history = pd.Series(np.arange(1.0, 41.0), index=index)
    origins = history.index[-12:]
    # Perfect predictions: every complete window scores zero error.
    paths = {"weekly_naive": [history.reindex(
        pd.date_range(day, periods=28, freq="D")).to_numpy()
        if len(history.reindex(
            pd.date_range(day, periods=28, freq="D")).dropna()) == 28
        else np.zeros(28) for day in origins]}
    rows = _horizon_rows(paths, history, origins, 8)
    by_horizon = {row["horizon"]: row for row in rows}
    assert by_horizon[7]["holdout_origins"] == 0
    assert by_horizon[7]["calibration_origins"] == 2
    # With origins shifted earlier, complete windows exist and score zero.
    origins2 = history.index[10:22]
    paths2 = {"weekly_naive": [history.reindex(
        pd.date_range(day, periods=28, freq="D")).to_numpy()
        for day in origins2]}
    rows2 = _horizon_rows(paths2, history, origins2, 8)
    by_horizon2 = {row["horizon"]: row for row in rows2}
    assert by_horizon2[7]["holdout_origins"] == 4
    assert by_horizon2[7]["mae_kwh"] == 0.0
    assert by_horizon2[7]["coverage_80"] == 1.0


def test_stage3_wattwise_1_reduced_tournament_real_data(tmp_path):
    from pathlib import Path

    from wattwise.decision_analysis import interval_quality, read_intervals
    root = Path(__file__).resolve().parents[1] / "demo"
    intervals = read_intervals(root / "pseg_interval_15min.csv")
    daily, _quality = interval_quality(intervals)
    result = run_tournament(daily, root / "household_weather.csv",
                            n_origins=10, n_selection=6)
    assert result["available"] is True
    assert result["protocol"]["origins"] == 10
    for row in result["one_step"]:
        assert np.isfinite(row["holdout_mae"])
    for row in result["horizons"]:
        assert row["holdout_origins"] <= 4
        assert row["mae_kwh"] is None or np.isfinite(row["mae_kwh"])
