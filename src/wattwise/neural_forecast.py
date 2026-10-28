"""Neural forecaster as a benchmark candidate, never an automatic winner.

Protocol mirrors ``decision_analysis.forecast_analysis`` exactly: the same
usable-day history, the same 84 walk-forward origins, one-step-ahead
predictions plus recursive 7/14/28-day paths, the same chronological split
(first 56 origins select, final 28 stay untouched until reporting), and the
same calibration rule for empirical coverage (10%/90% quantiles of
calibration errors whose target windows end before the holdout).

The model is a small NumPy MLP (no torch dependency). Features at each
origin use only past meter data plus the target day's realized cooling
degree-days: lags 1/2/3/7/14 daily totals, rolling 7- and 28-day means,
weekday one-hot, and CDD at base 65 F. Degree-days use realized ERA5
reanalysis for the target day, exactly the conditional caveat the cooling
analysis carries: this is not a forecast issued in advance, and the result
must never be read as one. Training-window statistics standardize features
and targets; nothing after an origin ever enters that origin's fit.

Three fixed seeds form an ensemble (mean path); per-seed spread is
reported. A small candidate set (hidden width 8/16/32) is selected on the
first 56 origins only. The final 28 days are evaluated once, after the
selection is frozen, and the candidate is reported win or lose. It is
never promoted over the baselines automatically.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .decision_analysis import predict_daily

LAGS = (1, 2, 3, 7, 14)
ROLLING = (7, 28)
N_FEATURES = len(LAGS) + len(ROLLING) + 7 + 1
N_ORIGINS = 84
N_SELECTION = 56
HORIZONS = (7, 14, 28)
BASELINE_METHODS = ("weekly_naive", "weekday_median", "ridge_ar")
CANDIDATE_CONFIGS = (
    {"hidden": 8, "lr": 0.01, "epochs": 300},
    {"hidden": 16, "lr": 0.01, "epochs": 300},
    {"hidden": 32, "lr": 0.01, "epochs": 300},
)
SEEDS = (0, 1, 2)


def build_cdd_series(weather_path):
    """Daily cooling degree-days at base 65 F, days with <20 hours dropped.

    Mirrors the quality bar in ``weather_analysis``: missing weather is
    never treated as 0 F, it is absent.
    """
    path = Path(weather_path)
    if not path.exists():
        return None
    frame = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")
    daily = frame.temperature_f.resample("D").agg(["mean", "count"])
    daily = daily[daily["count"] >= 20]
    return pd.Series(
        np.clip(daily["mean"].to_numpy() - 65.0, 0, None), index=daily.index
    )


def _day_features(past, cdd, day):
    """Feature vector for one target day from past meter data only.

    ``past`` is the usable-day history strictly before ``day``. Missing
    lags return NaN (callers decide: training skips the sample, prediction
    falls back to the training-window median). Rolling means use available
    past values only. CDD falls back to 0.0 only when the weather file has
    no qualifying day, and that fallback is documented in the output.
    """
    values = []
    for lag in LAGS:
        values.append(float(past.get(day - pd.Timedelta(days=lag), np.nan)))
    for window in ROLLING:
        tail = past[past.index < day].tail(window)
        values.append(float(tail.mean()) if len(tail) else np.nan)
    values.extend(1.0 if day.weekday() == d else 0.0 for d in range(7))
    cdd_value = cdd.get(day, np.nan) if cdd is not None else np.nan
    values.append(0.0 if pd.isna(cdd_value) else float(cdd_value))
    return np.asarray(values, dtype=float)


def _training_samples(full_past, train_window, cdd):
    """Feature/target pairs whose target day falls in the train window.

    Features may reach before the window (lags), but every feature value
    still comes from strictly before its target day, and no target day is
    ever after the origin. Samples with a missing lag are skipped rather
    than imputed.
    """
    rows, targets, skipped = [], [], 0
    for day in train_window.index:
        features = _day_features(full_past[full_past.index < day], cdd, day)
        if np.isnan(features).any():
            skipped += 1
            continue
        rows.append(features)
        targets.append(float(train_window.loc[day]))
    return np.asarray(rows), np.asarray(targets), skipped


class TinyMLP:
    """One hidden layer, tanh, linear output, full-batch Adam, NumPy only."""

    def __init__(self, n_inputs, hidden, lr, epochs, seed):
        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(2.0 / (n_inputs + hidden))
        scale2 = np.sqrt(2.0 / (hidden + 1))
        self.weights1 = rng.normal(0.0, scale1, (n_inputs, hidden))
        self.bias1 = np.zeros(hidden)
        self.weights2 = rng.normal(0.0, scale2, (hidden, 1))
        self.bias2 = np.zeros(1)
        self.lr = lr
        self.epochs = epochs

    def _forward(self, x):
        hidden = np.tanh(x @ self.weights1 + self.bias1)
        return hidden @ self.weights2 + self.bias2, hidden

    def fit(self, x, y):
        params = [self.weights1, self.bias1, self.weights2, self.bias2]
        first = [np.zeros_like(p) for p in params]
        second = [np.zeros_like(p) for p in params]
        n = len(y)
        for step in range(1, self.epochs + 1):
            pred, hidden = self._forward(x)
            error = pred[:, 0] - y
            grad_out = (2.0 / n) * error
            grad_w2 = hidden.T @ grad_out[:, None]
            grad_b2 = np.array([grad_out.sum()])
            grad_hidden = (grad_out[:, None] @ self.weights2.T) * (1.0 - hidden**2)
            grad_w1 = x.T @ grad_hidden
            grad_b1 = grad_hidden.sum(axis=0)
            grads = [grad_w1, grad_b1, grad_w2, grad_b2]
            for i, (param, grad) in enumerate(zip(params, grads)):
                first[i] = 0.9 * first[i] + 0.1 * grad
                second[i] = 0.999 * second[i] + 0.001 * grad * grad
                first_hat = first[i] / (1.0 - 0.9**step)
                second_hat = second[i] / (1.0 - 0.999**step)
                param -= self.lr * first_hat / (np.sqrt(second_hat) + 1e-8)
        return self

    def predict(self, x):
        pred, _ = self._forward(x)
        return pred[:, 0]


def _fit_origin_model(full_past, train_window, cdd, config, seed):
    """Fit one MLP for one origin. Returns None when no usable samples."""
    rows, targets, skipped = _training_samples(full_past, train_window, cdd)
    if len(rows) < 28:
        return None, skipped
    x_mean, x_std = rows.mean(axis=0), rows.std(axis=0)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)
    y_mean, y_std = float(targets.mean()), float(targets.std())
    y_std = y_std if y_std > 1e-8 else 1.0
    model = TinyMLP(rows.shape[1], config["hidden"], config["lr"],
                    config["epochs"], seed)
    model.fit((rows - x_mean) / x_std, (targets - y_mean) / y_std)
    return (model, x_mean, x_std, y_mean, y_std), skipped


def _predict_day(fitted, past, cdd, day, fallback):
    model, x_mean, x_std, y_mean, y_std = fitted
    features = _day_features(past, cdd, day)
    features = np.where(np.isnan(features), fallback, features)
    scaled = (features - x_mean) / x_std
    return max(float(model.predict(scaled[None, :])[0] * y_std + y_mean), 0.0)


def neural_path(history, cdd, origin, config, seed, horizon=28):
    """Recursive path from one origin: predictions feed later features.

    Only the origin day's realized CDD and later days' realized CDD enter
    (the conditional-weather caveat); meter features for future days come
    from the model's own earlier predictions, exactly like predict_daily.
    """
    full_past = history[history.index < origin]
    train_window = full_past.tail(240)
    fitted, _ = _fit_origin_model(full_past, train_window, cdd, config, seed)
    if fitted is None:
        return None
    fallback = float(train_window.median())
    extended = full_past.copy()
    path = []
    for day in pd.date_range(origin, periods=horizon, freq="D"):
        value = _predict_day(fitted, extended[extended.index < day], cdd, day,
                             fallback)
        path.append(value)
        extended.loc[day] = value
    return np.asarray(path)


def _horizon_rows(paths_by_model, history, origins, n_selection):
    """Cumulative-horizon errors with the forecast_analysis split rule.

    A horizon with no complete calibration window reports coverage_80 as
    null, and one with no complete holdout window reports its holdout
    metrics as null, instead of failing on an empty quantile. The full
    protocol always has complete windows; reduced test protocols may not.
    """
    rows = []
    for model, paths in paths_by_model.items():
        for horizon in HORIZONS:
            calibration_errors, held_errors = [], []
            held_observed = []
            for i, day in enumerate(origins):
                target_dates = pd.date_range(day, periods=horizon, freq="D")
                observed = history.reindex(target_dates)
                if observed.isna().any():
                    continue
                error = float(observed.sum() - paths[i][:horizon].sum())
                if i < n_selection and target_dates[-1] <= origins[n_selection - 1]:
                    calibration_errors.append(error)
                elif i >= n_selection:
                    held_errors.append(error)
                    held_observed.append(float(observed.sum()))
            held_errors = np.asarray(held_errors)
            if calibration_errors and len(held_errors):
                bounds = np.quantile(calibration_errors, [0.1, 0.9])
                coverage = float(np.mean((held_errors >= bounds[0])
                                         & (held_errors <= bounds[1])))
            else:
                coverage = None
            if len(held_errors):
                mae = float(np.mean(np.abs(held_errors)))
                bias = float(-np.mean(held_errors))
                wape = float(np.abs(held_errors).sum()
                             / max(sum(held_observed), 1e-9))
            else:
                mae = bias = wape = None
            rows.append({"model": model, "horizon": horizon,
                         "holdout_origins": len(held_errors),
                         "calibration_origins": len(calibration_errors),
                         "mae_kwh": mae, "bias_kwh": bias,
                         "wape": wape, "coverage_80": coverage})
    return rows


def _select_config(history, cdd, origins, n_selection, actual):
    """Lowest one-step selection MAE over the first selection origins.

    Selection sees only selection-origin predictions; holdout actuals can
    never change the chosen config.
    """
    selection = []
    for config in CANDIDATE_CONFIGS:
        seed_maes = []
        for seed in SEEDS:
            one_step = []
            for day in origins[:n_selection]:
                path = neural_path(history, cdd, day, config, seed, horizon=1)
                one_step.append(float(path[0]))
            seed_maes.append(float(np.mean(np.abs(np.array(one_step)
                                               - actual[:n_selection]))))
        selection.append({"config": config, "seed_selection_mae": seed_maes,
                          "selection_mae": float(np.mean(seed_maes))})
    chosen = min(selection, key=lambda row: row["selection_mae"])
    return selection, chosen["config"]


def evaluate_neural(daily, weather_path, n_origins=N_ORIGINS,
                    n_selection=N_SELECTION):
    """Run the full benchmark protocol. Returns a JSON-serializable dict."""
    history = daily.loc[daily.usable, "sum"]
    if len(history) < 180:
        return {"available": False,
                "reason": "At least 180 complete days are required"}
    cdd = build_cdd_series(weather_path)
    if cdd is None:
        return {"available": False, "reason": "No matched weather file"}
    origins = history.index[-n_origins:]
    actual = history.reindex(origins).to_numpy()

    baseline_paths = {}
    for method in BASELINE_METHODS:
        paths = []
        for day in origins:
            train = history[history.index < day].tail(240)
            paths.append(predict_daily(train, pd.date_range(day, periods=28,
                                                            freq="D"), method))
        baseline_paths[method] = paths

    selection, chosen_config = _select_config(history, cdd, origins,
                                               n_selection, actual)

    seed_paths = {}
    for seed in SEEDS:
        seed_paths[seed] = [neural_path(history, cdd, day, chosen_config, seed)
                            for day in origins]
    neural_paths = [np.mean([seed_paths[s][i] for s in SEEDS], axis=0)
                    for i in range(len(origins))]

    one_step_rows = []
    for model, paths in {**baseline_paths, "neural": neural_paths}.items():
        one_step = np.array([float(p[0]) for p in paths])
        holdout_errors = actual[n_selection:] - one_step[n_selection:]
        one_step_rows.append({
            "model": model,
            "selection_mae": float(np.mean(np.abs(
                one_step[:n_selection] - actual[:n_selection]))),
            "holdout_mae": float(np.mean(np.abs(holdout_errors))),
            "holdout_bias": float(np.mean(-holdout_errors)),
            "holdout_wape": float(np.abs(holdout_errors).sum()
                                  / max(actual[n_selection:].sum(), 1e-9)),
        })

    horizon_rows = _horizon_rows({**baseline_paths, "neural": neural_paths},
                                  history, origins, n_selection)

    seed_holdout_mae = []
    for seed in SEEDS:
        one_step = np.array([float(p[0]) for p in seed_paths[seed]])
        seed_holdout_mae.append(float(np.mean(np.abs(
            actual[n_selection:] - one_step[n_selection:]))))
    seed_horizon_mae = []
    for seed in SEEDS:
        errors = []
        for i, day in enumerate(origins[n_selection:], start=n_selection):
            target_dates = pd.date_range(day, periods=28, freq="D")
            observed = history.reindex(target_dates)
            if observed.isna().any():
                continue
            errors.append(abs(float(observed.sum())
                              - float(seed_paths[seed][i][:28].sum())))
        seed_horizon_mae.append(float(np.mean(errors)) if errors else None)
    present = [v for v in seed_horizon_mae if v is not None]

    return {
        "available": True,
        "protocol": {
            "origins": len(origins),
            "selection_origins": n_selection,
            "holdout_origins": len(origins) - n_selection,
            "train_window_days": 240,
            "features": ["lag1", "lag2", "lag3", "lag7", "lag14",
                         "rolling7_mean", "rolling28_mean",
                         "weekday_one_hot_7", "cdd_base65_realized"],
            "standardization": "training-window-only mean/std, per origin",
            "seeds": list(SEEDS),
            "ensemble": "mean path across seeds",
            "selection_rule": "lowest one-step selection MAE over the first "
                              "selection origins; holdout evaluated once after "
                              "selection is frozen",
            "origin_dates": [str(d.date()) for d in origins],
            "holdout_dates": [str(d.date()) for d in origins[n_selection:]],
        },
        "selection": selection,
        "chosen_config": chosen_config,
        "one_step": one_step_rows,
        "horizons": horizon_rows,
        "seed_spread": {
            "holdout_one_step_mae_by_seed": seed_holdout_mae,
            "holdout_one_step_mae_std": float(np.std(seed_holdout_mae)),
            "holdout_28day_mae_by_seed": seed_horizon_mae,
            "holdout_28day_mae_std": float(np.std(present)) if present else None,
        },
        "notes": [
            "Benchmark candidate only: reported win or lose, never promoted "
            "over the baselines automatically.",
            "Degree-days use realized ERA5 reanalysis for the target day, "
            "not a forecast available in advance; multiday paths are "
            "conditional on realized weather throughout.",
            "Origins overlap and are not independent. The 28-day holdout "
            "admits only one complete 28-day forecast per model.",
            "Baselines are recomputed on identical origins from the same "
            "predict_daily code path as forecast_analysis.",
        ],
    }
