"""Forecast model tournament: retain or replace weekly persistence.

Protocol mirrors ``decision_analysis.forecast_analysis`` and
``neural_forecast.evaluate_neural``: the same usable-day history, the
same frozen chronological origins (last 84 usable days, first 56 select,
final 28 stay untouched until reporting), the same 240-day train window,
and the same calibration rule for empirical coverage (10%/90% quantiles
of calibration errors whose target windows end before the holdout).

The tournament differs from FIXED-neural in one deliberate way: the
no-leaked-weather rule. FIXED-neural used the target day's realized
cooling degree-days, which is fine as a conditional benchmark but is not
a forecast available in advance. Here every feature value for a target
day comes from data strictly before that day. Recursive paths freeze the
weather features at their origin-day values for steps beyond day 1, so no
future weather is invented or assumed. The neural candidate is therefore
re-evaluated under this stricter rule; the FIXED-neural published result
stands for its own protocol and is not overturned by this tournament.

Candidates: weekly persistence (control, via ``predict_daily``),
weekday median (extra baseline, via ``predict_daily``), ridge
regression on the tournament features, a constrained regression tree
(NumPy CART, max depth 3, minimum leaf 20, deterministic), the neural
MLP (three fixed seeds, mean path, config selected on selection origins
only), and a simple ensemble (mean of the ridge, tree, and neural paths).

Decision rule, fixed before evaluation and applied mechanically: a
challenger replaces the control only if it beats weekly persistence by
at least 5 percent horizon MAE on at least 2 of the 3 horizons
(7/14/28 days), with interval score no more than 5 percent worse than
control on those horizons, and runtime within 10x of control. Otherwise
the control is retained. Thresholds live in module constants so any
post-hoc revision is visible in git.
"""

import resource
import time

import numpy as np
import pandas as pd

from .decision_analysis import predict_daily
from .neural_forecast import CANDIDATE_CONFIGS, TinyMLP, build_cdd_series

LAGS = (1, 2, 3, 7, 14)
ROLLING = (7, 28)
SEEDS = (0, 1, 2)
HORIZONS = (7, 14, 28)
N_ORIGINS = 84
N_SELECTION = 56
TRAIN_WINDOW_DAYS = 240

WIN_MARGIN = 0.05
MAX_INTERVAL_REGRESSION = 0.05
MAX_RUNTIME_MULTIPLE = 10.0
WIN_HORIZONS_REQUIRED = 2

TREE_MAX_DEPTH = 3
TREE_MIN_SAMPLES_LEAF = 20

FEATURE_NAMES = (
    ["lag1", "lag2", "lag3", "lag7", "lag14"]
    + ["rolling7_mean", "rolling28_mean"]
    + ["weekday_one_hot_7"]
    + ["cdd_lag1", "cdd_rolling7_mean"]
)

BASELINE_METHODS = ("weekly_naive", "weekday_median")


def tournament_day_features(past, cdd, day, frozen_cdd=None):
    """Feature vector for one target day from past data only.

    ``past`` is the meter history strictly before ``day``. ``cdd`` is the
    full daily cooling-degree-day series; only values strictly before
    ``day`` are read from it. ``frozen_cdd`` is a (lag1, rolling7) pair
    computed at an origin day, used for recursive steps beyond day 1 so
    no future weather is invented. Missing lags return NaN (callers
    decide: training skips the sample, prediction falls back to the
    training-window median).
    """
    values = []
    for lag in LAGS:
        values.append(float(past.get(day - pd.Timedelta(days=lag), np.nan)))
    for window in ROLLING:
        tail = past[past.index < day].tail(window)
        values.append(float(tail.mean()) if len(tail) else np.nan)
    values.extend(1.0 if day.weekday() == d else 0.0 for d in range(7))
    if frozen_cdd is not None:
        cdd_lag1, cdd_roll7 = frozen_cdd
    else:
        if cdd is None:
            cdd_lag1, cdd_roll7 = np.nan, np.nan
        else:
            cdd_lag1 = float(cdd.get(day - pd.Timedelta(days=1), np.nan))
            window = cdd[(cdd.index < day)
                         & (cdd.index >= day - pd.Timedelta(days=7))]
            cdd_roll7 = float(window.mean()) if len(window) else np.nan
    values.append(cdd_lag1)
    values.append(cdd_roll7)
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
        features = tournament_day_features(
            full_past[full_past.index < day], cdd, day)
        if np.isnan(features).any():
            skipped += 1
            continue
        rows.append(features)
        targets.append(float(train_window.loc[day]))
    return np.asarray(rows), np.asarray(targets), skipped


def _standardize(rows, targets):
    x_mean, x_std = rows.mean(axis=0), rows.std(axis=0)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)
    y_mean, y_std = float(targets.mean()), float(targets.std())
    y_std = y_std if y_std > 1e-8 else 1.0
    return x_mean, x_std, y_mean, y_std


def _fit_ridge(rows, targets, penalty=10.0):
    """Closed-form ridge on standardized features, centered target."""
    x_mean, x_std, y_mean, y_std = _standardize(rows, targets)
    scaled = (rows - x_mean) / x_std
    centered = (targets - y_mean) / y_std
    matrix = scaled.T @ scaled + penalty * np.eye(scaled.shape[1])
    coef = np.linalg.solve(matrix, scaled.T @ centered)
    return (coef, x_mean, x_std, y_mean, y_std)


def _predict_ridge(fitted, features):
    coef, x_mean, x_std, y_mean, y_std = fitted
    return float(((features - x_mean) / x_std) @ coef * y_std + y_mean)


class ConstrainedTreeRegressor:
    """Small deterministic CART regressor, NumPy only.

    Greedy split search over every feature and every midpoint between
    consecutive distinct values, scored by exact SSE via cumulative sums.
    Ties break toward the lowest feature index, then the lowest
    threshold, so fits are deterministic. Splits require at least
    ``min_samples_leaf`` on both sides; leaves predict the mean.
    """

    def __init__(self, max_depth=TREE_MAX_DEPTH,
                 min_samples_leaf=TREE_MIN_SAMPLES_LEAF):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree_ = None
        self.depth_ = 0
        self.leaf_sizes_ = []

    def _best_split(self, x, y):
        n = len(y)
        total = float(y.sum())
        total_sq = float((y * y).sum())
        best = None
        for feature in range(x.shape[1]):
            order = np.argsort(x[:, feature], kind="stable")
            values = x[order, feature]
            targets = y[order]
            cumsum = np.cumsum(targets)
            cumsq = np.cumsum(targets * targets)
            for cut in range(self.min_samples_leaf,
                             n - self.min_samples_leaf + 1):
                if values[cut - 1] == values[cut]:
                    continue
                left_n = cut
                left_sum = float(cumsum[cut - 1])
                left_sq = float(cumsq[cut - 1])
                right_n = n - cut
                right_sum = total - left_sum
                right_sq = total_sq - left_sq
                sse = (left_sq - left_sum * left_sum / left_n
                       + right_sq - right_sum * right_sum / right_n)
                if best is None or sse < best[0] - 1e-12:
                    threshold = 0.5 * (values[cut - 1] + values[cut])
                    best = (sse, feature, threshold)
        return best

    def _build(self, x, y, depth):
        self.depth_ = max(self.depth_, depth)
        node_sse = float(((y - y.mean()) ** 2).sum())
        if (depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf
                or node_sse <= 1e-12):
            self.leaf_sizes_.append(len(y))
            return ("leaf", float(y.mean()))
        split = self._best_split(x, y)
        if split is None or split[0] >= node_sse - 1e-12:
            self.leaf_sizes_.append(len(y))
            return ("leaf", float(y.mean()))
        _, feature, threshold = split
        left = x[:, feature] <= threshold
        if left.sum() < self.min_samples_leaf \
                or (~left).sum() < self.min_samples_leaf:
            self.leaf_sizes_.append(len(y))
            return ("leaf", float(y.mean()))
        return ("split", feature, threshold,
                self._build(x[left], y[left], depth + 1),
                self._build(x[~left], y[~left], depth + 1))

    def fit(self, x, y):
        self.depth_ = 0
        self.leaf_sizes_ = []
        self.tree_ = self._build(np.asarray(x, dtype=float),
                                np.asarray(y, dtype=float), 0)
        return self

    def _predict_one(self, node, features):
        while node[0] == "split":
            _, feature, threshold, left, right = node
            node = left if features[feature] <= threshold else right
        return node[1]

    def predict(self, x):
        rows = np.atleast_2d(np.asarray(x, dtype=float))
        return np.asarray([self._predict_one(self.tree_, row) for row in rows])


def _fit_tournament_neural(full_past, train_window, cdd, config, seed):
    """Fit one MLP for one origin on tournament features, or None."""
    rows, targets, skipped = _training_samples(full_past, train_window, cdd)
    if len(rows) < 28:
        return None, skipped
    x_mean, x_std, y_mean, y_std = _standardize(rows, targets)
    model = TinyMLP(rows.shape[1], config["hidden"], config["lr"],
                    config["epochs"], seed)
    model.fit((rows - x_mean) / x_std, (targets - y_mean) / y_std)
    return (model, x_mean, x_std, y_mean, y_std), skipped


def _learned_path(fitted_predict, history_before_origin, cdd, origin,
                  horizon, fallback):
    """Recursive path: predictions feed later meter features.

    Weather features freeze at their origin-day values for steps beyond
    day 1; no future weather is invented or assumed.
    """
    extended = history_before_origin.copy()
    frozen = None
    path = []
    for step, day in enumerate(pd.date_range(origin, periods=horizon,
                                             freq="D")):
        past = extended[extended.index < day]
        if step == 0:
            features = tournament_day_features(past, cdd, day)
            frozen = (float(features[14]), float(features[15]))
        else:
            features = tournament_day_features(past, cdd, day,
                                               frozen_cdd=frozen)
        features = np.where(np.isnan(features), fallback, features)
        value = max(float(fitted_predict(features)), 0.0)
        path.append(value)
        extended.loc[day] = value
    return np.asarray(path)


def _ridge_path(history, cdd, origin, horizon=28):
    full_past = history[history.index < origin]
    train_window = full_past.tail(TRAIN_WINDOW_DAYS)
    rows, targets, _ = _training_samples(full_past, train_window, cdd)
    fallback = float(train_window.median())
    if len(rows) < 28:
        return np.full(horizon, fallback)
    fitted = _fit_ridge(rows, targets)
    return _learned_path(lambda f: _predict_ridge(fitted, f), full_past,
                         cdd, origin, horizon, fallback)


def _tree_path(history, cdd, origin, horizon=28):
    full_past = history[history.index < origin]
    train_window = full_past.tail(TRAIN_WINDOW_DAYS)
    rows, targets, _ = _training_samples(full_past, train_window, cdd)
    fallback = float(train_window.median())
    if len(rows) < 28:
        return np.full(horizon, fallback)
    tree = ConstrainedTreeRegressor().fit(rows, targets)
    return _learned_path(lambda f: float(tree.predict(f[None])[0]),
                         full_past, cdd, origin, horizon, fallback)


def _neural_path(history, cdd, origin, config, seed, horizon=28):
    full_past = history[history.index < origin]
    train_window = full_past.tail(TRAIN_WINDOW_DAYS)
    fitted, _ = _fit_tournament_neural(full_past, train_window, cdd,
                                       config, seed)
    fallback = float(train_window.median())
    if fitted is None:
        return np.full(horizon, fallback)
    model, x_mean, x_std, y_mean, y_std = fitted

    def predict(features):
        scaled = (features - x_mean) / x_std
        return float(model.predict(scaled[None, :])[0] * y_std + y_mean)

    return _learned_path(predict, full_past, cdd, origin, horizon, fallback)


def _select_neural_config(history, cdd, origins, n_selection, actual):
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
                path = _neural_path(history, cdd, day, config, seed, horizon=1)
                one_step.append(float(path[0]))
            seed_maes.append(float(np.mean(np.abs(np.array(one_step)
                                               - actual[:n_selection]))))
        selection.append({"config": config, "seed_selection_mae": seed_maes,
                          "selection_mae": float(np.mean(seed_maes))})
    chosen = min(selection, key=lambda row: row["selection_mae"])
    return selection, chosen["config"]


def _interval_score(observed, predicted, calibration_errors, alpha=0.2):
    """Mean interval score for a central (1 - alpha) interval.

    The interval for each holdout window is the prediction shifted by the
    10%/90% quantiles of calibration errors; the score penalizes width
    plus misses at rate 2/alpha.
    """
    if not len(calibration_errors) or not len(observed):
        return None
    low, high = np.quantile(calibration_errors, [alpha / 2, 1 - alpha / 2])
    lower = predicted + low
    upper = predicted + high
    width = upper - lower
    below = np.where(observed < lower, (2.0 / alpha) * (lower - observed), 0.0)
    above = np.where(observed > upper, (2.0 / alpha) * (observed - upper), 0.0)
    return float(np.mean(width + below + above))


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
            held_observed, held_predicted = [], []
            for i, day in enumerate(origins):
                target_dates = pd.date_range(day, periods=horizon, freq="D")
                observed = history.reindex(target_dates)
                if observed.isna().any() or paths[i] is None:
                    continue
                error = float(observed.sum() - paths[i][:horizon].sum())
                if i < n_selection and target_dates[-1] <= origins[n_selection - 1]:
                    calibration_errors.append(error)
                elif i >= n_selection:
                    held_errors.append(error)
                    held_observed.append(float(observed.sum()))
                    held_predicted.append(float(paths[i][:horizon].sum()))
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
                score = _interval_score(np.asarray(held_observed),
                                        np.asarray(held_predicted),
                                        calibration_errors)
            else:
                mae = bias = wape = score = None
            rows.append({"model": model, "horizon": horizon,
                         "holdout_origins": len(held_errors),
                         "calibration_origins": len(calibration_errors),
                         "mae_kwh": mae, "bias_kwh": bias,
                         "wape": wape, "coverage_80": coverage,
                         "interval_score": score})
    for row in rows:
        same_horizon = [r for r in rows if r["horizon"] == row["horizon"]
                        and r["mae_kwh"] is not None]
        best = min((r["mae_kwh"] for r in same_horizon), default=None)
        control = next((r["mae_kwh"] for r in same_horizon
                        if r["model"] == "weekly_naive"), None)
        row["regret_vs_best"] = (None if best is None or row["mae_kwh"] is None
                                 else float(row["mae_kwh"] - best))
        row["regret_vs_control"] = (None if control is None
                                    or row["mae_kwh"] is None
                                    else float(row["mae_kwh"] - control))
    return rows


def decide(metrics, runtimes):
    """Apply the prespecified decision rule to a metric table.

    ``metrics`` maps model to horizon to {"mae", "interval_score"}.
    ``runtimes`` maps model to wall seconds. Returns the decision, the
    winning challenger (or None), and per-model verdicts.
    """
    control = metrics["weekly_naive"]
    control_runtime = runtimes.get("weekly_naive", 0.0) or 0.0
    verdicts = {}
    for model, table in metrics.items():
        if model == "weekly_naive":
            continue
        wins = 0
        horizon_detail = {}
        for horizon in HORIZONS:
            entry = table.get(horizon, {})
            control_entry = control.get(horizon, {})
            mae, control_mae = entry.get("mae"), control_entry.get("mae")
            score = entry.get("interval_score")
            control_score = control_entry.get("interval_score")
            if mae is None or control_mae is None or control_mae <= 0:
                horizon_detail[horizon] = "missing"
                continue
            mae_win = (control_mae - mae) / control_mae >= WIN_MARGIN
            score_ok = True
            if (score is not None and control_score is not None
                    and control_score > 0):
                score_ok = ((score - control_score) / control_score
                            <= MAX_INTERVAL_REGRESSION)
            horizon_detail[horizon] = "win" if (mae_win and score_ok) else "no"
            wins += 1 if (mae_win and score_ok) else 0
        runtime_ok = runtimes.get(model, 0.0) <= (
            MAX_RUNTIME_MULTIPLE * control_runtime if control_runtime > 0
            else True)
        improvements = []
        for horizon in HORIZONS:
            entry = table.get(horizon, {})
            control_entry = control.get(horizon, {})
            if (entry.get("mae") is not None
                    and control_entry.get("mae") not in (None, 0)):
                improvements.append((control_entry["mae"] - entry["mae"])
                                    / control_entry["mae"])
        verdicts[model] = {
            "horizon_wins": wins,
            "runtime_ok": bool(runtime_ok),
            "mean_horizon_improvement": (float(np.mean(improvements))
                                         if improvements else None),
            "horizon_detail": horizon_detail,
        }
    winners = [m for m, v in verdicts.items()
               if v["horizon_wins"] >= WIN_HORIZONS_REQUIRED
               and v["runtime_ok"]]
    if winners:
        winner = max(winners, key=lambda m: verdicts[m]["mean_horizon_improvement"])
        return {"decision": "REPLACE", "winner": winner,
                "per_model": verdicts}
    return {"decision": "RETAIN", "winner": None, "per_model": verdicts}


def _peak_rss_mb():
    """Process peak RSS watermark in MB (monotonic, not per-candidate)."""
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)


def run_tournament(daily, weather_path, n_origins=N_ORIGINS,
                   n_selection=N_SELECTION):
    """Run the full tournament protocol. Returns a JSON-serializable dict."""
    history = daily.loc[daily.usable, "sum"]
    if len(history) < 180:
        return {"available": False,
                "reason": "At least 180 complete days are required"}
    cdd = build_cdd_series(weather_path)
    if cdd is None:
        return {"available": False, "reason": "No matched weather file"}
    origins = history.index[-n_origins:]
    actual = history.reindex(origins).to_numpy()

    paths_by_model = {}
    runtimes = {}
    peak_rss = {}

    def timed(name, build):
        start = time.perf_counter()
        result = build()
        runtimes[name] = float(time.perf_counter() - start)
        peak_rss[name] = _peak_rss_mb()
        return result

    for method in BASELINE_METHODS:
        def build(method=method):
            return [predict_daily(history[history.index < day].tail(240),
                                  pd.date_range(day, periods=28, freq="D"),
                                  method)
                    for day in origins]
        paths_by_model[method] = timed(method, build)

    paths_by_model["ridge"] = timed(
        "ridge",
        lambda: [_ridge_path(history, cdd, day) for day in origins])
    paths_by_model["constrained_tree"] = timed(
        "constrained_tree",
        lambda: [_tree_path(history, cdd, day) for day in origins])

    selection, chosen_config = _select_neural_config(
        history, cdd, origins, n_selection, actual)
    seed_paths = {}
    for seed in SEEDS:
        seed_paths[seed] = timed(
            f"neural_seed_{seed}",
            lambda seed=seed: [_neural_path(history, cdd, day, chosen_config,
                                            seed) for day in origins])
    neural_paths = [np.mean([seed_paths[s][i] for s in SEEDS], axis=0)
                    for i in range(len(origins))]
    paths_by_model["neural"] = neural_paths
    runtimes["neural"] = float(sum(runtimes[f"neural_seed_{s}"] for s in SEEDS))
    peak_rss["neural"] = max(peak_rss[f"neural_seed_{s}"] for s in SEEDS)
    for seed in SEEDS:
        del runtimes[f"neural_seed_{seed}"]
        del peak_rss[f"neural_seed_{seed}"]

    paths_by_model["ensemble"] = [
        np.mean([paths_by_model["ridge"][i],
                 paths_by_model["constrained_tree"][i],
                 neural_paths[i]], axis=0)
        for i in range(len(origins))]
    runtimes["ensemble"] = float(runtimes["ridge"]
                                 + runtimes["constrained_tree"]
                                 + runtimes["neural"])
    peak_rss["ensemble"] = max(peak_rss["ridge"],
                               peak_rss["constrained_tree"],
                               peak_rss["neural"])

    one_step_rows = []
    for model, paths in paths_by_model.items():
        one_step = np.array([float(p[0]) if p is not None else np.nan
                             for p in paths])
        valid = ~np.isnan(one_step)
        selection_errors = one_step[:n_selection][valid[:n_selection]] \
            - actual[:n_selection][valid[:n_selection]]
        holdout_errors = actual[n_selection:][valid[n_selection:]] \
            - one_step[n_selection:][valid[n_selection:]]
        calibration_errors = list(selection_errors)
        score = _interval_score(
            actual[n_selection:][valid[n_selection:]],
            one_step[n_selection:][valid[n_selection:]],
            calibration_errors) if len(holdout_errors) else None
        one_step_rows.append({
            "model": model,
            "selection_mae": float(np.mean(np.abs(selection_errors)))
            if len(selection_errors) else None,
            "holdout_mae": float(np.mean(np.abs(holdout_errors)))
            if len(holdout_errors) else None,
            "holdout_bias": float(np.mean(-holdout_errors))
            if len(holdout_errors) else None,
            "holdout_wape": float(np.abs(holdout_errors).sum()
                                  / max(actual[n_selection:][valid[n_selection:]].sum(), 1e-9))
            if len(holdout_errors) else None,
            "interval_score": score,
        })

    horizon_rows = _horizon_rows(paths_by_model, history, origins, n_selection)

    metrics = {}
    for row in horizon_rows:
        metrics.setdefault(row["model"], {})[row["horizon"]] = {
            "mae": row["mae_kwh"], "interval_score": row["interval_score"]}
    decision = decide(metrics, runtimes)

    return {
        "available": True,
        "protocol": {
            "origins": len(origins),
            "selection_origins": n_selection,
            "holdout_origins": len(origins) - n_selection,
            "train_window_days": TRAIN_WINDOW_DAYS,
            "features": list(FEATURE_NAMES),
            "feature_rule": "Every feature value for a target day comes "
                            "from data strictly before that day. Recursive "
                            "steps beyond day 1 freeze the weather features "
                            "at their origin-day values; no future weather "
                            "is invented or assumed.",
            "standardization": "training-window-only mean/std, per origin",
            "seeds": list(SEEDS),
            "ensemble": "mean of ridge, tree, neural paths",
            "selection_rule": "Neural config selected on lowest one-step "
                              "selection MAE over the first selection "
                              "origins; holdout evaluated once after "
                              "selection is frozen.",
            "decision_rule": {
                "win_margin": WIN_MARGIN,
                "max_interval_regression": MAX_INTERVAL_REGRESSION,
                "max_runtime_multiple": MAX_RUNTIME_MULTIPLE,
                "win_horizons_required": WIN_HORIZONS_REQUIRED,
                "horizons": list(HORIZONS),
            },
            "origin_dates": [str(d.date()) for d in origins],
            "holdout_dates": [str(d.date()) for d in origins[n_selection:]],
        },
        "neural_selection": selection,
        "chosen_config": chosen_config,
        "one_step": one_step_rows,
        "horizons": horizon_rows,
        "runtimes_seconds": runtimes,
        "peak_rss_mb": peak_rss,
        "decision": decision,
        "notes": [
            "Tournament candidate only: the decision rule is applied "
            "mechanically; a REPLACE verdict does not promote a model "
            "automatically.",
            "The neural candidate is re-evaluated here under the stricter "
            "no-leaked-weather rule. The FIXED-neural published one-step "
            "result stands for its own protocol and is not overturned.",
            "Origins overlap and are not independent. The 28-day holdout "
            "admits only one complete 28-day forecast per model.",
            "Runtime is wall seconds for fitting plus predicting over all "
            "origins; peak RSS is the process watermark after each "
            "candidate, monotonic rather than per-candidate isolated.",
            "Selection origins are summer and holdout origins are fall in "
            "the full protocol; the comparison is fair because every "
            "candidate shares identical origins, while absolute MAE levels "
            "are season-mixed.",
        ],
    }
