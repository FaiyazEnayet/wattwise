"""Generalized household pipeline: run the analysis on any home's files.

The measured-household loaders in `wattwise.measured` and the frozen
report in `wattwise.decision_analysis` are specific to one household's
exports. This module is the reusable interface: documented schemas for
the files another household keeps privately on its own machine, a
uniform tariff interface, and summaries that keep every home's results
attributed to that home.

Schemas
-------
Interval CSV, canonical format (one row per interval):

    timestamp,kwh
    2026-01-05T00:00:00,0.1234

- `timestamp`: ISO-like local timestamp, parseable by pandas; rows are
  strictly increasing with no duplicates. The file carries no UTC
  offset; timestamps are read as the household's local wall time and
  are never silently converted.
- `kwh`: energy in that interval, finite and nonnegative.

Interval CSV, utility-export format (`input_format="pseg"`): the
`Start,kWh,kW` export shape parsed by `decision_analysis.read_intervals`
(`%m/%d/%Y %I:%M:%S %p` timestamps); the same ordering and sign rules
apply and the kW column is checked numeric, then dropped.

Bill CSV (one row per billed period, boundaries [start, end)):

    period_start,period_end,kwh,bill_usd

- dates are local calendar dates with `period_end` after
  `period_start`; periods do not overlap;
- `kwh` and `bill_usd` may be blank for a period the utility has not
  quantified yet; when present they are finite and nonnegative.

Any other columns, unparseable values or ordering violations raise
`SchemaError`: a file that does not match the documented schema fails
loudly instead of being priced as if it did.

Tariff interface
----------------
A tariff is any object with `name`, `marginal_price(timestamp)`
(all-in $/kWh for one more kWh at that time), `energy_charge(kwh,
timestamp)` and `fixed_charge(days)`. `FlatTariff` and
`TimeOfUseTariff` are declared-parameter implementations another
household can instantiate from its own rate sheet (or a JSON spec via
`tariff_from_spec`); `rate194_tariff` adapts the existing PSEG
`Rate194`. Tiered block pricing that needs a monthly total (Rate 180's
summer tier) keeps its own path in `wattwise.analyze`; it is not
forced into this per-interval interface.

Reporting rule
--------------
`build_household_report` reports `within_home` (one summary per
household, computed only from that household's files and tariff) and
`cross_home` (labeled aggregates over the per-home summaries: count,
attributed per-home totals, and min/median/max of mean daily use)
separately. Intervals are never pooled across homes, and cross-home
aggregates over a handful of homes are descriptive labels, not
population estimates. The household id is an anonymous label chosen by
the household (lowercase slug); it must not be a name or address.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "SchemaError", "FlatTariff", "TimeOfUseTariff", "rate194_tariff",
    "tariff_from_spec", "load_household_intervals", "load_household_bills",
    "HouseholdDataset", "summarize_household", "build_household_report",
    "report_bytes", "REPORT_SCHEMA",
]

REPORT_SCHEMA = "wattwise-household-report/1"

_INTERVAL_COLUMNS = {"timestamp", "kwh"}
_PSEG_COLUMNS = {"Start", "kWh", "kW"}
_BILL_COLUMNS = {"period_start", "period_end", "kwh", "bill_usd"}
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class SchemaError(ValueError):
    """A household file does not match its documented schema."""


def _check_columns(frame: pd.DataFrame, expected: set, what: str) -> None:
    got = set(frame.columns)
    if got != expected:
        raise SchemaError(
            f"{what} columns must be exactly {sorted(expected)}; "
            f"got {sorted(got)}")


def _validate_series(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise SchemaError("interval file has no rows")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise SchemaError("interval timestamps did not parse as timestamps")
    if frame.index.has_duplicates:
        raise SchemaError("duplicate interval timestamps")
    if not frame.index.is_monotonic_increasing:
        raise SchemaError("interval timestamps are not strictly increasing")
    values = frame["kwh"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise SchemaError("interval kwh contains non-finite values")
    if (values < 0).any():
        raise SchemaError("interval kwh contains negative values")
    return frame


def load_household_intervals(path, input_format: str = "canonical") -> pd.DataFrame:
    """Load and validate an interval CSV (see module docstring).

    Returns a DataFrame indexed by timestamp with one column, `kwh`.
    """
    if input_format == "canonical":
        frame = pd.read_csv(path)
        _check_columns(frame, _INTERVAL_COLUMNS, "interval")
        try:
            stamps = pd.to_datetime(frame["timestamp"])
        except (ValueError, TypeError) as exc:
            raise SchemaError(f"unparseable interval timestamp: {exc}") from exc
        if stamps.isna().any():
            raise SchemaError("interval file has a blank timestamp")
        try:
            kwh = pd.to_numeric(frame["kwh"], errors="raise")
        except (ValueError, TypeError) as exc:
            raise SchemaError(f"non-numeric interval kwh: {exc}") from exc
        out = pd.DataFrame({"kwh": kwh.to_numpy(dtype=float)},
                           index=pd.DatetimeIndex(stamps))
        out.index.name = "timestamp"
        return _validate_series(out)
    if input_format == "pseg":
        frame = pd.read_csv(path)
        _check_columns(frame, _PSEG_COLUMNS, "pseg interval")
        try:
            stamps = pd.to_datetime(frame["Start"],
                                    format="%m/%d/%Y %I:%M:%S %p")
            kwh = pd.to_numeric(frame["kWh"], errors="raise")
            pd.to_numeric(frame["kW"], errors="raise")
        except (ValueError, TypeError) as exc:
            raise SchemaError(f"invalid pseg interval export: {exc}") from exc
        out = pd.DataFrame({"kwh": kwh.to_numpy(dtype=float)},
                           index=pd.DatetimeIndex(stamps))
        out.index.name = "timestamp"
        return _validate_series(out)
    raise SchemaError(f"unknown interval format {input_format!r}")


def load_household_bills(path) -> pd.DataFrame:
    """Load and validate a bill CSV (see module docstring)."""
    frame = pd.read_csv(path)
    _check_columns(frame, _BILL_COLUMNS, "bill")
    try:
        starts = pd.to_datetime(frame["period_start"])
        ends = pd.to_datetime(frame["period_end"])
        kwh = pd.to_numeric(frame["kwh"], errors="raise")
        usd = pd.to_numeric(frame["bill_usd"], errors="raise")
    except (ValueError, TypeError) as exc:
        raise SchemaError(f"invalid bill file: {exc}") from exc
    if starts.isna().any() or ends.isna().any():
        raise SchemaError("bill file has a blank period date")
    out = pd.DataFrame({
        "period_start": starts, "period_end": ends,
        "kwh": kwh.astype(float), "bill_usd": usd.astype(float),
    }).sort_values("period_start").reset_index(drop=True)
    for _, row in out.iterrows():
        if not row["period_end"] > row["period_start"]:
            raise SchemaError("bill period_end must be after period_start")
        for col in ("kwh", "bill_usd"):
            value = row[col]
            if pd.notna(value) and (not math.isfinite(value) or value < 0):
                raise SchemaError(f"bill {col} must be finite and nonnegative")
    for prev, cur in zip(out.itertuples(), out.iloc[1:].itertuples()):
        if cur.period_start < prev.period_end:
            raise SchemaError("bill periods overlap")
    return out


def _nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


@dataclass(frozen=True)
class FlatTariff:
    """One all-in flat rate plus a daily service charge."""

    name: str
    delivery_per_kwh: float
    supply_per_kwh: float
    daily_service_charge: float

    def __post_init__(self):
        for f in ("delivery_per_kwh", "supply_per_kwh", "daily_service_charge"):
            _nonnegative(f, getattr(self, f))

    def marginal_price(self, timestamp=None) -> float:
        return self.delivery_per_kwh + self.supply_per_kwh

    def energy_charge(self, kwh: float, timestamp=None) -> float:
        return float(kwh) * self.marginal_price(timestamp)

    def fixed_charge(self, days: int) -> float:
        return days * self.daily_service_charge


@dataclass(frozen=True)
class TimeOfUseTariff:
    """Weekday peak-window time-of-use tariff from declared parameters.

    Peak is `peak_start_hour <= hour < peak_end_hour` on
    `peak_weekdays` (Mon=0), excluding any date in `holidays`.
    Prices are all-in $/kWh (delivery + supply kept separate so a rate
    sheet maps onto the fields directly).
    """

    name: str
    peak_delivery_per_kwh: float
    offpeak_delivery_per_kwh: float
    peak_supply_per_kwh: float
    offpeak_supply_per_kwh: float
    daily_service_charge: float
    peak_start_hour: int = 15
    peak_end_hour: int = 19
    peak_weekdays: tuple = (0, 1, 2, 3, 4)
    holidays: frozenset = field(default_factory=frozenset)

    def __post_init__(self):
        for f in ("peak_delivery_per_kwh", "offpeak_delivery_per_kwh",
                  "peak_supply_per_kwh", "offpeak_supply_per_kwh",
                  "daily_service_charge"):
            _nonnegative(f, getattr(self, f))
        if not (0 <= self.peak_start_hour < self.peak_end_hour <= 24):
            raise ValueError("peak window must satisfy 0 <= start < end <= 24")

    def is_peak(self, timestamp) -> bool:
        ts = pd.Timestamp(timestamp)
        if ts.weekday() not in self.peak_weekdays:
            return False
        if ts.date() in self.holidays:
            return False
        return self.peak_start_hour <= ts.hour < self.peak_end_hour

    def marginal_price(self, timestamp) -> float:
        if self.is_peak(timestamp):
            return self.peak_delivery_per_kwh + self.peak_supply_per_kwh
        return self.offpeak_delivery_per_kwh + self.offpeak_supply_per_kwh

    def energy_charge(self, kwh: float, timestamp) -> float:
        return float(kwh) * self.marginal_price(timestamp)

    def fixed_charge(self, days: int) -> float:
        return days * self.daily_service_charge


class Rate194Tariff:
    """Adapter exposing PSEG Rate 194 through the uniform interface."""

    def __init__(self, supply_rate=None, holidays=None):
        from .rates import DEFAULT_SUPPLY_RATE, Rate194
        self._rate = Rate194(
            supply_rate=DEFAULT_SUPPLY_RATE if supply_rate is None else supply_rate,
            holidays=holidays)
        self.name = self._rate.name

    def is_peak(self, timestamp) -> bool:
        from .rates import is_peak_hour
        return is_peak_hour(pd.Timestamp(timestamp).to_pydatetime(),
                            self._rate.holidays)

    def marginal_price(self, timestamp) -> float:
        return self._rate.marginal_price(pd.Timestamp(timestamp).to_pydatetime())

    def energy_charge(self, kwh: float, timestamp) -> float:
        return self._rate.energy_charge(
            float(kwh), pd.Timestamp(timestamp).to_pydatetime())

    def fixed_charge(self, days: int) -> float:
        return self._rate.fixed_charge(days)


def rate194_tariff(supply_rate=None, holidays=None) -> Rate194Tariff:
    """The existing PSEG Rate 194 as a uniform-interface tariff."""
    return Rate194Tariff(supply_rate=supply_rate, holidays=holidays)


def tariff_from_spec(spec: dict):
    """Build a tariff from a JSON-able spec dict.

    ``{"kind": "flat", ...}`` takes the FlatTariff rate fields;
    ``{"kind": "tou", ...}`` takes the TimeOfUseTariff fields (peak
    hours and weekdays optional). Unknown kinds, missing fields and
    negative or non-finite rates raise SchemaError.
    """
    if not isinstance(spec, dict) or "kind" not in spec:
        raise SchemaError("tariff spec must be a dict with a 'kind'")
    kind = spec["kind"]
    fields_by_kind = {
        "flat": ("name", "delivery_per_kwh", "supply_per_kwh",
                 "daily_service_charge"),
        "tou": ("name", "peak_delivery_per_kwh", "offpeak_delivery_per_kwh",
                "peak_supply_per_kwh", "offpeak_supply_per_kwh",
                "daily_service_charge"),
    }
    if kind not in fields_by_kind:
        raise SchemaError(f"unknown tariff kind {kind!r}")
    required = fields_by_kind[kind]
    missing = [f for f in required if f not in spec]
    if missing:
        raise SchemaError(f"tariff spec is missing field(s) {missing}")
    kwargs = {}
    for key, value in spec.items():
        if key in ("kind",):
            continue
        if key in ("peak_weekdays",):
            value = tuple(int(v) for v in value)
        elif key not in ("name",):
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise SchemaError(f"tariff field {key} is not numeric") from exc
            if not math.isfinite(value) or value < 0:
                raise SchemaError(f"tariff field {key} must be finite and nonnegative")
        kwargs[key] = value
    try:
        return FlatTariff(**kwargs) if kind == "flat" else TimeOfUseTariff(**kwargs)
    except TypeError as exc:
        raise SchemaError(f"invalid tariff spec: {exc}") from exc


@dataclass
class HouseholdDataset:
    """One household's private files, loaded and validated.

    `household_id` is an anonymous label (lowercase slug) chosen by the
    household; it is the only identifier the report carries.
    """

    household_id: str
    intervals: pd.DataFrame
    tariff: object
    bills: pd.DataFrame = None

    def __post_init__(self):
        if not _ID_RE.match(self.household_id or ""):
            raise SchemaError(
                "household_id must be an anonymous lowercase slug "
                "(letters, digits, '-' or '_'), not a name or address")
        if not isinstance(self.intervals, pd.DataFrame) \
                or "kwh" not in self.intervals.columns \
                or not isinstance(self.intervals.index, pd.DatetimeIndex) \
                or self.intervals.empty:
            raise SchemaError("intervals must be a validated interval frame")
        _validate_series(self.intervals)


def _price_series(intervals: pd.DataFrame, tariff) -> pd.Series:
    return pd.Series(
        [tariff.marginal_price(ts) for ts in intervals.index],
        index=intervals.index, dtype=float)


def summarize_household(dataset: HouseholdDataset) -> dict:
    """Deterministic within-home summary of one household's files."""
    iv = dataset.intervals
    tariff = dataset.tariff
    first, last = iv.index.min(), iv.index.max()
    days = (last.date() - first.date()).days + 1
    total = float(iv["kwh"].sum())
    prices = _price_series(iv, tariff)
    energy = float((iv["kwh"] * prices).sum())
    fixed = float(tariff.fixed_charge(days))

    diffs = iv.index.to_series().diff().dropna()
    cadence = None
    if len(diffs):
        cadence = float(diffs.mode().iloc[0].total_seconds() / 60.0)
        cadence = int(cadence) if cadence == int(cadence) else cadence

    peak = None
    is_peak = getattr(tariff, "is_peak", None)
    if callable(is_peak):
        mask = np.array([bool(is_peak(ts)) for ts in iv.index])
        peak_kwh = float(iv["kwh"].to_numpy()[mask].sum())
        peak = {"peak_kwh": round(peak_kwh, 4),
                "peak_kwh_pct": round(100 * peak_kwh / total, 2) if total else 0.0}

    reconciliation = None
    if dataset.bills is not None:
        reconciliation = []
        for row in dataset.bills.itertuples():
            a, b = row.period_start, row.period_end
            seg = iv[(iv.index >= a) & (iv.index < b)]
            interval_kwh = float(seg["kwh"].sum())
            entry = {
                "period_start": str(a.date()), "period_end": str(b.date()),
                "interval_kwh": round(interval_kwh, 4),
                "billed_kwh": None if pd.isna(row.kwh) else float(row.kwh),
                "difference_pct": None,
                "reconstructed_usd": None,
                "billed_usd": None if pd.isna(row.bill_usd) else float(row.bill_usd),
                "reconstruction_difference_usd": None,
            }
            if pd.notna(row.kwh) and row.kwh:
                entry["difference_pct"] = float(
                    100 * (interval_kwh / float(row.kwh) - 1))
            if pd.notna(row.bill_usd):
                seg_prices = _price_series(seg, tariff) if len(seg) else prices.iloc[0:0]
                reconstructed = float((seg["kwh"] * seg_prices).sum()) \
                    + float(tariff.fixed_charge((b - a).days))
                entry["reconstructed_usd"] = reconstructed
                entry["reconstruction_difference_usd"] = float(
                    reconstructed - float(row.bill_usd))
            reconciliation.append(entry)

    return {
        "household_id": dataset.household_id,
        "tariff": str(getattr(tariff, "name", type(tariff).__name__)),
        "first": str(first), "last": str(last), "days": int(days),
        "intervals": int(len(iv)), "cadence_minutes": cadence,
        "total_kwh": round(total, 4),
        "mean_daily_kwh": round(total / days, 4),
        "peak": peak,
        "bill_estimate": {
            "energy_usd": round(energy, 2), "fixed_usd": round(fixed, 2),
            "total_usd": round(energy + fixed, 2),
        },
        "bill_reconciliation": reconciliation,
    }


def build_household_report(datasets) -> dict:
    """Within-home and cross-home results, reported separately.

    `within_home` maps each household id to its own summary.
    `cross_home` carries only labeled aggregates over those summaries;
    intervals are never pooled across homes.
    """
    datasets = list(datasets)
    if not datasets:
        raise SchemaError("at least one household is required")
    ids = [d.household_id for d in datasets]
    if len(set(ids)) != len(ids):
        raise SchemaError("household ids must be unique")
    within = {d.household_id: summarize_household(d)
              for d in sorted(datasets, key=lambda d: d.household_id)}
    daily = sorted(s["mean_daily_kwh"] for s in within.values())
    n = len(daily)
    median = daily[n // 2] if n % 2 else (daily[n // 2 - 1] + daily[n // 2]) / 2
    cross = {
        "household_count": len(within),
        "household_ids": sorted(within),
        "total_kwh_by_home": {h: within[h]["total_kwh"] for h in sorted(within)},
        "mean_daily_kwh": {"min": daily[0], "median": float(median),
                           "max": daily[-1]},
        "note": ("Aggregates are computed over the per-home summaries "
                 "above. Homes are never pooled into one series, and "
                 "these figures describe only the homes listed."),
    }
    return {
        "schema": REPORT_SCHEMA,
        "scope": ("Each home is summarized on its own files and its own "
                  "tariff. Cross-home figures are labeled aggregates "
                  "over the per-home summaries, reported separately "
                  "from the within-home results."),
        "within_home": within,
        "cross_home": cross,
    }


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
            raise ValueError("non-finite value cannot enter a household report")
        number = float(f"{number:.9g}")
        return 0.0 if number == 0 else number
    raise TypeError(f"unserializable value in report: {type(value)!r}")


def report_bytes(report: dict) -> bytes:
    """Canonical JSON bytes for a household report.

    Sorted keys and 9-significant-digit float rendering, the same
    convention as `decision_analysis.canonical_json_bytes`, so repeated
    runs over the same inputs produce identical bytes.
    """
    return json.dumps(_canonicalize(report), sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
