"""Tariff assumption sweep (E2): whether to recommend reviewing alternatives.

Control: the current standardized Rate 194 (the delivery schedule in
``decision_analysis.tariff_replay``, schedule version 2026.1, base supply
$0.139822/kWh). Alternatives: Rate 180 (flat) and Rate 195 (overnight).
Only documented power-supply (PSC) assumptions are perturbed: the supply
grid is the 13 realized monthly blended supply rates from the billed
periods plus the two published scalars (September 2026 base $0.139822/kWh,
October 2026 $0.1392/kWh).

The calculator prices each month from the documented standardized rate
table (delivery $/kWh and supply multiplier by plan, season and bucket).
Rate 194 prices peak hours at the peak rate and every other hour at the
off-peak rate, so the off-peak bucket covers night hours as well. Rate 195
prices night (10 PM-6 AM), peak and day off-peak separately. Rate 180
prices all kWh at the flat delivery rate, tiered in summer above 250 kWh.

Outputs are differentials (alternative minus control) and regret (the
positive part of a differential), never a promised saving. The
recommendation rule is fixed before evaluation: review alternatives when
any alternative costs less than the control at any documented supply
value, when the exact break-even supply falls inside the documented
range, or when the annual differential at the base supply is within
$10 of zero (a near tie, so the choice is load-shape sensitive).
Otherwise the control stands across the documented range. Every result
carries the eligibility caveats: taxes and adjustments excluded,
billing-cycle tier proration excluded, standardized scenario only,
enrollment eligibility applies.
"""

from __future__ import annotations

import math
import os

import pandas as pd

from .decision_analysis import DEFAULT_SUPPLY, monthly_supply_history, peak_mask

CONTROL_PLAN = "194"
ALTERNATIVE_PLANS = ("180", "195")
PLANS = ("180", "194", "195")

# Published power-supply scalars ($/kWh): September 2026 base PSC and
# October 2026 published figure (see rates.DEFAULT_SUPPLY_RATE).
PUBLISHED_SUPPLY_SCALARS = (0.139822, 0.1392)

# Service charge of the standardized scenario ($/day), matching the
# subtotal in decision_analysis.tariff_replay and build_report. It is
# identical for every plan, so it cancels out of every differential.
SERVICE_USD_PER_DAY = 0.56

# An annual differential this close to zero at the base supply counts as
# a near tie: the choice is load-shape sensitive and worth reviewing.
NEAR_TIE_USD = 10.0

# (delivery $/kWh, supply multiplier) by (summer, bucket), transcribed
# from the standardized schedule in decision_analysis.tariff_replay.
_RATE_TABLE = {
    "194": {
        (True, "peak"): (0.2217, 1.9505),
        (False, "peak"): (0.1885, 2.134),
        (True, "offpeak"): (0.1093, 0.83),
        (False, "offpeak"): (0.0929, 0.83),
    },
    "195": {
        (True, "night"): (0.0452, 0.60),
        (False, "night"): (0.045, 0.60),
        (True, "peak"): (0.2979, 1.7419),
        (False, "peak"): (0.2440, 1.9688),
        (True, "offpeak"): (0.1388, 1.0),
        (False, "offpeak"): (0.0929, 1.0),
    },
}

# Rate 180 flat delivery $/kWh: nonsummer flat, summer tiered above 250 kWh.
_RATE180_DELIVERY = 0.1064
_RATE180_SUMMER_HIGH = 0.1348
_RATE180_TIER_KWH = 250.0

CAVEATS = (
    "Standardized 2026 delivery schedule replay; energy and service "
    "subtotal only.",
    "Taxes, adjustments, and billing-cycle tier proration are excluded; "
    "enrollment eligibility for Rate 180 and Rate 195 applies.",
    "Differentials are conditional on the supply assumption shown; no "
    "figure here is a promised saving.",
    "The control is the current standardized Rate 194 at the base supply "
    "$0.139822/kWh.",
)

_DEMO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "..", "demo")


def _validate_supply(supply: float) -> None:
    if not math.isfinite(supply) or supply < 0:
        raise ValueError("Supply rate must be finite and nonnegative")


def _validate_plan(plan: str) -> None:
    if plan not in PLANS:
        raise ValueError(f"Unknown plan {plan!r}; expected one of "
                         "180, 194, 195")


def _validate_kwh(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _validate_month(row: dict) -> None:
    for key in ("month", "days", "summer", "peak_kwh", "offpeak_kwh",
                "night_kwh"):
        if key not in row:
            raise ValueError(f"month is missing {key!r}")
    if int(row["days"]) <= 0:
        raise ValueError("month days must be positive")
    for key in ("peak_kwh", "offpeak_kwh", "night_kwh"):
        _validate_kwh(float(row[key]), key)


def _validate_months(months) -> list:
    months = list(months)
    if not months:
        raise ValueError("sweep needs at least one month")
    for row in months:
        _validate_month(row)
    return months


def documented_supply_grid(details_path=None) -> list:
    """Documented supply grid: realized monthly rates plus published scalars.

    The realized rates are the blended supply price (supply_usd / kWh)
    of each billed period in demo/pseg_rates_detailed.csv, via
    monthly_supply_history. The published scalars are the September 2026
    base PSC and the October 2026 published figure. Returns a sorted list.
    """
    if details_path is None:
        details_path = os.path.join(_DEMO_DIR, "pseg_rates_detailed.csv")
    details = pd.read_csv(details_path)
    history = monthly_supply_history(details)
    grid = sorted(set(history.values()) | set(PUBLISHED_SUPPLY_SCALARS))
    if not grid or any((not math.isfinite(v)) or v <= 0 for v in grid):
        raise ValueError("documented supply grid must be positive and finite")
    return grid


def monthly_energy_cost(*, peak_kwh: float, offpeak_kwh: float,
                        night_kwh: float, summer: bool, plan: str,
                        supply: float) -> float:
    """Energy cost for one month from the documented standardized table.

    ``offpeak_kwh`` is day off-peak energy only (non-peak, non-night).
    Rate 194 prices every non-peak hour at the off-peak rate, so its
    off-peak bucket is ``offpeak_kwh + night_kwh``.
    """
    _validate_plan(plan)
    _validate_supply(supply)
    _validate_kwh(peak_kwh, "peak_kwh")
    _validate_kwh(offpeak_kwh, "offpeak_kwh")
    _validate_kwh(night_kwh, "night_kwh")
    if plan == "194":
        d_pk, m_pk = _RATE_TABLE["194"][(bool(summer), "peak")]
        d_op, m_op = _RATE_TABLE["194"][(bool(summer), "offpeak")]
        return (peak_kwh * (d_pk + m_pk * supply)
                + (offpeak_kwh + night_kwh) * (d_op + m_op * supply))
    if plan == "195":
        d_nt, m_nt = _RATE_TABLE["195"][(bool(summer), "night")]
        d_pk, m_pk = _RATE_TABLE["195"][(bool(summer), "peak")]
        d_op, m_op = _RATE_TABLE["195"][(bool(summer), "offpeak")]
        return (night_kwh * (d_nt + m_nt * supply)
                + peak_kwh * (d_pk + m_pk * supply)
                + offpeak_kwh * (d_op + m_op * supply))
    total = peak_kwh + offpeak_kwh + night_kwh
    if summer:
        low = min(total, _RATE180_TIER_KWH)
        high = max(total - _RATE180_TIER_KWH, 0.0)
        delivery = low * _RATE180_DELIVERY + high * _RATE180_SUMMER_HIGH
    else:
        delivery = total * _RATE180_DELIVERY
    return delivery + total * supply


def monthly_profile_from_intervals(iv, start, end) -> list:
    """Monthly load profile from measured 15-minute intervals.

    ``iv`` is an intervals frame as read by decision_analysis.read_intervals
    (DatetimeIndex, ``kwh`` column). Months are calendar months in
    [start, end). Peak hours follow peak_mask; night is 10 PM-6 AM; day
    off-peak is the remainder. Each row carries month, days, summer,
    peak_kwh, offpeak_kwh, night_kwh and total_kwh.
    """
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if start >= end:
        raise ValueError("interval range must be non-empty")
    seg = iv[(iv.index >= start) & (iv.index < end)]
    if len(seg) == 0:
        raise ValueError("no intervals in the requested range")
    rows = []
    for (year, month), group in seg.groupby([seg.index.year,
                                             seg.index.month]):
        pm = peak_mask(group.index)
        nm = (group.index.hour >= 22) | (group.index.hour < 6)
        peak = float(group.loc[pm, "kwh"].sum())
        night = float(group.loc[nm, "kwh"].sum())
        total = float(group.kwh.sum())
        offpeak = total - peak - night
        if offpeak < -1e-6:
            raise ValueError("peak and night shares exceed the month total")
        rows.append({
            "month": f"{year}-{month:02d}",
            "days": int(group.index.normalize().nunique()),
            "summer": month in (6, 7, 8, 9),
            "peak_kwh": peak,
            "offpeak_kwh": max(offpeak, 0.0),
            "night_kwh": night,
            "total_kwh": total,
        })
    return rows


def annual_energy_cost(months, plan: str, supply: float) -> float:
    """Annual energy cost for a monthly profile under one plan."""
    _validate_plan(plan)
    _validate_supply(supply)
    months = _validate_months(months)
    return sum(monthly_energy_cost(
        peak_kwh=row["peak_kwh"], offpeak_kwh=row["offpeak_kwh"],
        night_kwh=row["night_kwh"], summer=row["summer"], plan=plan,
        supply=supply) for row in months)


def annual_differential_usd(months, plan: str, supply: float) -> float:
    """Annual cost of an alternative plan minus the Rate 194 control.

    The service charge is identical for every plan, so it cancels out
    of the differential.
    """
    _validate_plan(plan)
    if plan == CONTROL_PLAN:
        raise ValueError("the control plan cannot be an alternative")
    return (annual_energy_cost(months, plan, supply)
            - annual_energy_cost(months, CONTROL_PLAN, supply))


def _with_peak_share(months, peak_share: float) -> list:
    """Rebuild a profile with a uniform peak share of each month's total.

    Night kWh stays fixed; day off-peak absorbs the change. Raises when
    the share would push day off-peak below zero in any month.
    """
    rebuilt = []
    for row in months:
        total = (row["peak_kwh"] + row["offpeak_kwh"] + row["night_kwh"])
        peak = total * peak_share
        offpeak = total - peak - row["night_kwh"]
        if offpeak < -1e-9:
            raise ValueError("peak share exceeds the non-night load")
        rebuilt.append({**row, "peak_kwh": peak,
                        "offpeak_kwh": max(offpeak, 0.0)})
    return rebuilt


def _annual_differential(months, plan: str, supply: float,
                         peak_share=None) -> float:
    use = months if peak_share is None else _with_peak_share(months,
                                                             peak_share)
    return (annual_energy_cost(use, plan, supply)
            - annual_energy_cost(use, CONTROL_PLAN, supply))


def break_even_supply(months, plan: str):
    """Exact supply rate where an alternative ties the Rate 194 control.

    The annual differential is affine in the supply rate, so the
    break-even is the exact linear solve. Returns None when the two
    plans are parallel (no crossing). The value may fall outside the
    documented range; callers flag that separately.
    """
    _validate_plan(plan)
    if plan == CONTROL_PLAN:
        raise ValueError("the control plan cannot be an alternative")
    months = _validate_months(months)
    at_zero = _annual_differential(months, plan, 0.0)
    slope = _annual_differential(months, plan, 1.0) - at_zero
    if slope == 0:
        return None
    return -at_zero / slope


def peak_share_threshold(months, plan: str, supply: float):
    """Uniform peak share where an alternative ties the control.

    Holds each month's total and night kWh fixed and varies the peak
    share uniformly across months; day off-peak absorbs the change.
    Returns None when no feasible uniform share (0 to the largest share
    every month allows) ties the plans.
    """
    _validate_plan(plan)
    if plan == CONTROL_PLAN:
        raise ValueError("the control plan cannot be an alternative")
    _validate_supply(supply)
    months = _validate_months(months)
    caps = []
    for row in months:
        total = (row["peak_kwh"] + row["offpeak_kwh"] + row["night_kwh"])
        if total > 0:
            caps.append(1.0 - row["night_kwh"] / total)
    if not caps:
        return None
    p_max = min(caps)
    if p_max <= 0:
        return None
    at_zero = _annual_differential(months, plan, supply, peak_share=0.0)
    slope = (_annual_differential(months, plan, supply, peak_share=p_max)
             - at_zero) / p_max
    if slope == 0:
        return None
    threshold = -at_zero / slope
    return threshold if 0.0 <= threshold <= p_max else None


def sweep_tariff_assumptions(months, *, plans=ALTERNATIVE_PLANS,
                             supply_grid=None,
                             base_supply=DEFAULT_SUPPLY,
                             service_per_day=SERVICE_USD_PER_DAY) -> dict:
    """Sweep documented supply assumptions for the alternative plans.

    ``months`` is a monthly profile (see monthly_profile_from_intervals).
    Returns a dict with the per-plan differentials, break-even supplies,
    regret exposure and worst case at every documented supply value, the
    peak-share sensitivity at the base supply, the eligibility caveats,
    and a recommendation of "review" or "stay".
    """
    months = _validate_months(months)
    for plan in plans:
        _validate_plan(plan)
        if plan == CONTROL_PLAN:
            raise ValueError("the control plan cannot be an alternative")
    _validate_supply(base_supply)
    if not math.isfinite(service_per_day) or service_per_day < 0:
        raise ValueError("service charge must be finite and nonnegative")
    grid = sorted(supply_grid if supply_grid is not None
                  else documented_supply_grid())
    if not grid or any((not math.isfinite(v)) or v <= 0 for v in grid):
        raise ValueError("supply grid must be positive and finite")

    plan_results = {}
    for plan in plans:
        by_supply = []
        for supply in grid:
            monthly_rows = []
            for row in months:
                control_energy = monthly_energy_cost(
                    peak_kwh=row["peak_kwh"],
                    offpeak_kwh=row["offpeak_kwh"],
                    night_kwh=row["night_kwh"], summer=row["summer"],
                    plan=CONTROL_PLAN, supply=supply)
                alt_energy = monthly_energy_cost(
                    peak_kwh=row["peak_kwh"],
                    offpeak_kwh=row["offpeak_kwh"],
                    night_kwh=row["night_kwh"], summer=row["summer"],
                    plan=plan, supply=supply)
                service = int(row["days"]) * service_per_day
                control = control_energy + service
                alternative = alt_energy + service
                differential = alternative - control
                monthly_rows.append({
                    "month": row["month"],
                    "control_usd": control,
                    "alternative_usd": alternative,
                    "differential_usd": differential,
                    "regret_usd": max(differential, 0.0),
                })
            annual_control = sum(r["control_usd"] for r in monthly_rows)
            annual_alt = sum(r["alternative_usd"] for r in monthly_rows)
            by_supply.append({
                "supply_usd_per_kwh": supply,
                "annual_control_usd": annual_control,
                "annual_alternative_usd": annual_alt,
                "annual_differential_usd": annual_alt - annual_control,
                "annual_regret_usd": sum(r["regret_usd"]
                                         for r in monthly_rows),
                "monthly": monthly_rows,
            })
        break_even = break_even_supply(months, plan)
        in_range = break_even is not None and grid[0] <= break_even <= grid[-1]
        worst_regret = max(r["annual_regret_usd"] for r in by_supply)
        max_monthly = max(r["regret_usd"] for r in by_supply
                          for r in r["monthly"])
        base_diff = _annual_differential(months, plan, base_supply)
        wins_anywhere = any(r["annual_differential_usd"] < 0
                            for r in by_supply)
        near_tie = abs(base_diff) <= NEAR_TIE_USD
        threshold = peak_share_threshold(months, plan, base_supply)
        review = bool(in_range or wins_anywhere or near_tie)
        if in_range:
            reason = (f"Rate {plan} ties Rate 194 at a supply of "
                      f"${break_even:.4f}/kWh, inside the documented range "
                      f"${grid[0]:.4f}-${grid[-1]:.4f}/kWh, so the choice "
                      "depends on the supply assumption and is worth "
                      "reviewing.")
        elif wins_anywhere:
            reason = (f"Rate {plan} costs less than Rate 194 at some "
                      "documented supply values "
                      f"(annual differential ${base_diff:,.2f} at the base "
                      "supply), so the choice is worth reviewing; the "
                      "figure is conditional on the supply assumption.")
        elif near_tie:
            reason = (f"Rate {plan} is within ${NEAR_TIE_USD:.2f} a year of "
                      f"Rate 194 at the base supply (${base_diff:,.2f}), so "
                      "the choice is load-shape sensitive and worth "
                      "reviewing.")
        else:
            reason = (f"Rate 194 costs less than Rate {plan} at every "
                      "documented supply value and by more than "
                      f"${NEAR_TIE_USD:.2f} a year at the base supply "
                      f"(${base_diff:,.2f}).")
        plan_results[plan] = {
            "by_supply": by_supply,
            "break_even_supply_usd_per_kwh": break_even,
            "break_even_in_documented_range": in_range,
            "worst_case_regret_usd": worst_regret,
            "max_monthly_regret_usd": max_monthly,
            "peak_share_threshold_at_base_supply": threshold,
            "recommends_review": review,
            "review_reason": reason,
        }
    review_plans = [p for p, r in plan_results.items()
                    if r["recommends_review"]]
    if review_plans:
        recommendation = "review"
        recommendation_reason = (
            "Review " + ", ".join(f"Rate {p}" for p in review_plans) +
            " against the Rate 194 control: the choice turns on the "
            "supply assumption or the load shape, under the caveats below.")
    else:
        recommendation = "stay"
        recommendation_reason = (
            "Rate 194 costs less than every alternative at every "
            "documented supply value by a clear margin; no review is "
            "indicated under the documented assumptions.")
    return {
        "control": CONTROL_PLAN,
        "base_supply_usd_per_kwh": base_supply,
        "supply_grid_usd_per_kwh": grid,
        "service_usd_per_day": service_per_day,
        "caveats": list(CAVEATS),
        "plans": plan_results,
        "recommendation": recommendation,
        "recommendation_reason": recommendation_reason,
    }
