"""E2: tariff assumption sweep, whether to recommend reviewing alternatives.

The control is the current standardized Rate 194. The sweep perturbs only
documented power-supply (PSC) assumptions, the realized monthly blended
supply rates from the billed periods plus the two published scalars, and
compares Rate 194 with the documented alternatives Rate 180 (flat) and
Rate 195 (overnight). It reports the differential, the exact break-even
supply, monthly negative-regret exposure and the worst case, and decides
whether the household should review switching plans.

The calculator is checked against an independent line-item oracle: the
existing tariff replay engine pricing synthetic 15-minute intervals built
from the same monthly profile. The oracle and the calculator share only
the documented rate table; their computation paths are independent.

Fixtures are synthetic seeded counterfactuals, except for the one test
that runs the sweep on the household's actual 12-month load from the
demo intervals. No savings figure is presented as promised anywhere:
results report differentials and regret, and every recommendation
carries the eligibility caveats.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wattwise.decision_analysis import (DEFAULT_SUPPLY, peak_mask,
                                        read_intervals, tariff_replay)
from wattwise.tariff_sweep import (
    documented_supply_grid,
    monthly_energy_cost,
    monthly_profile_from_intervals,
    sweep_tariff_assumptions,
)

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260114
GRID = documented_supply_grid()
BASE_SUPPLY = DEFAULT_SUPPLY
TOLERANCE_USD = 0.01


def _seeded_profiles(seed=SEED, n=20):
    """20 annual counterfactual profiles: 12 months of seeded load."""
    rng = np.random.default_rng(seed)
    profiles = []
    for _ in range(n):
        months = []
        for m in range(1, 13):
            total = float(rng.uniform(200, 800))
            peak = total * float(rng.uniform(0.05, 0.60))
            night = min(total * float(rng.uniform(0.05, 0.50)),
                        total - peak)
            months.append({
                "month": f"2026-{m:02d}",
                "days": pd.Timestamp(2026, m, 1).days_in_month,
                "summer": m in (6, 7, 8, 9),
                "peak_kwh": peak,
                "offpeak_kwh": total - peak - night,
                "night_kwh": night,
                "total_kwh": total,
            })
        profiles.append(months)
    return profiles


def _oracle_monthly_costs(profile, plan, supply):
    """Line-item oracle: tariff replay on synthetic intervals.

    Each month's peak, night and day off-peak kWh are spread evenly
    over the matching 15-minute slots of that month, then priced by
    the existing replay engine. Returns {month: energy cost}.
    """
    index = pd.date_range("2026-01-01", "2027-01-01", freq="15min",
                          inclusive="left")
    peak = peak_mask(index)
    night = (index.hour >= 22) | (index.hour < 6)
    kwh = np.zeros(len(index))
    by_month = {}
    for row in profile:
        m = int(row["month"].split("-")[1])
        in_month = index.month == m
        pk_idx = in_month & peak
        nt_idx = in_month & night
        op_idx = in_month & ~peak & ~night
        assert pk_idx.sum() > 0 and nt_idx.sum() > 0 and op_idx.sum() > 0
        kwh[pk_idx] = row["peak_kwh"] / pk_idx.sum()
        kwh[nt_idx] = row["night_kwh"] / nt_idx.sum()
        kwh[op_idx] = row["offpeak_kwh"] / op_idx.sum()
        by_month[row["month"]] = m
    frame = pd.DataFrame({"kwh": kwh}, index=index)
    costs, _ = tariff_replay(frame, plan, supply)
    monthly = costs.groupby(costs.index.to_period("M")).sum()
    return {f"2026-{m:02d}": float(monthly[pd.Period(f"2026-{m:02d}", "M")])
            for m in by_month.values()}


def test_sweep_matches_oracle_on_seeded_fixtures():
    """On 20 seeded fixtures, calculator matches the oracle within $0.01."""
    grid_points = [GRID[0], BASE_SUPPLY, GRID[-1]]
    for i, profile in enumerate(_seeded_profiles()):
        for plan in ("180", "195"):
            for supply in grid_points:
                oracle = _oracle_monthly_costs(profile, plan, supply)
                for row in profile:
                    got = monthly_energy_cost(
                        peak_kwh=row["peak_kwh"],
                        offpeak_kwh=row["offpeak_kwh"],
                        night_kwh=row["night_kwh"],
                        summer=row["summer"], plan=plan, supply=supply)
                    want = oracle[row["month"]]
                    assert abs(got - want) < TOLERANCE_USD, (
                        f"fixture {i} plan {plan} supply {supply} "
                        f"month {row['month']}: {got:.4f} vs oracle {want:.4f}")
                annual_got = sum(
                    monthly_energy_cost(
                        peak_kwh=row["peak_kwh"],
                        offpeak_kwh=row["offpeak_kwh"],
                        night_kwh=row["night_kwh"],
                        summer=row["summer"], plan=plan, supply=supply)
                    for row in profile)
                assert abs(annual_got - sum(oracle.values())) < TOLERANCE_USD, (
                    f"fixture {i} plan {plan} supply {supply}: annual "
                    f"{annual_got:.4f} vs oracle {sum(oracle.values()):.4f}")


def test_break_even_is_exact_and_reversal_found():
    """Break-even supply is exact, and at least one reversal is found."""
    reversals = 0
    for i, profile in enumerate(_seeded_profiles()):
        result = sweep_tariff_assumptions(profile, supply_grid=GRID)
        for plan in ("180", "195"):
            entry = result["plans"][plan]
            be = entry["break_even_supply_usd_per_kwh"]
            assert be is not None, (
                f"fixture {i} plan {plan}: no break-even supply found")
            from wattwise.tariff_sweep import annual_differential_usd
            if be >= 0:
                assert abs(annual_differential_usd(profile, plan, be)) < 0.01, (
                    f"fixture {i} plan {plan}: differential at break-even "
                    f"{be:.4f} is not zero")
                low = annual_differential_usd(
                    profile, plan, max(be - 0.005, 0.0))
                high = annual_differential_usd(profile, plan, be + 0.005)
                assert (low > 0) != (high > 0), (
                    f"fixture {i} plan {plan}: differential does not "
                    "change sign across the break-even supply")
            else:
                # Crossing below the physical range: one sign throughout.
                low = annual_differential_usd(profile, plan, 0.0)
                high = annual_differential_usd(profile, plan, GRID[-1])
                assert (low > 0) == (high > 0), (
                    f"fixture {i} plan {plan}: negative break-even but "
                    "the differential still changes sign in range")
            if entry["break_even_in_documented_range"]:
                reversals += 1
                row = next(r for r in entry["by_supply"]
                           if r["supply_usd_per_kwh"] == GRID[0])
                top = next(r for r in entry["by_supply"]
                           if r["supply_usd_per_kwh"] == GRID[-1])
                assert (row["annual_differential_usd"] > 0) != (
                    top["annual_differential_usd"] > 0), (
                    f"fixture {i} plan {plan}: break-even in range but "
                    "the differential does not change sign across the grid")
    assert reversals >= 1, (
        "no fixture has a break-even supply inside the documented range")


def _real_profile():
    iv = read_intervals(ROOT / "demo" / "pseg_interval_15min.csv")
    return monthly_profile_from_intervals(iv, "2025-10-01", "2026-10-01")


def test_real_household_load():
    """The actual 12-month load: 180 is a near tie, 195 wins conditionally."""
    profile = _real_profile()
    assert len(profile) == 12
    result = sweep_tariff_assumptions(profile, supply_grid=GRID)
    base_180 = next(r for r in result["plans"]["180"]["by_supply"]
                    if r["supply_usd_per_kwh"] == BASE_SUPPLY)
    base_195 = next(r for r in result["plans"]["195"]["by_supply"]
                    if r["supply_usd_per_kwh"] == BASE_SUPPLY)
    # Rate 180 loses to Rate 194 by a few dollars a year: a near tie, so
    # the choice is load-shape sensitive and worth reviewing.
    assert 0.0 < base_180["annual_differential_usd"] < 10.0
    # Rate 195 wins the annual total, but only conditionally: the
    # differential is reported per assumption, never as promised saving.
    assert -20.0 < base_195["annual_differential_usd"] < 0.0
    # Even the winning plan has losing months: monthly regret is positive.
    assert result["plans"]["195"]["max_monthly_regret_usd"] > 0.0
    # 180 breaks even below the documented range; 195 at no physical supply.
    assert (result["plans"]["180"]["break_even_supply_usd_per_kwh"]
            < GRID[0])
    assert (result["plans"]["195"]["break_even_supply_usd_per_kwh"] < 0.0)
    # The household sits just below the 180 peak-share threshold.
    threshold = result["plans"]["180"]["peak_share_threshold_at_base_supply"]
    assert threshold is not None and 0.0 < threshold < 1.0
    actual_share = (sum(r["peak_kwh"] for r in profile)
                    / sum(r["total_kwh"] for r in profile))
    assert actual_share < threshold
    assert result["recommendation"] == "review"
    assert result["caveats"]


def test_documented_supply_grid():
    """The grid is the realized monthly rates plus the published scalars."""
    assert GRID == sorted(GRID)
    assert len(GRID) == 15
    assert abs(GRID[0] - 0.109953) < 1e-6
    assert abs(GRID[-1] - 0.179409) < 1e-6
    assert BASE_SUPPLY in GRID
    assert 0.1392 in GRID
    assert all(v > 0 for v in GRID)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError, match="nonnegative"):
        monthly_energy_cost(peak_kwh=-1.0, offpeak_kwh=0.0, night_kwh=0.0,
                            summer=False, plan="194", supply=0.14)
    with pytest.raises(ValueError, match="nonnegative"):
        monthly_energy_cost(peak_kwh=0.0, offpeak_kwh=0.0, night_kwh=0.0,
                            summer=False, plan="194", supply=-0.1)
    with pytest.raises(ValueError, match="finite"):
        monthly_energy_cost(peak_kwh=0.0, offpeak_kwh=0.0, night_kwh=0.0,
                            summer=False, plan="194", supply=float("nan"))
    with pytest.raises(ValueError, match="Unknown plan"):
        monthly_energy_cost(peak_kwh=0.0, offpeak_kwh=0.0, night_kwh=0.0,
                            summer=False, plan="199", supply=0.14)
    with pytest.raises(ValueError, match="at least one month"):
        sweep_tariff_assumptions([])
    with pytest.raises(ValueError, match="Unknown plan"):
        sweep_tariff_assumptions(_seeded_profiles()[0], plans=("199",))


def test_no_promised_saving_language():
    """No figure is presented as a promised saving anywhere in the result."""
    import json
    result = sweep_tariff_assumptions(_seeded_profiles()[0],
                                      supply_grid=GRID)
    text = json.dumps(result, default=str)
    assert "guaranteed" not in text.lower()

    def _keys(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                yield key
                yield from _keys(value)
        elif isinstance(obj, list):
            for value in obj:
                yield from _keys(value)
    assert not any("saving" in key for key in _keys(result))
    assert result["caveats"]
    assert "tax" in " ".join(result["caveats"]).lower()
    assert "eligib" in " ".join(result["caveats"]).lower()


def test_peak_share_threshold_is_exact():
    """The peak-share threshold is verified by rebuilding at that share."""
    from wattwise.tariff_sweep import annual_differential_usd
    found = 0
    for profile in _seeded_profiles():
        result = sweep_tariff_assumptions(profile, supply_grid=GRID)
        for plan in ("180", "195"):
            threshold = result["plans"][plan][
                "peak_share_threshold_at_base_supply"]
            if threshold is None:
                continue
            found += 1
            rebuilt = []
            for row in profile:
                total = row["total_kwh"]
                peak = total * threshold
                rebuilt.append({**row, "peak_kwh": peak,
                                "offpeak_kwh": total - peak
                                - row["night_kwh"]})
            assert abs(annual_differential_usd(rebuilt, plan,
                                               BASE_SUPPLY)) < 0.01, (
                f"plan {plan}: differential at peak-share threshold "
                f"{threshold:.4f} is not zero")
    assert found >= 1
