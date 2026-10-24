"""PSEG Long Island residential rate structures.

Modeled from PSEG Long Island's published residential rate documents:
Rate 194 (Time-of-Day, the standard rate since January 2024) and
Rate 180 (optional flat rate). Both carry a $0.54/day service charge.

Rate 194: peak is weekdays 3 PM - 7 PM, excluding federal holidays.
Off-peak is every other hour. Delivery and power-supply charges both
vary by season (summer = June-September) and by peak/off-peak.
Rate 180: one flat delivery charge (tiered in summer), 100% of the
monthly published power-supply rate.

The power-supply charge floats monthly with energy markets. The default
used here ($0.1392/kWh) is the published figure effective October 1,
2026; pass supply_rate to model any other month.
"""

import csv
import datetime as dt
from functools import lru_cache
import os

# Default power-supply rate: published PSEG LI figure, effective 2026-10-01.
DEFAULT_SUPPLY_RATE = 0.1392  # $/kWh

DAILY_SERVICE_CHARGE = 0.54  # $/day, both Rate 180 and Rate 194

# Federal holidays 2026 (peak hours do not apply on these days).
FEDERAL_HOLIDAYS_2026 = frozenset(
    [
        dt.date(2026, 1, 1),
        dt.date(2026, 1, 19),
        dt.date(2026, 2, 16),
        dt.date(2026, 5, 25),
        dt.date(2026, 6, 19),
        dt.date(2026, 7, 3),  # Independence Day observed
        dt.date(2026, 9, 7),
        dt.date(2026, 10, 12),
        dt.date(2026, 11, 11),
        dt.date(2026, 11, 26),
        dt.date(2026, 12, 25),
    ]
)

def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    """The n-th `weekday` (Mon=0) of the month."""
    first = dt.date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + dt.timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> dt.date:
    if month == 12:
        last = dt.date(year, 12, 31)
    else:
        last = dt.date(year, month + 1, 1) - dt.timedelta(days=1)
    return last - dt.timedelta(days=(last.weekday() - weekday) % 7)


def _observed(d: dt.date) -> dt.date:
    """Federal observed date: Sat -> Fri before, Sun -> Mon after."""
    if d.weekday() == 5:
        return d - dt.timedelta(days=1)
    if d.weekday() == 6:
        return d + dt.timedelta(days=1)
    return d


@lru_cache(maxsize=None)
def federal_holidays(year: int) -> frozenset:
    """US federal holidays whose observed date falls in `year`.

    Rule-based (weekday rules plus observed shifts for fixed-date
    holidays), matching the OPM schedule. Reproduces the archived
    FEDERAL_HOLIDAYS_2026 exactly for 2026; a fixed-date holiday whose
    observed date lands in the previous year (only Jan 1 on a Saturday,
    observed Dec 31) is counted in the year it is observed.
    """
    dates = {
        _observed(dt.date(year, 1, 1)),          # New Year's Day
        _nth_weekday(year, 1, 0, 3),             # MLK Day
        _nth_weekday(year, 2, 0, 3),             # Washington's Birthday
        _last_weekday(year, 5, 0),               # Memorial Day
        _observed(dt.date(year, 6, 19)),         # Juneteenth
        _observed(dt.date(year, 7, 4)),          # Independence Day
        _nth_weekday(year, 9, 0, 1),             # Labor Day
        _nth_weekday(year, 10, 0, 2),            # Columbus Day
        _observed(dt.date(year, 11, 11)),        # Veterans Day
        _nth_weekday(year, 11, 3, 4),            # Thanksgiving
        _observed(dt.date(year, 12, 25)),        # Christmas
    }
    dates = {d for d in dates if d.year == year}
    # Jan 1 of next year on a Saturday is observed Fri Dec 31 here.
    if dt.date(year + 1, 1, 1).weekday() == 5:
        dates.add(dt.date(year, 12, 31))
    return frozenset(dates)



def is_summer(d: dt.date) -> bool:
    """PSEG LI summer season is June through September."""
    return 6 <= d.month <= 9


def is_peak_hour(timestamp: dt.datetime, holidays=None) -> bool:
    """True if a timestamp falls in Rate 194 peak hours.

    Peak = weekdays 3 PM-7 PM, excluding federal holidays.
    `holidays=None` resolves the federal holidays of the timestamp's
    own year (WW4); pass an explicit set to pin a calendar.
    """
    if holidays is None:
        holidays = federal_holidays(timestamp.year)
    d = timestamp.date()
    if d in holidays:
        return False
    if timestamp.weekday() >= 5:  # Saturday, Sunday
        return False
    return 15 <= timestamp.hour < 19


class Rate180:
    """PSEG LI optional flat residential rate."""

    name = "Rate 180 (flat)"

    def __init__(self, supply_rate: float = DEFAULT_SUPPLY_RATE):
        self.supply_rate = supply_rate

    def delivery_rate(self, monthly_kwh: float, summer: bool) -> float:
        """Delivery $/kWh. The summer tier keys off the monthly total."""
        if summer:
            return 0.1021 if monthly_kwh <= 250 else 0.1294
        return 0.1021

    def energy_charge(self, monthly_kwh: float, summer: bool) -> float:
        """Total energy charge for a month: delivery + 100% supply."""
        if monthly_kwh < 0:
            raise ValueError("Energy must be nonnegative")
        # Incremental blocks: crossing 250 kWh does not reprice the first block.
        low = min(monthly_kwh, 250.0)
        high = max(monthly_kwh - 250.0, 0.0)
        delivery = (low * .1021 + high * .1294) if summer else monthly_kwh * .1021
        return delivery + monthly_kwh * self.supply_rate

    def marginal_price(self, when: dt.datetime, monthly_kwh: float = 0.0) -> float:
        """All-in $/kWh. Flat, so load shifting never pays under Rate 180."""
        return self.delivery_rate(monthly_kwh, is_summer(when.date())) + self.supply_rate

    def fixed_charge(self, days: int) -> float:
        return days * DAILY_SERVICE_CHARGE


class Rate194:
    """PSEG LI standard time-of-day residential rate."""

    name = "Rate 194 (time-of-day)"

    # (delivery $/kWh) by (season, peak?)
    _DELIVERY = {
        (True, True): 0.2127,  # summer peak
        (True, False): 0.1049,  # summer off-peak
        (False, True): 0.1809,  # nonsummer peak
        (False, False): 0.0891,  # nonsummer off-peak
    }
    # supply multiplier of the monthly published rate
    _SUPPLY_MULT = {
        (True, True): 1.9465,
        (True, False): 0.83,
        (False, True): 2.1036,
        (False, False): 0.83,
    }

    def __init__(self, supply_rate: float = DEFAULT_SUPPLY_RATE,
                 holidays=None):
        # holidays=None resolves the federal holidays of each
        # timestamp's own year (WW4); an explicit set is still honored.
        # The previous default pinned the 2026 set for every year.
        self.supply_rate = supply_rate
        self.holidays = holidays

    def energy_charge(self, kwh: float, when: dt.datetime) -> float:
        summer = is_summer(when.date())
        holidays = (self.holidays if self.holidays is not None
                    else federal_holidays(when.year))
        peak = is_peak_hour(when, holidays)
        delivery = self._DELIVERY[(summer, peak)]
        supply = self._SUPPLY_MULT[(summer, peak)] * self.supply_rate
        return kwh * (delivery + supply)

    def marginal_price(self, when: dt.datetime) -> float:
        """All-in $/kWh for one more kWh at a given hour (for the optimizer)."""
        return self.energy_charge(1.0, when)

    def fixed_charge(self, days: int) -> float:
        return days * DAILY_SERVICE_CHARGE


_DEMO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "..", "demo")


class Rate194History:
    """Exact per-period Rate 194 pricing from measured PSEG LI bills.

    Loads demo/pseg_rates_detailed.csv: the delivery and power-supply
    rates (peak and off-peak), basic service and MFC charges, DER rate,
    and the taxes/adjustments total, transcribed line-by-line from the
    utility's own bill details for 13 billed periods (rates only, no
    account or personal data).

    This prices a historical period exactly: basic service + MFC +
    energy (delivery + supply, peak and off-peak) + DER + the period's
    actual taxes and adjustments. Use it to validate the tariff model
    against utility reality instead of the published approximations.
    """

    name = "Rate 194 (time-of-day, measured history)"

    def __init__(self, path=None):
        self.path = path or os.path.join(_DEMO_DIR,
                                         "pseg_rates_detailed.csv")
        self.periods = {}
        with open(self.path) as f:
            for row in csv.DictReader(f):
                self.periods[row["bill_date"]] = {
                    k: float(row[k]) for k in (
                        "days", "kwh", "peak_kwh", "offpeak_kwh",
                        "basic_daily_rate", "basic_usd", "mfc_rate",
                        "mfc_usd", "del_peak_rate", "del_offpeak_rate",
                        "sup_peak_rate", "sup_offpeak_rate",
                        "delivery_usd", "supply_usd", "der_rate",
                        "der_usd", "other_taxes_usd", "taxes_usd",
                        "bill_usd")
                }
                self.periods[row["bill_date"]]["period_start"] = row[
                    "period_start"]
                self.periods[row["bill_date"]]["period_end"] = row[
                    "period_end"]

    def bill_for(self, bill_date, peak_kwh, offpeak_kwh) -> float:
        """Exact bill in dollars for a period's measured peak/off-peak kWh.

        bill_date is the YYYY-MM-DD bill date string from the CSV.
        """
        p = self.periods[bill_date]
        energy = (peak_kwh * (p["del_peak_rate"] + p["sup_peak_rate"])
                  + offpeak_kwh * (p["del_offpeak_rate"]
                                   + p["sup_offpeak_rate"]))
        return (p["basic_usd"] + p["mfc_usd"] + energy + p["der_usd"]
                + p["other_taxes_usd"])

    def validation_report(self):
        """Modeled vs actual bill for every stored period.

        Returns a list of dicts with bill_date, modeled, actual, error.
        """
        rows = []
        for bill_date, p in sorted(self.periods.items()):
            modeled = self.bill_for(bill_date, p["peak_kwh"],
                                    p["offpeak_kwh"])
            rows.append({
                "bill_date": bill_date,
                "period": f"{p['period_start']} to {p['period_end']}",
                "modeled": round(modeled, 2),
                "actual": round(p["bill_usd"], 2),
                "error": round(modeled - p["bill_usd"], 4),
            })
        return rows

    def max_error(self) -> float:
        """Worst absolute modeled-vs-actual error across all periods."""
        return max(abs(r["error"]) for r in self.validation_report())
