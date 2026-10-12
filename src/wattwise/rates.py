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

import datetime as dt

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


def is_summer(d: dt.date) -> bool:
    """PSEG LI summer season is June through September."""
    return 6 <= d.month <= 9


def is_peak_hour(timestamp: dt.datetime, holidays=frozenset()) -> bool:
    """True if a timestamp falls in Rate 194 peak hours.

    Peak = weekdays 3 PM-7 PM, excluding federal holidays.
    """
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
        delivery = self.delivery_rate(monthly_kwh, summer)
        return monthly_kwh * (delivery + self.supply_rate)

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
                 holidays=frozenset(FEDERAL_HOLIDAYS_2026)):
        self.supply_rate = supply_rate
        self.holidays = holidays

    def energy_charge(self, kwh: float, when: dt.datetime) -> float:
        summer = is_summer(when.date())
        peak = is_peak_hour(when, self.holidays)
        delivery = self._DELIVERY[(summer, peak)]
        supply = self._SUPPLY_MULT[(summer, peak)] * self.supply_rate
        return kwh * (delivery + supply)

    def marginal_price(self, when: dt.datetime) -> float:
        """All-in $/kWh for one more kWh at a given hour (for the optimizer)."""
        return self.energy_charge(1.0, when)

    def fixed_charge(self, days: int) -> float:
        return days * DAILY_SERVICE_CHARGE
