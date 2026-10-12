"""End-use appliance library.

Each appliance carries the metadata the optimizer and the disaggregator
need: typical draw, whether its load is shiftable, and the hours it may
run in. Flexible loads (dishwasher, laundry, water heater top-up) are
the levers the LP optimizer pulls; inflexible loads stay put.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Appliance:
    name: str
    power_kw: float          # typical draw while running
    flexible: bool           # can the optimizer reschedule it?
    window: tuple = (0, 24)  # allowed hours [start, end)
    notes: str = ""


APPLIANCES = [
    Appliance("baseload", 0.25, False, (0, 24),
              "Fridge, standby, always-on"),
    Appliance("hvac", 3.50, False, (0, 24),
              "Heating/cooling, comfort-bound"),
    Appliance("water_heater", 4.50, True, (0, 24),
              "Thermal storage: heat any time, use any time"),
    Appliance("dishwasher", 1.10, True, (0, 24),
              "Run after dinner, or overnight"),
    Appliance("laundry", 2.80, True, (0, 24),
              "Washer+dryer, 4x a week"),
    Appliance("cooking", 1.60, False, (0, 24),
              "Meals happen when they happen"),
    Appliance("lighting_plug", 0.55, False, (0, 24),
              "Evening lights and devices"),
]

FLEXIBLE_NAMES = [a.name for a in APPLIANCES if a.flexible]

BY_NAME = {a.name: a for a in APPLIANCES}


def flexible_daily_demand(end_use, date) -> dict:
    """Total kWh per flexible appliance on a given date (from end-use data)."""
    day = end_use.loc[end_use.index.date == date]
    return {name: float(day[name].sum()) for name in FLEXIBLE_NAMES}
