"""wattwise: home energy analytics for a Long Island house on PSEG.

Bill forecasting, time-of-use vs flat rate comparison, a
linear-programming load-shifting optimizer, appliance disaggregation,
anomaly detection, weather correlation, and what-if scenarios
(solar, EV, heat pump) — all against real PSEG Long Island
residential rate structures.
"""

from .rates import Rate180, Rate194, is_peak_hour, is_summer
from .greenbutton import read_greenbutton_xml
from .profile import make_demo_profile, make_demo_weather
from .appliances import APPLIANCES, FLEXIBLE_NAMES
from .disagg import disaggregate, learn_shapes
from .forecast import BillForecaster, forecast_month_bill
from .anomalies import daily_anomalies, explain_anomaly, typical_weekday_profile
from .weather import temperature_sensitivity, degree_days
from .optimize import optimize_load_shift, optimize_month
from .whatif import solar_scenario, ev_scenario, heat_pump_scenario
from .analyze import (monthly_bill, compare_plans, peak_share,
                      appliance_breakdown, savings_waterfall)

__all__ = [
    "Rate180", "Rate194", "is_peak_hour", "is_summer",
    "read_greenbutton_xml",
    "make_demo_profile", "make_demo_weather",
    "APPLIANCES", "FLEXIBLE_NAMES",
    "disaggregate", "learn_shapes",
    "BillForecaster", "forecast_month_bill",
    "daily_anomalies", "explain_anomaly", "typical_weekday_profile",
    "temperature_sensitivity", "degree_days",
    "optimize_load_shift", "optimize_month",
    "solar_scenario", "ev_scenario", "heat_pump_scenario",
    "monthly_bill", "compare_plans", "peak_share",
    "appliance_breakdown", "savings_waterfall",
]

__version__ = "0.1.0"
