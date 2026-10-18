# Changelog

## v1.1.0

### Fixed
- Forecast time coordinates now persist across fit and predict: the model
  stores the calendar origin from fit and transforms every timestamp into
  elapsed days from it. Trend and Fourier phase are continuous at the
  fit/predict boundary instead of resetting to zero.
- For 56 days of exact 10+2t load, the next-week forecast is now
  122,124,...,134 kWh/day (was 10,12,...,22).

### Tests
- 2 new regression tests in `tests/test_forecast.py`; full suite 57/57 green.
