# Changelog

## v1.2.0

### Added
- Real measured PSEG Long Island billing history in `demo/pseg_bills.csv`
  and `demo/pseg_appliance_estimates.csv`: six billed periods
  (Feb-Oct 2026, $715.52 total), measured kWh for Jul/Aug/Sep 2026
  (619.39 / 427.24 / 320.25), and the utility's appliance dollar
  estimates per period. All identifying fields stripped; aggregates
  only.
- `wattwise.measured` module: `load_billing_history`,
  `load_appliance_estimates`, `history_summary`, and
  `make_measured_profile()`, which anchors the seeded generator to
  the fully specified Sep 2026 period (320.25 kWh, $109.01 bill,
  $17 peak cost). Period totals, appliance mix and bills are
  measured; the intraday shape is modeled (seeded), and that split
  is stated everywhere the numbers appear.
- `demo/real_metrics.json` + `demo/real_profile.csv`: the engine run
  on the measured-calibrated Sep 2026 profile. Two honest findings
  surfaced: the tariff model prices the measured month $7.40 below
  the actual bill (taxes/fees/supply-rate timing it does not carry),
  and the measured end-use mix disagrees with the synthetic demo --
  real Sep usage is baseload 22%, HVAC 22%, laundry 19.5%, while the
  seeded profile put 33.9% into water heating. Modeled
  load-shifting saving on the measured mix is $1.12 (1.31%), far
  below the synthetic $5.18 (6.21%).

### Tests
- 6 new regression tests in `tests/test_measured.py` (history
  integrity, anchor calibration, totals reconciliation, and a PII
  scan of the shipped data files); full suite 63/63 green.

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
