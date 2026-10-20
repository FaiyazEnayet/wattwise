# Changelog

## v1.3.0

### Added
- `demo/pseg_rates_detailed.csv`: the full line-item rate breakdown of
  13 billed periods (Oct 2025-Oct 2026), transcribed from the utility's
  own bill detail pages: delivery and power-supply rates for peak and
  off-peak, basic service and MFC charges, the DER rate, and the
  taxes/adjustments total per period. Rates and amounts only; no
  account or personal data.
- `Rate194History` in `wattwise.rates`: stores the exact per-period
  delivery + supply rates and prices any historical period exactly
  (basic service + MFC + energy + DER + actual taxes/adjustments).
  Validation: it reproduces all 13 measured bills with a worst error
  of $0.0051 (about half a cent, the utility's own per-line rounding),
  where the tariff model built from published approximations priced
  the September 2026 anchor month $7.43 low.
- `demo/pseg_daily_oct2026.csv`: daily peak demand (kW) and average
  temperature for Oct 5-19 2026 from the portal's demand chart
  (labeled peaks 3.72 / 2.18 / 1.65 / 1.08 kW are exact; other bars
  are chart estimates). The portal also reports 10.38 kWh for Oct 18
  (22% below the 90-day average) and 71.22 kWh for the week ending
  Oct 11.
- The new data files joined the pattern-based PII scan in
  `tests/test_measured.py`.

### Tests
- 4 new tests in `tests/test_rates.py` (history loads 13 periods,
  spot-checks the September 2026 rates, reproduces the September 2026
  bill within $0.50, reproduces all 13 bills within a cent);
  full suite 71/71 green.

### Measured interval data (Oct 19 follow-up)
- `demo/pseg_interval_15min.csv`: a full year of the household's
  15-minute smart-meter intervals (37,976 rows, 2025-09-19 to
  2026-10-19, 4,086.55 kWh; timestamps, kWh and kW only). This
  supersedes the seeded intraday shape: `make_measured_profile()`
  now builds the Sep 2026 profile from the real interval shape
  (each date keeps its own measured hourly profile where covered;
  Sep 2-18 use the measured fall season/day-type average). Daily
  totals still anchor to the measured sub-period trend, appliance
  totals to the utility's dollar-share mix, and the month to
  exactly 320.25 kWh. Only the appliance split within each hour
  remains modeled.
- `demo/pseg_interval_validation.csv`: per-period reconciliation of
  the interval sums against the billed kWh, plus the
  interval-derived peak kWh (Rate 194 window). Full-coverage periods
  run +0.7% to +3.7% above billed, consistent with per-interval
  rounding in the export; the Jan 2026 period is the outlier at
  +9.6% against a rounded chart estimate. The holiday-adjusted
  weekday 3-7 PM peak reproduces the utility's billed peak kWh
  within a few kWh per period (Sep 2026: 40.52 derived vs 39
  billed), validating the rate window in `wattwise.rates`.
- `demo/pseg_hourly_oct17.csv`: the portal's hourly costs for
  Saturday 2026-10-17 (all off-peak, $2.15 total), the first real
  intraday shape obtained. Against it, the seeded generator's
  Saturday stacked 47% of the day into 5-9 PM versus 23% measured;
  the intraday windows in `wattwise.profile` were corrected (weekend
  laundry late morning, wider evening spread, late-night tail).
- Measured seasonal shapes from the intervals: winter 7.1 kWh/day
  peaking mid-afternoon, summer 15.3 kWh/day with 29% of use between
  midnight and 6 AM (overnight cooling), spring 8.9, fall 10.0.
- `demo/pseg_appliance_estimates.csv`: Jan 2026 rows from the portal
  bill comparison (laundry $17, entertainment $10, cooking $8,
  refrigeration $6, lighting $3, taxes/fees $23); Aug 2026 values
  updated to the comparison values that reconcile exactly to the
  portal's $99 Aug energy charges. The Aug 2025 bill now carries its
  measured 472 kWh from the portal comparison.
- Honest finding: running the optimizer on the real interval shape
  cut the modeled load-shifting saving from $1.12 (seeded shape) to
  $0.58 (0.71%), because the real household puts less energy in the
  weekday peak window than the seeded profile assumed.
- The portal rate-plan page was verified to match `is_peak_hour` in
  `wattwise.rates` exactly (weekdays 3-7 PM peak, weekends and
  federal holidays all off-peak); no code change was needed.

### Tests
- 5 new tests in `tests/test_measured.py` (interval export loads
  37,976 rows / 4,086.55 kWh, per-period reconciliation errors match
  the measured values, interval peak reproduces billed peak kWh,
  the measured profile uses the real interval shape, hourly Oct 17
  loader); the Aug 2025 monthly row now asserts its 472 kWh from
  the portal comparison. Full suite 76/76 green.

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
  $19 peak cost). Period totals, appliance mix and bills are
  measured; the intraday shape is modeled (seeded), and that split
  is stated everywhere the numbers appear.
- `demo/real_metrics.json` + `demo/real_profile.csv`: the engine run
  on the measured-calibrated Sep 2026 profile. Two honest findings
  surfaced: the tariff model prices the measured month $7.43 below
  the actual bill (taxes/fees/supply-rate timing it does not carry),
  and the measured end-use mix disagrees with the synthetic demo --
  real Sep usage is HVAC 22.1%, baseload 21.9%, laundry 19.4%, while the
  seeded profile put 33.9% into water heating. Modeled
  load-shifting saving on the measured mix is $1.12 (1.31%), far
  below the synthetic $5.18 (6.21%).

### Data expansion (Oct 19 follow-up)
- Billing history extended to 13 billed periods (Aug 2025-Oct 2026,
  $1,394.07 total) by recovering seven earlier bills from payment
  confirmation emails; period dates for those are inferred from a
  ~9-day payment lag calibrated on periods where both bill and
  payment exist, and are labeled as inferred in the CSV.
- `demo/pseg_subperiod.csv`: 12 measured sub-period kWh increments
  (7-31 days each, Jul 2-Oct 17 2026) derived by differencing the
  cumulative weekly usage snapshots. `make_measured_profile()` now
  also anchors each sub-period's daily mean, so the September
  profile reproduces the measured within-month decline instead of a
  flat seeded trend.
- `demo/pseg_rates.csv`: tariff facts from utility emails
  (Time-of-Day Rate 194, peak 3-7 PM weekdays at ~2x, per-period
  peak-hour spend, and the Feb 2026 power-supply price increase).
- Correction: the Sep 2026 peak-hour cost is $19 (energy-insights
  email subject), not $17; the $17 figure in the email body is the
  above/below-average comparison value, previously misread.

### Portal data (Oct 19 evening follow-up)
- Fourteen exact bills from the utility portal's bill history
  (periods 2025-08-01 through 2026-10-02, $1,493.31 total) replace
  the payment-lag-inferred rows: exact bill dates and days-in-bill,
  with period dates chained backward from the verified 2026-07-02
  anchor, matching every portal day count and the verified
  2026-02-03 and 2025-10-02 boundaries. `demo/pseg_bills.csv` adds a
  bill_date column and a kwh_source label; a regression test
  asserts the full chain.
- `demo/pseg_monthly.csv` (new): monthly kWh totals, peak/off-peak
  splits and average temperature per billing period, read from the
  portal's yearly energy chart. Nine pre-Jul-2026 periods are
  chart estimates (2,400 kWh total), labeled as such; Jul/Aug/Sep
  2026 keep their measured totals.
- The September 2026 anchor is now fully specified from the
  portal's monthly summary: 281 kWh off-peak ($63) and 39 kWh peak
  ($19), $82 of energy charges plus $27 taxes/fees on the $109.01
  bill; peak usage down 7% versus the prior period. New module
  constants and `load_monthly()` in `wattwise.measured`;
  `history_summary()` now reports measured vs chart-estimated kWh
  and the seasonal envelope (619.39 kWh in the Jul 2026 period vs
  ~195 kWh in the Dec-Jan period, a 3.2x summer-to-winter swing
  tracking the chart's temperature curve).
- `demo/pseg_appliance_estimates.csv` carries the $27 taxes/fees row
  for Sep 2026 for reconciliation; the appliance-share mapping
  excludes it (taxes are not an end use).
- `demo/real_metrics.json` rebuilt: unchanged model outputs on the
  Sep profile ($101.58 TOU / $102.22 flat, $1.12 shifting saving),
  plus the measured peak/off-peak split, charge composition and
  the 14-period history.

### Tests
- 10 regression tests in `tests/test_measured.py` (history
  integrity with the chained portal periods, anchor calibration
  with the measured peak/off-peak split and charge composition,
  monthly chart loader with seasonal assertions, totals
  reconciliation, sub-period reconciliation and trend, rates file,
  and a pattern-based PII scan of the shipped data files that
  needs no real tokens); full suite 67/67 green.

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
