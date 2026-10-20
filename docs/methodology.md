# WattWise analysis methods

WattWise is a household electricity decision tool built around original smart-meter intervals, utility bill line items and explicitly labeled counterfactuals. Its purpose is to identify changes worth testing and to measure their results. A fitted chart or a lower seasonal bill is not evidence that an intervention saved money.

## Household boundary

The household is in Port Jefferson, New York, ZIP 11777. Cooling uses individual room AC units. Space heating is paid by the apartment building and is outside the electricity-saving boundary. There is no EV. Solar ownership has not been confirmed. No heating, EV or solar savings are silently added to the household results.

## Sources and precedence

- `demo/pseg_interval_15min.csv` contains 37,976 original quarter-hour readings from September 19, 2025 through October 19, 2026 at 14:45, totaling 4,086.55 kWh. The separately supplied shorter export is an exact 2,940-row subset, not additional energy.
- `demo/pseg_rates_detailed.csv` contains 13 detailed bills and their energy rates, service, MFC, DER and other adjustment amounts. These line-item kWh supersede earlier chart estimates for the same period. An additional earlier bill remains in the existing billing-history file.
- `demo/pseg_appliance_estimates.csv` contains utility estimates in dollars. These are not measured appliance energy. Cost shares are never treated as measured kWh shares in the new analysis.
- `demo/household_weather.csv` contains town-level ERA5 weather reanalysis supplied by Open-Meteo. Its metadata file records coordinates, query, provider and units. This is modeled weather, not a sensor inside the home.
- The 2026 residential rate booklet is linked from [PSEG rate information](https://www.psegliny.com/aboutpseglongisland/ratesandtariffs/rateinformation). The snapshot comparison uses its delivery schedule and a declared base power-supply assumption. Rates and tariff eligibility can change.

All model results and input fingerprints are frozen in `demo/analysis.json`. The HTML embeds that report, the chart library and fonts once. It needs no application server or network connection to render its main content.

## Time and data quality

The CSV uses local wall-clock timestamps without UTC offsets or repeated-hour flags. The analysis preserves those original values. America/New_York is the assumed service-area timezone for expected daily counts, not a claim that the original export has established a UTC mapping.

The spring transition has four absent wall-clock slots, including 01:45, which does not match a simple conventional 02:00 through 02:45 skip. The autumn transition has 96 rows rather than a clearly represented 100-interval local day. Both transition dates are excluded from daily models pending utility confirmation. The partial final export day is also excluded from daily training, and remains visible in raw charts. Missing readings are not silently replaced by zero. Duplicate timestamps, negative imports without explicit import/export semantics, or nonfinite values fail with an actionable error.

Billing uses start-inclusive and end-exclusive periods. The previous reconciliation included the following period's first day. After correcting this and using detailed bill kWh instead of older chart reads, the 12 fully covered detailed periods are all within 0.5% of their billed energy. September 2 to October 2 contains 320.75 raw kWh against 320 billed kWh, and 39.08 raw peak-window kWh against 39 billed kWh. Remaining rounding or export differences are disclosed rather than rescaling the original measurements.

The published weekday 15:00 through 19:00 peak window uses a federal holiday calendar across every represented year. A 2026-only holiday set must not price Thanksgiving 2025 as an ordinary weekday.

## Bill reconstruction and changes

Energy is calculated separately for peak and off-peak kWh using the recorded delivery plus supply rate. Basic service, MFC, DER and remaining adjustments are then added. The maximum unrounded reconstruction difference across the 13 detailed bills is approximately $0.005136. This is an accounting check using stored bill inputs, not independent out-of-sample forecasting.

The portal's rounded September grouping of approximately $82 energy plus $27 fees is not treated as $27 of pure taxes. The detailed ledger separates service, usage-linked fees and other adjustments. Rate-specific future taxes are not inferred by applying an unexplained flat percentage.

Bill-change attribution uses exact Shapley arithmetic over four factors: energy quantity, peak share, energy prices, and service/other line items. It averages each factor's incremental effect over all 24 factor orders. Contributions sum to the reconstructed bill difference, up to floating-point error. This explains a financial identity; it does not prove why occupants changed their usage. The final category includes changes in period length and usage-linked non-energy charges, which are named in the ledger.

## Rate comparison and regret

The same October 2025 through September 2026 calendar-year load is replayed under Rates 180, 194 and 195. The comparison applies the published 2026 delivery schedule to all those readings with one editable base supply assumption. At the initial $0.139822/kWh assumption, annual energy-plus-service subtotals are approximately $1,152.62, $1,148.17 and $1,135.45 respectively. Thus the cheapest scenario is only $12.71 below Rate 194. These are modeled subtotals, not the historical bills that alternative enrollment would have produced.

Taxes, adjustments, eligibility and changing monthly supply prices are excluded from this standardized comparison. Summer Rate 180 blocks reset on calendar months; actual bill-period threshold proration requires the complete applicable tariff. The first block is priced incrementally. Crossing 250 kWh must not apply the higher marginal delivery rate retroactively to every kWh. The chart shows each month's difference against Rate 194 so a small annual advantage cannot hide seasonal losses.

## Savings and scheduling

The savings laboratory works on raw intervals within a selected fully covered bill. The user declares removable persistent watts, the fraction of peak energy that might be flexible, a scenario total-load limit, and optional equipment cost.

Persistent reduction is applied first and is capped by each interval's measured energy. Flexible energy is then removed proportionally from the remaining peak-window load. The same energy is placed into available off-peak intervals on that day within the chosen total-load limit. The aggregate experiment allows earlier or later off-peak placement on the same day; it is not a release-time-aware appliance schedule. The separate task scheduler enforces an explicit earliest start, finish deadline and consecutive quarter-hour cycle.

Counterfactual money uses that period's recorded average peak/off-peak rates. Service stays fixed. Removed kWh also reduce the stored usage-linked MFC and DER amount per kWh; other adjustments are held fixed. No tariff-tax reduction is promised. Shifting conserves energy. Combined persistent reduction and shifting operate on the same modified load to avoid double counting. Existing intervals above a chosen load cap are preserved and disclosed, rather than pretending the scenario made the original home feasible.

An annual extrapolation is clearly a same-rate daily extrapolation from the selected period, not a seasonal annual forecast. Payback inherits that assumption and excludes financing, maintenance and degradation. Do not recommend buying equipment when the scenario benefit is too small or uncertain to justify its cost. Essential refrigeration and other necessary always-on loads are not automatically removable.

A two-scenario decision calculator compares intervention versus no purchase and computes the expected value of perfect information. Probabilities are user-declared assumptions. Benefits use an explicit horizon, discount rate, upfront cost and constant observed September energy-price mix. Perfect information is an upper bound, not the value of an actual imperfect meter.

The AC calculator uses user-entered input power, avoided runtime and affected days. It reports an off-peak to peak energy-cost range from stored rates. Default input values are assumptions. Metered temperature association is not converted into a claimed thermostat or AC saving.

## Cooling and load patterns

The descriptive weather model relates daily energy to weekday indicators and cooling degree-days above 65°F. Its cooling slope is constrained nonnegative. Heating degree-days are not assigned to the household's heating cost. Two hundred moving-week bootstrap resamples report an 80% slope range.

On this dataset, the fitted cooling slope is approximately 0.728 kWh per cooling degree-day, with an 80% bootstrap range of 0.584 to 0.853. The final 56-day conditional test has MAE approximately 2.744 kWh/day, worse than the fixed earlier-weekday baseline at 1.846 kWh/day. The dashboard shows that failure prominently. The model remains descriptive and is not promoted as a verified AC attribution model. The conditional evaluation uses realized weather, not weather forecasts available at issuance time.

Daily-pattern clustering uses three deterministic k-means groups on normalized complete 24-hour profiles. Groups describe load shapes only. They are not occupancy labels or discovered appliances. Weather residual alerts use a robust three-MAD threshold on the descriptive fit; they are candidates for investigation, not validated live alerts or evidence of waste.

## Forecast protocol

The candidates are weekly persistence, the median of recent same weekdays, and a small regularized autoregression using earlier lags, a recent mean and weekday indicators. Every walk-forward origin sees only earlier meter observations.

The first 56 origins select the model. The final 28 origins are held out for reporting. In this run, autoregression wins the selection period, while weekly persistence is slightly better on the final holdout: approximately 1.568 versus 1.564 kWh/day MAE. Both results remain visible; no claim of neural-model superiority is made.

Cumulative 7-, 14- and 28-day errors are also evaluated. Every calibration target ends before the final holdout. The final holdout admits 22 complete seven-day origins, 15 fourteen-day origins and one twenty-eight-day origin per model; overlapping origins are dependent and one monthly result is not a calibration study. Seven-day total bounds use prior cumulative errors directly.

Future multiday forecasts are recursive. Their empirical bands resample seven-day blocks of selection-period residuals, keeping errors within each sampled block together. The reported daily holdout coverage is 100% against an 80% target, so the bands are conservative in this holdout. One-step daily error is not misreported as seven-day forecast accuracy. A sum of daily interval bounds in the open-bill panel is labeled a conservative envelope, not a calibrated monthly prediction interval.

The open bill is the recorded October 2 to November 3 period. The last partial day is completed using a recent same-weekday daily estimate minus energy already observed. Future energy is priced using the declared tariff assumption and the open period's observed weighted price mix. This is a planning subtotal excluding taxes, adjustments and supply-price uncertainty, not an issued bill.

## Intervention notebook

Interventions, dates, cost and confounder notes are stored in local browser storage, with JSON import and export. No cloud synchronization or appliance-control permission is implied.

The basic comparison needs at least 14 complete days in each 28-day window before and after the actual intervention date. It reports an unadjusted daily energy difference with a moving-week variation band. The separate matched comparison searches the previous 90 days for controls on the same weekday and within 5°F. It uses up to four nearest controls, requires at least three per treated day, and refuses an adjusted estimate unless at least 14 post-change days match.

Matching uses only earlier controls. Its block band is conditional on those matches and does not incorporate control-selection uncertainty or eliminate unmeasured confounding. Household changes, occupants, travel, equipment and seasonal drift remain alternative explanations. WattWise never labels an observational decrease a verified causal dollar saving automatically. A small change may require a plug meter or longer repeated trial to distinguish it from normal variation.

## Reproduce

```sh
python -m pip install -e ".[dev]"
python -m wattwise analyze --root . --out demo/analysis.json
python scripts/make_dashboard.py
python -m pytest tests -q
node --test tests/test_dashboard_math.cjs
```

For UI-only changes, `python scripts/make_dashboard.py --reuse-analysis` can reuse the frozen report. Rebuild without that flag after any data or analysis change. Weather fetching is an explicit optional step: `python scripts/fetch_household_weather.py`. No scheduled ingestion is silently enabled. The original `wattwise insights` command and legacy model modules remain available; their heuristic bands and seeded examples are distinct from this validated dashboard path.

## What would strengthen the evidence next

1. Confirm the utility export's DST labeling, acquire original timestamp-offset metadata, and replace exclusion with a validated UTC mapping.
2. Add more seasons and independently logged interventions. One household and approximately one annual cycle cannot establish broad population generalization.
3. Obtain plug-meter AC input and runtime measurements, plus optional indoor temperature, before optimizing comfort or claiming appliance-level savings.
4. Archive tariff revisions and forecast-issuance weather snapshots; validate full bill adjustments before making enrollment or monthly-dollar promises.
5. Measure multiday forecast errors on many historical origins and calibrate horizon-specific cumulative bands. Add a conservative recommendation gate based on lower-bound benefit and observed adherence.
6. Use repeated, comfortable on/off or timing trials where practical, and disclose experiment exclusions before reading the outcome. There is no promise of publication or a universal portfolio score.
