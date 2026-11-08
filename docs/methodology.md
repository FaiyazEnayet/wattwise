# WattWise analysis methods

WattWise is a household electricity decision tool. It works from original smart-meter intervals and utility bill line items, and every counterfactual it produces is labeled as one. The point is to find changes worth testing, then measure whether they did anything. A fitted chart or a lower bill in a mild month is not evidence that an intervention saved money.

## Household boundary

The household is on Long Island, New York. Cooling uses individual room AC units. No heating, EV or solar savings are silently added to the household results.

## Sources and precedence

- `demo/pseg_interval_15min.csv` contains 39,800 original quarter-hour readings from September 19, 2025 through November 7, 2026 at 14:45, totaling 4,264.91 kWh. The separately supplied shorter export is an exact 2,940-row subset, not additional energy. A later export, supplied in November 2026, repeats those same 2,940 readings exactly and extends the file by 1,824 new readings; in its stitched overlap regions it also carried a zero-valued placeholder row beside the real reading for each timestamp, and the placeholders were dropped rather than summed.
- `demo/pseg_rates_detailed.csv` contains 13 detailed bills and their energy rates, service, MFC, DER and other adjustment amounts. These line-item kWh supersede earlier chart estimates for the same period. An additional earlier bill remains in the existing billing-history file.
- `demo/pseg_appliance_estimates.csv` contains utility estimates in dollars. They are estimates, not measurements, and a cost share is never treated as a kWh share in the analysis.
- `demo/household_weather.csv` contains town-level ERA5 weather reanalysis supplied by Open-Meteo. Its metadata file records coordinates, query, provider and units. This is modeled weather, not a sensor inside the home.
- The 2026 residential rate booklet is linked from [PSEG rate information](https://www.psegliny.com/aboutpseglongisland/ratesandtariffs/rateinformation). The snapshot comparison uses its delivery schedule and a declared base power-supply assumption. Rates and tariff eligibility can change.

All model results and input fingerprints are frozen in `demo/analysis.json`. The HTML embeds that report, the chart library and fonts once. It needs no application server or network connection to render its main content.

## Time and data quality

The CSV uses local wall-clock timestamps without UTC offsets or repeated-hour flags. The analysis preserves those original values as exported, without inventing offsets.

Billing uses start-inclusive and end-exclusive periods. The previous reconciliation included the following period's first day. After correcting this and using detailed bill kWh instead of older chart reads, the 12 fully covered detailed periods are all within 0.5% of their billed energy. September 2 to October 2 contains 320.75 raw kWh against 320 billed kWh, and 39.08 raw peak-window kWh against 39 billed kWh. Remaining rounding or export differences are disclosed rather than rescaling the original measurements.

The published weekday 15:00 through 19:00 peak window uses a federal holiday calendar across every represented year. A 2026-only holiday set must not price Thanksgiving 2025 as an ordinary weekday.

## Bill reconstruction and changes

Peak and off-peak kWh are priced separately at the recorded delivery plus supply rates. Basic service, MFC, DER and the remaining adjustments are then added. The maximum unrounded reconstruction difference across the 13 detailed bills is approximately $0.005136. That is an accounting check on stored inputs, not an out-of-sample forecast.

The portal's rounded September grouping of approximately $82 energy plus $27 fees is not treated as $27 of pure taxes. The detailed ledger separates service, usage-linked fees and other adjustments. Rate-specific future taxes are not inferred by applying an unexplained flat percentage.

Bill-change attribution uses exact Shapley arithmetic over four factors: energy quantity, peak share, energy prices, and service/other line items. It averages each factor's incremental effect over all 24 factor orders. Contributions sum to the reconstructed bill difference, up to floating-point error. This explains a financial identity; it does not prove why occupants changed their usage. The final category includes changes in period length and usage-linked non-energy charges, which are named in the ledger.

## Rate comparison and regret

The same October 2025 through September 2026 calendar-year load is replayed under Rates 180, 194 and 195. The comparison applies the published 2026 delivery schedule to all those readings with one editable base supply assumption. At the initial $0.139822/kWh assumption, annual energy-plus-service subtotals are approximately $1,152.62, $1,148.17 and $1,135.45 respectively. So the cheapest scenario beats Rate 194 by only $12.71 over the year. These are modeled subtotals, not the historical bills that alternative enrollment would have produced.

Taxes, adjustments, eligibility and changing monthly supply prices are excluded from this standardized comparison. Summer Rate 180 blocks reset on calendar months; actual bill-period threshold proration requires the complete applicable tariff. The first block is priced incrementally. Crossing 250 kWh must not apply the higher marginal delivery rate retroactively to every kWh. The chart shows each month's difference against Rate 194 so a small annual advantage cannot hide seasonal losses.

## Savings and scheduling

The savings laboratory works on raw intervals within a selected fully covered bill. The user declares removable persistent watts, the fraction of peak energy that might be flexible, a scenario total-load limit, and optional equipment cost.

Persistent reduction is applied first and is capped by each interval's measured energy. Flexible energy is then removed proportionally from the remaining peak-window load. The same energy is placed into available off-peak intervals on that day within the chosen total-load limit. The aggregate experiment allows earlier or later off-peak placement on the same day; it is not a release-time-aware appliance schedule. The separate task scheduler enforces an explicit earliest start, finish deadline and consecutive quarter-hour cycle.

Counterfactual money uses that period's recorded average peak/off-peak rates. Service stays fixed. Removed kWh also reduce the stored usage-linked MFC and DER amount per kWh; other adjustments are held fixed. Shifting conserves energy, and no tax reduction is promised on the moved kWh. Combined persistent reduction and shifting operate on the same modified load to avoid double counting. Existing intervals above a chosen load cap are preserved and disclosed, rather than pretending the scenario made the original home feasible.

Annual figures are a flat extrapolation of the selected period at the same rates, not a seasonal forecast, and payback inherits that assumption (financing, maintenance and degradation excluded). Equipment is not worth buying when the scenario benefit is too small or too uncertain to justify its cost. Essential refrigeration and other necessary always-on loads are not automatically removable.

A two-scenario decision calculator compares intervention versus no purchase and computes the expected value of perfect information. Probabilities are user-declared assumptions. Benefits use an explicit horizon, discount rate, upfront cost and constant observed September energy-price mix. Perfect information is an upper bound, not the value of an actual imperfect meter.

The AC calculator uses user-entered input power, avoided runtime and affected days. It reports an off-peak to peak energy-cost range from stored rates. Default input values are assumptions. Metered temperature association is not converted into a claimed thermostat or AC saving.

## Cooling and load patterns

The descriptive weather model relates daily energy to weekday indicators and cooling degree-days above 65°F. Its cooling slope is constrained nonnegative. Heating degree-days are not assigned to the household's heating cost. Two hundred moving-week bootstrap resamples report an 80% slope range.

On this dataset, the fitted cooling slope is approximately 0.724 kWh per cooling degree-day, with an 80% bootstrap range of 0.584 to 0.862. The final 56-day conditional test has MAE approximately 1.990 kWh/day, worse than the fixed earlier-weekday baseline at 1.544 kWh/day. The dashboard shows that failure as-is. The model stays in the tool as a descriptive one; it is not promoted as verified AC attribution. The conditional evaluation uses realized weather, not weather forecasts available at issuance time.

Daily-pattern clustering uses three deterministic k-means groups on normalized complete 24-hour profiles. Groups describe load shapes only. They are not occupancy labels or discovered appliances. Weather residual alerts use a robust three-MAD threshold on the descriptive fit; they are candidates for investigation, not validated live alerts or evidence of waste.

## Forecast protocol

The candidates are weekly persistence, the median of recent same weekdays, and a small regularized autoregression using earlier lags, a recent mean and weekday indicators. Every walk-forward origin sees only earlier meter observations.

The first 56 origins select the model. The final 28 origins are held out for reporting. In this run, autoregression wins the selection period, while weekly persistence is slightly better on the final holdout: approximately 1.753 versus 1.749 kWh/day MAE. Both numbers stay in the report, and the simpler model winning is stated plainly.

Cumulative 7-, 14- and 28-day errors are also evaluated. Every calibration target ends before the final holdout. The final holdout admits 17 complete seven-day origins and 10 fourteen-day origins per model, and no complete twenty-eight-day origin: every 28-day window in the current holdout crosses the excluded November DST day, so that horizon reports null metrics instead of an estimate from incomplete windows. Overlapping origins are dependent and one monthly result is not a calibration study. Seven-day total bounds use prior cumulative errors directly.

Future multiday forecasts are recursive. Their empirical bands resample seven-day blocks of selection-period residuals, keeping errors within each sampled block together. The reported daily holdout coverage is 75% against an 80% target, so the bands run slightly narrow in this holdout. One-step daily error is not misreported as seven-day forecast accuracy. A sum of daily interval bounds in the open-bill panel is labeled a conservative envelope, not a calibrated monthly prediction interval.

The open-bill projection runs only while the billing history declares a period as unbilled. The October 2 to November 3 period was projected that way (its last partial day completed from a recent same-weekday daily estimate minus energy already observed, future energy priced from the declared tariff assumption and the period's observed weighted price mix) until its bill was issued in November 2026. The history currently declares no open period, so the report carries the projection as unavailable rather than inventing a period boundary. A projection is a planning subtotal excluding taxes, adjustments and supply-price uncertainty, not an issued bill.

## Intervention notebook

Interventions, dates, cost and confounder notes are stored in local browser storage, with JSON import and export.

The basic comparison needs at least 14 complete days in each 28-day window before and after the actual intervention date. It reports an unadjusted daily energy difference with a moving-week variation band. The separate matched comparison searches the previous 90 days for controls on the same weekday and within 5°F. It uses up to four nearest controls, requires at least three per treated day, and refuses an adjusted estimate unless at least 14 post-change days match.

Matching uses only earlier controls. Its block band is conditional on those matches and does not incorporate control-selection uncertainty or eliminate unmeasured confounding. Household changes, occupants, travel, equipment and seasonal drift remain alternative explanations. An observational decrease is never auto-labeled a verified causal saving. A small change may require a plug meter or longer repeated trial to distinguish it from normal variation.

## Reproduce

```sh
python -m pip install -e ".[dev]"
python -m wattwise analyze --root . --out demo/analysis.json
python scripts/make_dashboard.py
python -m pytest tests -q
node --test tests/test_dashboard_math.cjs
```

For UI-only changes, `python scripts/make_dashboard.py --reuse-analysis` can reuse the frozen report. Rebuild without that flag after any data or analysis change. Weather fetching is an explicit optional step: `python scripts/fetch_household_weather.py --latitude <lat> --longitude <lon>` (coordinates are private; the script has no committed defaults). No scheduled ingestion is silently enabled. The original `wattwise insights` command and legacy model modules remain available; their heuristic bands and seeded examples are distinct from this validated dashboard path.

## Provenance appendix: evidence manifest and frozen outputs

Every build of the decision report records an evidence manifest inside `demo/analysis.json`. For each input the report reads (the 15-minute intervals, the line-item bill details, the billing history, the utility appliance estimates and the ERA5 weather file) the manifest records the SHA-256, byte size, data-row count, source role, parser schema and timestamp convention. It also records the pricing assumptions (tariff schedule version 2026.1, the declared September 2026 base supply price and the published rate source), a SHA-256 fingerprint of the analysis source code, and the Python, pandas, NumPy and SciPy versions. A single manifest digest is derived over the canonical ordered entries. The manifest contains only repository-relative paths; it carries no coordinates or other private values. The git commit and the library environment at build time are recorded as annotations and are deliberately excluded from the digest and from byte-equivalence checks: the digest identifies what the report was built from (inputs, pricing, code), not the machine state it was built on, because the commit that records the frozen outputs always differs from the commit they were generated at and library patch versions drift between machines.

Two checks keep the frozen outputs honest. First, `scripts/make_dashboard.py --reuse-analysis` recomputes the manifest from the current inputs and code and refuses to reuse the stored report when the digests differ, or when the stored report has no manifest. Second, `scripts/verify_frozen_outputs.py` rebuilds the report and dashboard in a temporary copy of the worktree and fails when the committed `demo/analysis.json` differs from the clean rebuild (canonical JSON comparison: sorted keys, canonical float rendering, non-finite values rejected) or when the compiled dashboard structure differs (a normalized digest with the embedded report payload masked; `--strict` compares the dashboard byte for byte). The workflow step that runs the verifier in CI is a separate one-line addition, recorded with the change that introduced the verifier; with that step in place, a stale committed report fails the build instead of sitting unnoticed.

All values published from the frozen report reconcile to the sections above: regenerating the report after the tariff versioning work changed no published number; it added the tariff version fields and this manifest, and nothing else.

